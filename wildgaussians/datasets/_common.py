import warnings
import logging
import os
import struct
import numpy as np
import PIL.Image
import PIL.ExifTags
from tqdm import tqdm
from typing import Optional, TypeVar, Tuple, Union, List, Sequence, Dict, cast, overload
from ..types import Dataset, Literal, Cameras, UnloadedDataset


def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (np.random.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=a.dtype,
    )
    return np.eye(3, dtype=a.dtype) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def viewmatrix(
    lookdir,
    up,
    position,
    lock_up = False,
):
    """Construct lookat view matrix."""
    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def orthogonal_dir(a, b): 
        return normalize(np.cross(a, b))

    vecs = [None, normalize(up), normalize(lookdir)]
    # x-axis is always the normalized cross product of `lookdir` and `up`.
    vecs[0] = orthogonal_dir(vecs[1], vecs[2])
    # Default is to lock `lookdir` vector, if lock_up is True lock `up` instead.
    ax = 2 if lock_up else 1
    # Set the not-locked axis to be orthogonal to the other two.
    vecs[ax] = orthogonal_dir(vecs[(ax + 1) % 3], vecs[(ax + 2) % 3])
    m = np.stack(vecs + [position], axis=1)
    return m


TDataset = TypeVar("TDataset", bound=Union[Dataset, UnloadedDataset])


def single(xs):
    out = None
    for x in xs:
        if out is not None:
            raise ValueError("Expected single value, got multiple")
        out = x
    if out is None:
        raise ValueError("Expected single value, got none")
    return out


def get_transform_poses_pca(poses):
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is positive
    if poses_recentered.mean(axis=0)[2, 1] > 0:
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return transform


def focus_point_fn(poses, xnp = np):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = xnp.eye(3) - directions * xnp.transpose(directions, [0, 2, 1])
    mt_m = xnp.transpose(m, [0, 2, 1]) @ m
    focus_pt = xnp.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def get_default_viewer_transform(poses, dataset_type: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    if dataset_type == "object-centric":
        transform = get_transform_poses_pca(poses)

        poses = apply_transform(transform, poses)
        lookat = focus_point_fn(poses)

        poses[:, :3, 3] -= lookat
        transform[:3, 3] -= lookat
        return transform, poses[0][..., :3, :4]

    elif dataset_type == "forward-facing":
        raise NotImplementedError("Forward-facing dataset type is not supported")
    elif dataset_type is None:
        # Unknown dataset type
        # We move all center the scene on the mean of the camera origins
        # and reorient the scene so that the average camera up is up
        origins = poses[..., :3, 3]
        mean_origin = np.mean(origins, 0)
        translation = mean_origin
        up = np.mean(poses[:, :3, 1], 0)
        up = -up / np.linalg.norm(up)

        rotation = rotation_matrix(up, np.array([0, 0, 1], dtype=up.dtype))
        transform = np.concatenate([rotation, rotation @ -translation[..., None]], -1)
        transform = np.concatenate([transform, np.array([[0, 0, 0, 1]], dtype=transform.dtype)], 0)

        # Scale so that cameras fit in a 2x2x2 cube centered at the origin
        maxlen = np.quantile(np.abs(poses[..., 0:3, 3] - mean_origin[None]).max(-1), 0.95) * 1.1
        dataparser_scale = float(1 / maxlen)
        transform = np.diag([dataparser_scale, dataparser_scale, dataparser_scale, 1]) @ transform

        camera = apply_transform(transform, poses[0])
        return transform, camera[..., :3, :4]
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported")


METADATA_COLUMNS = ["exposure"]
DatasetType = Literal["object-centric", "forward-facing"]


def get_scene_scale(cameras: Cameras, dataset_type: Optional[DatasetType]):
    if dataset_type == "object-centric":
        return float(np.percentile(np.linalg.norm(cameras.poses[..., :3, 3] - cameras.poses[..., :3, 3].mean(), axis=-1), 90))

    elif dataset_type == "forward-facing":
        assert cameras.nears_fars is not None, "Forward-facing dataset must set z-near and z-far"
        return float(cameras.nears_fars.mean())

    elif dataset_type is None:
        return float(np.percentile(np.linalg.norm(cameras.poses[..., :3, 3] - cameras.poses[..., :3, 3].mean(), axis=-1), 90))
    
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported")


def get_image_metadata(image: PIL.Image.Image):
    # Metadata format: [ exposure, ]
    values = {}
    try:
        exif_pil = image.getexif()
    except AttributeError:
        exif_pil = image._getexif()  # type: ignore
    if exif_pil is not None:
        exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif_pil.items() if k in PIL.ExifTags.TAGS}
        if "ExposureTime" in exif and "ISOSpeedRatings" in exif:
            shutters = exif["ExposureTime"]
            isos = exif["ISOSpeedRatings"]
            exposure = shutters * isos / 1000.0
            values["exposure"] = exposure
    return np.array([values.get(c, np.nan) for c in METADATA_COLUMNS], dtype=np.float32)


def _dataset_rescale_intrinsics(dataset: Dataset, image_sizes: np.ndarray):
    cameras = dataset["cameras"]
    if np.any(cameras.image_sizes != image_sizes):
        logging.info("Image sizes do not match camera sizes. Resizing cameras to match image sizes.")

        if np.any(cameras.image_sizes % image_sizes != 0):
            warnings.warn("Downscaled image sizes are not a multiple of camera sizes.")

        multx, multy = np.moveaxis(
            image_sizes.astype(np.float64) / cameras.image_sizes.astype(np.float64), -1, 0)

        if "downscale_factor" in dataset["metadata"]:
            # Downscale factor is passed, we will use it for focal lengths
            # Not for the center of the image, because there could have been rounding
            # which would move the center from the center of the image
            downscale_factor = dataset["metadata"]["downscale_factor"]
            low = np.floor(cameras.image_sizes * np.stack([multx, multy], -1))
            high = np.ceil(cameras.image_sizes * np.stack([multx, multy], -1))
            if np.any(image_sizes < low) or np.any(image_sizes > high):
                raise RuntimeError(f"Downscaled image sizes do not match the downscale_factor of {downscale_factor}.")

        # NOTE: In previous versions of NerfBaselines, we scaled the parameters differently
        # We used:
        #   cx <- cx * multx,  cy <- cy * multy
        #   fx <- fx * multx,  fy <- fy * multx
        # This renders changes slightly the results on the MipNeRF 360 dataset

        multipliers = np.stack([multx, multy, multx, multy], -1)
        dataset["cameras"] = cameras.replace(
            image_sizes=image_sizes, 
            intrinsics=(cameras.intrinsics * multipliers).astype(cameras.intrinsics.dtype))


def dataset_load_features(
    dataset: UnloadedDataset, features=None, supported_camera_models=None
) -> Dataset:
    if features is None:
        features = frozenset(("color",))
    if supported_camera_models is None:
        supported_camera_models = frozenset(("pinhole",))
    images: List[np.ndarray] = []
    image_sizes = []
    all_metadata = []
    resize = dataset["metadata"].get("downscale_loaded_factor")
    if resize == 1:
        resize = None

    i = 0
    logging.info(f"Loading images from {dataset.get('image_paths_root')}")

    for p in tqdm(dataset["image_paths"], desc="loading images", dynamic_ncols=True):
        if str(p).endswith(".bin"):
            assert dataset["metadata"]["color_space"] == "linear"
            with open(p, "rb") as f:
                data_bytes = f.read()
                h, w = struct.unpack("ii", data_bytes[:8])
                image = (
                    np.frombuffer(
                        data_bytes, dtype=np.float16, count=h * w * 4, offset=8
                    )
                    .astype(np.float32)
                    .reshape([h, w, 4])
                )
            metadata = np.array(
                [np.nan for _ in range(len(METADATA_COLUMNS))], dtype=np.float32
            )
        else:
            assert dataset["metadata"]["color_space"] == "srgb"
            pil_image = PIL.Image.open(p)
            metadata = get_image_metadata(pil_image)
            if resize is not None:
                w, h = pil_image.size
                new_size = round(w/resize), round(h/resize)
                pil_image = pil_image.resize(new_size, PIL.Image.Resampling.BICUBIC)
                warnings.warn(f"Resized image with a factor of {resize}")

            image = np.array(pil_image, dtype=np.uint8)
        images.append(image)
        image_sizes.append([image.shape[1], image.shape[0]])
        all_metadata.append(metadata)
        i += 1

    logging.debug(f"Loaded {len(images)} images")

    if dataset["sampling_mask_paths"] is not None:
        sampling_masks = []
        for p in tqdm(dataset["sampling_mask_paths"], desc="loading sampling masks", dynamic_ncols=True):
            sampling_mask = PIL.Image.open(p).convert("L")
            if resize is not None:
                w, h = sampling_mask.size
                new_size = round(w*resize), round(h*resize)
                sampling_mask = sampling_mask.resize(new_size, PIL.Image.Resampling.NEAREST)
                warnings.warn(f"Resized sampling mask with a factor of {resize}")

            sampling_masks.append(np.array(sampling_mask, dtype=np.uint8).astype(bool))
        dataset["sampling_masks"] = sampling_masks  # padded_stack(sampling_masks)
        logging.debug(f"Loaded {len(sampling_masks)} sampling masks")

    if resize is not None:
        # Replace all paths with the resized paths
        dataset["image_paths"] = [
            os.path.join("/resized", os.path.relpath(p, dataset["image_paths_root"])) 
            for p in dataset["image_paths"]]
        dataset["image_paths_root"] = "/resized"
        if dataset["sampling_mask_paths"] is not None:
            dataset["sampling_mask_paths"] = [
                os.path.join("/resized-sampling-masks", os.path.relpath(p, dataset["sampling_mask_paths_root"])) 
                for p in dataset["sampling_mask_paths"]]
            dataset["sampling_mask_paths_root"] = "/resized-sampling-masks"

    dataset["images"] = images  # padded_stack(images)

    # Replace image sizes and metadata
    image_sizes = np.array(image_sizes, dtype=np.int32)

    _dataset_rescale_intrinsics(cast(Dataset, dataset), image_sizes)

    if supported_camera_models is not None and supported_camera_models != set(("pinhole",)):
        raise RuntimeError(
            "Some cameras models are not supported by the method."
        )
    return cast(Dataset, dataset)


class DatasetNotFoundError(Exception):
    pass


class MultiDatasetError(DatasetNotFoundError):
    def __init__(self, errors, message):
        self.errors = errors
        self.message = message
        super().__init__(message + "\n" + "".join(f"\n  {name}: {error}" for name, error in errors.items()))

    def write_to_logger(self, color=True, terminal_width=None):
        if terminal_width is None:
            terminal_width = 120
            try:
                terminal_width = min(os.get_terminal_size().columns, 120)
            except OSError:
                pass
        message = self.message
        if color:
            message = "\33[0m\33[31m" + message + "\33[0m"
        for name, error in self.errors.items():
            prefix = f"   {name}: "
            mlen = terminal_width - len(prefix)
            prefixlen = len(prefix)
            if color:
                prefix = f"\33[96m{prefix}\33[0m"
            rows = [error[i : i + mlen] for i in range(0, len(error), mlen)]
            mdetail = f'\n{" "*prefixlen}'.join(rows)
            message += f"\n{prefix}{mdetail}"
        logging.error(message)


def dataset_index_select(dataset: TDataset, i: Union[slice, int, list, np.ndarray]) -> TDataset:
    assert isinstance(i, (slice, int, list, np.ndarray))
    dataset_len = len(dataset["image_paths"])

    def index(key, obj):
        if obj is None:
            return None
        if key == "cameras":
            if len(obj) == 1:
                return obj if isinstance(i, int) else obj
            return obj[i]
        if isinstance(obj, np.ndarray):
            if obj.shape[0] == 1:
                return obj[0] if isinstance(i, int) else obj
            obj = obj[i]
            return obj
        if isinstance(obj, list):
            indices = np.arange(dataset_len)[i]
            if indices.ndim == 0:
                return obj[indices]
            return [obj[i] for i in indices]
        raise ValueError(f"Cannot index object of type {type(obj)} at key {key}")

    _dataset = cast(Dict, dataset.copy())
    _dataset.update({k: index(k, v) for k, v in dataset.items() if k not in {
        "image_paths_root", 
        "sampling_mask_paths_root", 
        "points3D_xyz", 
        "points3D_rgb", 
        "metadata"}})
    return cast(TDataset, _dataset)


@overload
def new_dataset(*,
                cameras: Cameras,
                image_paths: Sequence[str],
                image_paths_root: Optional[str] = ...,
                images: Union[np.ndarray, List[np.ndarray]],
                sampling_mask_paths: Optional[Sequence[str]] = ...,
                sampling_mask_paths_root: Optional[str] = None,
                sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]] = ...,  # [N][H, W]
                points3D_xyz: Optional[np.ndarray] = ...,  # [M, 3]
                points3D_rgb: Optional[np.ndarray] = ...,  # [M, 3]
                images_points3D_indices: Optional[Sequence[np.ndarray]] = None,  # [N][<M]
                metadata: Dict) -> Dataset:
    ...


@overload
def new_dataset(*,
                cameras: Cameras,
                image_paths: Sequence[str],
                image_paths_root: Optional[str] = ...,
                images: Literal[None] = None,
                sampling_mask_paths: Optional[Sequence[str]] = ...,
                sampling_mask_paths_root: Optional[str] = None,
                sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]] = ...,  # [N][H, W]
                points3D_xyz: Optional[np.ndarray] = ...,  # [M, 3]
                points3D_rgb: Optional[np.ndarray] = ...,  # [M, 3]
                images_points3D_indices: Optional[Sequence[np.ndarray]] = None,  # [N][<M]
                metadata: Dict) -> UnloadedDataset:
    ...


def new_dataset(*,
                cameras: Cameras,
                image_paths: Sequence[str],
                image_paths_root: Optional[str] = None,
                images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,  # [N][H, W, 3]
                sampling_mask_paths: Optional[Sequence[str]] = None,
                sampling_mask_paths_root: Optional[str] = None,
                sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]] = None,  # [N][H, W]
                points3D_xyz: Optional[np.ndarray] = None,  # [M, 3]
                points3D_rgb: Optional[np.ndarray] = None,  # [M, 3]
                images_points3D_indices: Optional[Sequence[np.ndarray]] = None,  # [N][<M]
                metadata: Dict) -> Union[UnloadedDataset, Dataset]:
    if image_paths_root is None:
        image_paths_root = os.path.commonpath(image_paths)
    if sampling_mask_paths_root is None and sampling_mask_paths is not None:
        sampling_mask_paths_root = os.path.commonpath(sampling_mask_paths)
    if image_paths_root is None:
        image_paths_root = os.path.commonpath(image_paths)
    return UnloadedDataset(
        cameras=cameras,
        image_paths=list(image_paths),
        sampling_mask_paths=list(sampling_mask_paths) if sampling_mask_paths is not None else None,
        sampling_mask_paths_root=sampling_mask_paths_root,
        image_paths_root=image_paths_root,
        images=images,
        sampling_masks=sampling_masks,
        points3D_xyz=points3D_xyz,
        points3D_rgb=points3D_rgb,
        images_points3D_indices=list(images_points3D_indices) if images_points3D_indices is not None else None,
        metadata=metadata
    )


def get_transform_and_scale(transform):
    assert len(transform.shape) == 2, "Transform should be a 4x4 or a 3x4 matrix."
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    assert np.allclose(scale, scale[0], rtol=1e-3, atol=0)
    scale = float(np.mean(scale).item())
    transform = transform.copy()
    transform[:3, :] /= scale
    return transform, scale


def apply_transform(transform, poses):
    transform, scale = get_transform_and_scale(transform)
    poses = unpad_poses(transform @ pad_poses(poses))
    poses[..., :3, 3] *= scale
    return poses


def invert_transform(transform, has_scale=False):
    scale = None
    if has_scale:
        transform, scale = get_transform_and_scale(transform)
    else:
        transform = transform.copy()
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    transform[..., :3, :] = np.concatenate([R.T, -np.matmul(R.T, t[..., None])], axis=-1)
    if scale is not None:
        transform[..., :3, :3] *= 1/scale
    return transform

