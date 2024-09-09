from functools import wraps
import base64
import struct
from datetime import datetime
import json
import contextlib
import tarfile
import time
import io
import logging
import os
import typing
from typing import Dict, Union, Iterable, TypeVar, Optional, cast, List, Callable
import numpy as np
from pathlib import Path

from tqdm import tqdm

from .datasets import new_dataset, dataset_index_select
from .utils import (
    read_image, 
    convert_image_dtype, 
    image_to_srgb,
    save_image,
    visualize_depth,
    open_any_directory,
    serialize_nb_info,
    save_depth,
)
from .types import (
    Literal, 
    Dataset,
    RenderOutput, 
    EvaluationProtocol, 
    Cameras,
    Method,
    Trajectory,
    camera_model_to_int,
    new_cameras,
)
try:
    from typeguard import suppress_type_checks  # type: ignore
except ImportError:
    from contextlib import nullcontext as suppress_type_checks


def assert_not_none(x):
    assert x is not None, "value must not be None"
    return x


OutputType = Literal["color", "depth"]
T = TypeVar("T")


def _wrap_metric_arbitrary_shape(fn):
    @wraps(fn)
    def wrapped(a, b, **kwargs):
        bs = a.shape[:-3]
        a = np.reshape(a, (-1, *a.shape[-3:]))
        b = np.reshape(b, (-1, *b.shape[-3:]))
        out = fn(a, b, **kwargs)
        return np.reshape(out, bs)

    return wrapped


@_wrap_metric_arbitrary_shape
def dmpix_ssim(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_val: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
    filter_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Computes the structural similarity index (SSIM) between image pairs.

    This function is based on the standard SSIM implementation from:
    Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
    "Image quality assessment: from error visibility to structural similarity",
    in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.

    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Note: the true SSIM is only defined on grayscale. This function does not
    perform any colorspace transform. If the input is in a color space, then it
    will compute the average SSIM.

    NOTE: This function exactly matches dm_pix.ssim

    Args:
        a: First image (or set of images).
        b: Second image (or set of images).
        max_val: The maximum magnitude that `a` or `b` can have.
        kernel_size: Window size (>= 1). Image dims must be at least this small.
        sigma: The bandwidth of the Gaussian used for filtering (> 0.).
        k1: One of the SSIM dampening parameters (> 0.).
        k2: One of the SSIM dampening parameters (> 0.).
        return_map: If True, will cause the per-pixel SSIM "map" to be returned.
        precision: The numerical precision to use when performing convolution.

    Returns:
        Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    # DO NOT REMOVE - Logging usage.

    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"

    if filter_fn is None:
        # Construct a 1D Gaussian blur filter.
        hw = kernel_size // 2
        shift = (2 * hw - kernel_size + 1) / 2
        f_i = ((np.arange(kernel_size) - hw + shift) / sigma) ** 2
        filt = np.exp(-0.5 * f_i)
        filt /= np.sum(filt)

        # Construct a 1D convolution.
        def filter_fn_1(z):
            return np.convolve(z, filt, mode="valid")

        # jax.vmap(filter_fn_1)
        filter_fn_vmap = lambda x: np.stack([filter_fn_1(y) for y in x], 0)  # noqa: E731

        # Apply the vectorized filter along the y axis.
        def filter_fn_y(z):
            z_flat = np.moveaxis(z, -3, -1).reshape((-1, z.shape[-3]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-2],
                z.shape[-1],
                -1,
            )
            z_filtered = np.moveaxis(filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -3)
            return z_filtered

        # Apply the vectorized filter along the x axis.
        def filter_fn_x(z):
            z_flat = np.moveaxis(z, -2, -1).reshape((-1, z.shape[-2]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-3],
                z.shape[-1],
                -1,
            )
            z_filtered = np.moveaxis(filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -2)
            return z_filtered

        # Apply the blur in both x and y.
        filter_fn = lambda z: filter_fn_y(filter_fn_x(z))  # noqa: E731

    mu0 = filter_fn(a)
    mu1 = filter_fn(b)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filter_fn(a**2) - mu00
    sigma11 = filter_fn(b**2) - mu11
    sigma01 = filter_fn(a * b) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    epsilon = np.finfo(np.float32).eps ** 2
    sigma00 = np.maximum(epsilon, sigma00)
    sigma11 = np.maximum(epsilon, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim_value = np.mean(ssim_map, tuple(range(-3, 0)))
    return ssim_map if return_map else ssim_value


def _normalize_input(a):
    return np.clip(a, 0, 1).astype(np.float32)


def ssim(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Structural Similarity Index Measure (the higher the better).
    Args:
        a: Tensor of prediction images [B, H, W, C].
        b: Tensor of target images [B, H, W, C].
    Returns:
        Tensor of mean SSIM values for each image [B].
    """
    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"
    a = _normalize_input(a)
    b = _normalize_input(b)
    return dmpix_ssim(a, b)


def mse(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Mean Squared Error (the lower the better).
    Args:
        a: Tensor of prediction images [B, H, W, C].
        b: Tensor of target images [B, H, W, C].
    Returns:
        Tensor of mean squared error values for each image [B].
    """
    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"
    a = _normalize_input(a)
    b = _normalize_input(b)
    return _mean((a - b) ** 2)


def _mean(metric):
    return np.mean(metric, (-3, -2, -1))


def mae(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Mean Absolute Error (the lower the better).
    Args:
        a: Tensor of prediction images [B, H, W, C].
        b: Tensor of target images [B, H, W, C].
    Returns:
        Tensor of mean absolute error values for each image [B].
    """
    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"
    a = _normalize_input(a)
    b = _normalize_input(b)
    return _mean(np.abs(a - b))


def psnr(a: Union[np.ndarray, np.float32, np.float64], b: Optional[np.ndarray] = None) -> Union[np.ndarray, np.float32, np.float64]:
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    It can reuse computed MSE values if b is None.
    Args:
        a: Tensor of prediction images [B, H, W, C] or a tensor of MSE values [B] (b must be None in that case).
        b: Tensor of target images [B, H, W, C] or None (if a are MSE values).
    Returns:
        Tensor of PSNR values for each image [B].
    """
    mse_value = a if b is None else mse(cast(np.ndarray, a), b)
    return -10 * np.log10(mse_value)


_LPIPS_CACHE = {}
_LPIPS_GPU_AVAILABLE = None


def _lpips(a, b, net, version="0.1"):
    global _LPIPS_GPU_AVAILABLE
    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"

    import torch

    lp_net = _LPIPS_CACHE.get(net)
    if lp_net is None:
        from ._metrics_lpips import LPIPS

        lp_net = LPIPS(net=net, version=version)
        _LPIPS_CACHE[net] = lp_net

    device = torch.device("cpu")
    if _LPIPS_GPU_AVAILABLE is None:
        _LPIPS_GPU_AVAILABLE = torch.cuda.is_available()
        if _LPIPS_GPU_AVAILABLE:
            try:
                lp_net.cuda()
                torch.zeros((1,), device="cuda").cpu()
            except Exception:
                _LPIPS_GPU_AVAILABLE = False

    if _LPIPS_GPU_AVAILABLE:
        device = torch.device("cuda")

    batch_shape = a.shape[:-3]
    img_shape = a.shape[-3:]
    a = _normalize_input(a)
    b = _normalize_input(b)
    with torch.no_grad():
        a = torch.from_numpy(a).float().view(-1, *img_shape).permute(0, 3, 1, 2).mul_(2).sub_(1).to(device)
        b = torch.from_numpy(b).float().view(-1, *img_shape).permute(0, 3, 1, 2).mul_(2).sub_(1).to(device)
        out = cast(torch.Tensor, lp_net.to(device).forward(a, b))
        out = out.detach().cpu().numpy().reshape(batch_shape)
        return out


def lpips_alex(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Learned Perceptual Image Patch Similarity (the lower the better).
    Args:
        a: Tensor of prediction images [B..., H, W, C].
        b: Tensor of target images [B..., H, W, C].
    Returns:
        Tensor of LPIPS values for each image [B...].
    """
    return _lpips(a, b, net="alex")


def lpips_vgg(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Learned Perceptual Image Patch Similarity (the lower the better).
    Args:
        a: Tensor of prediction images [B..., H, W, C].
        b: Tensor of target images [B..., H, W, C].
    Returns:
        Tensor of LPIPS values for each image [B...].
    """
    return _lpips(a, b, net="vgg")


lpips = lpips_alex


@typing.overload
def compute_metrics(pred: np.ndarray, gt: np.ndarray, *, reduce: Literal[True] = True, run_lpips_vgg: bool = ...) -> Dict[str, float]:
    ...


@typing.overload
def compute_metrics(pred: np.ndarray, gt: np.ndarray, *, reduce: Literal[False], run_lpips_vgg: bool = ...) -> Dict[str, np.ndarray]:
    ...


def compute_metrics(pred, gt, *, reduce: bool = True, run_lpips_vgg: bool = False):
    # NOTE: we blend with black background here!
    def reduction(x):
        if reduce:
            return x.mean().item()
        else:
            return x

    pred = pred[..., : gt.shape[-1]]
    pred = convert_image_dtype(pred, np.float32)
    gt = convert_image_dtype(gt, np.float32)
    mse_ = mse(pred, gt)
    out = {
        "psnr": reduction(psnr(mse_)),
        "ssim": reduction(ssim(gt, pred)),
        "mae": reduction(mae(gt, pred)),
        "mse": reduction(mse_),
        "lpips": reduction(lpips(gt, pred)),
    }
    if run_lpips_vgg:
        out["lpips_vgg"] = reduction(lpips_vgg(gt, pred))
    return out


def evaluate(predictions: str, 
             output: str, 
             *,
             description: str = "evaluating", 
             evaluation_protocol: EvaluationProtocol):
    """
    Evaluate a set of predictions.

    Args:
        predictions: Path to a directory containing the predictions.
        output: Path to a json file where the results will be written.
        description: Description of the evaluation, used for progress bar.
        evaluation_protocol: The evaluation protocol to use. If None, the protocol from info.json will be used.
    Returns:
        A dictionary containing the results.
    """
    if os.path.exists(output):
        raise FileExistsError(f"{output} already exists")

    with open_any_directory(predictions, "r") as _predictions_path:
        predictions_path = Path(_predictions_path)
        with open(predictions_path / "info.json", "r", encoding="utf8") as f:
            nb_info = json.load(f)

        logging.info(f"Using evaluation protocol {evaluation_protocol.get_name()}")

        # Run the evaluation
        metrics_lists = {}
        relpaths = [str(x.relative_to(predictions_path / "color")) for x in (predictions_path / "color").glob("**/*") if x.is_file()]
        relpaths.sort()

        def read_predictions() -> Iterable[RenderOutput]:
            # Load the prediction
            for relname in relpaths:
                yield {
                    "color": read_image(predictions_path / "color" / relname)
                }

        gt_images = [
            read_image(predictions_path / "gt-color" / name) for name in relpaths
        ]
        with suppress_type_checks():
            from pprint import pprint
            pprint(nb_info)
            dataset = new_dataset(
                cameras=typing.cast(Cameras, None),
                image_paths=relpaths,
                image_paths_root=str(predictions_path / "color"),
                metadata=typing.cast(Dict, nb_info.get("render_dataset_metadata", nb_info.get("dataset_metadata", {}))),
                images=gt_images)

            # Evaluate the prediction
            with tqdm(desc=description, dynamic_ncols=True, total=len(relpaths)) as progress:
                def collect_metrics_lists():
                    for i, pred in enumerate(read_predictions()):
                        dataset_slice = dataset_index_select(dataset, [i])
                        data = evaluation_protocol.evaluate(pred, dataset_slice)

                        for k, v in data.items():
                            if k not in metrics_lists:
                                metrics_lists[k] = []
                            metrics_lists[k].append(v)
                        progress.update(1)
                        if "psnr" in metrics_lists:
                            psnr_val = np.mean(metrics_lists["psnr"][-1])
                            progress.set_postfix(psnr=f"{psnr_val:.4f}")
                        yield data

                metrics = evaluation_protocol.accumulate_metrics(collect_metrics_lists())

        # If output is specified, write the results to a file
        if os.path.exists(str(output)):
            raise FileExistsError(f"{output} already exists")

        out = save_evaluation_results(str(output),
                                      metrics=metrics, 
                                      metrics_lists=metrics_lists, 
                                      evaluation_protocol=evaluation_protocol.get_name(),
                                      nb_info=nb_info)
        return out


class DefaultEvaluationProtocol(EvaluationProtocol):
    _name = "default"
    _lpips_vgg = False

    def __init__(self):
        pass

    def render(self, method: Method, dataset: Dataset) -> RenderOutput:
        return method.render(dataset["cameras"].item())

    def get_name(self):
        return self._name

    def evaluate(self, predictions: RenderOutput, dataset: Dataset) -> Dict[str, Union[float, int]]:
        assert len(dataset["images"]) == 1, "Only single image evaluation is supported"
        background_color = dataset["metadata"].get("background_color")
        color_space = dataset["metadata"]["color_space"]

        pred = predictions["color"]
        gt = dataset["images"][0]
        pred = image_to_srgb(pred, np.uint8, color_space=color_space, background_color=background_color)
        gt = image_to_srgb(gt, np.uint8, color_space=color_space, background_color=background_color)
        pred_f = convert_image_dtype(pred, np.float32)
        gt_f = convert_image_dtype(gt, np.float32)
        return compute_metrics(pred_f[None], gt_f[None], run_lpips_vgg=self._lpips_vgg, reduce=True)

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        acc = {}
        for i, data in enumerate(metrics):
            for k, v in data.items():
                # acc[k] = (acc.get(k, 0) * i + v) / (i + 1)
                acc[k] = acc.get(k, 0) * (i / (i + 1)) + v / (i + 1)
        return acc


class NerfEvaluationProtocol(DefaultEvaluationProtocol):
    _name = "nerf"
    _lpips_vgg = True


def render_all_images(
    method: Method,
    dataset: Dataset,
    output: str,
    *,
    description: str = "rendering all images",
    nb_info: Optional[dict] = None,
    evaluation_protocol: EvaluationProtocol,
) -> Iterable[RenderOutput]:
    logging.info(f"Rendering images with evaluation protocol {evaluation_protocol.get_name()}")
    background_color =  dataset["metadata"].get("background_color", None)
    if background_color is not None:
        background_color = convert_image_dtype(background_color, np.uint8)
    if nb_info is None:
        nb_info = {}
    else:
        nb_info = nb_info.copy()
        dataset_colorspace = dataset["metadata"].get("color_space", "srgb")
        assert dataset_colorspace == nb_info.get("color_space", "srgb"), \
            f"Dataset color space {dataset_colorspace} != method color space {nb_info['color_space']}"
        if "dataset_background_color" in nb_info:
            info_background_color = nb_info.get("dataset_background_color")
            if info_background_color is not None:
                info_background_color = np.array(info_background_color, np.uint8)
            assert info_background_color is None or (background_color is not None and np.array_equal(info_background_color, background_color)), \
                f"Dataset background color {background_color} != method background color {info_background_color}"
    nb_info["evaluation_protocol"] = evaluation_protocol.get_name()

    with tqdm(desc=description, total=len(dataset["image_paths"]), dynamic_ncols=True) as progress:
        for val in save_predictions(output,
                                    (
                                        evaluation_protocol.render(method, dataset_index_select(dataset, [i])) 
                                         for i in range(len(dataset["image_paths"]))
                                    ),
                                    dataset=dataset,
                                    nb_info=nb_info):
            progress.update(1)
            yield val


def render_frames(
    method: Method,
    cameras: Cameras,
    output: Union[str, Path],
    fps: float,
    embeddings: Optional[List[np.ndarray]] = None,
    description: str = "rendering frames",
    output_type: OutputType = "color",
    nb_info: Optional[dict] = None,
) -> None:
    output = Path(output)
    assert cameras.image_sizes is not None, "cameras.image_sizes must be set"
    render = method.render
    color_space = "srgb"
    background_color = nb_info.get("background_color") if nb_info is not None else None
    expected_scene_scale = nb_info.get("expected_scene_scale") if nb_info is not None else None

    def _predict_all(allow_transparency=True):
        predictions = (render(cam, options={"embedding": embeddings[i] if embeddings is not None else None}) for i, cam in enumerate(cameras))
        for i, pred in enumerate(tqdm(predictions, desc=description, total=len(cameras), dynamic_ncols=True)):
            pred_image = image_to_srgb(pred["color"], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
            if output_type == "color":
                yield pred_image
            elif output_type == "depth":
                assert "depth" in pred, "Method does not output depth"
                depth_rgb = visualize_depth(pred["depth"], near_far=cameras.nears_fars[i] if cameras.nears_fars is not None else None, expected_scale=expected_scene_scale)
                yield convert_image_dtype(depth_rgb, np.uint8)
            else:
                raise RuntimeError(f"Output type {output_type} is not supported.")

    if str(output).endswith(".tar.gz"):
        with tarfile.open(output, "w:gz") as tar:
            for i, frame in enumerate(_predict_all()):
                rel_path = f"{i:05d}.png"
                tarinfo = tarfile.TarInfo(name=rel_path)
                tarinfo.mtime = int(time.time())
                with io.BytesIO() as f:
                    f.name = rel_path
                    tarinfo.size = f.tell()
                    f.seek(0)
                    save_image(f, frame)
                    tar.addfile(tarinfo=tarinfo, fileobj=f)
    elif str(output).endswith(".mp4") or str(output).endswith(".gif"):
        # Handle video
        import mediapy

        w, h = cameras.image_sizes[0]
        codec = 'h264'
        if str(output).endswith(".gif"):
            codec = "gif"
        with mediapy.VideoWriter(output, (h, w), metadata=mediapy.VideoMetadata(len(cameras), (h, w), fps, bps=None), fps=fps, codec=codec) as writer:
            for i, frame in enumerate(_predict_all(allow_transparency=False)):
                writer.add_image(frame)
    else:
        os.makedirs(output, exist_ok=True)
        for i, frame in enumerate(_predict_all()):
            rel_path = f"{i:05d}.png"
            with open(os.path.join(output, rel_path), "wb") as f:
                save_image(f, frame)


def trajectory_get_cameras(trajectory: Trajectory) -> Cameras:
    if trajectory["camera_model"] != "pinhole":
        raise NotImplementedError("Only pinhole camera model is supported")
    poses = np.stack([x["pose"] for x in trajectory["frames"]])
    intrinsics = np.stack([x["intrinsics"] for x in trajectory["frames"]])
    camera_models = np.array([camera_model_to_int(trajectory["camera_model"])]*len(poses), dtype=np.int32)
    image_sizes = np.array([list(trajectory["image_size"])]*len(poses), dtype=np.int32)
    return new_cameras(poses=poses, 
                       intrinsics=intrinsics, 
                       camera_models=camera_models, 
                       image_sizes=image_sizes,
                       distortion_parameters=np.zeros((len(poses), 0), dtype=np.float32),
                       nears_fars=None, 
                       metadata=None)


def trajectory_get_embeddings(method: Method, trajectory: Trajectory) -> Optional[List[np.ndarray]]:
    appearances = list(trajectory.get("appearances") or [])
    appearance_embeddings: List[Optional[np.ndarray]] = [None] * len(appearances)

    # Fill in embedding images
    for i, appearance in enumerate(appearances):
        if appearance.get("embedding") is not None:
            appearance_embeddings[i] = appearance.get("embedding")
        elif appearance.get("embedding_train_index") is not None:
            appearance_embeddings[i] = method.get_train_embedding(assert_not_none(appearance.get("embedding_train_index")))
    if all(x is None for x in appearance_embeddings):
        return None
    if not all(x is not None for x in appearance_embeddings):
        raise ValueError("Either all embeddings must be provided or all must be missing")
    if all(x.get("appearance_weights") is None for x in trajectory["frames"]):
        return None
    if not all(x.get("appearance_weights") is not None for x in trajectory["frames"]):
        raise ValueError("Either all appearance weights must be provided or all must be missing")
    appearance_embeddings_np = np.stack(cast(List[np.ndarray], appearance_embeddings))

    # Interpolate embeddings
    out = []
    for frame in trajectory["frames"]:
        embedding = frame.get("appearance_weights") @ appearance_embeddings_np
        out.append(embedding)
    return out


def _encode_values(values: List[float]) -> str:
    return base64.b64encode(b"".join(struct.pack("f", v) for v in values)).decode("ascii")


def serialize_evaluation_results(metrics: Dict, 
                                 metrics_lists, 
                                 evaluation_protocol: str, 
                                 nb_info: Dict):
    precision = 5
    nb_info = serialize_nb_info(nb_info)
    out = {}
    render_datetime = nb_info.pop("render_datetime", None)
    if render_datetime is not None:
        out["render_datetime"] = render_datetime
    render_dataset_metadata = nb_info.pop("render_dataset_metadata", None)
    if render_dataset_metadata is not None:
        out["render_dataset_metadata"] = render_dataset_metadata
    out.update({
        "nb_info": nb_info,
        "evaluate_datetime": datetime.utcnow().isoformat(timespec="seconds"),
        "metrics": {k: round(v, precision) for k, v in metrics.items()},
        "metrics_raw": {k: _encode_values(metrics_lists[k]) for k in metrics_lists},
        "evaluation_protocol": evaluation_protocol,
    })
    return out



def save_evaluation_results(file,
                            metrics: Dict, 
                            metrics_lists, 
                            evaluation_protocol: str, 
                            nb_info: Dict):
    if isinstance(file, str):
        if os.path.exists(file):
            raise FileExistsError(f"{file} already exists")
        with open(file, "w", encoding="utf8") as f:
            return save_evaluation_results(f, metrics, metrics_lists, evaluation_protocol, nb_info)

    else:
        out = serialize_evaluation_results(metrics, metrics_lists, evaluation_protocol, nb_info)
        json.dump(out, file, indent=2)
        return out


def save_cameras_npz(file, cameras):
    numpy_arrays = {}
    def extract_array(arr, name):
        numpy_arrays[name] = arr
        return arr
    cameras.apply(extract_array)
    np.savez(file, **numpy_arrays)


def save_predictions(output: str, predictions: Iterable[RenderOutput], dataset: Dataset, *, nb_info=None) -> Iterable[RenderOutput]:
    background_color =  dataset["metadata"].get("background_color", None)
    assert background_color is None or background_color.dtype == np.uint8, "background_color must be None or uint8"
    color_space = dataset["metadata"]["color_space"]
    expected_scene_scale = dataset["metadata"].get("expected_scene_scale")
    allow_transparency = True

    def _predict_all(open_fn) -> Iterable[RenderOutput]:
        for i, (pred, (w, h)) in enumerate(zip(predictions, assert_not_none(dataset["cameras"].image_sizes))):
            gt_image = image_to_srgb(dataset["images"][i][:h, :w], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
            pred_image = image_to_srgb(pred["color"], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
            assert gt_image.shape[:-1] == pred_image.shape[:-1], f"gt size {gt_image.shape[:-1]} != pred size {pred_image.shape[:-1]}"
            relative_name = Path(dataset["image_paths"][i])
            if dataset["image_paths_root"] is not None:
                relative_name = relative_name.relative_to(Path(dataset["image_paths_root"]))
            with open_fn(f"gt-color/{relative_name.with_suffix('.png')}") as f:
                save_image(f, gt_image)
            with open_fn(f"color/{relative_name.with_suffix('.png')}") as f:
                save_image(f, pred_image)

            with open_fn(f"cameras/{relative_name.with_suffix('.npz')}") as f:
                save_cameras_npz(f, dataset["cameras"][i])
            # with open_fn(f"gt-color/{relative_name.with_suffix('.npy')}") as f:
            #     np.save(f, dataset["images"][i][:h, :w])
            # with open_fn(f"color/{relative_name.with_suffix('.npy')}") as f:
            #     np.save(f, pred["color"])
            if "depth" in pred:
                with open_fn(f"depth/{relative_name.with_suffix('.bin')}") as f:
                    save_depth(f, pred["depth"])
                depth_rgb = visualize_depth(pred["depth"], near_far=dataset["cameras"].nears_fars[i] if dataset["cameras"].nears_fars is not None else None, expected_scale=expected_scene_scale)
                with open_fn(f"depth-rgb/{relative_name.with_suffix('.png')}") as f:
                    save_image(f, depth_rgb)
            if color_space == "linear":
                # Store the raw linear image as well
                with open_fn(f"gt-color-linear/{relative_name.with_suffix('.bin')}") as f:
                    save_image(f, dataset["images"][i][:h, :w])
                with open_fn(f"color-linear/{relative_name.with_suffix('.bin')}") as f:
                    save_image(f, pred["color"])
            yield pred

    def write_metadata(open_fn):
        from pprint import pprint
        pprint(nb_info)
        with open_fn("info.json") as fp:
            background_color = dataset["metadata"].get("background_color", None)
            if isinstance(background_color, np.ndarray):
                background_color = background_color.tolist()
            fp.write(
                json.dumps(
                    serialize_nb_info(
                        {
                            **(nb_info or {}),
                            "render_datetime": datetime.utcnow().isoformat(timespec="seconds"),
                            "render_dataset_metadata": dataset["metadata"],
                        }),
                    indent=2,
                ).encode("utf-8")
            )

    if str(output).endswith(".tar.gz"):
        with tarfile.open(output, "w:gz") as tar:

            @contextlib.contextmanager
            def open_fn_tar(path):
                rel_path = path
                path = os.path.join(output, path)
                tarinfo = tarfile.TarInfo(name=rel_path)
                tarinfo.mtime = int(time.time())
                with io.BytesIO() as f:
                    f.name = path
                    yield f
                    tarinfo.size = f.tell()
                    f.seek(0)
                    tar.addfile(tarinfo=tarinfo, fileobj=f)

            write_metadata(open_fn_tar)
            yield from _predict_all(open_fn_tar)
    else:

        def open_fn_fs(path):
            path = os.path.join(output, path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return open(path, "wb")

        write_metadata(open_fn_fs)
        yield from _predict_all(open_fn_fs)


