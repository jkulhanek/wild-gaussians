import os
from datetime import datetime
import time
import zipfile
import contextlib
from pathlib import Path
import tempfile
import logging
import click
import sys
import math
import struct
import tarfile
import requests
from typing import Any, Optional, Dict, Union, Iterator
from typing import BinaryIO, Tuple, IO
from tqdm import tqdm
import numpy as np
from PIL import Image
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


OpenMode = Literal["r", "w"]


def assert_not_none(x):
    assert x is not None
    return x


class Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        levelname = record.levelname[0]
        message = record.getMessage()
        if levelname == "D":
            return f"\033[0;36mdebug:\033[0m {message}"
        elif levelname == "I":
            return f"\033[1;36minfo:\033[0m {message}"
        elif levelname == "W":
            return f"\033[0;1;33mwarning: {message}\033[0m"
        elif levelname == "E":
            return f"\033[0;1;31merror: {message}\033[0m"
        else:
            return message


def setup_logging(verbose: bool):
    kwargs: Dict[str, Any] = {}
    if sys.version_info >= (3, 8):
        kwargs["force"] = True
    if verbose >= 1:
        logging.basicConfig(level=logging.DEBUG, **kwargs)
        logging.getLogger('PIL').setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, **kwargs)
    for handler in logging.root.handlers:
        handler.setFormatter(Formatter())
    logging.captureWarnings(True)


class Indices:
    def __init__(self, steps):
        self._steps = steps
        self.total: Optional[int] = None

    def __contains__(self, x):
        if isinstance(self._steps, list):
            steps = self._steps
            if any(x < 0 for x in self._steps):
                assert self.total is not None, "total must be specified for negative steps"
                steps = set(x if x >= 0 else self.total + x for x in self._steps)
            return x in steps
        elif isinstance(self._steps, slice):
            start: int = self._steps.start or 0
            if start < 0:
                assert self.total is not None, "total must be specified for negative start"
                start = self.total - start
            stop: Optional[int] = self._steps.stop or self.total
            if stop is not None and stop < 0:
                assert self.total is not None, "total must be specified for negative stop"
                stop = self.total - stop
            step: int = self._steps.step or 1
            return x >= start and (stop is None or x < stop) and (x - start) % step == 0

    @classmethod
    def every_iters(cls, iters: int, zero: bool = False):
        start = iters if zero else 0
        return cls(slice(start, None, iters))

    def __repr__(self):
        if isinstance(self._steps, list):
            return ",".join(map(str, self._steps))
        elif isinstance(self._steps, slice):
            out = f"{self._steps.start or ''}:{self._steps.stop or ''}"
            if self._steps.step is not None:
                out += f":{self._steps.step}"
            return out
        else:
            return repr(self._steps)

    def __str__(self):
        return repr(self)


def convert_image_dtype(image: np.ndarray, dtype) -> np.ndarray:
    if image.dtype == dtype:
        return image
    if image.dtype != np.uint8 and dtype != np.uint8:
        return image.astype(dtype)
    if image.dtype == np.uint8 and dtype != np.uint8:
        return image.astype(dtype) / 255.0
    if image.dtype != np.uint8 and dtype == np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"cannot convert image from {image.dtype} to {dtype}")


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def image_to_srgb(tensor, dtype, color_space: Optional[str] = None, allow_alpha: bool = False, background_color: Optional[np.ndarray] = None):
    # Remove alpha channel in uint8
    if color_space is None:
        color_space = "srgb"
    if tensor.shape[-1] == 4 and not allow_alpha:
        # NOTE: here we blend with black background
        if tensor.dtype == np.uint8:
            tensor = convert_image_dtype(tensor, np.float32)
        alpha = tensor[..., -1:]
        tensor = tensor[..., :3] * tensor[..., -1:]
        # Default is black background [0, 0, 0]
        if background_color is not None:
            tensor += (1 - alpha) * convert_image_dtype(background_color, np.float32)

    if color_space == "linear":
        tensor = convert_image_dtype(tensor, np.float32)
        tensor = linear_to_srgb(tensor)

    # Round to 8-bit for fair comparisons
    tensor = convert_image_dtype(tensor, np.uint8)
    tensor = convert_image_dtype(tensor, dtype)
    return tensor


def save_image(file: Union[BinaryIO, str, Path], tensor: np.ndarray):
    if isinstance(file, (str, Path)):
        with open(file, "wb") as f:
            return save_image(f, tensor)
    path = Path(file.name)
    if str(path).endswith(".bin"):
        if tensor.shape[2] < 4:
            tensor = np.dstack((tensor, np.ones([tensor.shape[0], tensor.shape[1], 4 - tensor.shape[2]])))
        file.write(struct.pack("ii", tensor.shape[0], tensor.shape[1]))
        file.write(tensor.astype(np.float16).tobytes())
    else:
        from PIL import Image

        tensor = convert_image_dtype(tensor, np.uint8)
        image = Image.fromarray(tensor)
        image.save(file, format="png")


def read_image(file: Union[BinaryIO, str, Path]) -> np.ndarray:
    if isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            return read_image(f)
    path = Path(file.name)
    if str(path).endswith(".bin"):
        h, w = struct.unpack("ii", file.read(8))
        itemsize = 2
        img = np.frombuffer(file.read(h * w * 4 * itemsize), dtype=np.float16, count=h * w * 4, offset=8).reshape([h, w, 4])
        assert img.itemsize == itemsize
        return img.astype(np.float32)
    else:
        from PIL import Image

        return np.array(Image.open(file))


def save_depth(file: Union[BinaryIO, str, Path], tensor: np.ndarray):
    if isinstance(file, (str, Path)):
        with open(file, "wb") as f:
            return save_depth(f, tensor)
    path = Path(file.name)
    assert str(path).endswith(".bin")
    file.write(struct.pack("ii", tensor.shape[0], tensor.shape[1]))
    file.write(tensor.astype(np.float16).tobytes())


def mark_host(fn):
    fn.__host__ = True
    return fn


def _zipnerf_power_transformation(x, lam: float):
    m = abs(lam - 1) / lam
    return (((x / abs(lam - 1)) + 1) ** lam - 1) * m


def apply_colormap(array: np.ndarray, *, pallete: str = "viridis", invert: bool = False) -> np.ndarray:
    # TODO: remove matplotlib dependency
    import matplotlib
    import matplotlib.colors

    # Map to a color scale
    array_long = (array * 255).astype(np.int32).clip(0, 255)
    colormap = matplotlib.colormaps[pallete]
    colormap_colors = None
    if isinstance(colormap, matplotlib.colors.ListedColormap):
        colormap_colors = colormap.colors
    else:
        colormap_colors = [list(colormap(i / 255))[:3] for i in range(256)]
    pallete_array = np.array(colormap_colors, dtype=np.float32)  # type: ignore
    if invert:
        array_long = 255 - array_long
    out = pallete_array[array_long]
    return (out * 255).astype(np.uint8)


def visualize_depth(depth: np.ndarray, expected_scale: Optional[float] = None, near_far: Optional[np.ndarray] = None, pallete: str = "viridis") -> np.ndarray:
    # We will squash the depth to range [0, 1] using Barron's power transformation
    xnp = np
    eps = xnp.finfo(xnp.float32).eps  # type: ignore
    if near_far is not None:
        depth_squashed = (depth - near_far[0]) / (near_far[1] - near_far[0])
    elif expected_scale is not None:
        depth = depth / max(0.3 * expected_scale, eps)

        # We use the power series -> for lam=-1.5 the limit is 5/3
        depth_squashed = _zipnerf_power_transformation(depth, -1.5) / (5 / 3)
    else:
        depth_squashed = depth
    depth_squashed = depth_squashed.clip(0, 1)

    # Map to a color scale
    return apply_colormap(depth_squashed, pallete=pallete)


def make_image_grid(*images: np.ndarray, ncol=None, padding=2, max_width=1920, background: Union[None, Tuple[float, float, float], np.ndarray] = None):
    if ncol is None:
        ncol = len(images)
    dtype = images[0].dtype
    if background is None:
        background = np.full((3,), 255 if dtype == np.uint8 else 1, dtype=dtype)
    elif isinstance(background, tuple):
        background = np.array(background, dtype=dtype)
    elif isinstance(background, np.ndarray):
        background = convert_image_dtype(background, dtype=dtype)
    else:
        raise ValueError(f"Invalid background type {type(background)}")
    nrow = int(math.ceil(len(images) / ncol))
    scale_factor = 1
    height, width = tuple(map(int, np.max([x.shape[:2] for x in images], axis=0).tolist()))
    if max_width is not None:
        scale_factor = min(1, (max_width - padding * (ncol - 1)) / (ncol * width))
        height = int(height * scale_factor)
        width = int(width * scale_factor)

    def interpolate(image) -> np.ndarray:
        img = Image.fromarray(image)
        img_width, img_height = img.size
        aspect = img_width / img_height
        img_width = int(min(width, aspect * height))
        img_height = int(img_width / aspect)
        img = img.resize((img_width, img_height))
        return np.array(img)

    images = tuple(map(interpolate, images))
    grid: np.ndarray = np.ndarray(
        (height * nrow + padding * (nrow - 1), width * ncol + padding * (ncol - 1), images[0].shape[-1]),
        dtype=dtype,
    )
    grid[..., :] = background
    for i, image in enumerate(images):
        x = i % ncol
        y = i // ncol
        h, w = image.shape[:2]
        offx = x * (width + padding) + (width - w) // 2
        offy = y * (height + padding) + (height - h) // 2
        grid[offy : offy + h, 
             offx : offx + w] = image
    return grid


class IndicesClickType(click.ParamType):
    name = "indices"

    def convert(self, value, param, ctx):
        del param, ctx
        if value is None:
            return None
        if isinstance(value, Indices):
            return value
        if ":" in value:
            parts = [int(x) if x else None for x in value.split(":")]
            assert len(parts) <= 3, "too many parts in slice"
            return Indices(slice(*parts))
        return Indices([int(x) for x in value.split(",")])


class SetParamOptionType(click.ParamType):
    name = "key-value"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, tuple):
            return value
        if "=" not in value:
            self.fail(f"expected key=value pair, got {value}", param, ctx)
        k, v = value.split("=", 1)
        return k, v


MetricAccumulationMode = Literal["average", "last", "sum"]


class MetricsAccumulator:
    def __init__(
        self,
        options: Optional[Dict[str, MetricAccumulationMode]] = None,
    ):
        self.options = options or {}
        self._state = None

    def update(self, metrics: Dict[str, Union[int, float]]) -> None:
        if self._state is None:
            self._state = {}
        state = self._state
        n_iters_since_update = state["n_iters_since_update"] = state.get("n_iters_since_update", {})
        for k, v in metrics.items():
            accumulation_mode = self.options.get(k, "average")
            n_iters_since_update[k] = n = n_iters_since_update.get(k, 0) + 1
            if k not in state:
                state[k] = 0
            if accumulation_mode == "last":
                state[k] = v
            elif accumulation_mode == "average":
                state[k] = state[k] * ((n - 1) / n) + v / n
            elif accumulation_mode == "sum":
                state[k] += v
            else:
                raise ValueError(f"Unknown accumulation mode {accumulation_mode}")

    def pop(self) -> Dict[str, Union[int, float]]:
        if self._state is None:
            return {}
        state = self._state
        self._state = None
        state.pop("n_iters_since_update", None)
        return state


@contextlib.contextmanager
def open_any(
    path: Union[str, Path, BinaryIO], mode: OpenMode = "r"
) -> Iterator[IO[bytes]]:
    if not isinstance(path, (str, Path)):
        yield path
        return

    path = str(path)
    components = path.split("/")
    zip_parts = [i for i, c in enumerate(components[:-1]) if c.endswith(".zip")]
    if zip_parts:
        with open_any("/".join(components[: zip_parts[-1] + 1]), mode=mode) as f:
            if components[zip_parts[-1]].endswith(".tar.gz"):
                # Extract from tar.gz
                rest = "/".join(components[zip_parts[-1] + 1 :])
                with tarfile.open(fileobj=f, mode=mode + ":gz") as tar:
                    if mode == "r":
                        with assert_not_none(tar.extractfile(rest)) as f:
                            yield f
                    elif mode == "w":
                        _, extension = os.path.split(rest)
                        with tempfile.TemporaryFile("wb", suffix=extension) as tmp:
                            yield tmp
                            tmp.flush()
                            tmp.seek(0)
                            tarinfo = tarfile.TarInfo(name=rest)
                            tarinfo.mtime = int(time.time())
                            tarinfo.mode = 0o644
                            tarinfo.size = tmp.tell()
                            tar.addfile(
                                tarinfo=tarinfo,
                                fileobj=tmp,
                            )

            else:
                # Extract from zip
                with zipfile.ZipFile(f, mode=mode) as zip, zip.open(
                    "/".join(components[zip_parts[-1] + 1 :]), mode=mode
                ) as f:
                    yield f
        return

    # Download from url
    if path.startswith("http://") or path.startswith("https://"):
        assert mode == "r", "Only reading from remote files is supported."
        response = requests.get(path, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True, desc="Downloading"
        )
        name = path.split("/")[-1]
        with tempfile.TemporaryFile("rb+", suffix=name) as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
            file.flush()
            file.seek(0)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                logging.error(
                    f"Failed to download {path}. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes."
                )
            yield file
        return

    # Normal file
    if mode == "w":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode=mode + "b") as f:
        yield f


@contextlib.contextmanager
def open_any_directory(path: Union[str, Path], mode: OpenMode = "r") -> Iterator[str]:
    path = str(path)
    path = os.path.abspath(path)

    components = path.split("/")
    compressed_parts = [
        i
        for i, c in enumerate(components)
        if c.endswith(".zip") or c.endswith(".tar.gz")
    ]
    if compressed_parts:
        with open_any(
            "/".join(components[: compressed_parts[-1] + 1]), mode=mode
        ) as f, tempfile.TemporaryDirectory() as tmpdir:
            rest = "/".join(components[compressed_parts[-1] + 1 :])
            if components[compressed_parts[-1]].endswith(".tar.gz"):
                with tarfile.open(fileobj=f, mode=mode + ":gz") as tar:
                    if mode == "r":
                        for member in tar.getmembers():
                            if not member.name.startswith(rest):
                                continue
                            if member.isdir():
                                os.makedirs(
                                    os.path.join(tmpdir, member.name), exist_ok=True
                                )
                            else:
                                tar.extract(member, tmpdir)
                        yield os.path.join(tmpdir, rest)
                    elif mode == "w":
                        tmp_path = Path(tmpdir) / rest
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        yield os.path.join(tmpdir, rest)

                        for root, dirs, files in os.walk(tmp_path):
                            for dir in dirs:
                                tar.add(
                                    os.path.join(root, dir),
                                    arcname=os.path.relpath(
                                        os.path.join(root, dir), tmp_path
                                    ),
                                )
                            for file in files:
                                tar.add(
                                    os.path.join(root, file),
                                    arcname=os.path.relpath(
                                        os.path.join(root, file), tmp_path
                                    ),
                                )
                    else:
                        raise RuntimeError(f"Unsupported mode {mode} for tar.gz files.")
            else:
                with zipfile.ZipFile(f, mode=mode) as zip:
                    # Extract from zip
                    if mode == "r":
                        for member in zip.infolist():
                            if not member.filename.startswith(rest):
                                continue
                            if member.is_dir():
                                os.makedirs(
                                    os.path.join(tmpdir, member.filename), exist_ok=True
                                )
                            else:
                                zip.extract(member, tmpdir)
                                # Fix mtime
                                extracted_path = os.path.join(tmpdir, member.filename)
                                date_time = datetime(*member.date_time)
                                mtime = time.mktime(date_time.timetuple())
                                os.utime(extracted_path, (mtime, mtime))

                        yield os.path.join(tmpdir, rest)
                    elif mode == "w":
                        tmp_path = Path(tmpdir) / rest
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        yield os.path.join(tmpdir, rest)

                        for root, dirs, files in os.walk(tmp_path):
                            for dir in dirs:
                                zip.write(
                                    os.path.join(root, dir),
                                    arcname=os.path.relpath(
                                        os.path.join(root, dir), tmp_path
                                    ),
                                )
                            for file in files:
                                zip.write(
                                    os.path.join(root, file),
                                    arcname=os.path.relpath(
                                        os.path.join(root, file), tmp_path
                                    ),
                                )
                    else:
                        raise RuntimeError(f"Unsupported mode {mode} for zip files.")
        return

    if path.startswith("http://") or path.startswith("https://"):
        raise RuntimeError(
            "Only tar.gz and zip files are supported for remote directories."
        )

    # Normal file
    Path(path).mkdir(parents=True, exist_ok=True)
    yield str(Path(path).absolute())
    return


def serialize_nb_info(info: dict) -> dict:
    info = info.copy()

    def fix_dm(dm):
        if dm is None:
            return None
        dm = dm.copy()
        if isinstance(dm.get("background_color"), np.ndarray):
            dm["background_color"] = dm["background_color"].tolist()
        if "viewer_initial_pose" in dm and isinstance(dm["viewer_initial_pose"], np.ndarray):
            dm["viewer_initial_pose"] = np.round(dm["viewer_initial_pose"][:3, :4].astype(np.float64), 6).tolist()
        if "viewer_transform" in dm and isinstance(dm["viewer_transform"], np.ndarray):
            dm["viewer_transform"] = np.round(dm["viewer_transform"][:3, :4].astype(np.float64), 6).tolist()
        if dm.get("expected_scene_scale") is not None:
            dm["expected_scene_scale"] = round(dm["expected_scene_scale"], 6)
        return dm

    if "dataset_metadata" in info:
        info["dataset_metadata"] = fix_dm(info["dataset_metadata"])
    if "render_dataset_metadata" in info:
        info["render_dataset_metadata"] = fix_dm(info["render_dataset_metadata"])

    def ts(x):
        _ = info
        if isinstance(x, np.ndarray):
            raise NotImplementedError("Numpy arrays are not supported in nb-info")
        if isinstance(x, dict):
            return {k: ts(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return [ts(v) for v in x]
        else:
            return x
    ts(info)
    return info


def deserialize_nb_info(info: dict) -> dict:
    info = info.copy()
    def fix_dm(dm):
        if dm is None:
            return None
        dm = dm.copy()
        if dm.get("background_color") is not None:
            dm["background_color"] = np.array(dm["background_color"], dtype=np.uint8)
        if "viewer_initial_pose" in dm:
            dm["viewer_initial_pose"] = np.array(dm["viewer_initial_pose"], dtype=np.float32)
        if "viewer_transform" in dm:
            dm["viewer_transform"] = np.array(dm["viewer_transform"], dtype=np.float32)
        return dm
    if "dataset_metadata" in info:
        info["dataset_metadata"] = fix_dm(info["dataset_metadata"])
    if "render_dataset_metadata" in info:
        info["render_dataset_metadata"] = fix_dm(info["render_dataset_metadata"])
    return info


def new_nb_info(train_dataset_metadata, 
                method, 
                config_overrides, 
                evaluation_protocol=None,
                resources_utilization_info=None,
                total_train_time=None):
    dataset_metadata = train_dataset_metadata.copy()
    model_info = method.get_info()

    if evaluation_protocol is None:
        evaluation_protocol = "default"
        evaluation_protocol = dataset_metadata.get("evaluation_protocol", evaluation_protocol)
    if not isinstance(evaluation_protocol, str):
        evaluation_protocol = evaluation_protocol.get_name()
    method_id = model_info.get("method_id", model_info.get("name"))
    return {
        "method": method_id,
        "num_iterations": model_info["num_iterations"],
        "total_train_time": round(total_train_time, 5) if total_train_time is not None else None,
        "resources_utilization": resources_utilization_info,
        # Date time in ISO 8601 format
        "datetime": datetime.utcnow().isoformat(timespec="seconds"),
        "config_overrides": config_overrides,
        "dataset_metadata": dataset_metadata,
        "evaluation_protocol": evaluation_protocol,

        # Store hparams
        "hparams": method.get_info().get("hparams"),
    }
