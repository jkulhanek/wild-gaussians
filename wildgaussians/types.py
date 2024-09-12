import sys
from abc import abstractmethod
import typing
from typing import Optional, Iterable, List, Dict, Any, cast, Union, Sequence, TYPE_CHECKING, overload, TypeVar, Iterator, Callable, Tuple
from dataclasses import dataclass
import dataclasses
import os
import numpy as np
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
try:
    from typing import Generic
except ImportError:
    from typing_extensions import Generic
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
try:
    from typing import runtime_checkable
except ImportError:
    from typing_extensions import runtime_checkable
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
try:
    from typing import get_args as get_args
    from typing import get_origin as get_origin
except ImportError:
    from typing_extensions import get_args as get_args
    from typing_extensions import get_origin as get_origin
try:
    from typing import NotRequired
    from typing import Required
    from typing import TypedDict
except ImportError:
    from typing_extensions import NotRequired
    from typing_extensions import Required
    from typing_extensions import TypedDict
try:
    from typing import FrozenSet
except ImportError:
    from typing_extensions import FrozenSet


if TYPE_CHECKING:
    import torch


WG_PREFIX = os.path.expanduser(os.environ.get("WILD_GAUSSIANS_PREFIX", "~/.cache/wild-gaussians"))
ColorSpace = Literal["srgb", "linear"]
CameraModel = Literal["pinhole", "opencv", "opencv_fisheye", "full_opencv"]
DatasetFeature = Literal["color", "points3D_xyz", "points3D_rgb"]
TTensor = TypeVar("TTensor", np.ndarray, "torch.Tensor")
TTensor_co = TypeVar("TTensor_co", np.ndarray, "torch.Tensor", covariant=True)


@overload
def _get_xnp(tensor: np.ndarray):
    return np


@overload
def _get_xnp(tensor: 'torch.Tensor'):
    return cast('torch', sys.modules["torch"])


def _get_xnp(tensor: TTensor):
    if isinstance(tensor, np.ndarray):
        return np
    if tensor.__module__ == "torch":
        return cast('torch', sys.modules["torch"])
    raise ValueError(f"Unknown tensor type {type(tensor)}")


def camera_model_to_int(camera_model: CameraModel) -> int:
    camera_models = get_args(CameraModel)
    if camera_model not in camera_models:
        raise ValueError(f"Unknown camera model {camera_model}, known models are {camera_models}")
    return get_args(CameraModel).index(camera_model)


def camera_model_from_int(i: int) -> CameraModel:
    camera_models = get_args(CameraModel)
    if i >= len(camera_models):
        raise ValueError(f"Unknown camera model with index {i}, known models are {camera_models}")
    return get_args(CameraModel)[i]


class GenericCameras(Protocol[TTensor_co]):
    @property
    def poses(self) -> TTensor_co:
        """Camera-to-world matrices, [N, (R, t)]"""
        ...

    @property
    def intrinsics(self) -> TTensor_co:
        """Intrinsics, [N, (fx,fy,cx,cy)]"""
        ...

    @property
    def camera_models(self) -> TTensor_co:
        """Camera types, [N]"""
        ...

    @property
    def distortion_parameters(self) -> TTensor_co:
        """Distortion parameters, [N, num_params]"""
        ...

    @property
    def image_sizes(self) -> TTensor_co:
        """Image sizes, [N, 2]"""
        ...

    @property
    def nears_fars(self) -> Optional[TTensor_co]:
        """Near and far planes, [N, 2]"""
        ...

    @property
    def metadata(self) -> Optional[TTensor_co]:
        """Metadata, [N, ...]"""
        ...

    def __len__(self) -> int:
        ...

    def item(self) -> Self:
        """Returns a single camera if there is only one. Otherwise raises an error."""
        ...

    def __getitem__(self, index) -> Self:
        ...

    def __setitem__(self, index, value: Self) -> None:
        ...

    def __iter__(self) -> Iterator[Self]:
        ...

    @classmethod
    def cat(cls, values: Sequence[Self]) -> Self:
        ...

    def replace(self, **changes) -> Self:
        ...

    def apply(self, fn: Callable[[TTensor_co, str], TTensor]) -> 'GenericCameras[TTensor]':
        ...


@runtime_checkable
class Cameras(GenericCameras[np.ndarray], Protocol):
    pass


@dataclass(frozen=True)
class GenericCamerasImpl(Generic[TTensor_co]):
    poses: TTensor_co  # [N, (R, t)]
    intrinsics: TTensor_co  # [N, (fx,fy,cx,cy)]

    camera_models: TTensor_co  # [N]
    distortion_parameters: TTensor_co  # [N, num_params]
    image_sizes: TTensor_co  # [N, 2]

    nears_fars: Optional[TTensor_co]  # [N, 2]
    metadata: Optional[TTensor_co] = None

    def __len__(self) -> int:
        return 1 if len(self.poses.shape) == 2 else len(self.poses)

    def item(self):
        assert len(self) == 1, "Cameras must have exactly one element to be converted to a single camera"
        return self if len(self.poses.shape) == 2 else self[0]

    def __getitem__(self, index):
        return type(self)(
            poses=self.poses[index],
            intrinsics=self.intrinsics[index],
            camera_models=self.camera_models[index],
            distortion_parameters=self.distortion_parameters[index],
            image_sizes=self.image_sizes[index],
            nears_fars=self.nears_fars[index] if self.nears_fars is not None else None,
            metadata=self.metadata[index] if self.metadata is not None else None,
        )

    def __setitem__(self, index, value: Self) -> None:
        assert (self.image_sizes is None) == (value.image_sizes is None), "Either both or none of the cameras must have image sizes"
        assert (self.nears_fars is None) == (value.nears_fars is None), "Either both or none of the cameras must have nears and fars"
        self.poses[index] = value.poses
        self.intrinsics[index] = value.intrinsics
        self.camera_models[index] = value.camera_models
        self.distortion_parameters[index] = value.distortion_parameters
        self.image_sizes[index] = value.image_sizes
        if self.nears_fars is not None:
            self.nears_fars[index] = cast(TTensor_co, value.nears_fars)
        if self.metadata is not None:
            self.metadata[index] = cast(TTensor_co, value.metadata)

    def __iter__(self) -> Iterator[Self]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def cat(cls, values: Sequence[Self]) -> Self:
        xnp = _get_xnp(values[0].poses)
        nears_fars: Optional[TTensor_co] = None
        metadata: Optional[TTensor_co] = None
        if any(v.nears_fars is not None for v in values):
            assert all(v.nears_fars is not None for v in values), "Either all or none of the cameras must have nears and fars"
            nears_fars = xnp.concatenate([cast(TTensor_co, v.nears_fars) for v in values])
        if any(v.metadata is not None for v in values):
            assert all(v.metadata is not None for v in values), "Either all or none of the cameras must have metadata"
            metadata = xnp.concatenate([cast(TTensor_co, v.metadata) for v in values])
        return cls(
            poses=xnp.concatenate([v.poses for v in values]),
            intrinsics=xnp.concatenate([v.intrinsics for v in values]),
            camera_models=xnp.concatenate([v.camera_models for v in values]),
            distortion_parameters=xnp.concatenate([v.distortion_parameters for v in values]),
            image_sizes=xnp.concatenate([cast(TTensor_co, v.image_sizes) for v in values]),
            nears_fars=nears_fars,
            metadata=metadata,
        )

    def replace(self, **changes) -> Self:
        return dataclasses.replace(self, **changes)

    def apply(self, fn: Callable[[TTensor_co, str], TTensor]) -> 'GenericCamerasImpl[TTensor]':
        return GenericCamerasImpl[TTensor](
            poses=fn(self.poses, "poses"),
            intrinsics=fn(self.intrinsics, "intrinsics"),
            camera_models=fn(self.camera_models, "camera_models"),
            distortion_parameters=fn(self.distortion_parameters, "distortion_parameters"),
            image_sizes=fn(self.image_sizes, "image_sizes"),
            nears_fars=fn(cast(TTensor_co, self.nears_fars), "nears_fars") if self.nears_fars is not None else None,
            metadata=fn(cast(TTensor_co, self.metadata), "metadata") if self.metadata is not None else None,
        )


def new_cameras(
    *,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    camera_models: np.ndarray,
    distortion_parameters: np.ndarray,
    image_sizes: np.ndarray,
    nears_fars: Optional[np.ndarray] = None,
    metadata: Optional[np.ndarray] = None,
) -> Cameras:
    return GenericCamerasImpl[np.ndarray](
        poses=poses,
        intrinsics=intrinsics,
        camera_models=camera_models,
        distortion_parameters=distortion_parameters,
        image_sizes=image_sizes,
        nears_fars=nears_fars,
        metadata=metadata)
    

class _IncompleteDataset(TypedDict, total=True):
    cameras: Cameras  # [N]

    image_paths: List[str]
    image_paths_root: str
    sampling_mask_paths: Optional[List[str]]
    sampling_mask_paths_root: Optional[str]
    metadata: Dict
    sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]]  # [N][H, W]
    points3D_xyz: Optional[np.ndarray]  # [M, 3]
    points3D_rgb: Optional[np.ndarray]  # [M, 3]
    images_points3D_indices: Optional[List[np.ndarray]]  # [N][<M]


class UnloadedDataset(_IncompleteDataset):
    images: NotRequired[Optional[Union[np.ndarray, List[np.ndarray]]]]  # [N][H, W, 3]


class Dataset(_IncompleteDataset):
    images: Union[np.ndarray, List[np.ndarray]]  # [N][H, W, 3]


class RenderOutput(TypedDict, total=False):
    color: Required[np.ndarray]  # [h w 3]
    depth: np.ndarray  # [h w]
    accumulation: np.ndarray  # [h w]


class OptimizeEmbeddingOutput(TypedDict):
    embedding: np.ndarray
    render_output: RenderOutput
    metrics: NotRequired[Dict[str, Sequence[float]]]


class MethodInfo(TypedDict, total=False):
    method_id: Required[str]
    required_features: FrozenSet[DatasetFeature]
    supported_camera_models: FrozenSet


class ModelInfo(TypedDict, total=False):
    method_id: Required[str]
    num_iterations: Required[int]
    loaded_step: Optional[int]
    loaded_checkpoint: Optional[str]
    batch_size: int
    eval_batch_size: int
    required_features: FrozenSet[DatasetFeature]
    supported_camera_models: FrozenSet
    hparams: Dict[str, Any]


@runtime_checkable
class Method(Protocol):
    def __init__(self, 
                 *,
                 checkpoint: Union[str, None] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[Dict[str, Any]] = None):
        pass

    @classmethod
    def install(cls):
        """
        Install the method.
        """
        pass

    @classmethod
    @abstractmethod
    def get_method_info(cls) -> MethodInfo:
        """
        Get method info needed to initialize the datasets.

        Returns:
            Method info.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """
        Get method defaults for the trainer.

        Returns:
            Method info.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        """
        Get the embedding for the given image index.

        Args:
            index: Image index.

        Returns:
            Image embedding.
        """
        return None

    @abstractmethod
    def optimize_embedding(self, dataset: Dataset, *, embedding: Optional[np.ndarray] = None) -> OptimizeEmbeddingOutput:
        """
        Optimize embeddings for single image (passed as dataset slice).

        Args:
            dataset: Dataset.
            embedding: Optional initial embedding.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self, camera: Cameras, *, options: Optional[Dict] = None) -> RenderOutput:  # [h w c]
        """
        Render images.

        Args:
            cameras: Cameras.
            options: Render options
        """
        raise NotImplementedError()

    @abstractmethod
    def train_iteration(self, step: int):
        """
        Train one iteration.

        Args:
            step: Current step.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str):
        """
        Save model.

        Args:
            path: Path to save.
        """
        raise NotImplementedError()


@runtime_checkable
class EvaluationProtocol(Protocol):
    def get_name(self) -> str:
        ...
        
    def render(self, method: Method, dataset: Dataset) -> RenderOutput:
        ...

    def evaluate(self, predictions: RenderOutput, dataset: Dataset) -> Dict[str, Union[float, int]]:
        ...

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        ...


class DatasetSpecMetadata(TypedDict, total=False):
    id: str
    name: str
    description: str
    paper_title: str
    paper_authors: List[str]
    paper_link: str
    link: str
    metrics: List[str]
    default_metric: str
    scenes: List[Dict[str, str]]


class LoadDatasetFunction(Protocol):
    def __call__(self, 
                 path: str, 
                 split: str, 
                 features: Optional[FrozenSet[DatasetFeature]] = None, 
                 **kwargs) -> UnloadedDataset:
        ...


class DownloadDatasetFunction(Protocol):
    def __call__(self, 
                 path: str, 
                 output: str) -> None:
        ...


class TrajectoryFrameAppearance(TypedDict, total=False):
    embedding: Optional[np.ndarray]
    embedding_train_index: Optional[int]


class TrajectoryFrame(TypedDict, total=True):
    pose: np.ndarray
    intrinsics: np.ndarray
    appearance_weights: NotRequired[np.ndarray]


class TrajectoryKeyframe(TypedDict, total=True):
    pose: np.ndarray
    fov: Optional[float]
    transition_duration: NotRequired[Optional[float]]
    appearance: NotRequired[TrajectoryFrameAppearance]


TrajectoryInterpolationType = Literal["kochanek-bartels", "none"]


class ImageSetInterpolationSource(TypedDict, total=True):
    type: Literal["interpolation"]
    interpolation: Literal["none"]
    keyframes: List[TrajectoryKeyframe]
    default_fov: float
    default_transition_duration: float
    default_appearance: NotRequired[Optional[TrajectoryFrameAppearance]]


class KochanekBartelsInterpolationSource(TypedDict, total=True):
    type: Literal["interpolation"]
    interpolation: Literal["kochanek-bartels"]
    is_cycle: bool
    tension: float
    keyframes: List[TrajectoryKeyframe]
    default_fov: float
    default_transition_duration: float
    default_appearance: NotRequired[Optional[TrajectoryFrameAppearance]]


TrajectoryInterpolationSource = Union[ImageSetInterpolationSource, KochanekBartelsInterpolationSource]


class Trajectory(TypedDict, total=True):
    camera_model: CameraModel
    image_size: Tuple[int, int]
    frames: List[TrajectoryFrame]
    appearances: NotRequired[List[TrajectoryFrameAppearance]]
    fps: float
    source: NotRequired[Optional[TrajectoryInterpolationSource]]


@runtime_checkable
class LoggerEvent(Protocol):
    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        ...

    def add_text(self, tag: str, text: str) -> None:
        ...

    def add_image(self, tag: str, image: np.ndarray, display_name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> None:
        ...

    def add_embedding(self, tag: str, embeddings: np.ndarray, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        ...

    def add_plot(self, tag: str, *data: np.ndarray,
                 axes_labels: Optional[Sequence[str]] = None, 
                 title: Optional[str] = None,
                 **kwargs) -> None:
        ...

    def add_histogram(self, tag: str, values: np.ndarray, *, num_bins: Optional[int] = None) -> None:
        ...


@runtime_checkable
class Logger(Protocol):
    def add_event(self, step: int) -> typing.ContextManager[LoggerEvent]:
        ...

    def add_scalar(self, tag: str, value: Union[float, int], step: int) -> None:
        ...

    def add_text(self, tag: str, text: str, step: int) -> None:
        ...

    def add_image(self, tag: str, image: np.ndarray, step: int, *, display_name: Optional[str] = None, description: Optional[str] = None) -> None:
        ...

    def add_embedding(self, tag: str, embeddings: np.ndarray, step: int, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        ...
