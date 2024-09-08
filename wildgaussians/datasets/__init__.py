from typing import Union, Optional, overload
import logging
from pathlib import Path
from ..types import Dataset, DatasetFeature, CameraModel, FrozenSet, WG_PREFIX
from ._common import dataset_load_features as dataset_load_features
from ._common import dataset_index_select as dataset_index_select
from ._common import new_dataset as new_dataset
from ..types import UnloadedDataset, Literal


@overload
def load_dataset(
        path: Union[Path, str], 
        split: str, 
        features: Optional[FrozenSet[DatasetFeature]] = ...,
        supported_camera_models: Optional[FrozenSet[CameraModel]] = ...,
        load_features: Literal[True] = ...,
        **kwargs) -> Dataset:
    ...


@overload
def load_dataset(
        path: Union[Path, str], 
        split: str, 
        features: Optional[FrozenSet[DatasetFeature]] = ...,
        supported_camera_models: Optional[FrozenSet[CameraModel]] = ...,
        load_features: Literal[False] = ...,
        **kwargs) -> UnloadedDataset:
    ...



def load_dataset(
        path: Union[Path, str], 
        split: str, 
        features: Optional[FrozenSet[DatasetFeature]] = None,
        supported_camera_models: Optional[FrozenSet[CameraModel]] = None,
        load_features: bool = True,
        *,
        load_dataset_fn,
        download_dataset_fn=None,
        evaluation_protocol: Optional[str] = None,
        **kwargs,
        ) -> Union[Dataset, UnloadedDataset]:
    path = str(path)
    if features is None:
        features = frozenset(("color",))
    if supported_camera_models is None:
        supported_camera_models = frozenset(("pinhole",))
    # If path is and external path, we download the dataset first
    if path.startswith("external://") and download_dataset_fn is not None:
        dataset = path.split("://", 1)[1]
        path = Path(WG_PREFIX) / "datasets" / dataset
        if not path.exists():
            download_dataset_fn(dataset, path)
        path = str(path)

    dataset_instance = None
    dataset_instance = load_dataset_fn(path, split=split, features=features, **kwargs)
    logging.info(f"loaded dataset from path {path}")

    # Set correct eval protocol
    if evaluation_protocol is not None:
        dataset_instance["metadata"]["evaluation_protocol"] = evaluation_protocol or "default"

    if load_features:
        return dataset_load_features(dataset_instance, features=features, supported_camera_models=supported_camera_models)
    return dataset_instance
