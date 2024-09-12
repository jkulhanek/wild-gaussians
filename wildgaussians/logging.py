import shutil
import math
import warnings
import os
from PIL import Image
import numpy as np
import io
import contextlib

from pathlib import Path
import typing
from typing import Optional, Union, List, Dict, Sequence, Any, cast
from typing import TYPE_CHECKING
from .types import Logger, LoggerEvent
from .utils import convert_image_dtype

if TYPE_CHECKING:
    import wandb.sdk.wandb_run


def _flatten_simplify_hparams(hparams: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    def simplify(v):
        if isinstance(v, (tuple, list)):
            chars = "()" if isinstance(v, tuple) else "[]"
            flat[k] = chars[0] + ",".join(simplify(x) for x in v) + chars[1]
        if isinstance(v, dict):
            return "{" + ",".join(f"{k}:{simplify(v)}" for k, v in v.items()) + "}"
        if isinstance(v, (float, int)):
            return str(v)
        if isinstance(v, (str, Path)):
            return f"'{v}'"
        return str(v)

    for k, v in hparams.items():
        if prefix:
            k = f"{prefix}/{k}"
        if isinstance(v, Path):
            flat[k] = str(v)
        elif isinstance(v, (tuple, list, dict)):
            flat[k] = simplify(v)
        else:
            flat[k] = v
    return flat

class BaseLoggerEvent(LoggerEvent):
    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        raise NotImplementedError()
    
    def add_text(self, tag: str, text: str) -> None:
        raise NotImplementedError()
    
    def add_image(self, tag: str, image: np.ndarray, display_name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> None:
        raise NotImplementedError()
    
    def add_embedding(self, tag: str, embeddings: np.ndarray, *,
                        images: Optional[List[np.ndarray]] = None, 
                        labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        raise NotImplementedError()
    
    def add_plot(self, tag: str, *data: np.ndarray,
                 axes_labels: Optional[Sequence[str]] = None, 
                 title: Optional[str] = None,
                 colors: Optional[Sequence[np.ndarray]] = None,
                 labels: Optional[Sequence[str]] = None,
                 **kwargs) -> None:
        assert len(data) > 0, "At least one data array should be provided"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        assert all(len(d.shape) == 2 for d in data), "All data should have two dimensions"
        assert all(d.shape[1] == data[0].shape[1] for d in data), "All data should have the same number of columns"
        num_dim = data[0].shape[1]
        if axes_labels is None:
            axes_labels = ["x", "y", "z"][:num_dim]
        else:
            assert num_dim == len(axes_labels), "All data should have the same number of columns as axes_labels"
        assert data[0].shape[1] == 2, "Only 2D plots are supported"

        colors_mpl = None
        if colors is not None:
            assert len(colors) == len(data), "Number of colors should match number of data arrays"
            assert all(c.shape == (3,) for c in colors), "All colors should be RGB"
            colors_mpl = [tuple((c / 255).tolist()) for c in colors]
        
        if labels is not None:
            assert len(labels) == len(data), "Number of labels should match number of data arrays"

        # Render the image using matplotlib
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        for i, d in enumerate(data):
            x, y = d.T
            kwargs = {}
            if colors_mpl is not None:
                kwargs["color"] = colors_mpl[i]
            if labels is not None:
                kwargs["label"] = labels[i]
            ax.plot(x, y, **kwargs)
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])

        # Render plot as np array
        fig.canvas.draw()
        with io.BytesIO() as img_buf:
            fig.savefig(img_buf, format='png')
            img_buf.seek(0)
            plot_img = np.array(Image.open(img_buf))
        plt.close(fig)
        self.add_image(tag, plot_img, display_name=title, description=title)


class BaseLogger(Logger):
    def add_event(self, step: int) -> typing.ContextManager[LoggerEvent]:
        raise NotImplementedError()

    def add_scalar(self, tag: str, value: Union[float, int], step: int):
        with self.add_event(step) as event:
            event.add_scalar(tag, value)

    def add_text(self, tag: str, text: str, step: int):
        with self.add_event(step) as event:
            event.add_text(tag, text)

    def add_image(self, tag: str, image: np.ndarray, step: int, *, display_name: Optional[str] = None, description: Optional[str] = None, **kwargs):
        with self.add_event(step) as event:
            event.add_image(tag, image, display_name, description, **kwargs)

    def add_embedding(self, tag: str, embeddings: np.ndarray, step: int, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        with self.add_event(step) as event:
            event.add_embedding(tag, embeddings, images=images, labels=labels)

    def add_hparams(self, hparams: Dict[str, Any]):
        raise NotImplementedError()


class WandbLoggerEvent(BaseLoggerEvent):
    def __init__(self, commit):
        self._commit: Dict[str, Any] = commit

    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        self._commit[tag] = value

    def add_text(self, tag: str, text: str) -> None:
        self._commit[tag] = text

    def add_image(self, tag: str, image: np.ndarray, display_name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> None:
        import wandb
        self._commit[tag] = [wandb.Image(image, caption=description)]

    def add_histogram(self, tag: str, values: np.ndarray, *, num_bins: Optional[int] = None) -> None:
        import wandb
        if num_bins is not None:
            self._commit[tag] = wandb.Histogram(cast(Any, values), num_bins=num_bins)
        else:
            self._commit[tag] = wandb.Histogram(cast(Any, values))

    def add_embedding(self, tag: str, embeddings: np.ndarray, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        import wandb
        table = wandb.Table([])
        table.add_column("embedding", embeddings)
        if labels is not None:
            if isinstance(labels[0], dict):
                for key in labels[0].keys():
                    table.add_column(key, [cast(dict, label)[key] for label in labels])
            else:
                table.add_column("label", labels)
        if images is not None:
            table.add_column("image", [wandb.Image(image) for image in images])
        self._commit[tag] = table

    def add_plot(self, tag: str, *data: np.ndarray, 
                 axes_labels: Optional[Sequence[str]] = None, 
                 title: Optional[str] = None, 
                 colors: Optional[Sequence[np.ndarray]] = None, 
                 labels: Optional[Sequence[str]] = None, **kwargs) -> None:
        if len(data) == 0 or data[0].shape[-1] != 2 or colors is not None:
            # Not supported by WandbLogger
            return super().add_plot(tag, *data, axes_labels=axes_labels, title=title, colors=colors, labels=labels, **kwargs)

        import wandb

        if colors is not None:
            warnings.warn("WandbLogger does not support colors for add_plot, ignoring them")

        xlabel, ylabel = axes_labels or ["x", "y"]
        if len(data) == 1:
            table = wandb.Table(data=[[x, y] for x, y in data[0]], columns=[xlabel, ylabel])
            self._commit[tag] = wandb.plot.line(  # type: ignore
                table,
                x=xlabel,
                y=ylabel,
                title=title
            )
        else:
            self._commit[tag] = wandb.plot.line_series(  # type: ignore
                [x[:, 0] for x in data],
                [x[:, 1] for x in data],
                keys=labels,
                title=title,
                xname=xlabel,
            )
    

class WandbLogger(BaseLogger):
    def __init__(self, output: Union[str, Path], **kwargs):
        # wandb does not support python 3.7, therefore, we patch it
        try:
            from typing import Literal
        except ImportError:
            from typing_extensions import Literal
            typing.Literal = Literal  # type: ignore
                    
        import wandb
        wandb_run: "wandb.sdk.wandb_run.Run" = typing.cast(
            "wandb.sdk.wandb_run.Run", 
            wandb.init(dir=str(output), **kwargs))
        self._wandb_run = wandb_run
        self._wandb = wandb

    @contextlib.contextmanager
    def add_event(self, step: int):
        commit = {}
        yield WandbLoggerEvent(commit)
        self._wandb_run.log(commit, step=step)

    def add_hparams(self, hparams: Dict[str, Any]):
        self._wandb_run.config.update(_flatten_simplify_hparams(hparams))

    def __str__(self):
        return "wandb"


if TYPE_CHECKING:
    _ConcatLoggerEventBase = LoggerEvent
else:
    _ConcatLoggerEventBase = object


class ConcatLoggerEvent(_ConcatLoggerEventBase):
    def __init__(self, events):
        self.events = events

    def __getattr__(self, name):
        callbacks = []
        for event in self.events:
            callbacks.append(getattr(event, name))
        def call(*args, **kwargs):
            for callback in callbacks:
                callback(*args, **kwargs)
        return call


class ConcatLogger(BaseLogger):
    def __init__(self, loggers):
        self.loggers = loggers

    def __bool__(self):
        return bool(self.loggers)

    @contextlib.contextmanager
    def add_event(self, step: int):
        def enter_event(loggers, events):
            if loggers:
                with loggers[0].add_event(step) as event:
                    yield from enter_event(loggers[1:], [event] + events)
            else:
                yield ConcatLoggerEvent(events)  # type: ignore
        yield from enter_event(self.loggers, [])

    def add_hparams(self, hparams: Dict[str, Any], **kwargs):
        for logger in self.loggers:
            logger.add_hparams(hparams, **kwargs)

    def __str__(self):
        if not self:
            return "[]"
        return ",".join(map(str, self.loggers))


class TensorboardLoggerEvent(BaseLoggerEvent):
    def __init__(self, logdir, summaries, step):
        os.makedirs(logdir, exist_ok=True)
        self._step = step
        self._logdir = logdir
        self._summaries = summaries

        # Create default bins for histograms, see generate_testdata.py in tensorflow/tensorboard
        v = 1e-12
        buckets = []
        neg_buckets = []
        while v < 1e20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets

    @staticmethod
    def _encode(rawstr):
        # I'd use urllib but, I'm unsure about the differences from python3 to python2, etc.
        retval = rawstr
        retval = retval.replace("%", f"%{ord('%'):02x}")
        retval = retval.replace("/", f"%{ord('/'):02x}")
        retval = retval.replace("\\", "%%%02x" % (ord("\\")))
        return retval

    def add_image(
        self,
        tag: str,
        image: np.ndarray,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> None:
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.plugins.image.metadata import create_summary_metadata

        metadata = None
        if display_name is not None or description is not None:
            metadata = create_summary_metadata(
                display_name=display_name or "",
                description=description or "",
            )
        with io.BytesIO() as simg:
            image = convert_image_dtype(image, np.uint8)
            Image.fromarray(image).save(simg, format="png")
            self._summaries.append(
                Summary.Value(  # type: ignore
                    tag=tag,
                    metadata=metadata,
                    image=Summary.Image(  # type: ignore
                        encoded_image_string=simg.getvalue(),
                        height=image.shape[0],
                        width=image.shape[1],
                    ),
                )
            )

    def add_text(self, tag: str, text: str) -> None:
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData

        plugin_data = SummaryMetadata.PluginData(  # type: ignore
            plugin_name="text", content=TextPluginData(version=0).SerializeToString()  # type: ignore
        )  # type: ignore
        smd = SummaryMetadata(plugin_data=plugin_data)  # type: ignore
        tensor = TensorProto(
            dtype="DT_STRING",  # type: ignore
            string_val=[text.encode("utf8")],  # type: ignore
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),  # type: ignore
        )  # type: ignore
        self._summaries.append(Summary.Value(tag=tag, metadata=smd, tensor=tensor))  # type: ignore

    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        from tensorboard.compat.proto.summary_pb2 import Summary

        assert isinstance(value, (float, int))
        self._summaries.append(Summary.Value(tag=tag, simple_value=value))  # type: ignore

    def add_embedding(self, tag: str, embeddings: np.ndarray, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
        from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo
        from tensorboard.compat import tf
        from google.protobuf import text_format

        def make_sprite(label_img, save_path):
            # Background white for white tensorboard theme and black for dark theme
            background = 255

            # this ensures the sprite image has correct dimension as described in
            # https://www.tensorflow.org/get_started/embedding_viz
            nrow = int(math.ceil(len(label_img) ** 0.5))
            label_img = [convert_image_dtype(img, np.uint8) for img in label_img]
            mh, mw = max(img.shape[0] for img in label_img), max(img.shape[1] for img in label_img)

            arranged_augment_square_HWC = np.full((mh * nrow, mw * nrow, 3), background, dtype=np.uint8)
            for i, image in enumerate(label_img):
                img_width = ow = image.shape[1]
                img_height = oh = image.shape[0]
                aspect = img_width / img_height
                img_width = int(min(mw, aspect * mh))
                img_height = int(img_width / aspect)
                if img_width != ow or img_height != oh:
                    img = Image.fromarray(image)
                    img = img.resize((img_width, img_height))
                    image = np.array(img)
                x = i % nrow
                y = i // nrow
                h, w = image.shape[:2]
                offx = x * mw + (mw - w) // 2
                offy = y * mh + (mh - h) // 2
                arranged_augment_square_HWC[offy : offy + h, offx : offx + w] = image

            im = Image.fromarray(arranged_augment_square_HWC)
            im.save(os.path.join(str(save_path), "sprite.png"))
            return mw, mh

        # Maybe we should encode the tag so slashes don't trip us up?
        # I don't think this will mess us up, but better safe than sorry.
        subdir = Path(f"{str(self._step).zfill(5)}/{self._encode(tag)}")
        save_path = Path(self._logdir) / subdir

        if save_path.exists():
            shutil.rmtree(str(save_path))
            warnings.warn(f"Removing existing log directory: {save_path}")
        save_path.mkdir(parents=True)

        if labels is not None:
            assert len(labels) == len(embeddings), "#labels should equal with #data points"
            tsv = []
            if len(labels) > 0:
                if isinstance(labels[0], dict):
                    metadata_header = list(labels[0].keys())
                    metadata = [metadata_header] + [[str(cast(Dict, x).get(k, "")) for k in metadata_header] for x in labels]
                    tsv = ["\t".join(str(e) for e in ln) for ln in metadata]
                else:
                    metadata = labels
                    tsv = [str(x) for x in metadata]
            metadata_bytes = tf.compat.as_bytes("\n".join(tsv) + "\n")
            with (save_path / "metadata.tsv").open("wb") as f:
                f.write(metadata_bytes)

        label_img_size = None
        if images is not None:
            assert (
                len(images) == embeddings.shape[0]
            ), "#images should equal with #data points"
            label_img_size = make_sprite(images, save_path)

        assert (
            embeddings.ndim == 2
        ), "mat should be 2D, where mat.size(0) is the number of data points"
        with (save_path / "tensors.tsv").open("wb") as f:
            for x in embeddings:
                x = [str(i.item()) for i in x]
                f.write(tf.compat.as_bytes("\t".join(x) + "\n"))

        projector_config: Any = ProjectorConfig()
        if (Path(self._logdir) / "projector_config.pbtxt").exists():
            message_bytes = (Path(self._logdir) / "projector_config.pbtxt").read_bytes()
            projector_config = text_format.Parse(message_bytes, projector_config)

        embedding_info: Any = EmbeddingInfo()  # type: ignore
        embedding_info.tensor_name = f"{tag}:{str(self._step).zfill(5)}"
        embedding_info.tensor_path = str(subdir / "tensors.tsv")
        if labels is not None:
            embedding_info.metadata_path = str(subdir / "metadata.tsv")
        if images is not None:
            assert label_img_size is not None, "label_img_size should not be None"
            embedding_info.sprite.image_path = str(subdir / "sprite.png")
            embedding_info.sprite.single_image_dim.extend(label_img_size)
        projector_config.embeddings.extend([embedding_info])

        config_pbtxt = text_format.MessageToString(projector_config)
        with (Path(self._logdir) / "projector_config.pbtxt").open("wb") as f:
            f.write(tf.compat.as_bytes(config_pbtxt))


    def add_histogram(self,
                      tag: str,
                      values: np.ndarray,
                      *,
                      num_bins: Optional[int] = None):
        """Add histogram to summary.

        Args:
            tag: Data identifier
            values: Values to build histogram
        """
        from tensorboard.compat.proto.summary_pb2 import Summary, HistogramProto

        max_bins = num_bins
        bins = self.default_bins

        def make_histogram(values, bins, max_bins=None):
            """Convert values into a histogram proto using logic from histogram.cc."""
            if values.size == 0:
                raise ValueError("The input has no element.")
            values = values.reshape(-1)
            counts, limits = np.histogram(values, bins=bins)
            num_bins = len(counts)
            if max_bins is not None and num_bins > max_bins:
                subsampling = num_bins // max_bins
                subsampling_remainder = num_bins % subsampling
                if subsampling_remainder != 0:
                    counts = np.pad(
                        counts,
                        pad_width=[[0, subsampling - subsampling_remainder]],
                        mode="constant",
                        constant_values=0,
                    )
                counts = counts.reshape(-1, subsampling).sum(axis=-1)
                new_limits = np.empty((counts.size + 1,), limits.dtype)
                new_limits[:-1] = limits[:-1:subsampling]
                new_limits[-1] = limits[-1]
                limits = new_limits

            # Find the first and the last bin defining the support of the histogram:

            cum_counts = np.cumsum(np.greater(counts, 0))
            start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
            start = int(start)
            end = int(end) + 1
            del cum_counts

            # TensorBoard only includes the right bin limits. To still have the leftmost limit
            # included, we include an empty bin left.
            # If start == 0, we need to add an empty one left, otherwise we can just include the bin left to the
            # first nonzero-count bin:
            counts = (
                counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
            )
            limits = limits[start : end + 1]

            if counts.size == 0 or limits.size == 0:
                raise ValueError("The histogram is empty, please file a bug report.")

            sum_sq = values.dot(values)
            return HistogramProto(  # type: ignore
                min=values.min(),  # type: ignore
                max=values.max(),  # type: ignore
                num=len(values),  # type: ignore
                sum=values.sum(),  # type: ignore
                sum_squares=sum_sq,  # type: ignore
                bucket_limit=limits.tolist(),  # type: ignore
                bucket=counts.tolist(),  # type: ignore
            )

        hist = make_histogram(values, bins, max_bins)
        self._summaries.append(Summary.Value(tag=tag, histo=hist))  # type: ignore


def _tensorboard_hparams(hparam_dict=None, metrics_list=None, hparam_domain_discrete=None):
    from tensorboard.plugins.hparams.api_pb2 import ( # type: ignore
        DataType,
        Experiment,
        HParamInfo, # type: ignore
        MetricInfo,
        MetricName,
        Status,
    ) # type: ignore
    from tensorboard.plugins.hparams.metadata import (
        EXPERIMENT_TAG,
        PLUGIN_DATA_VERSION,
        PLUGIN_NAME,
        SESSION_END_INFO_TAG,
        SESSION_START_INFO_TAG,
    )
    from tensorboard.plugins.hparams.plugin_data_pb2 import (
        HParamsPluginData,
        SessionEndInfo,
        SessionStartInfo,
    )
    from tensorboard.compat.proto.summary_pb2 import (
        Summary,
        SummaryMetadata,
    )
    from google.protobuf import struct_pb2

    hparam_dict = hparam_dict or {}
    hparam_domain_discrete = hparam_domain_discrete or {}
    for k, v in hparam_domain_discrete.items():
        if (
            k not in hparam_dict
            or not isinstance(v, list)
            or not all(isinstance(d, type(hparam_dict[k])) for d in v)
        ):
            raise TypeError(
                f"parameter: hparam_domain_discrete[{k}] should be a list of same type as hparam_dict[{k}]."
            )
    hps = []

    ssi: Any = SessionStartInfo()
    for k, v in hparam_dict.items():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            ssi.hparams[k].number_value = v

            if k in hparam_domain_discrete:
                domain_discrete: Optional[struct_pb2.ListValue] = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(number_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            hps.append(
                HParamInfo(  # type: ignore
                    name=k,  # type: ignore
                    type=DataType.Value("DATA_TYPE_FLOAT64"),  # type: ignore
                    domain_discrete=domain_discrete,  # type: ignore
                )
            )
            continue

        if isinstance(v, str):
            ssi.hparams[k].string_value = v

            if k in hparam_domain_discrete:
                domain_discrete = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(string_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            hps.append(
                HParamInfo(  # type: ignore
                    name=k,  # type: ignore
                    type=DataType.Value("DATA_TYPE_STRING"),  # type: ignore
                    domain_discrete=domain_discrete,  # type: ignore
                )
            )
            continue

        if isinstance(v, bool):
            ssi.hparams[k].bool_value = v

            if k in hparam_domain_discrete:
                domain_discrete = struct_pb2.ListValue(  # type: ignore
                    values=[
                        struct_pb2.Value(bool_value=d)  # type: ignore
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            hps.append(
                HParamInfo(  # type: ignore
                    name=k,  # type: ignore
                    type=DataType.Value("DATA_TYPE_BOOL"),  # type: ignore
                    domain_discrete=domain_discrete,  # type: ignore
                )
            )
            continue

        if isinstance(v, np.ndarray):
            ssi.hparams[k].number_value = v.item()
            hps.append(HParamInfo(name=k, type=DataType.Value("DATA_TYPE_FLOAT64")))  # type: ignore
            continue
        raise ValueError(
            "value should be one of int, float, str, bool, or np.ndarray"
        )

    content = HParamsPluginData(session_start_info=ssi, version=PLUGIN_DATA_VERSION)  # type: ignore
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(  # type: ignore
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()  # type: ignore
        )  # type: ignore
    )
    ssi = Summary(value=[Summary.Value(tag=SESSION_START_INFO_TAG, metadata=smd)])  # type: ignore

    mts = [MetricInfo(name=MetricName(tag=k)) for k in metrics_list]  # type: ignore

    exp = Experiment(hparam_infos=hps, metric_infos=mts)  # type: ignore

    content = HParamsPluginData(experiment=exp, version=PLUGIN_DATA_VERSION)  # type: ignore
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(  # type: ignore
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()  # type: ignore
        )
    )  # type: ignore
    exp = Summary(value=[Summary.Value(tag=EXPERIMENT_TAG, metadata=smd)])  # type: ignore

    sei = SessionEndInfo(status=Status.Value("STATUS_SUCCESS"))  # type: ignore
    content = HParamsPluginData(session_end_info=sei, version=PLUGIN_DATA_VERSION)  # type: ignore
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(  # type: ignore
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()  # type: ignore
        )
    )
    sei = Summary(value=[Summary.Value(tag=SESSION_END_INFO_TAG, metadata=smd)])  # type: ignore

    return exp, ssi, sei


class TensorboardLogger(BaseLogger):
    def __init__(self, 
                 output: Union[str, Path], 
                 hparam_plugin_metrics: Optional[Sequence[str]] = None,
                 subdirectory: Optional[str] = "tensorboard"):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        output = str(output)
        if subdirectory is not None:
            output = os.path.join(output, subdirectory)

        self._output = output
        self._writer = EventFileWriter(output)
        self._hparam_plugin_metrics = hparam_plugin_metrics or []

    @contextlib.contextmanager
    def add_event(self, step: int):
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.compat.proto.event_pb2 import Event

        summaries = []
        yield TensorboardLoggerEvent(self._writer.get_logdir(), summaries, step=step)
        summary = Summary(value=summaries)  # type: ignore
        self._writer.add_event(Event(summary=summary, step=step))  # type: ignore

    def add_hparams(self, hparams: Dict[str, Any]):
        from tensorboard.compat.proto.event_pb2 import Event
        if not isinstance(hparams, dict):
            raise TypeError("hparam should be dictionary.")
        hparam_domain_discrete = {}
        hparams = _flatten_simplify_hparams(hparams)
        exp, ssi, sei = _tensorboard_hparams(hparams, self._hparam_plugin_metrics or [], hparam_domain_discrete)
        self._writer.add_event(Event(summary=exp, step=0))  # type: ignore
        self._writer.add_event(Event(summary=ssi, step=0))  # type: ignore
        self._writer.add_event(Event(summary=sei, step=0))  # type: ignore

    def __str__(self):
        return "tensorboard"


def log_metrics(logger: Logger, metrics, *, prefix: str = "", step: int):
    with logger.add_event(step) as event:
        for k, val in metrics.items():
            tag = f"{prefix}{k}"
            if isinstance(val, (int, float)):
                event.add_scalar(tag, val)
            elif isinstance(val, str):
                event.add_text(tag, val)
            else:
                raise ValueError(f"Unknown metric type for {tag}: {val}")

