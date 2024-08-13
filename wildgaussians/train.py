import json
from functools import partial
import logging
import math
import time
import shutil
from omegaconf import OmegaConf
import os
from typing import Dict, cast, Optional, List, Tuple
from tqdm import tqdm
from pathlib import Path
import numpy as np
import click
from .types import FrozenSet, Method, Dataset, DatasetFeature, EvaluationProtocol, Logger
from .evaluation import render_all_images, evaluate, compute_metrics, DefaultEvaluationProtocol
from .logging import TensorboardLogger
from .datasets import load_dataset
from .datasets.phototourism import horizontal_half_dataset
from . import datasets
from .utils import (
    Indices, 
    setup_logging, 
    image_to_srgb, 
    make_image_grid, 
    visualize_depth,
    IndicesClickType, 
    SetParamOptionType,
    MetricsAccumulator,
)
from .wildgaussians_spec import WildGaussiansMethodSpec
from .method import WildGaussians


def eval_all(method: Method, logger: Logger, dataset: Dataset, *, output: str, step: int, evaluation_protocol: EvaluationProtocol, split: str, nb_info):
    total_rays = 0
    metrics: Optional[Dict[str, float]] = {} if logger else None
    expected_scene_scale = dataset["metadata"].get("expected_scene_scale")

    # Store predictions, compute metrics, etc.
    prefix = dataset["image_paths_root"]
    if prefix is None:
        prefix = Path(os.path.commonpath(dataset["image_paths"]))

    if split != "test":
        output_metrics = os.path.join(output, f"results-{step}-{split}.json")
        output = os.path.join(output, f"predictions-{step}-{split}.tar.gz")
    else:
        output_metrics = os.path.join(output, f"results-{step}.json")
        output = os.path.join(output, f"predictions-{step}.tar.gz")

    if os.path.exists(output):
        if os.path.isfile(output):
            os.unlink(output)
        else:
            shutil.rmtree(output)
        logging.warning(f"removed existing predictions at {output}")

    if os.path.exists(output_metrics):
        os.unlink(output_metrics)
        logging.warning(f"removed existing results at {output_metrics}")

    start = time.perf_counter()
    num_vis_images = 16
    vis_images: List[Tuple[np.ndarray, np.ndarray]] = []
    vis_depth: List[np.ndarray] = []
    for (i, gt), pred, (w, h) in zip(
        enumerate(dataset["images"]),
        render_all_images(
            method,
            dataset,
            output=output,
            description=f"rendering all images at step={step}",
            nb_info=nb_info,
            evaluation_protocol=evaluation_protocol,
        ),
        dataset["cameras"].image_sizes,
    ):
        if len(vis_images) < num_vis_images:
            color = pred["color"]
            background_color = dataset["metadata"].get("background_color", None)
            dataset_colorspace = dataset["metadata"].get("color_space", "srgb")
            color_srgb = image_to_srgb(color, np.uint8, color_space=dataset_colorspace, background_color=background_color)
            gt_srgb = image_to_srgb(gt[:h, :w], np.uint8, color_space=dataset_colorspace, background_color=background_color)
            vis_images.append((gt_srgb, color_srgb))
            if "depth" in pred:
                near_far = dataset["cameras"].nears_fars[i] if dataset["cameras"].nears_fars is not None else None
                vis_depth.append(visualize_depth(pred["depth"], expected_scale=expected_scene_scale, near_far=near_far))
    elapsed = time.perf_counter() - start

    # Compute metrics
    info = evaluate(
        output, 
        output_metrics, 
        evaluation_protocol=evaluation_protocol,
        description=f"evaluating all images at step={step}")
    metrics = info["metrics"]

    if logger:
        assert metrics is not None, "metrics must be computed"
        logging.debug(f"logging metrics to {logger}")
        metrics["fps"] = len(dataset["cameras"]) / elapsed
        metrics["rays-per-second"] = total_rays / elapsed
        metrics["time"] = elapsed
        with logger.add_event(step) as event:
            for k, v in metrics.items():
                event.add_scalar(f"eval-all-{split}/{k}", v)

        num_cols = int(math.sqrt(len(vis_images)))
        color_vis = make_image_grid(
            make_image_grid(*[x[0] for x in vis_images], ncol=num_cols),
            make_image_grid(*[x[1] for x in vis_images], ncol=num_cols),
        )

        logger.add_image(f"eval-all-{split}/color", 
                         color_vis, 
                         display_name="color", 
                         description="left: gt, right: prediction", 
                         step=step)



def eval_few_custom(method: WildGaussians, logger: Logger, dataset: Dataset, split: str, step: int, evaluation_protocol: EvaluationProtocol):
    disable_tqdm = False

    embeddings = None
    evaluation_dataset = dataset
    metrics = MetricsAccumulator()
    result_no_optim = None
    optim = None
    optim_metrics = None
    i = 0

    eval_few_rows: List[List[np.ndarray]] = [[] for _ in range(len(dataset["cameras"]))]
    if evaluation_protocol.get_name() == "nerfw":
        # dataset = datasets.dataset_index_select(dataset, slice(None, 1))
        optimization_dataset = horizontal_half_dataset(dataset, left=True)
        embeddings = []
        for i, optim in tqdm(enumerate(method.optimize_embeddings(optimization_dataset)), desc="optimizing embeddings", total=len(dataset["cameras"]), disable=disable_tqdm):
            embeddings.append(optim["embedding"])
            if optim_metrics is None and "metrics" in optim:
                optim_metrics = optim["metrics"]

        evaluation_dataset = horizontal_half_dataset(dataset, left=False)
        images_f = [image_to_srgb(img, dtype=np.float32) for img in evaluation_dataset["images"]]
        for i, result_no_optim in tqdm(enumerate(method.render(evaluation_dataset["cameras"])), desc="rendering", total=len(dataset["cameras"]), disable=disable_tqdm):
            metrics.update({
                k + "-nopt": v for k, v in compute_metrics(image_to_srgb(result_no_optim["color"], dtype=np.float32), images_f[i]).items()
            })
            eval_few_rows[i].append(image_to_srgb(result_no_optim["color"], dtype=np.uint8))
    else:
        images_f = [image_to_srgb(img, dtype=np.float32) for img in evaluation_dataset["images"]]

    for i in range(len(evaluation_dataset["cameras"])):
        eval_few_rows[i].insert(0, evaluation_dataset["images"][i])

    result_optim = None
    renders = []
    for i, result_optim in tqdm(enumerate(method.render(evaluation_dataset["cameras"], embeddings=embeddings)), desc="rendering", total=len(dataset["cameras"]), disable=disable_tqdm):
        metrics.update(compute_metrics(image_to_srgb(result_optim["color"], dtype=np.float32), images_f[i]))
        renders.append(image_to_srgb(result_optim["color"], dtype=np.uint8))
        eval_few_rows[i].append(image_to_srgb(result_optim["color"], dtype=np.uint8))
    assert result_optim is not None
    cast(Dict, evaluation_dataset)["renders"] = renders

    with logger.add_event(step) as event:
        for k, v in metrics.pop().items():
            event.add_scalar(f"eval-few-{split}/{k}", v)
        # optimization_dataset.images[i], 
        # image_to_srgb(optim["render_output"]["color"], dtype=np.uint8),
        if evaluation_protocol.get_name() == "nerfw":
            assert result_no_optim is not None
            event.add_image(f"eval-few-{split}/color", 
                            make_image_grid(*[x for y in eval_few_rows for x in y], ncol=4),
                            description="left: gt, middle: nopt, right: opt")
        else:
            event.add_image(f"eval-few-{split}/color", 
                            make_image_grid(*[x for y in eval_few_rows for x in y], ncol=3),
                            description="left: gt, right: render")
        
        # Render optimization graph for PSNR, MSE
        if optim_metrics is not None:
            for k in ["psnr", "mse"]:
                metric = optim_metrics[k]
                event.add_plot(
                    f"eval-few-{split}/optimization-{k}",
                    np.stack((np.arange(len(metric)), metric), -1),
                    axes_labels=("iteration", k),
                    title=f"Optimization of {k} over iterations",
                )


@click.command("train")
@click.option("--data", type=str, required=True)
@click.option("--output", type=str, default=".")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--dataset-type", type=click.Choice(["default", "nerfonthego", "phototourism"]), default="default")
@click.option("--eval-few-iters", type=IndicesClickType(), default=Indices.every_iters(2_000), help="When to evaluate on few images")
@click.option("--set", "config_overrides", help="Override a parameter in the method.", type=SetParamOptionType(), multiple=True, default=None)
def train_command(
    data,
    output,
    verbose,
    eval_few_iters,
    dataset_type="default",
    config_overrides=None,
    debug=False,
):
    if debug:
        import torch
        torch.autograd.set_detect_anomaly(True)  # type: ignore[attr-defined]
        config_overrides = config_overrides or ()
        config_overrides = config_overrides + (("iterations", "100"),)
        print(config_overrides)
        eval_few_iters = Indices.every_iters(70)
    logging.basicConfig(level=logging.INFO if not debug else logging.DEBUG)
    setup_logging(verbose)

    if config_overrides is not None and isinstance(config_overrides, (list, tuple)):
        config_overrides = dict(config_overrides)
    config_overrides = {**WildGaussiansMethodSpec["dataset_overrides"].get(dataset_type, {}), **(config_overrides or {})}

    # Load dataset
    method_info = WildGaussians.get_method_info()
    features: FrozenSet[DatasetFeature] = frozenset({"color", "points3D_xyz"})
    if dataset_type == "phototourism":
        assert config_overrides["config"] == "phototourism.yml"
        from .datasets.phototourism import load_phototourism_dataset, download_phototourism_dataset, NerfWEvaluationProtocol

        evaluation_protocol = NerfWEvaluationProtocol()
        load_dataset_fn = partial(
            load_dataset,
            load_dataset_fn=load_phototourism_dataset,
            download_dataset_fn=download_phototourism_dataset,
            evaluation_protocol=evaluation_protocol.get_name(),
        )
    else:
        if dataset_type == "nerfonthego":
            assert config_overrides["config"] == "nerfonthego.yml"
        from .datasets.colmap import load_colmap_dataset
        evaluation_protocol = DefaultEvaluationProtocol()
        load_dataset_fn = partial(
            load_dataset, 
            load_dataset_fn=load_colmap_dataset, 
            images_path="images",
            evaluation_protocol=evaluation_protocol.get_name()
        )

    test_dataset = load_dataset_fn(data, "test", features, load_features=False)
    train_dataset = load_dataset_fn(data, "train", features, load_features=False)
    if dataset_type == "nerfonthego":
        dataset_not_official = "Please use the dataset provided for the WG paper"
        assert os.path.exists(os.path.join(data, "nb-info.json")), dataset_not_official
        with open(os.path.join(data, "nb-info.json"), "r") as f:
            info = json.load(f)
            assert info.pop("loader", None) == "colmap", dataset_not_official
            info.pop("loader_kwargs", None)
            assert info["name"] == "nerfonthego-undistorted", dataset_not_official
            test_dataset["metadata"].update(info)
            train_dataset["metadata"].update(info)
        assert train_dataset["metadata"]["name"] == "nerfonthego-undistorted"
        assert test_dataset["metadata"]["name"] == "nerfonthego-undistorted"
    if debug:
        train_dataset = datasets.dataset_index_select(train_dataset, slice(None, 8))
        test_dataset = datasets.dataset_index_select(test_dataset, slice(None, 8))
    supported_camera_models = method_info.get("supported_camera_models", frozenset(("pinhole",)))
    train_dataset = datasets.dataset_load_features(train_dataset, supported_camera_models=supported_camera_models)
    train_dataset["images"] = [x[..., :3] for x in train_dataset["images"]]
    train_images = train_dataset.get("images")
    test_dataset = datasets.dataset_load_features(test_dataset, supported_camera_models=supported_camera_models)
    test_dataset["images"] = [x[..., :3] for x in test_dataset["images"]]
    train_images_thumbnails = [img[::8, ::8].copy() for img in train_dataset["images"]]

    output_path = Path(output)

    # Store a slice of train dataset used for eval_few
    train_dataset_eval_few = datasets.dataset_index_select(train_dataset, [0, 1, 2, 3])
    test_dataset_eval_few = datasets.dataset_index_select(test_dataset, [0, 1, 2, 3])

    method = WildGaussians(
        checkpoint=None,
        train_dataset=train_dataset,
        config_overrides=config_overrides,
    )
    info = method.get_info()
    train_dataset["images"] = train_images

    # Init logger
    logger = TensorboardLogger(output_path / "tensorboard", ["eval-all-test/ssim", "eval-all-test/psnr"])

    # Log hparams
    logger.add_hparams(cast(Dict, OmegaConf.to_container(method.config, resolve=True)))

    info = method.get_info()
    acc_metrics = MetricsAccumulator()

    num_iterations = info["num_iterations"]
    assert num_iterations is not None, "num_iterations must be set in the config"
    step = 0
    for step in (pbar := tqdm(range(num_iterations), miniters=10, desc="training", disable=debug)):
        metrics = method.train_iteration(step)
        step += 1

        acc_metrics.update(metrics)

        # Log metrics
        if step % 100 == 0:
            acc_metrics_values = acc_metrics.pop()
            with logger.add_event(step) as event:
                for k, val in acc_metrics_values.items():
                    event.add_scalar(f"train/{k}", val)
            pbar.set_postfix({ "train/loss": f"{acc_metrics_values['loss']:.4f}", "psnr": f"{acc_metrics_values['psnr']:.4f}" })

        if step % 10_000 == 0:
            path = output_path / f"checkpoint-{step}"  # pyright: ignore[reportCallIssue]
            if path.exists():
                shutil.rmtree(path)
                logging.warning(f"removed existing checkpoint at {path}")
            method.save(str(path))
            logging.info(f"checkpoint saved at step={step}")

        if step in eval_few_iters:
            logging.info(f"evaluating on few images at step {step}")
            eval_few_custom(method, logger, train_dataset_eval_few, split="train", step=step, evaluation_protocol=evaluation_protocol)
            eval_few_custom(method, logger, test_dataset_eval_few, split="test", step=step, evaluation_protocol=evaluation_protocol)

        if step % 10_000 == 0:
            # Display embeddings
            logging.info(f"logging embeddings at step {step}")
            labels = [{
                "name": os.path.relpath(x, train_dataset["image_paths_root"]),
                "id": i,
            } for i, x in enumerate(train_dataset["image_paths"])]
            if method.model.appearance_embeddings is not None:
                logger.add_embedding("train/appearance-embeddings", 
                                     method.model.appearance_embeddings.detach().cpu().numpy(), 
                                     images=train_images_thumbnails, 
                                     labels=labels,
                                     step=step)

    logging.info(f"evaluating on all images as step {step}")
    eval_all(method, logger, test_dataset, split="test", step=step, output=str(output_path), evaluation_protocol=evaluation_protocol, nb_info={})
    if evaluation_protocol.get_name() == "nerfw":
        eval_all(method, logger, train_dataset_eval_few, split="trainsubset", step=step, output=str(output_path), evaluation_protocol=evaluation_protocol, nb_info={})
    else:
        eval_all(method, logger, train_dataset, split="train", step=step, output=str(output_path), evaluation_protocol=evaluation_protocol, nb_info={})

    # Save final checkpoint
    if step % 10_000 != 0:
        path = output_path / f"checkpoint-{step}"  # pyright: ignore[reportCallIssue]
        if path.exists():
            path.unlink()
            logging.warning(f"removed existing checkpoint at {path}")
        method.save(str(path))
        logging.info(f"checkpoint saved at step={step}")


if __name__ == "__main__":
    train_command()  # pylint: disable=no-value-for-parameter
