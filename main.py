# -*- coding: utf-8 -*-

import os
import time         # 用来统计时间
import random       # 用于随机选类别 & 控制随机性
import numpy as np
import matplotlib.pyplot as plt

import torch
from os import path
from tqdm import tqdm
from sklearn.manifold import TSNE

from config import load_args, ALL_METHODS
from models import load_backbone
from typing import Any, Dict, List, Tuple, Optional
from datasets import Features, load_dataset
from utils import set_determinism, validate
from torch._prims_common import DeviceLikeType
from torch.utils.data import Dataset, DataLoader


def make_dataloader(
    dataset: Dataset,
    shuffle: bool = False,
    batch_size: int = 256,
    num_workers: int = 8,
    device: Optional[DeviceLikeType] = None,
    persistent_workers: bool = False,
) -> DataLoader:
    pin_memory = (device is not None) and (torch.device(device).type == "cuda")
    config = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "pin_memory_device": str(device) if pin_memory else "",
        "persistent_workers": persistent_workers,
    }
    try:
        from prefetch_generator import BackgroundGenerator

        class DataLoaderX(DataLoader):
            def __iter__(self):
                return BackgroundGenerator(super().__iter__())

        return DataLoaderX(dataset, **config)
    except ImportError:
        return DataLoader(dataset, **config)


def check_cache_features(root: str) -> bool:
    files_list = ["X_train.pt", "y_train.pt", "X_test.pt", "y_test.pt"]
    for file in files_list:
        if not path.isfile(path.join(root, file)):
            return False
    return True


@torch.no_grad()
def cache_features(
    backbone: torch.nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: Optional[DeviceLikeType] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    X_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    for X, y in tqdm(dataloader, "Caching"):
        X: torch.Tensor = backbone(X.to(device))
        y: torch.Tensor = y.to(torch.int16, non_blocking=True)
        X_all.append(X.cpu())
        y_all.append(y.cpu())
    return torch.cat(X_all), torch.cat(y_all)


@torch.no_grad()
def tsne_visualize(
    learner: Any,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: Optional[DeviceLikeType],
    save_path: str,
    max_samples_per_class: Optional[int] = None,   # None = 不限制
    mode: str = "logit",
    base_class_ids: Optional[List[int]] = None,
    cil_class_ids: Optional[List[int]] = None,
    num_base_classes: int = 10,
    num_cil_classes: int = 4,
    tsne_seed: Optional[int] = None,
) -> None:
    """
    t-SNE visualization (final enhanced version)
    - no axis
    - no title
    - bold legend
    - reproducible (seed)
    - outer black border
    - save both PNG & SVG
    - unique color per class
    """

    # =========================================================
    # 1) Random seed (reproducibility)
    # =========================================================
    if tsne_seed is not None:
        random.seed(tsne_seed)
        np.random.seed(tsne_seed)
        torch.manual_seed(tsne_seed)

    # =========================================================
    # 2) Select Base / CIL classes
    # =========================================================
    allowed_classes = None
    selected_base, selected_cil = [], []

    if base_class_ids:
        selected_base = random.sample(
            base_class_ids, min(num_base_classes, len(base_class_ids))
        )

    if cil_class_ids:
        selected_cil = random.sample(
            cil_class_ids, min(num_cil_classes, len(cil_class_ids))
        )

    if selected_base or selected_cil:
        allowed_classes = set(selected_base + selected_cil)
        print(f"[t-SNE] Base classes: {selected_base}, CIL classes: {selected_cil}")
    else:
        print("[t-SNE] Using all classes")

    # =========================================================
    # 3) Feature extraction (logit or backbone feature)
    # =========================================================
    if mode == "logit":
        net = getattr(learner, "model", None)
        if not isinstance(net, torch.nn.Module):
            net = learner if isinstance(learner, torch.nn.Module) else None
    else:
        net = None

    if net is None:
        backbone = getattr(learner, "backbone", None) or getattr(
            learner.model, "backbone", None
        )
        if backbone is None:
            raise RuntimeError("learner has no backbone")
        backbone.eval()
    else:
        net.eval()

    features = []
    labels = []
    per_class_counts = {}

    for X, y in tqdm(dataloader, desc=f"t-SNE Collect ({mode})"):
        X = X.to(device, non_blocking=True)
        y = y.cpu()

        # ---- feature extraction ----
        if net is not None:
            out = net(X)
            if isinstance(out, (tuple, list)):
                out = out[0]
            batch_feats = out.detach().cpu()
        else:
            batch_feats = backbone(X).cpu()

        # ---- sample filtering ----
        for i in range(len(y)):
            cls = int(y[i])

            if allowed_classes is not None and cls not in allowed_classes:
                continue

            if max_samples_per_class is not None:
                cnt = per_class_counts.get(cls, 0)
                if cnt >= max_samples_per_class:
                    continue
                per_class_counts[cls] = cnt + 1

            features.append(batch_feats[i])
            labels.append(cls)

    if len(features) == 0:
        print("⚠️ t-SNE: No samples to visualize")
        return

    features_np = torch.stack(features).numpy()
    labels_np = np.array(labels)

    unique_labels = np.unique(labels_np)
    print(f"[t-SNE] Samples: {len(labels_np)}, Classes: {len(unique_labels)}")

    # =========================================================
    # 4) Run t-SNE
    # =========================================================
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=tsne_seed if tsne_seed is not None else 0,
        learning_rate="auto",
        perplexity=30,
    )
    feats_2d = tsne.fit_transform(features_np)

    # =========================================================
    # 5) Plotting (unique color per class)
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 8))

    # ---- discrete colormap (high contrast, paper-friendly) ----
    cmap = plt.get_cmap("tab20")
    num_colors = cmap.N

    label2color = {
        cls: cmap(i % num_colors)
        for i, cls in enumerate(unique_labels)
    }

    for cls in unique_labels:
        mask = labels_np == cls
        ax.scatter(
            feats_2d[mask, 0],
            feats_2d[mask, 1],
            s=5,
            color=label2color[cls],
            label=str(cls),
        )

    # ---- legend (bold) ----
    if len(unique_labels) <= 20:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize="medium",
            prop={"weight": "bold"},
            frameon=False,
        )

    # ---- remove axis ----
    ax.set_xticks([])
    ax.set_yticks([])

    # ---- outer black border ----
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

    plt.tight_layout()

    # =========================================================
    # 6) Save PNG & SVG
    # =========================================================
    os.makedirs(path.dirname(save_path), exist_ok=True)

    # PNG
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # SVG
    svg_path = save_path.replace(".png", ".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")

    plt.close()

    print(f"[t-SNE] PNG saved to → {save_path}")
    print(f"[t-SNE] SVG saved to → {svg_path}")

def main(args: Dict[str, Any]):
    backbone_name = args["backbone"]

    # Select device
    if args["cpu_only"] or not torch.cuda.is_available():
        main_device = torch.device("cpu")
        all_gpus = None
    elif args["gpus"] is not None:
        gpus = args["gpus"]
        main_device = torch.device(f"cuda:{gpus[0]}")
        all_gpus = [torch.device(f"cuda:{gpu}") for gpu in gpus]
    else:
        main_device = torch.device("cuda:0")
        all_gpus = None

    # --------- 全局随机种子（训练 + t-SNE 都保持一致） ----------
    if args["seed"] is not None:
        set_determinism(args["seed"])        # 原本就有
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

    if "backbone_path" in args:
        assert path.isfile(
            args["backbone_path"]
        ), f"Backbone file \"{args['backbone_path']}\" doesn't exist."
        preload_backbone = True
        backbone, _, feature_size = torch.load(
            args["backbone_path"], map_location=main_device, weights_only=False
        )
    else:
        # Load model pre-train on ImageNet if there is no base training dataset.
        preload_backbone = False
        load_pretrain = args["base_ratio"] == 0 or "ImageNet" not in args["dataset"]
        backbone, _, feature_size = load_backbone(backbone_name, pretrain=load_pretrain)
        if load_pretrain:
            assert args["dataset"] != "ImageNet", "Data may leak!!!"
    backbone = backbone.to(main_device, non_blocking=True)

    dataset_args = {
        "name": args["dataset"],
        "root": args["data_root"],
        "base_ratio": args["base_ratio"],
        "num_phases": args["phases"],
        "shuffle_seed": args["dataset_seed"] if "dataset_seed" in args else None,
    }
    dataset_train = load_dataset(train=True, augment=True, **dataset_args)
    dataset_test = load_dataset(train=False, augment=False, **dataset_args)

    # Select algorithm
    assert args["method"] in ALL_METHODS, f"Unknown method: {args['method']}"
    learner = ALL_METHODS[args["method"]](
        args, backbone, feature_size, main_device, all_devices=all_gpus
    )

    # ---- 统计 base training 时间（如果有）----
    base_train_time = 0.0

    # Base training
    if args["base_ratio"] > 0 and not preload_backbone:
        train_subset = dataset_train.subset_at_phase(0)
        test_subset = dataset_test.subset_at_phase(0)
        train_loader = make_dataloader(
            train_subset,
            True,
            args["batch_size"],
            args["num_workers"],
            device=main_device,
        )
        test_loader = make_dataloader(
            test_subset,
            False,
            args["batch_size"],
            args["num_workers"],
            device=main_device,
        )
        t0 = time.time()
        learner.base_training(
            train_loader,
            test_loader,
            dataset_train.base_size,
        )
        base_train_time = time.time() - t0
        print(f"[Base Training] time: {base_train_time:.2f} s")

    # Load dataset
    if args["cache_features"]:
        if "cache_path" not in args or args["cache_path"] is None:
            args["cache_path"] = args["saving_root"]
        if not check_cache_features(args["cache_path"]):
            backbone = learner.backbone.eval()
            dataset_train = load_dataset(
                args["dataset"], args["data_root"], True, 1, 0, augment=False
            )
            dataset_test = load_dataset(
                args["dataset"], args["data_root"], False, 1, 0, augment=False
            )
            train_loader = make_dataloader(
                dataset_train.subset_at_phase(0),
                False,
                args["batch_size"],
                args["num_workers"],
                device=main_device,
            )
            test_loader = make_dataloader(
                dataset_test.subset_at_phase(0),
                False,
                args["batch_size"],
                args["num_workers"],
                device=main_device,
            )

            if all_gpus is not None and len(all_gpus) > 1:
                backbone = torch.nn.DataParallel(backbone, device_ids=all_gpus)
            X_train, y_train = cache_features(
                backbone, train_loader, device=main_device
            )
            X_test, y_test = cache_features(backbone, test_loader, device=main_device)
            torch.save(X_train, path.join(args["cache_path"], "X_train.pt"))
            torch.save(y_train, path.join(args["cache_path"], "y_train.pt"))
            torch.save(X_test, path.join(args["cache_path"], "X_test.pt"))
            torch.save(y_test, path.join(args["cache_path"], "y_test.pt"))
        dataset_train = Features(
            args["cache_path"],
            train=True,
            base_ratio=args["base_ratio"],
            num_phases=args["phases"],
            augment=False,
        )
        dataset_test = Features(
            args["cache_path"],
            train=False,
            base_ratio=args["base_ratio"],
            num_phases=args["phases"],
            augment=False,
        )
        learner.backbone = torch.nn.Identity()
        learner.model.backbone = torch.nn.Identity()
    else:
        dataset_train = load_dataset(train=True, augment=False, **dataset_args)
        dataset_test = load_dataset(train=False, augment=False, **dataset_args)

    # Incremental learning
    sum_acc = 0
    log_file_path = path.join(args["saving_root"], "IL.csv")
    log_file = open(log_file_path, "w", buffering=1)

    # CSV 头：把每个 phase 的 train / inference 时间也记下来
    print(
        "phase",
        "acc@avg",
        "acc@1",
        "acc@5",
        "f1-micro",
        "loss",
        "train_time",
        "inference_time",
        file=log_file,
        sep=",",
    )

    total_train_time = 0.0
    total_infer_time = 0.0

    for phase in range(0, args["phases"] + 1):
        train_subset = dataset_train.subset_at_phase(phase)
        test_subset = dataset_test.subset_until_phase(phase)
        train_loader = make_dataloader(
            train_subset,
            True,
            args["IL_batch_size"],
            args["num_workers"],
            device=main_device,
        )
        test_loader = make_dataloader(
            test_subset,
            False,
            args["IL_batch_size"],
            args["num_workers"],
            device=main_device,
        )

        # ---------- 统计本 phase 的训练时间 ----------
        t_train_start = time.time()
        if phase == 0:
            learner.learn(train_loader, dataset_train.base_size, "Re-align")
        else:
            learner.learn(train_loader, dataset_train.phase_size)
        train_time_phase = time.time() - t_train_start
        total_train_time += train_time_phase

        learner.before_validation()

        # ---------- 统计本 phase 的推理时间（验证） ----------
        t_infer_start = time.time()
        val_meter = validate(
            learner,
            test_loader,
            dataset_train.num_classes,
            desc=f"Phase {phase}",
        )
        inference_time_phase = time.time() - t_infer_start
        total_infer_time += inference_time_phase

        sum_acc += val_meter.accuracy
        print(
            f"Phase {phase}",
            f"loss: {val_meter.loss:.4f}",
            f"acc@1: {val_meter.accuracy * 100:.3f}%",
            f"acc@5: {val_meter.accuracy5 * 100:.3f}%",
            f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
            f"acc@avg: {sum_acc / (phase + 1) * 100:.3f}%",
            f"train_time: {train_time_phase:.2f}s",
            f"infer_time: {inference_time_phase:.2f}s",
            sep="    ",
        )
        print(
            phase,
            sum_acc / (phase + 1),
            val_meter.accuracy,
            val_meter.accuracy5,
            val_meter.f1_micro,
            val_meter.loss,
            train_time_phase,
            inference_time_phase,
            file=log_file,
            sep=",",
        )

        # 在最后一个 phase 进行一次 t-SNE 可视化（使用 logits，区分不同方法）
        if phase == args["phases"]:
            method_name = args["method"].replace("/", "_")
            tsne_path = path.join(
                args["saving_root"],
                f"tsne_{method_name}_phase{phase}_logit.png",
            )

            # 前 base_size 个是 base 类，其余是 CIL 类
            base_ids = list(range(dataset_train.base_size))
            cil_ids = list(range(dataset_train.base_size, dataset_train.num_classes))

            tsne_visualize(
                learner,
                test_loader,
                main_device,
                tsne_path,
                max_samples_per_class=None,  # ⭐ 不限制每类样本数
                mode="logit",
                base_class_ids=base_ids,
                cil_class_ids=cil_ids,
                num_base_classes=10,
                num_cil_classes=4,
                tsne_seed=args["seed"] if args.get("seed", None) is not None else 0,
            )

    log_file.close()

    print("=" * 60)
    print(f"[Summary] Base training time: {base_train_time:.2f} s")
    print(f"[Summary] Incremental training time (sum over phases): {total_train_time:.2f} s")
    print(f"[Summary] Inference time (validation sum over phases): {total_infer_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main(load_args())