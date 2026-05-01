from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DIAGNOSIS_COLUMNS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
AGE_BINS = [0, 20, 30, 40, 50, 60, 70, 80, 120]
AGE_LABELS = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
REVIEW_COLUMNS = [
    "image",
    "review_status",
    "reviewed_diagnosis",
    "reviewed_age",
    "reviewed_sex",
    "reviewer_confidence",
    "reviewer_notes",
    "reviewed_at",
]

POST_TRAINING_REVIEW_COLUMNS = [
    "image",
    "diagnosis",
    "predicted_diagnosis",
    "age_group",
    "sex_clean",
    "age_x_sex",
    "confidence",
    "rereview_reason",
    "training_run_id",
]


@dataclass(frozen=True)
class IsicPaths:
    root: Path
    train_metadata: Path
    train_ground_truth: Path
    test_metadata: Path
    test_ground_truth: Path
    train_images: Path
    test_images: Path
    reviews: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "IsicPaths":
        root = Path(root)
        return cls(
            root=root,
            train_metadata=root / "ISIC_2019_Training_Metadata.csv",
            train_ground_truth=root / "ISIC_2019_Training_GroundTruth.csv",
            test_metadata=root / "ISIC_2019_Test_Metadata.csv",
            test_ground_truth=root / "ISIC_2019_Test_GroundTruth.csv",
            train_images=root / "ISIC_2019_Training_Input",
            test_images=root / "ISIC_2019_Test_Input",
            reviews=root / "human_reviews.csv",
        )


def add_diagnosis_label(ground_truth: pd.DataFrame) -> pd.DataFrame:
    diagnosis_cols = [col for col in DIAGNOSIS_COLUMNS if col in ground_truth.columns]
    out = ground_truth.copy()
    label_matrix = out[diagnosis_cols]
    out["diagnosis"] = label_matrix.idxmax(axis=1)
    out.loc[label_matrix.sum(axis=1).eq(0), "diagnosis"] = "missing_or_unlabeled"
    return out[["image", "diagnosis"] + diagnosis_cols]


def clean_metadata(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["sex_clean"] = (
        out["sex"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"nan": pd.NA, "": pd.NA, "unknown": pd.NA})
        .fillna("missing")
    )
    out["age"] = pd.to_numeric(out["age_approx"], errors="coerce")
    out["age_group"] = pd.cut(out["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    out["age_group"] = out["age_group"].astype("string").fillna("missing")
    out["age_x_sex"] = out["age_group"] + " | " + out["sex_clean"]
    return out


def load_isic_2019(paths: IsicPaths) -> pd.DataFrame:
    train_meta = pd.read_csv(paths.train_metadata)
    test_meta = pd.read_csv(paths.test_metadata)
    train_gt = pd.read_csv(paths.train_ground_truth)

    train = train_meta.merge(add_diagnosis_label(train_gt), on="image", how="left")
    train["split"] = "train"

    if paths.test_ground_truth.exists():
        test_gt = pd.read_csv(paths.test_ground_truth)
        test = test_meta.merge(add_diagnosis_label(test_gt), on="image", how="left")
    else:
        test = test_meta.copy()
        test["diagnosis"] = np.nan
    test["split"] = "test"

    diagnosis_cols = [col for col in DIAGNOSIS_COLUMNS if col in train.columns or col in test.columns]
    keep_cols = ["image", "age_approx", "sex", "split", "diagnosis"] + diagnosis_cols
    combined = pd.concat([train[keep_cols], test[keep_cols]], ignore_index=True)
    return clean_metadata(combined)


def normalized_entropy(proportions: Iterable[float]) -> float:
    p = np.asarray(list(proportions), dtype=float)
    p = p[p > 0]
    if len(p) <= 1:
        return 1.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def gini(values: Iterable[float]) -> float:
    x = np.sort(np.asarray(list(values), dtype=float))
    if len(x) == 0 or np.isclose(x.sum(), 0):
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def representation_metrics(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for split_name, split_df in data.groupby("split"):
        shares = split_df[group_col].value_counts(normalize=True, dropna=False)
        min_share = shares.min()
        max_share = shares.max()
        rows.append(
            {
                "split": split_name,
                "attribute": group_col,
                "n_groups": len(shares),
                "min_share_pct": min_share * 100,
                "max_share_pct": max_share * 100,
                "min_to_max_ratio": min_share / max_share if max_share else np.nan,
                "max_disparity_pp": (max_share - min_share) * 100,
                "normalized_entropy": normalized_entropy(shares.values),
                "gini_concentration": gini(shares.values),
                "smallest_group": shares.idxmin(),
                "largest_group": shares.idxmax(),
            }
        )
    return pd.DataFrame(rows)


def all_representation_metrics(data: pd.DataFrame) -> pd.DataFrame:
    fairness_cols = ["sex_clean", "age_group", "age_x_sex"]
    return pd.concat([representation_metrics(data, col) for col in fairness_cols], ignore_index=True)


def group_parity_table(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    counts = data.groupby(["split", group_col]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby("split")["n"].transform("sum")

    pivot = counts.pivot(index=group_col, columns="split", values=["n", "share"]).fillna(0)
    pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
    pivot = pivot.reset_index()

    overall = data[group_col].value_counts(normalize=True).rename("overall_share")
    pivot = pivot.merge(overall, left_on=group_col, right_index=True, how="left")
    pivot["uniform_share"] = 1 / data[group_col].nunique(dropna=False)

    for split in ["train", "test"]:
        share_col = f"share_{split}"
        if share_col in pivot:
            pivot[f"{split}_ratio_to_uniform"] = pivot[share_col] / pivot["uniform_share"]
            pivot[f"{split}_ratio_to_overall"] = pivot[share_col] / pivot["overall_share"]

    if {"share_train", "share_test"}.issubset(pivot.columns):
        pivot["test_minus_train_pp"] = (pivot["share_test"] - pivot["share_train"]) * 100
        pivot["test_to_train_ratio"] = pivot["share_test"] / pivot["share_train"].replace(0, np.nan)

    ratio_cols = [col for col in pivot.columns if col.endswith("ratio_to_uniform")]
    pivot["underrepresented_vs_uniform"] = pivot[ratio_cols].lt(0.8).any(axis=1)
    return pivot.sort_values("overall_share", ascending=False)


def make_review_queue(
    data: pd.DataFrame,
    representation_threshold: float = 0.8,
    rare_combo_threshold: int = 25,
) -> pd.DataFrame:
    train_df = data[data["split"] == "train"].copy()

    group_counts = train_df["age_x_sex"].value_counts()
    group_share = train_df["age_x_sex"].value_counts(normalize=True)
    uniform_share = 1 / train_df["age_x_sex"].nunique(dropna=False)

    train_df["age_x_sex_n"] = train_df["age_x_sex"].map(group_counts)
    train_df["age_x_sex_share"] = train_df["age_x_sex"].map(group_share)
    train_df["age_x_sex_ratio_to_uniform"] = train_df["age_x_sex_share"] / uniform_share

    dx_group_counts = (
        train_df.groupby(["diagnosis", "age_x_sex"]).size().rename("diagnosis_age_x_sex_n")
    )
    train_df = train_df.merge(dx_group_counts, on=["diagnosis", "age_x_sex"], how="left")

    train_df["review_reason"] = np.select(
        [
            train_df["sex_clean"].eq("missing") | train_df["age_group"].eq("missing"),
            train_df["age_x_sex_ratio_to_uniform"].lt(representation_threshold),
            train_df["diagnosis_age_x_sex_n"].lt(rare_combo_threshold),
        ],
        [
            "missing age or sex metadata",
            "underrepresented age x sex group",
            "rare diagnosis within age x sex group",
        ],
        default="not prioritized",
    )

    queue = train_df[train_df["review_reason"] != "not prioritized"].copy()
    queue["priority_score"] = (
        (1 / queue["age_x_sex_ratio_to_uniform"].replace(0, np.nan)).fillna(100)
        + (rare_combo_threshold / queue["diagnosis_age_x_sex_n"].clip(lower=1))
        + np.where(queue["review_reason"].eq("missing age or sex metadata"), 3, 0)
    )

    columns = [
        "image",
        "diagnosis",
        "age",
        "age_group",
        "sex_clean",
        "age_x_sex",
        "age_x_sex_n",
        "age_x_sex_share",
        "age_x_sex_ratio_to_uniform",
        "diagnosis_age_x_sex_n",
        "review_reason",
        "priority_score",
    ]
    return queue[columns].sort_values("priority_score", ascending=False).reset_index(drop=True)


def load_reviews(path: Path) -> pd.DataFrame:
    if path.exists():
        reviews = pd.read_csv(path)
    else:
        reviews = pd.DataFrame(columns=REVIEW_COLUMNS)
    for col in REVIEW_COLUMNS:
        if col not in reviews.columns:
            reviews[col] = pd.NA
    return reviews[REVIEW_COLUMNS]


def save_review(path: Path, review: dict) -> None:
    reviews = load_reviews(path)
    reviews = reviews[reviews["image"] != review["image"]]
    reviews = pd.concat([reviews, pd.DataFrame([review])], ignore_index=True)
    reviews.to_csv(path, index=False)


def attach_review_status(queue: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        out = queue.copy()
        out["review_status"] = "pending"
        return out
    latest = reviews.drop_duplicates("image", keep="last")
    out = queue.merge(latest, on="image", how="left")
    out["review_status"] = out["review_status"].fillna("pending")
    return out


def image_path_for(image_id: str, paths: IsicPaths, split: str = "train") -> Path | None:
    image_dir = paths.train_images if split == "train" else paths.test_images
    candidate = image_dir / f"{image_id}.jpg"
    return candidate if candidate.exists() else None


def compute_training_weights(data: pd.DataFrame, reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    train_df = data[data["split"] == "train"].copy()
    if reviews is not None and not reviews.empty:
        latest = reviews.drop_duplicates("image", keep="last")
        train_df = train_df.merge(latest, on="image", how="left")
        train_df["final_diagnosis"] = train_df["reviewed_diagnosis"].where(
            train_df["reviewed_diagnosis"].notna(), train_df["diagnosis"]
        )
        train_df["training_gate"] = np.where(
            train_df["review_status"].isin(["exclude from training", "mark uncertain"]),
            "hold_out",
            "train",
        )
    else:
        train_df["final_diagnosis"] = train_df["diagnosis"]
        train_df["training_gate"] = "train"

    group_counts = train_df["age_x_sex"].value_counts()
    diagnosis_counts = train_df["final_diagnosis"].value_counts()

    train_df["group_weight"] = train_df["age_x_sex"].map(lambda value: 1 / group_counts[value])
    train_df["diagnosis_weight"] = train_df["final_diagnosis"].map(lambda value: 1 / diagnosis_counts[value])
    train_df["sample_weight"] = train_df["group_weight"] * train_df["diagnosis_weight"]
    train_df["sample_weight"] = train_df["sample_weight"] / train_df["sample_weight"].mean()
    train_df.loc[train_df["training_gate"].eq("hold_out"), "sample_weight"] = 0.0
    return train_df[
        [
            "image",
            "final_diagnosis",
            "age_group",
            "sex_clean",
            "age_x_sex",
            "training_gate",
            "group_weight",
            "diagnosis_weight",
            "sample_weight",
        ]
    ]


def load_post_training_rereview(path: Path) -> pd.DataFrame:
    if path.exists():
        rereview = pd.read_csv(path)
    else:
        rereview = pd.DataFrame(columns=POST_TRAINING_REVIEW_COLUMNS)
    for col in POST_TRAINING_REVIEW_COLUMNS:
        if col not in rereview.columns:
            rereview[col] = pd.NA
    return rereview[POST_TRAINING_REVIEW_COLUMNS]
