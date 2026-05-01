from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from isic_fairness_pipeline import (
    AGE_LABELS,
    DIAGNOSIS_COLUMNS,
    IsicPaths,
    attach_review_status,
    compute_training_weights,
    image_path_for,
    load_isic_2019,
    load_post_training_rereview,
    load_reviews,
    make_review_queue,
    save_review,
)


ROOT = Path(__file__).resolve().parent
PATHS = IsicPaths.from_root(ROOT)
TRAINING_STATUS = {
    "state": "idle",
    "message": "No training run has started yet.",
    "progress": 0,
    "interrupt_requested": False,
    "started_at": None,
    "finished_at": None,
    "metrics": None,
    "run_id": None,
}
TRAINING_LOCK = threading.Lock()


class TrainingInterrupted(Exception):
    pass


def check_interrupted() -> None:
    if TRAINING_STATUS.get("interrupt_requested"):
        raise TrainingInterrupted("Training was interrupted by the reviewer.")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def respond_json(handler: SimpleHTTPRequestHandler, payload: dict | list, status: int = 200) -> None:
    body = json.dumps(payload, default=json_default).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def dataset() -> pd.DataFrame:
    return load_isic_2019(PATHS)


def queue_with_reviews(representation_threshold: float = 0.8, rare_combo_threshold: int = 25) -> pd.DataFrame:
    queue = make_review_queue(dataset(), representation_threshold, rare_combo_threshold)
    reviews = load_reviews(PATHS.reviews)
    return attach_review_status(queue, reviews)


def image_features(image_id: str) -> np.ndarray:
    path = image_path_for(image_id, PATHS, split="train")
    if path is None:
        return np.zeros(18, dtype=float)

    with Image.open(path) as image:
        image = image.convert("RGB").resize((32, 32))
        arr = np.asarray(image, dtype=np.float32) / 255.0

    channels = arr.reshape(-1, 3)
    means = channels.mean(axis=0)
    stds = channels.std(axis=0)
    mins = channels.min(axis=0)
    maxs = channels.max(axis=0)
    q25 = np.quantile(channels, 0.25, axis=0)
    q75 = np.quantile(channels, 0.75, axis=0)
    return np.concatenate([means, stds, mins, maxs, q25, q75])


def build_feature_matrix(rows: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    image_matrix = np.vstack([image_features(image_id) for image_id in rows["image"]])
    image_cols = [f"image_feature_{idx}" for idx in range(image_matrix.shape[1])]

    metadata = rows[["age_group", "sex_clean"]].fillna("missing")
    metadata["age_group"] = pd.Categorical(metadata["age_group"], categories=AGE_LABELS + ["missing"])
    metadata["sex_clean"] = pd.Categorical(metadata["sex_clean"], categories=["female", "male", "missing"])
    metadata_matrix = pd.get_dummies(metadata, columns=["age_group", "sex_clean"], dtype=float)
    matrix = np.hstack([image_matrix, metadata_matrix.to_numpy(dtype=float)])
    return matrix, image_cols + metadata_matrix.columns.tolist()


def select_training_frame(max_train_rows: int, max_validation_rows: int) -> pd.DataFrame:
    data = dataset()
    reviews = load_reviews(PATHS.reviews)
    weights = compute_training_weights(data, reviews)
    train = data[data["split"] == "train"].merge(weights, on=["image", "age_group", "sex_clean", "age_x_sex"], how="inner")
    train = train[train["training_gate"] == "train"].copy()
    train = train[train["final_diagnosis"].isin([col for col in DIAGNOSIS_COLUMNS if col != "UNK"])]

    # Keep the local demo fast and deterministic while still training on real ISIC rows and images.
    cap = max_train_rows + max_validation_rows
    if len(train) > cap:
        sampled = []
        for _, group in train.groupby("final_diagnosis"):
            sampled.append(group.sample(min(len(group), max(80, cap // 8)), random_state=42))
        train = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=42).head(cap).reset_index(drop=True)
    return train


def run_training(max_train_rows: int = 5000, max_validation_rows: int = 1500) -> None:
    with TRAINING_LOCK:
        if TRAINING_STATUS["state"] == "running":
            return
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        TRAINING_STATUS.update(
            {
                "state": "running",
                "message": "Preparing weighted training data.",
                "progress": 5,
                "interrupt_requested": False,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "metrics": None,
                "run_id": run_id,
            }
        )

    try:
        frame = select_training_frame(max_train_rows, max_validation_rows)
        check_interrupted()
        TRAINING_STATUS["message"] = f"Extracting image and metadata features from {len(frame):,} rows."
        TRAINING_STATUS["progress"] = 20

        labels = frame["final_diagnosis"].astype(str)
        train_rows, val_rows = train_test_split(
            frame,
            test_size=min(max_validation_rows, max(1, int(len(frame) * 0.25))),
            random_state=42,
            stratify=labels,
        )

        x_train, feature_names = build_feature_matrix(train_rows)
        check_interrupted()
        TRAINING_STATUS["progress"] = 42
        x_val, _ = build_feature_matrix(val_rows)
        check_interrupted()
        TRAINING_STATUS["progress"] = 55

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        TRAINING_STATUS["message"] = "Training weighted image+metadata classifier."
        TRAINING_STATUS["progress"] = 65
        model = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=1200,
            tol=1e-3,
            random_state=42,
        )
        model.fit(x_train, train_rows["final_diagnosis"], sample_weight=train_rows["sample_weight"])
        check_interrupted()

        TRAINING_STATUS["message"] = "Evaluating validation groups and creating rereview flags."
        TRAINING_STATUS["progress"] = 82
        pred = model.predict(x_val)
        proba = model.predict_proba(x_val)
        confidence = proba.max(axis=1)

        val = val_rows.copy()
        val["predicted_diagnosis"] = pred
        val["confidence"] = confidence
        val["correct"] = val["final_diagnosis"].eq(val["predicted_diagnosis"])

        group_metrics = (
            val.groupby("age_x_sex")
            .agg(
                n=("image", "size"),
                accuracy=("correct", "mean"),
                mean_confidence=("confidence", "mean"),
            )
            .reset_index()
        )
        global_accuracy = accuracy_score(val["final_diagnosis"], val["predicted_diagnosis"])
        global_balanced_accuracy = balanced_accuracy_score(val["final_diagnosis"], val["predicted_diagnosis"])
        macro_recall = recall_score(
            val["final_diagnosis"],
            val["predicted_diagnosis"],
            average="macro",
            zero_division=0,
        )

        weak_groups = group_metrics[
            (group_metrics["n"] >= 12)
            & (
                (group_metrics["accuracy"] <= max(0.0, global_accuracy - 0.12))
                | (group_metrics["mean_confidence"] < 0.42)
            )
        ]["age_x_sex"].tolist()

        rereview = val[
            (~val["correct"])
            & (
                val["age_x_sex"].isin(weak_groups)
                | (val["confidence"] < 0.38)
            )
        ].copy()
        rereview["rereview_reason"] = np.where(
            rereview["age_x_sex"].isin(weak_groups),
            "post-training weak age x sex group",
            "post-training low confidence error",
        )
        rereview["diagnosis"] = rereview["final_diagnosis"]
        rereview["training_run_id"] = run_id
        rereview = rereview[
            [
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
        ].sort_values(["rereview_reason", "confidence"], ascending=[True, True])
        rereview.to_csv(ROOT / "post_training_rereview.csv", index=False)

        group_metrics.to_csv(ROOT / "training_group_metrics.csv", index=False)

        TRAINING_STATUS.update(
            {
                "state": "finished",
                "message": "Training complete. Post-training rereview queue is ready.",
                "progress": 100,
                "interrupt_requested": False,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "training_rows": int(len(train_rows)),
                    "validation_rows": int(len(val_rows)),
                    "feature_count": int(len(feature_names)),
                    "accuracy": float(global_accuracy),
                    "balanced_accuracy": float(global_balanced_accuracy),
                    "macro_recall": float(macro_recall),
                    "weak_group_count": int(len(weak_groups)),
                    "post_training_rereview_count": int(len(rereview)),
                },
            }
        )
    except TrainingInterrupted as exc:
        TRAINING_STATUS.update(
            {
                "state": "interrupted",
                "message": str(exc),
                "progress": 0,
                "interrupt_requested": False,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metrics": None,
            }
        )
    except Exception as exc:
        TRAINING_STATUS.update(
            {
                "state": "failed",
                "message": f"Training failed: {exc}",
                "progress": 0,
                "interrupt_requested": False,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metrics": None,
            }
        )


class PipelineHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/summary":
            data = dataset()
            queue = queue_with_reviews()
            reviews = load_reviews(PATHS.reviews)
            rereview = load_post_training_rereview(ROOT / "post_training_rereview.csv")
            payload = {
                "rows": data.groupby("split").size().to_dict(),
                "review_queue_count": int(len(queue)),
                "pending_review_count": int((queue["review_status"] == "pending").sum()),
                "reviewed_count": int(len(reviews)),
                "post_training_rereview_count": int(len(rereview)),
                "review_reasons": queue["review_reason"].value_counts().to_dict(),
            }
            respond_json(self, payload)
            return

        if parsed.path == "/api/queue":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["100"])[0])
            status = params.get("status", ["pending"])[0]
            queue = queue_with_reviews()
            if status != "all":
                queue = queue[queue["review_status"] == status]
            respond_json(self, queue.head(limit).to_dict(orient="records"))
            return

        if parsed.path == "/api/post-training-rereview":
            rereview = load_post_training_rereview(ROOT / "post_training_rereview.csv")
            respond_json(self, rereview.head(200).to_dict(orient="records"))
            return

        if parsed.path == "/api/training-status":
            respond_json(self, TRAINING_STATUS)
            return

        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/review":
            body = read_json(self)
            review = {
                "image": body["image"],
                "review_status": body.get("review_status", "confirm labels"),
                "reviewed_diagnosis": body.get("reviewed_diagnosis"),
                "reviewed_age": body.get("reviewed_age"),
                "reviewed_sex": body.get("reviewed_sex"),
                "reviewer_confidence": body.get("reviewer_confidence", "medium"),
                "reviewer_notes": body.get("reviewer_notes", ""),
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
            save_review(PATHS.reviews, review)
            respond_json(self, {"ok": True, "review": review})
            return

        if parsed.path == "/api/train":
            body = read_json(self)
            with TRAINING_LOCK:
                if TRAINING_STATUS["state"] == "running":
                    respond_json(self, TRAINING_STATUS, status=409)
                    return
            thread = threading.Thread(
                target=run_training,
                kwargs={
                    "max_train_rows": int(body.get("max_train_rows", 5000)),
                    "max_validation_rows": int(body.get("max_validation_rows", 1500)),
                },
                daemon=True,
            )
            thread.start()
            time.sleep(0.1)
            respond_json(self, TRAINING_STATUS)
            return

        if parsed.path == "/api/interrupt-training":
            with TRAINING_LOCK:
                if TRAINING_STATUS["state"] == "running":
                    TRAINING_STATUS["interrupt_requested"] = True
                    TRAINING_STATUS["message"] = "Interrupt requested. Stopping at the next safe point."
                    respond_json(self, TRAINING_STATUS)
                    return
            respond_json(self, {"ok": False, "message": "No active training run to interrupt."}, status=409)
            return

        respond_json(self, {"error": "Unknown endpoint"}, status=404)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8502), PipelineHandler)
    print("ISIC FairReview app running at http://127.0.0.1:8502")
    server.serve_forever()


if __name__ == "__main__":
    main()
