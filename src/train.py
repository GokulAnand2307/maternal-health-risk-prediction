"""Train and evaluate maternal-health risk classifiers.

Academic portfolio project only - not for clinical use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURES = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]
TARGET = "RiskLevel"
LABELS = ["low risk", "mid risk", "high risk"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PALETTE = {0: "#2fbf9f", 1: "#f0b35a", 2: "#e76f73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Input CSV path")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[FEATURES + [TARGET]].drop_duplicates().copy()
    data[TARGET] = data[TARGET].astype(str).str.strip().str.lower()
    unexpected = sorted(set(data[TARGET].dropna()) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")

    return data[FEATURES], data[TARGET].map(LABEL_TO_ID)


def save_eda_charts(X: pd.DataFrame, y: pd.Series, output: Path) -> None:
    """Save the exploratory charts developed for the original dissertation."""
    frame = X.copy()
    frame[TARGET] = y
    frame["RiskLabel"] = frame[TARGET].map(dict(enumerate(LABELS)))

    plt.figure(figsize=(9, 6))
    sns.histplot(
        data=frame,
        x="Age",
        hue="RiskLabel",
        hue_order=LABELS,
        multiple="stack",
        palette=[PALETTE[index] for index in range(3)],
        bins=18,
    )
    plt.title("Age distribution by maternal risk level")
    plt.xlabel("Age (years)")
    plt.ylabel("Observations")
    plt.tight_layout()
    plt.savefig(output / "age_distribution_by_risk.svg")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=frame,
        x="RiskLabel",
        y="BS",
        order=LABELS,
        hue="RiskLabel",
        palette=[PALETTE[index] for index in range(3)],
        inner="quartile",
        legend=False,
    )
    plt.title("Blood sugar distribution by maternal risk level")
    plt.xlabel("Risk level")
    plt.ylabel("Blood sugar (mmol/L)")
    plt.tight_layout()
    plt.savefig(output / "blood_sugar_by_risk.svg")
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=frame,
        x="SystolicBP",
        y="DiastolicBP",
        hue="RiskLabel",
        hue_order=LABELS,
        palette=[PALETTE[index] for index in range(3)],
        alpha=0.75,
        s=55,
    )
    plt.title("Systolic and diastolic blood pressure by risk level")
    plt.xlabel("Systolic blood pressure (mm Hg)")
    plt.ylabel("Diastolic blood pressure (mm Hg)")
    plt.tight_layout()
    plt.savefig(output / "blood_pressure_relationship.svg")
    plt.close()

    correlation = frame[FEATURES + [TARGET]].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(output / "correlation_heatmap.svg")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=frame,
        x="RiskLabel",
        y="HeartRate",
        order=LABELS,
        hue="RiskLabel",
        palette=[PALETTE[index] for index in range(3)],
        legend=False,
    )
    plt.title("Heart-rate distribution by maternal risk level")
    plt.xlabel("Risk level")
    plt.ylabel("Heart rate (bpm)")
    plt.tight_layout()
    plt.savefig(output / "heart_rate_by_risk.svg")
    plt.close()

    pair_data = frame[
        ["Age", "SystolicBP", "DiastolicBP", "BS", "HeartRate", "RiskLabel"]
    ].sample(n=min(100, len(frame)), random_state=42)
    pair = sns.pairplot(
        pair_data,
        hue="RiskLabel",
        hue_order=LABELS,
        palette=[PALETTE[index] for index in range(3)],
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.55, "s": 22},
    )
    pair.fig.suptitle("Pairwise feature relationships", y=1.02)
    pair.savefig(output / "pairwise_feature_relationships.svg")
    plt.close(pair.fig)


def make_pipeline(model: object, seed: int) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("smote", SMOTE(random_state=seed)),
            ("model", model),
        ]
    )


def save_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, title: str, output_path: Path
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.title(title)
    plt.xlabel("Predicted risk")
    plt.ylabel("Actual risk")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_feature_importance(model: object, title: str, output_path: Path) -> None:
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot.barh(color="#26a69a")
    plt.title(title)
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    X, y = load_data(args.data)
    save_eda_charts(X, y, args.output)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.seed,
        stratify=y,
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=args.seed,
            n_jobs=-1,
        ),
    }

    rows: list[dict[str, float | str]] = []
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator, args.seed)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "macro_precision": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "macro_recall": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "macro_f1": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
            }
        )

        save_confusion_matrix(
            y_test,
            predictions,
            f"{name.replace('_', ' ').title()} confusion matrix",
            args.output / f"confusion_matrix_{name}.svg",
        )
        save_feature_importance(
            pipeline.named_steps["model"],
            f"{name.replace('_', ' ').title()} feature importance",
            args.output / f"{name}_feature_importance.svg",
        )

    metrics = pd.DataFrame(rows).sort_values("macro_recall", ascending=False)
    metrics.to_csv(args.output / "model_metrics.csv", index=False)

    comparison = metrics.assign(
        model=metrics["model"].str.replace("_", " ").str.title()
    ).set_index("model")[
        ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    ]
    comparison.columns = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]
    comparison.plot.bar(figsize=(10, 6), color=["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"])
    plt.title("Model performance comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.output / "model_performance_comparison.svg")
    plt.close()
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
"""Train and evaluate maternal-health risk classifiers.

Academic portfolio project only - not for clinical use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURES = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]
TARGET = "RiskLevel"
LABELS = ["low risk", "mid risk", "high risk"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Input CSV path")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[FEATURES + [TARGET]].drop_duplicates().copy()
    data[TARGET] = data[TARGET].astype(str).str.strip().str.lower()
    unexpected = sorted(set(data[TARGET].dropna()) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")

    return data[FEATURES], data[TARGET].map(LABEL_TO_ID)


def make_pipeline(model: object, seed: int) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("smote", SMOTE(random_state=seed)),
            ("model", model),
        ]
    )


def save_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, title: str, output_path: Path
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.title(title)
    plt.xlabel("Predicted risk")
    plt.ylabel("Actual risk")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_feature_importance(model: object, title: str, output_path: Path) -> None:
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot.barh(color="#26a69a")
    plt.title(title)
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.seed,
        stratify=y,
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=args.seed,
            n_jobs=-1,
        ),
    }

    rows: list[dict[str, float | str]] = []
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator, args.seed)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "macro_precision": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "macro_recall": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "macro_f1": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
            }
        )

        save_confusion_matrix(
            y_test,
            predictions,
            f"{name.replace('_', ' ').title()} confusion matrix",
            args.output / f"confusion_matrix_{name}.png",
        )
        save_feature_importance(
            pipeline.named_steps["model"],
            f"{name.replace('_', ' ').title()} feature importance",
            args.output / f"{name}_feature_importance.png",
        )

    metrics = pd.DataFrame(rows).sort_values("macro_recall", ascending=False)
    metrics.to_csv(args.output / "model_metrics.csv", index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

