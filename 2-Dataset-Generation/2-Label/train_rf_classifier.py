import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold


DEFAULT_VALID_LABELS = ("spine", "shaft", "soma")


def log_hks_features(x):
    return np.log(np.clip(x, 1e-30, None))




def hks_feature_columns(df):
    cols = [col for col in df.columns if col.startswith("hks_")]
    if not cols:
        raise ValueError("No HKS feature columns found. Expected columns named hks_0 ... hks_31.")

    def hks_index(col):
        return int(col.rsplit("_", 1)[1])

    return sorted(cols, key=hks_index)


def majority_vote_predict(models, x):
    preds = np.array([model.predict(x) for model in models])
    return np.array([Counter(preds[:, i]).most_common(1)[0][0] for i in range(preds.shape[1])])


def train_groupkfold_ensemble(df, valid_labels=DEFAULT_VALID_LABELS, n_splits=5):
    filtered = df[df["tag"].isin(valid_labels)].copy()
    if filtered.empty:
        raise ValueError(f"No rows found with tag in {list(valid_labels)}.")

    feature_cols = hks_feature_columns(filtered)
    x = log_hks_features(filtered[feature_cols])
    y = filtered["tag"]
    groups = filtered["post_pt_root_id"]

    group_count = groups.nunique()
    if group_count < n_splits:
        raise ValueError(f"Need at least {n_splits} unique post_pt_root_id groups, found {group_count}.")

    print(f"Total dataset: {group_count} neurons ({len(x)} total compartments)")
    print(f"Features: {len(feature_cols)} columns ({feature_cols[0]} ... {feature_cols[-1]})")
    print("-" * 60)

    gkf = GroupKFold(n_splits=n_splits)
    ensemble_models = []
    fold_accuracies = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups=groups), 1):
        x_train_fold = x.iloc[train_idx]
        x_test_fold = x.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(x_train_fold, y_train_fold)

        y_pred = clf.predict(x_test_fold)
        acc = accuracy_score(y_test_fold, y_pred)

        fold_accuracies.append(acc)
        ensemble_models.append(clf)
        all_y_true.extend(y_test_fold)
        all_y_pred.extend(y_pred)

        num_test_neurons = filtered.iloc[test_idx]["post_pt_root_id"].nunique()
        print(f"Fold {fold}: Tested on {num_test_neurons} entirely new neurons (Accuracy: {acc:.2%})")

    label_order = list(ensemble_models[-1].classes_)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=label_order)

    print("-" * 60)
    print(f"Average True Holdout Accuracy: {np.mean(fold_accuracies):.2%}")
    print(f"Standard Deviation:            {np.std(fold_accuracies):.2%}\n")
    print("Overall Classification Report (Across all holdout folds):")
    print(classification_report(all_y_true, all_y_pred, labels=label_order))
    print("Confusion Matrix")
    print(f"labels: {label_order}")
    print(cm)

    return ensemble_models, {
        "feature_cols": feature_cols,
        "fold_accuracies": fold_accuracies,
        "labels": label_order,
        "confusion_matrix": cm,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
    }


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "david" / "microns" / "ml_ready.csv"
    default_output = Path(__file__).resolve().parent / "rf_ensemble.pkl"

    parser = argparse.ArgumentParser(
        description="Train the HKS Random Forest ensemble used by mask_generation.py."
    )
    parser.add_argument("--input-csv", type=Path, default=default_input)
    parser.add_argument("--output-model", type=Path, default=default_output)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_VALID_LABELS),
        help="Valid class labels to train on. Defaults match the notebook.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    models, _ = train_groupkfold_ensemble(df, valid_labels=tuple(args.labels), n_splits=args.n_splits)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, args.output_model)
    print(f"\nSaved ensemble: {args.output_model}")


if __name__ == "__main__":
    main()
