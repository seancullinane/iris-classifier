# src/train.py
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate(test_size: float = 0.2, random_state: int = 42) -> float:
    # Load
    data = load_iris()
    X, y = data.data, data.target
    target_names = data.target_names

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

    # Ensure outputs/
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix PNG
    fig_path = out_dir / "confusion_matrix.png"
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Acc: {acc:.3f})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {fig_path.resolve()}")

    # Save trained model
    model_path = out_dir / "model.joblib"
    dump(clf, model_path)
    print(f"Saved model → {model_path.resolve()}")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree on the Iris dataset and save outputs."
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"Config → test_size={args.test_size}, random_state={args.random_state}")
    train_and_evaluate(test_size=args.test_size, random_state=args.random_state)
