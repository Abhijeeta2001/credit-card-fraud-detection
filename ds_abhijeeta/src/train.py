"""
src/train.py

CLI-style training entrypoint.
"""

from src.pipeline import train_best_model
from src.model import save_model, MODEL_PATH


def main():
    trained = train_best_model()
    save_model(trained, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print("Metrics:")
    for k, v in trained.metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
