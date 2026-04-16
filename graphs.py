from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOGS_CSV_DIR = Path("logs_csv")
RESULTS_DIR = Path("results/training_curves")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pick all Attention CNN and ViT CSVs
csv_files = sorted(list(LOGS_CSV_DIR.glob("attention_cnn_*.csv")) + list(LOGS_CSV_DIR.glob("vit_*.csv")))

for csv_path in csv_files:
    df = pd.read_csv(csv_path)

    tag = csv_path.stem
    ep = df["epoch"]

    # Convert accuracy to percentage for plotting
    train_acc = df["train_acc"] * 100
    val_acc = df["val_acc"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves - {tag}", fontsize=14)

    # Loss plot
    ax1.plot(ep, df["train_loss"], "o-", label="Train Loss")
    ax1.plot(ep, df["val_loss"], "o-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(ep, train_acc, "o-", label="Train Acc")
    ax2.plot(ep, val_acc, "o-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fig_path = RESULTS_DIR / f"training_curves_{tag}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved -> {fig_path}")