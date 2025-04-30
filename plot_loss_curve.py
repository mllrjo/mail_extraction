# File: plot_loss_curve.py
# Description: Plot TinyGNN training loss over epochs.

import os
import torch
import matplotlib.pyplot as plt

# === Smart model path based on SWDE_INDEX ===
mode = os.getenv("SWDE_INDEX", "swde_index.json")
model_tag = "curated" if "curated" in mode.lower() else "full"
MODEL_PATH = f"models/tinygnn_{model_tag}.pth"
print(f"📊 Loading loss from: {MODEL_PATH}")

def main():
    checkpoint = torch.load(MODEL_PATH)
    losses = checkpoint.get("losses", None)

    if losses is None:
        print("❌ No loss history found in checkpoint.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"TinyGNN Loss Curve ({model_tag})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

