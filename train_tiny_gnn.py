# File: train_tiny_gnn.py
# Description: Dynamically train TinyGNN based on detected domain and fields.

import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from utils import TinyGNN, GNNDataset, infer_domain_fields
from tqdm import tqdm

# === Smarter dynamic model naming ===
mode = os.getenv("SWDE_INDEX", "swde_index.json")
if "curated" in mode.lower():
    MODEL_PATH = "models/tinygnn_curated.pth"
else:
    MODEL_PATH = "models/tinygnn_full.pth"
# === Smart model saving based on SWDE_INDEX mode ===
mode = os.getenv("SWDE_INDEX", "swde_index.json")
model_tag = "curated" if "curated" in mode.lower() else "full"
MODEL_PATH = f"models/tinygnn_{model_tag}.pth"
print(f"💾 Saving model to: {MODEL_PATH}")



# === CONFIG ===
GRAPH_DIR = "gnn_data"
BATCH_SIZE = 8
EPOCHS = 50
HIDDEN_DIM = 64
LEARNING_RATE = 0.005
GT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "swde", "html_data", "groundtruth", "groundtruth"))

# === DETECT DOMAIN FROM DATA ===
sample_graphs = [f for f in os.listdir(GRAPH_DIR) if f.endswith(".pt")]
if not sample_graphs:
    raise ValueError(f"No .pt graph files found in {GRAPH_DIR}")

sample_data = torch.load(os.path.join(GRAPH_DIR, sample_graphs[0]))
if not hasattr(sample_data, "domain"):
    raise ValueError("Sample graph missing 'domain' attribute.")

domain = sample_data.domain
print(f"✅ Detected domain from graph: {domain}")

# === INFER LABELS FROM GROUNDTRUTH ===
domain_fields = infer_domain_fields(GT_BASE)
fields = sorted(domain_fields.get(domain, []))
label2idx = {label: i for i, label in enumerate(fields)}

print(f"✅ Training on fields: {fields}")

# === LOAD DATASET ===
dataset = GNNDataset(GRAPH_DIR)
input_dim = dataset[0].x.shape[1]
output_dim = len(label2idx)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === INIT MODEL ===
model = TinyGNN(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === TRAIN LOOP ===
print("🚀 Starting training...")
print("🚀 Starting training...")
losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        mask = (batch.y != -1)

        if mask.sum() == 0:
            continue  # Skip batches with no labels (safety)

        loss = F.cross_entropy(out[mask], batch.y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"🧪 Epoch {epoch:02d}: Loss = {avg_loss:.4f}")

    # === AFTER TRAINING ===
    # Save model and loss history
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "label2idx": label2idx,
        "losses": losses,   # ✅ Save losses for plotting later
    }, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

