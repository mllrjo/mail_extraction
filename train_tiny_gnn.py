# File: train_tiny_gnn.py
# Description: Dynamically train TinyGNN based on detected domain and fields.

import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from utils import TinyGNN, GNNDataset, infer_domain_fields
from tqdm import tqdm

# === CONFIG ===
GRAPH_DIR = "gnn_data"
MODEL_PATH = "models/tinygnn_model.pth"
BATCH_SIZE = 8
EPOCHS = 10
HIDDEN_DIM = 64
LEARNING_RATE = 0.01
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
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        mask = (batch.y != -1)
        loss = F.cross_entropy(out[mask], batch.y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"🧪 Epoch {epoch:02d}: Loss = {avg_loss:.4f}")

# === SAVE MODEL ===
os.makedirs("models", exist_ok=True)
torch.save({
    "input_dim": input_dim,
    "output_dim": output_dim,
    "label2idx": label2idx,
    "state_dict": model.state_dict()
}, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

