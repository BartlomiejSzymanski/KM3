import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from time import time
import numpy as np

# ---- Wczytaj dane z JSON ----
with open("data/imdb_dataset_prepared.json", "r") as f:
    data = json.load(f)

# ---- Konwersja danych do tensorów ----
X_train = torch.tensor(data['X_train'], dtype=torch.long)
y_train = torch.tensor(data['y_train'], dtype=torch.float32)
X_test = torch.tensor(data['X_test'], dtype=torch.long)
y_test = torch.tensor(data['y_test'], dtype=torch.float32)

embeddings_matrix = torch.tensor(data['embeddings'], dtype=torch.float32)

vocab_size = embeddings_matrix.shape[0]
embedding_dim = embeddings_matrix.shape[1]
X_train = np.clip(X_train, 0, vocab_size - 1)
X_test = np.clip(X_test, 0, vocab_size - 1)

# ---- Dataset & Loader ----
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Max index in X_train:", X_train.max().item())
print("Vocab size:", vocab_size)

# ---- Model ----
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weights, seq_len):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        conv_out_len = (seq_len - 3 + 1) // 2
        self.fc = nn.Linear(8 * conv_out_len, 1)

    def forward(self, x):
        x = self.embedding(x)                 # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                # (batch, embed_dim, seq_len)
        x = F.relu(self.conv1(x))            # (batch, 8, L)
        x = self.pool(x)                      # (batch, 8, L/2)
        x = x.view(x.size(0), -1)             # flatten
        x = torch.sigmoid(self.fc(x))        # (batch, 1)
        return x

seq_len = X_train.shape[1]
model = CNNModel(vocab_size, embedding_dim, embeddings_matrix, seq_len)

# ---- Loss & Optimizer ----
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# ---- Accuracy Metric ----
def binary_accuracy(preds, y):
    preds = preds > 0.5
    return (preds == y.bool()).float().mean().item()

# ---- Training Loop ----
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, total_acc, num_batches = 0, 0, 0
    start = time()

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.squeeze())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += binary_accuracy(preds, yb.squeeze())
        num_batches += 1

    train_loss = total_loss / num_batches
    train_acc = total_acc / num_batches

    # ---- Eval ----
    model.eval()
    with torch.no_grad():
        preds_test = model(X_test).squeeze()
        test_loss = criterion(preds_test, y_test.squeeze()).item()
        test_acc = binary_accuracy(preds_test, y_test.squeeze())

    elapsed = time() - start
    print(f"Epoch: {epoch} ({elapsed:.2f}s)\tTrain: (l: {train_loss:.4f}, a: {train_acc:.4f})\t"
          f"Test: (l: {test_loss:.4f}, a: {test_acc:.4f})")
