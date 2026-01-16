import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from rdkit import Chem
import dgl
from dgl.nn import GraphConv, AvgPooling
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os

# ==========================================
# 0. Graph Utilities
# ==========================================
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    g = dgl.graph(([], []))
    g.add_nodes(mol.GetNumAtoms())

    src, dst = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([u, v])
        dst.extend([v, u])
    g.add_edges(src, dst)

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(
            one_hot_encoding(atom.GetSymbol(),
                             ['C', 'N', 'O', 'F', 'S', 'Cl', 'Unknown'])
        )

    g.ndata['h'] = torch.tensor(atom_features, dtype=torch.float32)
    g = dgl.add_self_loop(g)
    return g

# ==========================================
# 1. Dataset
# ==========================================
class DrugDiscoveryDataset(Dataset):
    def __init__(self, csv_path, is_test=False):
        self.df = pd.read_csv(csv_path)
        self.is_test = is_test
        self.graphs = []
        self.labels = []

        print(f"Processing graphs for {csv_path}")
        for i, smi in enumerate(self.df['Drug']):
            g = smiles_to_graph(smi)
            if g is not None:
                self.graphs.append(g)
                if not is_test:
                    self.labels.append(int(self.df.iloc[i]['Label']))

        if not is_test:
            self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.is_test:
            return self.graphs[idx], self.df.iloc[idx]['id']
        return self.graphs[idx], self.labels[idx]

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    bg = dgl.batch(graphs)
    return bg, torch.tensor(labels)

# ==========================================
# 2. Model (slightly stabilized)
# ==========================================
class SimpleGNN(nn.Module):
    def __init__(self, in_feats, hidden_dim=64):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.pool = AvgPooling()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = self.pool(g, h)
        h = self.dropout(h)
        return self.fc(h).squeeze(1)

# ==========================================
# 3. Evaluation
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for bg, y in loader:
            bg = bg.to(device)
            y = y.to(device).float()

            logits = model(bg, bg.ndata['h'])
            p = torch.sigmoid(logits)

            probs.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    probs = np.array(probs)
    labels = np.array(labels)

    aupr = average_precision_score(labels, probs)
    preds = (probs >= 0.5).astype(int)

    print("Confusion matrix:\n", confusion_matrix(labels, preds))
    print(f"Pred stats → min:{probs.min():.4f} max:{probs.max():.4f} mean:{probs.mean():.4f}")

    return aupr, f1_score(labels, preds)

# ==========================================
# 4. Training (CORRECT WEIGHTED SAMPLER)
# ==========================================
def train_baseline():
    device = torch.device("cpu")
    print("Using device:", device)

    train_path = '../data/train.csv'
    test_path = '../data/test.csv'

    full_ds = DrugDiscoveryDataset(train_path)

    print("Label distribution:")
    print(pd.Series(full_ds.labels.numpy()).value_counts(normalize=True))

    train_idx, val_idx = train_test_split(
        range(len(full_ds)),
        test_size=0.2,
        random_state=42,
        stratify=full_ds.labels
    )

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    # ---- FIXED SAMPLER (TRAIN ONLY)
    train_labels = full_ds.labels[train_idx]
    class_counts = torch.bincount(train_labels)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=64, sampler=sampler, collate_fn=collate
    )

    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=collate
    )

    model = SimpleGNN(in_feats=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ❗ NO pos_weight when using sampler
    loss_fn = nn.BCEWithLogitsLoss()

    best_aupr = 0

    for epoch in range(20):
        model.train()
        total_loss = 0

        for bg, y in train_loader:
            bg = bg.to(device)
            y = y.to(device).float()

            logits = model(bg, bg.ndata['h'])
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_aupr, val_f1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1:02d} | "
              f"Loss {total_loss/len(train_loader):.4f} | "
              f"Val AUPR {val_aupr:.4f} | Val F1 {val_f1:.4f}")

        if val_aupr > best_aupr:
            best_aupr = val_aupr
            torch.save(model.state_dict(), 'best_model.pth')
            print(">>> Best model saved")

    # ==========================================
    # 5. Test Prediction
    # ==========================================
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_ds = DrugDiscoveryDataset(test_path, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)

    ids, preds = [], []

    with torch.no_grad():
        for bg, id_ in test_loader:
            bg = bg.to(device)
            p = torch.sigmoid(model(bg, bg.ndata['h']))
            preds.extend((p >= 0.5).int().cpu().numpy())
            ids.extend(id_)

    os.makedirs('../submissions', exist_ok=True)
    pd.DataFrame({'id': ids, 'target': preds}).to_csv(
        '../submissions/baseline_submission.csv', index=False
    )

    print("Submission saved.")

if __name__ == '__main__':
    train_baseline()
