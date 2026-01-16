# GNN-challenge-task
<<<<<<< HEAD
# GNN Challenge: Drug–Target Interaction Classification (Davis Dataset)

## 📌 Problem Definition
=======
##  Problem Definition
>>>>>>> cfa9cd718606814d8255874dc775c61d694292e7
This challenge focuses on **binary classification of drug–target interactions (DTIs)** using **Graph Neural Networks (GNNs)**.  
Given a drug molecule graph and a protein target graph, the goal is to predict whether the interaction is **strong or weak**.

This problem is covered under **DGL Lectures 1.1–4.6**, including:
- Message passing neural networks
- Graph-level classification
- Neighborhood aggregation
- Sampling and batching of graph data
- Imbalanced classification handling

---

<<<<<<< HEAD
## 🧠 Problem Type
=======
##  Problem Type
>>>>>>> cfa9cd718606814d8255874dc775c61d694292e7
- **Graph Classification**
- **Binary Classification**

---

<<<<<<< HEAD
## 📊 Dataset
=======
##  Dataset
>>>>>>> cfa9cd718606814d8255874dc775c61d694292e7
- **Dataset**: Davis Drug–Target Interaction Dataset
- **Source**: Public benchmark dataset used in DeepDTA and GraphDTA
- **Description**:
  - Drugs represented as molecular graphs
  - Targets represented as protein graphs
  - Continuous binding affinity values converted to binary labels

**Label Construction**:
- Strong interaction: KIBA/Davis score ≥ threshold
- Weak interaction: KIBA/Davis score < threshold

---

<<<<<<< HEAD
## 🎯 Objective Metric
=======
##  Objective Metric
>>>>>>> cfa9cd718606814d8255874dc775c61d694292e7
The primary evaluation metric is:

- **F1-score (macro)**

This metric is chosen because:
- The dataset is **class-imbalanced**
- Accuracy is misleading in this setting
- F1-score is difficult to optimize and more informative

Secondary metrics:
- Precision
- Recall
- ROC-AUC

---

<<<<<<< HEAD
## ⚙️ Constraints
=======
## Constraints
>>>>>>> cfa9cd718606814d8255874dc775c61d694292e7
- No external datasets allowed
- Must use only the Davis dataset
- Model must train within **reasonable time on a single GPU or CPU**
- Only methods covered in **DGL Lectures 1.1–4.6** may be used

---

<<<<<<< HEAD
## 🏗 Model Architecture
=======
##  Model Architecture
>>>>>>> cfa9cd718606814d8255874dc775c61d694292e7
- Drug encoder: **GIN (Graph Isomorphism Network)**
- Protein encoder: **GIN**
- Graph-level pooling: Global mean pooling
- Classifier: Fully connected layers
- Loss function: Binary Cross Entropy
- Class imbalance handled using **WeightedRandomSampler**

---

<<<<<<< HEAD
## 🧪 Training Details
=======
##  Training Details
>>>>>>> cfa9cd718606814d8255874dc775c61d694292e7
- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- Sampling: WeightedRandomSampler
- Framework: PyTorch + DGL

---

<<<<<<< HEAD
## 📦 Installation
=======
##  Installation
>>>>>>> cfa9cd718606814d8255874dc775c61d694292e7

```bash
conda create -n dgl_challenge python=3.9 -y
conda activate dgl_challenge
pip install -r requirements.txt
