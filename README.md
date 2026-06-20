# CrystalSSL-Benchmark

## A Controlled Comparison of Contrastive and Masked Self-Supervised Learning for Crystal Graph Representation Learning

---

## Overview

Self-supervised learning (SSL) has emerged as a powerful paradigm for learning representations from unlabeled data. While SSL has transformed computer vision and natural language processing, its effectiveness for crystalline materials remains an active area of research.

This project presents a controlled benchmark of two widely used SSL objectives for crystal graph neural networks:

* Contrastive Learning
* Masked Atom Modeling

Both methods are evaluated using the same graph construction pipeline, identical GNN architecture, identical optimization settings, and identical downstream evaluation protocol. By controlling every component except the self-supervised objective, performance differences can be attributed directly to the pretraining strategy rather than architectural variation.

The central question investigated is:

> Which self-supervised learning objective produces more useful crystal representations for downstream materials property prediction?

---

## Key Findings

* Built a controlled benchmark comparing Contrastive Learning and Masked Atom Modeling on the exact same crystal GNN backbone.
* Evaluated both frozen crystal representations and downstream band-gap prediction performance.
* Both SSL objectives learned informative crystal representations, achieving approximately 81% metal/non-metal classification accuracy under frozen-embedding evaluation.
* Masked Atom Modeling achieved the strongest downstream band-gap prediction performance.
* Masked SSL improved test RMSE by approximately 11.5% relative to a randomly initialized baseline.
* Results suggest that reconstruction-based objectives may align particularly well with chemically meaningful crystal representation learning.

---

## Architecture

```text
                    Crystal Structure
                            │
                            ▼
                    Graph Construction
                            │
                            ▼
                     Shared GNNEncoder
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
          ▼                                   ▼
 Contrastive Learning             Masked Atom Modeling
      (NT-Xent Loss)               (Atom Reconstruction)
          │                                   │
          └─────────────────┬─────────────────┘
                            │
                            ▼
                   Pretrained Encoder
                            │
                            ▼
                   Band Gap Fine-Tuning
                            │
                            ▼
                     RMSE Evaluation
```

---

## Experimental Design

To isolate the effect of the self-supervised objective, all experiments use:

* Identical graph construction
* Identical train/validation/test splits
* Identical GNN architecture
* Identical hidden dimensions
* Identical optimization settings
* Identical downstream regression head

The only component that changes between conditions is the pretraining objective.

This controlled setup allows differences in performance to be attributed directly to the SSL strategy rather than architectural variation.

---

## Dataset

### Source

Data was obtained from the Materials Project database through the official `mp_api` client.

Retrieved fields:

* Material ID
* Crystal Structure
* Reduced Formula
* Band Gap

### Filtering

To maintain manageable graph sizes while preserving chemical diversity:

* Maximum 80 atoms per unit cell
* Between 1 and 8 distinct elements
* Structures with missing band-gap values removed

### Final Dataset

```text
3553 crystal structures
```

### Why Materials Project?

Materials Project is one of the largest publicly available databases of DFT-computed crystal structures and materials properties, making it a widely used benchmark source for materials machine learning research.

---

## Crystal Graph Construction

Each crystal structure is converted into a graph representation.

### Nodes

Nodes correspond to atoms.

Initial node feature:

```text
Atomic Number
```

### Edges

Edges are generated using periodic-neighbor search with a 4.0 Å cutoff radius.

Periodic boundary conditions are explicitly respected so that bonds crossing unit-cell boundaries are correctly represented.

### Edge Features

Each edge stores:

```text
Interatomic Distance
```

### Connectivity Safeguard

Rare structures producing zero neighbors under the cutoff rule are connected using a k-nearest-neighbor fallback graph to ensure every crystal remains connected.

---

## Model Architecture

All experiments share the same backbone architecture.

### Shared GNN Encoder

The encoder consists of:

* Atomic embedding layer
* 4× GINEConv layers
* LayerNorm + ReLU activations
* Jumping Knowledge aggregation

### Why GINE?

GINEConv explicitly incorporates edge attributes during message passing.

For crystalline materials, bond length contains physically meaningful information that standard graph convolutions often ignore.

### Why Jumping Knowledge?

Crystal properties depend on both local atomic environments and larger structural context.

Jumping Knowledge preserves information from multiple message-passing depths by combining representations from every layer instead of relying solely on the final layer.

### Prediction Heads

#### Projection Head

Used only during contrastive pretraining.

#### Masked Atom Decoder

Predicts masked atomic identities during masked pretraining.

#### Regression Head

Used during downstream band-gap prediction.

---

## Phase 1: Self-Supervised Representation Learning

### Contrastive Learning

Two augmented views of each crystal graph are generated using stochastic node removal.

The encoder is trained using NT-Xent loss to:

* Pull representations of the same crystal together
* Push representations of different crystals apart

### Masked Atom Modeling

15% of atoms are replaced with a special MASK token.

The encoder is trained to reconstruct the original atomic identities at masked positions using cross-entropy loss.

---

## Phase 1 Evaluation

Representations are evaluated without downstream fine-tuning.

Methods:

* Logistic Regression (Metal vs Non-Metal)
* K-Means Clustering
* Adjusted Rand Index (ARI)
* Normalized Mutual Information (NMI)
* Silhouette Score
* t-SNE Visualization

### Results

| Method               | LogReg Accuracy |   ARI |   NMI | Silhouette |
| -------------------- | --------------: | ----: | ----: | ---------: |
| Contrastive Learning |           0.814 | 0.024 | 0.053 |      0.195 |
| Masked Atom Modeling |           0.810 | 0.105 | 0.078 |      0.184 |

Both methods produced informative crystal representations, with masked encoding showing stronger natural cluster alignment with the underlying metal/non-metal labels.

---

## Phase 2: Band Gap Prediction

The pretrained encoder is fine-tuned end-to-end for band-gap regression.

Three conditions are compared:

1. Contrastive-pretrained encoder
2. Masked-pretrained encoder
3. Randomly initialized encoder

### Training Protocol

* Train / Validation / Test split: 68% / 12% / 20%
* Validation-based checkpoint selection
* End-to-end fine-tuning of encoder and regression head
* Evaluation using Root Mean Squared Error (RMSE)

### Results

| Model                 | Test RMSE (eV) | Improvement vs Baseline |
| --------------------- | -------------: | ----------------------: |
| Contrastive SSL       |         0.9145 |                  -1.47% |
| Masked SSL            |         0.7980 |                 +11.46% |
| Random Initialization |         0.9013 |                   0.00% |

Masked Atom Modeling achieved the strongest downstream performance, reducing prediction error by more than 11% relative to training from scratch.

---

## Results Summary

### Representation Quality

Both SSL objectives successfully learned meaningful crystal representations.

### Downstream Prediction

Masked Atom Modeling produced the strongest band-gap prediction performance.

### Overall Observation

The results indicate that reconstruction-based objectives may capture chemically relevant information particularly well for crystalline materials.

---

## Repository Structure

```text
.
├── data_utils.py
├── models.py
├── pretrain_compare.py
├── finetune_regression.py
├── outputs
│   ├── phase1
│   └── phase2
├── requirements.txt
└── README.md
```

---

## Future Work

Potential extensions include:

* Larger crystal datasets
* Additional materials properties
* Domain-specific crystal graph augmentations
* Alternative self-supervised objectives
* Larger graph neural network backbones
* Multi-property prediction benchmarks

---

## Conclusion

This project presents a controlled benchmark of two self-supervised learning objectives for crystal graph neural networks.

Both objectives produced informative crystal representations. However, masked atom modeling delivered the strongest downstream band-gap prediction performance, improving test RMSE by approximately 11.5% relative to a randomly initialized baseline.

These results suggest that reconstruction-based objectives may be particularly effective for learning chemically meaningful crystal representations and transferring them to downstream materials property prediction tasks.

## Limitations

* All results are from a single random seed; the contrastive-vs-baseline
  gap (-1.47%) is small enough to require multi-seed confirmation before
  treating it as conclusive.
* SSL pretraining is transductive: the encoder sees test-set crystal
  structures (not labels) during pretraining. This is fair across all
  three conditions but means results reflect generalization to unseen
  *labels*, not unseen *structures*.
* No hyperparameter sweep was performed; all architecture/training
  configs were fixed rather than tuned per method.
