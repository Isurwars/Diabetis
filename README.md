# ENSANUT 2018 Diabetes Prediction via Graph Neural Networks (GNN)

Predicting diabetes risk in Mexican citizens using the **Encuesta Nacional de Salud y Nutrición (ENSANUT) 2018** published by INEGI / INSP. This project formulates patient risk prediction as a **Patient-Similarity Graph Attention Network (GATv2)** with survey expansion weights and spatial cluster-level validation.

---

## 🚀 Key Features

* **Consolidated SQLite Storage**: Seamlessly queries 9 ENSANUT modules (`adultos`, `residentes`, `hogares`, `viviendas`, `seguridad_alimentaria`, `ayuda_alimentaria`) indexed on `(upm, viv_sel, hogar, numren)` in [`ensanut_2018.db`](file:///home/isurwars/Projects/Diabetis/ensanut_2018.db).
* **Household Epidemiological Enrichment**: Integrates household size (`tam_hogar`), child count (`n_menores`), elderly count (`n_adultos_mayores`), ELCSA food insecurity (`inseguridad_alim_p1`), and nutritional assistance programs (`recibe_ayuda_alim`).
* **Multi-Relational Graph Architecture**: Over **1,042,853 directed edges** combining two complementary relation types:
  * **Relation 0 (Clinical $k$-NN)**: Connects clinically similar individuals (583,987 edges).
  * **Relation 1 (Local Community / Co-habitation)**: Connects individuals residing in the same `UPM` community (458,866 edges).
* **Zero Data Leakage**: Excludes post-diagnosis conditional fields (`P3_2` to `P3_18`) and gestational diabetes (`P3_1 = 2`) to ensure valid epidemiology.
* **Complex Survey Weights**: Incorporates official sampling expansion factors (`f_20mas`) directly into the training loss.
* **Spatial Clustered Validation**: Evaluates models strictly on held-out geographic sampling clusters (`UPM`) to prevent spatial snooping and measure real out-of-sample community generalization.
* **Clinical Explainability**: Leverages GATv2 multi-relational attention weights to explain *which similar peer patients and community neighbors* influenced a patient's risk score.

---

## 📊 Empirical Performance Benchmark

Evaluated on 6,584 citizens in held-out geographic sampling clusters:

| Model | Test ROC-AUC | Test PR-AUC | Screening Recall (Sensitivity) | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `0.8396` | `0.3490` | 74.3% | 76.8% |
| **XGBoost (Weighted)** | `0.8455` | `0.3761` | 76.5% | 77.2% |
| **LightGBM** | `0.8488` | `0.3765` | 77.1% | 77.8% |
| **GATv2 (Graph Attention)** | `0.8356` | `0.3509` | **81.86%** | **71.45%** |

*Note: GATv2 provides direct graph-level clinical explainability by identifying influential peer neighbors for every patient.*

---

## 🛠️ Project Structure

```
Diabetis/
├── configs/
│   └── default.yaml             # Central configuration (hyperparameters, paths)
├── src/
│   ├── config.py                # Typed dataclass config loader
│   ├── data/                    # Database connection & SQL extraction
│   │   └── database.py
│   ├── features/                # Imputation, scaling & survey weights
│   │   └── preprocessor.py
│   ├── graph/                   # k-NN graph construction & UPM splitting
│   │   └── builder.py
│   ├── models/                  # GATv2, Focal Loss & Tabular Baselines
│   │   ├── gatv2.py
│   │   └── baselines.py
│   └── evaluation/              # Threshold calibration & Attention explainer
│       ├── metrics.py
│       └── explainer.py
├── scripts/
│   ├── run_training.py          # Train & evaluate GATv2
│   ├── run_baselines.py         # Benchmark XGBoost, LightGBM, Logistic Regression
│   └── predict_patient.py       # Patient-level risk scoring & peer inspection CLI
├── checkpoints/                 # Saved PyTorch model weights (best_gatv2.pt)
├── consolidate_to_sqlite.py     # ETL CSV to SQLite pipeline
└── README.md
```

---

## 💻 Usage Guide

### 1. Environment Setup
```bash
# Uses uv with Python 3.12 (already prepared in .venv)
source .venv/bin/activate
```

### 2. Run GATv2 Training & Evaluation
```bash
# Train GATv2 using settings from configs/default.yaml
python scripts/run_training.py --epochs 30 --device cpu
```

### 3. Run Tabular Baselines Benchmark
```bash
# Benchmark Logistic Regression, LightGBM, and XGBoost on identical UPM splits
python scripts/run_baselines.py
```

### 4. Interactive Patient Risk & Explainability Inspection
```bash
# Inspect any patient (e.g. Node #4) to see risk score and top 5 influential graph peers:
python scripts/predict_patient.py --patient-idx 4 --top-peers 5
```
