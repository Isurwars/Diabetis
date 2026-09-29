#!/usr/bin/env python3
"""
Patient Risk Predictor & Clinical Explainability CLI
===================================================
Loads the trained GATv2 model and inspects any patient's risk profile,
predicting diabetes probability and identifying the top influential
peers in the patient-similarity graph.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.config import load_config
from src.data.database import load_raw_cohort
from src.features.preprocessor import FeaturePreprocessor
from src.graph.builder import build_knn_graph, build_multirelational_graph
from src.models.gatv2 import GATv2DiabetesClassifier
from src.evaluation.explainer import PatientAttentionExplainer
from torch_geometric.data import Data

def main():
    parser = argparse.ArgumentParser(description="Predict diabetes risk for an individual patient")
    parser.add_argument("--patient-idx", type=int, default=4, help="Index of patient node (0 to 43,018)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_gatv2.pt", help="Path to model checkpoint")
    parser.add_argument("--top-peers", type=int, default=5, help="Number of influential peer neighbors to display")
    parser.add_argument("--graph-type", type=str, default="knn", choices=["knn", "multirelational"], help="Graph topology: 'knn' (default) or 'multirelational' (ablation test)")
    args = parser.parse_args()

    cfg = load_config()
    
    print(f"[1/3] Loading cohort and building {args.graph_type.upper()} Graph...")
    df = load_raw_cohort(cfg.data.db_path, cfg.data.exclude_gestational)
    preprocessor = FeaturePreprocessor(cfg.features)
    X, y, upm_clusters, _ = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names()
    
    if args.graph_type == "multirelational":
        edge_index, edge_type = build_multirelational_graph(X, upm_clusters, cfg.graph)
        edge_dim = 16
    else:
        edge_index = build_knn_graph(X, cfg.graph)
        edge_type = None
        edge_dim = None

    pyg_data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor(y, dtype=torch.float32)
    )

    print(f"[2/3] Loading GATv2 model from {args.checkpoint}...")
    model = GATv2DiabetesClassifier(in_features=X.shape[1], config=cfg.model, edge_dim=edge_dim)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    else:
        print(f"[-] Checkpoint {args.checkpoint} not found! Please run scripts/run_training.py first.")
        sys.exit(1)

    print(f"\n[3/3] Analyzing Patient #{args.patient_idx}...")
    explainer = PatientAttentionExplainer(model, pyg_data, feature_names)
    explanation = explainer.explain_patient(args.patient_idx, top_n=args.top_peers)

    raw_patient = df.iloc[args.patient_idx]
    actual_diag = "Diagnosed Diabetes" if explanation["actual_label"] == 1 else "No Diabetes"
    risk = explanation["predicted_risk"]
    
    print("=" * 75)
    print(f"             CLINICAL RISK ASSESSMENT REPORT (PATIENT #{args.patient_idx})")
    print("=" * 75)
    print(f"  • Predicted Diabetes Risk Probability: {risk:.2%}")
    print(f"  • Ground Truth Registry Diagnosis    : {actual_diag}")
    print(f"  • Screening Recommendation            : {'HIGH RISK - Refer for Glucose / HbA1c screening' if risk >= 0.40 else 'LOW RISK - Standard annual follow-up'}")
    
    print(f"\nPatient Key Clinical Vitals & Household Environment:")
    print(f"  - Age (Edad)                : {int(raw_patient.get('edad', 0))} years")
    print(f"  - Sex                       : {'Male' if raw_patient.get('sexo') == 1 else 'Female'}")
    print(f"  - Habitual Weight           : {raw_patient.get('peso_habitual', 'N/A')} kg")
    print(f"  - Clinical Obesity History  : {'Yes' if raw_patient.get('dx_obesidad') == 1 else 'No'}")
    print(f"  - Hypertension Diagnosis    : {'Yes' if raw_patient.get('dx_hipertension') == 1 else 'No'}")
    print(f"  - Maternal Diabetes History : {'Yes' if raw_patient.get('ant_madre_diab') == 1 else 'No'}")
    print(f"  - Paternal Diabetes History : {'Yes' if raw_patient.get('ant_padre_diab') == 1 else 'No'}")
    print(f"  - Household Size            : {int(raw_patient.get('tam_hogar', 1))} members")
    print(f"  - Minor Children in Home    : {int(raw_patient.get('n_menores', 0))}")
    print(f"  - Food Insecurity Worry     : {'Yes' if raw_patient.get('inseguridad_alim_p1') == 1 else 'No'}")

    print(f"\nTop {args.top_peers} Influential Peer Neighbors (Attention Explainability):")
    for i, peer in enumerate(explanation["top_influential_peers"], 1):
        peer_diag = "Diabetic" if peer["peer_is_diabetic"] else "Non-Diabetic"
        print(f"  [{i}] Node #{peer['peer_node_id']} | Type: {peer['relation_type']:<28} | Attn: {peer['attention_weight']:.4f} | Outcome: {peer_diag}")
    print("=" * 75)

if __name__ == "__main__":
    main()
