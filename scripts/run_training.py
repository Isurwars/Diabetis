#!/usr/bin/env python3
import os
import sys
import argparse
import torch
from torch_geometric.data import Data

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.config import load_config
from src.data.database import load_raw_cohort
from src.features.preprocessor import FeaturePreprocessor
from src.graph.builder import build_knn_graph, build_multirelational_graph, create_clustered_splits
from src.models.gatv2 import GATv2DiabetesClassifier, FocalLoss
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.explainer import PatientAttentionExplainer

def main():
    parser = argparse.ArgumentParser(description="Run modular GATv2 training pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu or cuda)")
    parser.add_argument("--graph-type", type=str, default="knn", choices=["knn", "multirelational"], help="Graph topology: 'knn' (default, higher AUC) or 'multirelational' (ablation test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.device is not None:
        cfg.training.device = args.device

    print("=" * 65)
    print("      ENSANUT 2018 DIABETES GATv2 TRAINING PIPELINE")
    print("=" * 65)
    print(f"• Config File      : {args.config}")
    print(f"• Database         : {cfg.data.db_path}")
    print(f"• Graph Topology   : {args.graph_type.upper()} (k={cfg.graph.k} {cfg.graph.metric})")
    print(f"• Epochs           : {cfg.training.epochs}")
    print(f"• Target Device    : {cfg.training.device.upper()}")

    # 1. Load Data
    print("\n[1/5] Extracting cohort and household epidemiology from SQLite...")
    df = load_raw_cohort(cfg.data.db_path, cfg.data.exclude_gestational)
    print(f"      Loaded {len(df):,} adult records.")

    # 2. Preprocess features & extract survey weights
    print("\n[2/5] Preprocessing features and survey expansion factors...")
    preprocessor = FeaturePreprocessor(cfg.features)
    X, y, upm_clusters, sample_weights = preprocessor.fit_transform(df)
    n_nodes, in_features = X.shape
    print(f"      Feature Matrix: {n_nodes:,} nodes x {in_features} features (with household indicators).")
    print(f"      Prevalence: {y.sum():,} diagnosed diabetes ({y.mean()*100:.2f}% positive).")

    # 3. Construct Graph & Clustered Splits
    train_mask, val_mask, test_mask = create_clustered_splits(upm_clusters, n_nodes, cfg.split)
    
    if args.graph_type == "multirelational":
        print("\n[3/5] Building Multi-Relational Graph (Clinical k-NN + Community UPM edges)...")
        edge_index, edge_type = build_multirelational_graph(X, upm_clusters, cfg.graph)
        edge_dim = 16
        print(f"      Multi-Relational Graph: {n_nodes:,} nodes, {edge_index.shape[1]:,} directed edges.")
        print(f"      Relation 0 (Clinical k-NN): {(edge_type == 0).sum().item():,} edges")
        print(f"      Relation 1 (Community UPM): {(edge_type == 1).sum().item():,} edges")
    else:
        print("\n[3/5] Building Simplified Clinical k-NN Patient Graph (Optimal Homophily)...")
        edge_index = build_knn_graph(X, cfg.graph)
        edge_type = None
        edge_dim = None
        print(f"      Clinical k-NN Graph: {n_nodes:,} nodes, {edge_index.shape[1]:,} directed edges.")

    pyg_data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor(y, dtype=torch.float32),
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    pyg_data = pyg_data.to(device)

    # 4. Initialize Model, Loss, Optimizer
    model_desc = "Multi-Relational GATv2" if edge_dim else "Simplified GATv2"
    print(f"\n[4/5] Initializing {model_desc} on {device.upper()}...")
    model = GATv2DiabetesClassifier(in_features, cfg.model, edge_dim=edge_dim).to(device)
    criterion = FocalLoss(alpha=cfg.training.focal_loss_alpha, gamma=cfg.training.focal_loss_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    best_val_roc = 0.0
    best_state = None

    print(f"\n[5/5] Training for {cfg.training.epochs} epochs...")
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(pyg_data.x, pyg_data.edge_index, edge_type)
        
        # Apply survey-weighted focal loss
        train_weights = pyg_data.weights[pyg_data.train_mask] if cfg.features.use_survey_weights else None
        loss = criterion(logits[pyg_data.train_mask], pyg_data.y[pyg_data.train_mask], train_weights)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == cfg.training.epochs:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(pyg_data.x, pyg_data.edge_index, edge_type)
                val_probs = torch.sigmoid(val_logits[pyg_data.val_mask]).cpu().numpy()
                val_true = pyg_data.y[pyg_data.val_mask].cpu().numpy()
                val_metrics = evaluate_predictions(val_true, val_probs)
                
                if val_metrics["roc_auc"] > best_val_roc:
                    best_val_roc = val_metrics["roc_auc"]
                    best_state = model.state_dict().copy()

                print(f"      Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f} | Val PR-AUC: {val_metrics['pr_auc']:.4f}")

    # Evaluate Best Weights on Test Set
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(cfg.training.checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.training.checkpoint_path)
    print(f"\n[✔] Best model saved to: {cfg.training.checkpoint_path}")

    model.eval()
    with torch.no_grad():
        test_logits, _ = model(pyg_data.x, pyg_data.edge_index, edge_type)
        test_probs = torch.sigmoid(test_logits[pyg_data.test_mask]).cpu().numpy()
        test_true = pyg_data.y[pyg_data.test_mask].cpu().numpy()
        test_metrics = evaluate_predictions(test_true, test_probs)

    print("\n" + "=" * 65)
    print("             TEST SET EVALUATION METRICS (GATv2)             ")
    print("=" * 65)
    print(f"  • Test ROC-AUC                      : {test_metrics['roc_auc']:.4f}")
    print(f"  • Test PR-AUC (Average Precision)  : {test_metrics['pr_auc']:.4f}")
    print(f"  • Brier Score                       : {test_metrics['brier_score']:.4f}")
    print(f"  • Optimal Youden Threshold (Sens/Spec): {test_metrics['youden_threshold']:.4f}")
    print(f"  • Optimal F1 Threshold              : {test_metrics['f1_threshold']:.4f}")

    y_rep = test_metrics["report_youden"]
    print(f"\n[A] Performance at Screening Threshold (Youden's J = {test_metrics['youden_threshold']:.4f}):")
    print(f"    - Sensitivity / Recall (Diabetes) : {y_rep['Diagnosed Diabetes']['recall']*100:.2f}%")
    print(f"    - Specificity (Non-Diabetes)       : {y_rep['No Diabetes']['recall']*100:.2f}%")
    print(f"    - Overall Accuracy                 : {y_rep['accuracy']*100:.2f}%")
    print(f"    - Confusion Matrix                 : {test_metrics['cm_youden']}")

    f1_rep = test_metrics["report_f1"]
    print(f"\n[B] Performance at Balanced Threshold (F1 Max = {test_metrics['f1_threshold']:.4f}):")
    print(f"    - F1 Score (Diabetes)              : {f1_rep['Diagnosed Diabetes']['f1-score']:.4f}")
    print(f"    - Overall Accuracy                 : {f1_rep['accuracy']*100:.2f}%")
    print(f"    - Confusion Matrix                 : {test_metrics['cm_f1']}")
    print("=" * 65)

    # Sample Explainability demonstration
    print("\n[+] Demonstrating Clinical Explainability on high-risk patient...")
    test_node_indices = torch.where(pyg_data.test_mask)[0]
    # Pick a true positive test patient
    pos_test_nodes = test_node_indices[pyg_data.y[test_node_indices] == 1]
    if len(pos_test_nodes) > 0:
        query_node = int(pos_test_nodes[0].item())
        explainer = PatientAttentionExplainer(model, pyg_data, preprocessor.get_feature_names())
        explanation = explainer.explain_patient(query_node, top_n=3)
        print(f"    - Query Patient #{query_node}: Actual=Diabetic, Predicted Risk={explanation['predicted_risk']:.2%}")
        print(f"    - Top 3 Influential Peers in Graph:")
        for rank, peer in enumerate(explanation["top_influential_peers"], 1):
            status = "Diabetic" if peer["peer_is_diabetic"] else "Non-Diabetic"
            print(f"      {rank}. Node #{peer['peer_node_id']} (Attn Weight: {peer['attention_weight']:.4f}, Status: {status})")

if __name__ == "__main__":
    main()
