import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
import xgboost as xgb

from src.config import load_config
from src.data.database import get_db_connection
from src.graph.builder import create_clustered_splits

def load_enriched_cohort(db_path: str = "ensanut_2018.db"):
    conn = get_db_connection(db_path)
    query = """
    WITH res_agg AS (
        SELECT upm, viv_sel, hogar, 
               COUNT(*) as tam_hogar,
               SUM(CASE WHEN edad < 18 THEN 1 ELSE 0 END) as n_menores,
               SUM(CASE WHEN edad >= 60 THEN 1 ELSE 0 END) as n_adultos_mayores
        FROM residentes
        GROUP BY upm, viv_sel, hogar
    ),
    seg_agg AS (
        SELECT upm, viv_sel, hogar, p1 as inseguridad_alim_p1
        FROM seguridad_alimentaria
        GROUP BY upm, viv_sel, hogar
    ),
    ayu_agg AS (
        SELECT upm, viv_sel, hogar, p1 as recibe_ayuda_alim
        FROM ayuda_alimentaria
        GROUP BY upm, viv_sel, hogar
    )
    SELECT 
        a.upm, 
        a.viv_sel, 
        a.hogar, 
        a.numren,
        a.p3_1,
        a.f_20mas as factor_expansion,
        -- Demographics
        r.edad,
        r.sexo,
        r.nivel as nivel_educativo,
        r.estrato as estrato_socioeconomico,
        a.dominio,
        a.region,
        -- Anthropometrics & clinical history
        a.p1_1 as dx_obesidad,
        a.p1_4 as silueta_corporal,
        a.p1_5 as peso_habitual,
        a.p1_7 as cambio_peso,
        a.p1_8 as kg_cambio,
        a.p4_1 as dx_hipertension,
        a.p6_3 as dx_colesterol_trigliceridos,
        a.p7_1_1 as ant_padre_diab,
        a.p7_1_2 as ant_madre_diab,
        a.p7_1_3 as ant_hermano_diab,
        a.p7_2_1 as ant_padre_hta,
        a.p7_2_2 as ant_madre_hta,
        a.p7_2_3 as ant_hermano_hta,
        a.p13_1 as fuma_100_cigarros,
        a.p13_2 as fuma_actualmente,
        a.p14_1 as consume_alcohol,
        -- Household assets & structure
        h.p2_9_1 as tiene_refri,
        h.p2_9_2 as tiene_lavadora,
        h.p2_9_3 as tiene_auto,
        COALESCE(res_agg.tam_hogar, 1) as tam_hogar,
        COALESCE(res_agg.n_menores, 0) as n_menores,
        COALESCE(res_agg.n_adultos_mayores, 0) as n_adultos_mayores,
        COALESCE(seg_agg.inseguridad_alim_p1, 2) as inseguridad_alim_p1,
        COALESCE(ayu_agg.recibe_ayuda_alim, 2) as recibe_ayuda_alim
    FROM adultos a
    LEFT JOIN residentes r 
        ON a.upm = r.upm AND a.viv_sel = r.viv_sel AND a.hogar = r.hogar AND a.numren = r.numren
    LEFT JOIN hogares h 
        ON a.upm = h.upm AND a.viv_sel = h.viv_sel AND a.hogar = h.hogar
    LEFT JOIN res_agg 
        ON a.upm = res_agg.upm AND a.viv_sel = res_agg.viv_sel AND a.hogar = res_agg.hogar
    LEFT JOIN seg_agg 
        ON a.upm = seg_agg.upm AND a.viv_sel = seg_agg.viv_sel AND a.hogar = seg_agg.hogar
    LEFT JOIN ayu_agg 
        ON a.upm = ayu_agg.upm AND a.viv_sel = ayu_agg.viv_sel AND a.hogar = ayu_agg.hogar
    WHERE a.p3_1 IN (1, 3);
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def engineer_features(df: pd.DataFrame):
    y = (df["p3_1"] == 1).astype(int).values
    upm_clusters = df["upm"].values

    raw_weights = df["factor_expansion"].fillna(1.0).values.astype(np.float32)
    sample_weights = raw_weights / np.mean(raw_weights)

    # Base cleaned variables
    f_df = pd.DataFrame()

    # Binary clinical flags (1=Yes, 0=No)
    for col in ["dx_obesidad", "dx_hipertension", "dx_colesterol_trigliceridos",
                "ant_padre_diab", "ant_madre_diab", "ant_hermano_diab",
                "ant_padre_hta", "ant_madre_hta", "ant_hermano_hta",
                "fuma_100_cigarros", "tiene_refri", "tiene_lavadora", "tiene_auto",
                "inseguridad_alim_p1", "recibe_ayuda_alim"]:
        f_df[col] = (df[col] == 1).astype(float)

    f_df["sexo"] = (df["sexo"] == 1).astype(float)
    f_df["fuma_actualmente"] = (df["fuma_actualmente"] == 1).astype(float)
    f_df["consume_alcohol"] = (df["consume_alcohol"] == 1).astype(float)
    f_df["rural"] = (df["dominio"] == 2).astype(float)

    # Continuous variables with median imputation
    for col in ["edad", "peso_habitual", "kg_cambio", "silueta_corporal",
                "nivel_educativo", "estrato_socioeconomico", "region",
                "tam_hogar", "n_menores", "n_adultos_mayores"]:
        series = df[col].copy()
        series.loc[series >= 888] = np.nan
        f_df[col] = series.fillna(series.median()).values

    # === ADVANCED ENGINEERED CLINICAL FEATURES ===
    # 1. Genetic Burden Score (Dosage)
    f_df["genetic_burden_diab"] = (
        2.0 * f_df["ant_padre_diab"] + 
        2.0 * f_df["ant_madre_diab"] + 
        1.5 * f_df["ant_hermano_diab"]
    )
    f_df["both_parents_diab"] = (f_df["ant_padre_diab"] * f_df["ant_madre_diab"]).astype(float)
    f_df["fam_hta_burden"] = f_df["ant_padre_hta"] + f_df["ant_madre_hta"] + f_df["ant_hermano_hta"]

    # 2. Metabolic-Age Interactions
    age_norm = f_df["edad"] / 50.0
    weight_norm = f_df["peso_habitual"] / 70.0
    f_df["age_x_weight"] = age_norm * weight_norm
    f_df["age_x_hipertension"] = age_norm * f_df["dx_hipertension"]
    f_df["age_x_dyslipidemia"] = age_norm * f_df["dx_colesterol_trigliceridos"]
    f_df["age_x_genetic"] = age_norm * f_df["genetic_burden_diab"]

    # 3. Comorbidity & Adiposity Phenotype
    f_df["comorbidity_count"] = (
        f_df["dx_obesidad"] + f_df["dx_hipertension"] + f_df["dx_colesterol_trigliceridos"]
    )
    f_df["high_adiposity_silhouette"] = (f_df["silueta_corporal"] >= 6).astype(float)
    f_df["severe_adiposity_silhouette"] = (f_df["silueta_corporal"] >= 7).astype(float)

    # 4. Socioeconomic Deprivation & Environmental Vulnerability
    f_df["asset_deprivation"] = (
        (1.0 - f_df["tiene_refri"]) + 
        (1.0 - f_df["tiene_lavadora"]) + 
        (1.0 - f_df["tiene_auto"])
    )
    f_df["ses_vulnerability"] = (
        f_df["asset_deprivation"] + 
        f_df["inseguridad_alim_p1"] + 
        f_df["recibe_ayuda_alim"]
    )

    # 5. Lifestyle Cumulative Risk
    f_df["lifestyle_risk"] = f_df["fuma_actualmente"] + f_df["consume_alcohol"]

    return f_df, y, upm_clusters, sample_weights

def main():
    cfg = load_config("configs/default.yaml")
    print("Loading data and testing advanced feature engineering...")
    df = load_enriched_cohort()
    f_df, y, upm_clusters, sample_weights = engineer_features(df)
    print(f"Engineered Feature Matrix: {f_df.shape[0]:,} records x {f_df.shape[1]} features (from original 24)")

    n_nodes = len(f_df)
    train_mask, val_mask, test_mask = create_clustered_splits(upm_clusters, n_nodes, cfg.split)
    train_idx = np.where(train_mask.numpy())[0]
    val_idx = np.where(val_mask.numpy())[0]
    test_idx = np.where(test_mask.numpy())[0]

    X = f_df.values
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Evaluate LightGBM with engineered features
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=180,
        max_depth=6,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_test_probs = lgb_model.predict_proba(X_test)[:, 1]

    roc_lgb = roc_auc_score(y_test, lgb_test_probs)
    pr_lgb = average_precision_score(y_test, lgb_test_probs)
    brier_lgb = brier_score_loss(y_test, lgb_test_probs)

    print(f"\n[LightGBM with Advanced Features]:")
    print(f"  • Test ROC-AUC : {roc_lgb:.4f}  (Baseline was: 0.8491)")
    print(f"  • Test PR-AUC  : {pr_lgb:.4f}  (Baseline was: 0.3781)")
    print(f"  • Brier Score  : {brier_lgb:.4f}")

    # Top Feature Importances
    imp = pd.Series(lgb_model.feature_importances_, index=f_df.columns).sort_values(ascending=False)
    print("\nTop 12 Most Influential Features:")
    for rank, (feat, val) in enumerate(imp.head(12).items(), 1):
        print(f"  {rank:2d}. {feat:<28} : {val}")

if __name__ == "__main__":
    main()
