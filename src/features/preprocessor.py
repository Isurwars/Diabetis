import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from ..config import FeaturesConfig

class FeaturePreprocessor:
    """
    Transforms raw ENSANUT survey records into preprocessed feature vectors.
    Includes advanced epidemiological indices: genetic dosage burdens,
    metabolic-age interactions, comorbidity counts, adiposity phenotypes,
    and socioeconomic deprivation scores.
    """
    def __init__(self, config: FeaturesConfig = None):
        self.config = config or FeaturesConfig()
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms raw DataFrame into:
        - X: Preprocessed feature matrix (N, D)
        - y: Binary targets (N,)
        - upm_clusters: Spatial cluster IDs for grouped splitting (N,)
        - sample_weights: Normalized survey expansion weights (N,)
        """
        # Target: 1 for diagnosed diabetes, 0 for negative
        y = (df["p3_1"] == 1).astype(int).values
        upm_clusters = df["upm"].values
        
        # Survey weights normalized to mean 1.0
        raw_weights = df["factor_expansion"].fillna(1.0).values.astype(np.float32)
        sample_weights = raw_weights / np.mean(raw_weights)

        f_df = pd.DataFrame()

        # Binary clinical & household flags (1=Yes, 0=No)
        binary_cols = [
            "dx_obesidad", "dx_hipertension", "dx_colesterol_trigliceridos",
            "ant_padre_diab", "ant_madre_diab", "ant_hermano_diab",
            "fuma_100_cigarros", "tiene_refri", "tiene_lavadora", "tiene_auto",
            "inseguridad_alim_p1", "recibe_ayuda_alim"
        ]
        # Optional columns from database if present
        for col in ["ant_padre_hta", "ant_madre_hta", "ant_hermano_hta"]:
            if col in df.columns:
                binary_cols.append(col)

        for col in binary_cols:
            if col in df.columns:
                f_df[col] = (df[col] == 1).astype(float)
            else:
                f_df[col] = 0.0

        f_df["sexo"] = (df["sexo"] == 1).astype(float)
        f_df["fuma_actualmente"] = (df["fuma_actualmente"] == 1).astype(float)
        
        if "consume_alcohol" in df.columns:
            f_df["consume_alcohol"] = (df["consume_alcohol"] == 1).astype(float)
        else:
            f_df["consume_alcohol"] = 0.0

        if "dominio" in df.columns:
            f_df["rural"] = (df["dominio"] == 2).astype(float)
        else:
            f_df["rural"] = 0.0

        # Continuous / ordinal columns: replace refusal codes (>=888) with NaN and impute median
        continuous_cols = [
            "edad", "peso_habitual", "kg_cambio", "silueta_corporal",
            "nivel_educativo", "estrato_socioeconomico",
            "tam_hogar", "n_menores", "n_adultos_mayores"
        ]
        if "region" in df.columns:
            continuous_cols.append("region")

        for col in continuous_cols:
            if col in df.columns:
                series = df[col].copy()
                series.loc[series >= 888] = np.nan
                f_df[col] = series.fillna(series.median()).values
            else:
                f_df[col] = 0.0

        if "cambio_peso" in df.columns:
            f_df["cambio_peso"] = df["cambio_peso"].fillna(3).values
        else:
            f_df["cambio_peso"] = 3.0

        # =====================================================================
        # ADVANCED CLINICAL & EPIDEMIOLOGICAL FEATURE ENGINEERING
        # =====================================================================
        # 1. Genetic Burden Scores (Dosage Effect)
        f_df["genetic_burden_diab"] = (
            2.0 * f_df["ant_padre_diab"] + 
            2.0 * f_df["ant_madre_diab"] + 
            1.5 * f_df["ant_hermano_diab"]
        )
        f_df["both_parents_diab"] = (f_df["ant_padre_diab"] * f_df["ant_madre_diab"]).astype(float)
        
        if "ant_padre_hta" in f_df.columns:
            f_df["fam_hta_burden"] = f_df["ant_padre_hta"] + f_df["ant_madre_hta"] + f_df["ant_hermano_hta"]
        else:
            f_df["fam_hta_burden"] = 0.0

        # 2. Metabolic-Age Interactions (multiplicative biological deterioration)
        age_norm = f_df["edad"] / 50.0
        weight_norm = f_df["peso_habitual"] / 70.0
        f_df["age_x_weight"] = age_norm * weight_norm
        f_df["age_x_hipertension"] = age_norm * f_df["dx_hipertension"]
        f_df["age_x_dyslipidemia"] = age_norm * f_df["dx_colesterol_trigliceridos"]
        f_df["age_x_genetic"] = age_norm * f_df["genetic_burden_diab"]

        # 3. Comorbidity & Adiposity Phenotypes
        f_df["comorbidity_count"] = (
            f_df["dx_obesidad"] + f_df["dx_hipertension"] + f_df["dx_colesterol_trigliceridos"]
        )
        f_df["high_adiposity_silhouette"] = (f_df["silueta_corporal"] >= 6).astype(float)
        f_df["severe_adiposity_silhouette"] = (f_df["silueta_corporal"] >= 7).astype(float)

        # 4. Socioeconomic Deprivation & Environmental Vulnerability Index
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

        # 5. Cumulative Lifestyle Risk
        f_df["lifestyle_risk"] = f_df["fuma_actualmente"] + f_df["consume_alcohol"]

        self.feature_names = list(f_df.columns)
        X_scaled = self.scaler.fit_transform(f_df)
        return X_scaled, y, upm_clusters, sample_weights

    def get_feature_names(self) -> List[str]:
        return self.feature_names
