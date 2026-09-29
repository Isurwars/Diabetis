import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from ..config import FeaturesConfig

class FeaturePreprocessor:
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
        
        # Survey weights normalized to mean 1.0 (so loss magnitudes remain standard)
        raw_weights = df["factor_expansion"].fillna(1.0).values.astype(np.float32)
        sample_weights = raw_weights / np.mean(raw_weights)

        # Select feature subsets based on config
        self.feature_names = [
            "edad", "sexo", "nivel_educativo", "estrato_socioeconomico",
            "dx_obesidad", "silueta_corporal", "peso_habitual",
            "cambio_peso", "kg_cambio", "dx_hipertension",
            "dx_colesterol_trigliceridos", "ant_padre_diab",
            "ant_madre_diab", "ant_hermano_diab", "fuma_100_cigarros",
            "fuma_actualmente", "tiene_refri", "tiene_lavadora", "tiene_auto",
            "tam_hogar", "n_menores", "n_adultos_mayores",
            "inseguridad_alim_p1", "recibe_ayuda_alim"
        ]

        X_df = df[self.feature_names].copy()

        # Binary columns: INEGI 1=Yes, 2=No, 9=NS/NR
        binary_cols = [
            "dx_obesidad", "dx_hipertension", "dx_colesterol_trigliceridos",
            "ant_padre_diab", "ant_madre_diab", "ant_hermano_diab",
            "fuma_100_cigarros", "tiene_refri", "tiene_lavadora", "tiene_auto",
            "inseguridad_alim_p1", "recibe_ayuda_alim"
        ]
        for col in binary_cols:
            X_df[col] = (X_df[col] == 1).astype(float)

        # Sex: Male=1, Female=0
        X_df["sexo"] = (X_df["sexo"] == 1).astype(float)

        # Continuous columns: replace refusal codes (>=888) with NaN and impute median
        continuous_cols = [
            "edad", "peso_habitual", "kg_cambio", "silueta_corporal",
            "nivel_educativo", "estrato_socioeconomico",
            "tam_hogar", "n_menores", "n_adultos_mayores"
        ]
        for col in continuous_cols:
            X_df.loc[X_df[col] >= 888, col] = np.nan
            median_val = X_df[col].median()
            X_df[col] = X_df[col].fillna(median_val)

        # Categorical remaining
        X_df["cambio_peso"] = X_df["cambio_peso"].fillna(3)
        X_df["fuma_actualmente"] = (X_df["fuma_actualmente"] == 1).astype(float)

        X_scaled = self.scaler.fit_transform(X_df)
        return X_scaled, y, upm_clusters, sample_weights

    def get_feature_names(self) -> List[str]:
        return self.feature_names
