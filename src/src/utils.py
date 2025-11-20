"""
utils.py

Funções utilitárias:
 - haversine_km: distância entre duas coordenadas
 - load_model_and_ic: carrega modelo e parâmetros IC (pickle)
 - build_feature_array: monta array numpy na ordem de FEATURES esperada
 - safe_get_feature_names: extrai nomes transformados do ColumnTransformer (quando disponível)
"""

import os
import joblib
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import Tuple, List, Any

# Lista de features padrão (deve ser mantida sincronizada com o pipeline)
FEATURES = [
    "area_privativa",
    "num_quartos",
    "num_suites",
    "num_vagas",
    "idade_imovel",
    "estado_conservacao",
    "latitude",
    "longitude",
    "dist_centro_comercial",
    "dist_transporte_publico",
    "idh_setor_censitario",
    "score_seguranca",
    "lazer_completo",
    "valor_condominio"
]

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula distância (km) entre dois pontos usando fórmula haversine.
    """
    R = 6371.0  # raio da Terra em km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def load_model_and_ic(model_path: str = None, ic_path: str = None):
    """
    Carrega o pipeline/modelo e os parâmetros de intervalo de confiança (ic_params).
    Retorna (model, ic_params_dict)
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_precificacao.pkl")
    if ic_path is None:
        ic_path = os.path.join(os.path.dirname(__file__), "..", "models", "ic_params.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
    if not os.path.exists(ic_path):
        raise FileNotFoundError(f"IC params não encontrado em: {ic_path}")

    model = joblib.load(model_path)
    ic_params = joblib.load(ic_path)
    return model, ic_params

def build_feature_array(input_dict: dict, features: List[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Monta o array numpy no formato (1, n_features) na ordem `features`.
    Também retorna um DataFrame pandas de uma linha com as colunas originais (útil para transformação via pipeline).
    Exemplo de uso:
        X_array, X_df = build_feature_array(dados_imovel, FEATURES)
    """
    if features is None:
        features = FEATURES

    # Assegurar que todas as features existam no dict; se faltar, usar None
    row = {f: input_dict.get(f, None) for f in features}
    df = pd.DataFrame([row], columns=features)
    arr = df.values.astype(object)  # deixamos object para permitir None; o preprocessor tratará
    return arr, df

def safe_get_feature_names(preprocessor) -> List[str]:
    """
    Tenta extrair nomes de features transformadas a partir do ColumnTransformer.
    Se não for possível, retorna a lista FEATURES original.
    """
    try:
        # sklearn >= 1.0
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        # fallback: retornar FEATURES (ordem original)
        return FEATURES

# Exemplo rápido (para debug)
if __name__ == "__main__":
    print("Utils carregado. FEATURES:", FEATURES)
