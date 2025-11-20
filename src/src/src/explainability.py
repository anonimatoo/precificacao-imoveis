"""
explainability.py

Funções para calcular explicabilidade local com SHAP e retornar top N features.

Uso:
    from src.explainability import get_shap_top_features
    top = get_shap_top_features(model, preprocessor, X_df, top_n=3)
"""

import numpy as np
import pandas as pd
import shap
from typing import List, Dict, Any
from src.utils import safe_get_feature_names

def get_shap_top_features(model, preprocessor, X_df: pd.DataFrame, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Calcula valores SHAP (local) para a linha X_df (1 amostra) e retorna as top_n features
    com maior impacto absoluto.

    Parâmetros:
      - model: o regressor (por exemplo, LGBM dentro de Pipeline: model.named_steps["regressor"])
      - preprocessor: o ColumnTransformer/step de preprocess (por exemplo, model.named_steps["preprocess"])
      - X_df: DataFrame com 1 linha contendo as features originais (antes do preprocess)
      - top_n: quantas features retornar

    Retorno:
      Lista de dicionários: [{"feature": nome, "impacto_reais": valor_shap}, ...]
      O valor de impacto está na mesma unidade da predição (ex: R$).
    """
    # 1. Transformar a amostra
    try:
        X_transformed = preprocessor.transform(X_df)
    except Exception as e:
        # Em alguns pipelines complexos, transform pode exigir ajustes. Re-raise com mensagem clara.
        raise RuntimeError(f"Erro ao transformar X_df: {e}")

    # 2. Extrair o regressor real (se pipeline)
    regressor = model
    # Se passou a pipeline inteira, tentar extrair 'regressor'
    if hasattr(model, "named_steps"):
        regressor = model.named_steps.get("regressor", model)

    # 3. Criar explainer - TreeExplainer é rápido para modelos de árvore
    try:
        explainer = shap.TreeExplainer(regressor)
    except Exception:
        # fallback genérico
        explainer = shap.Explainer(regressor)

    # 4. Obter shap values
    shap_values = explainer.shap_values(X_transformed)
    # shap_values pode ser array ou lista (multiclasse). Normalizar para array 2D.
    if isinstance(shap_values, list):
        # regressão normalmente retorna 2D, mas se for lista pegue primeiro
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    # Garantir que estamos tratando 1 amostra
    if shap_vals.ndim == 2:
        shap_for_row = shap_vals[0]
    elif shap_vals.ndim == 1:
        shap_for_row = shap_vals
    else:
        raise RuntimeError("Formato inesperado de shap_values")

    # 5. Nome das features transformadas (quando aplicável)
    try:
        feature_names = safe_get_feature_names(preprocessor)
    except Exception:
        feature_names = list(X_df.columns)

    # Em alguns casos a transformação pode expandir features (onehot) e safe_get_feature_names cuidará disso.
    # Ajustamos o tamanho: se mismatch, usamos índices simples com fallback para nomes originais.
    if len(feature_names) != len(shap_for_row):
        # fallback: criar nomes genéricos mantendo colunas originais quando possível
        fallback_names = []
        for i in range(len(shap_for_row)):
            if i < len(feature_names):
                fallback_names.append(feature_names[i])
            else:
                fallback_names.append(f"f_{i}")
        feature_names = fallback_names

    # 6. Top N por impacto absoluto
    abs_vals = np.abs(shap_for_row)
    top_idx = np.argsort(abs_vals)[-top_n:][::-1]  # índices das top features

    top_features = []
    for idx in top_idx:
        nome = feature_names[idx] if idx < len(feature_names) else f"f_{idx}"
        impacto = float(shap_for_row[idx])
        top_features.append({"feature": nome, "impacto_reais": impacto})

    return top_features
