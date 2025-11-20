import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
import joblib

# Defina colunas, etc.
FEATURES = [
    "area_privativa", "num_quartos", "num_suites", "num_vagas",
    "idade_imovel", "estado_conservacao", "latitude", "longitude",
    "dist_centro_comercial", "dist_transporte_publico",
    "idh_setor_censitario", "score_seguranca",
    "lazer_completo", "valor_condominio"
]
TARGET = "preco_venda"

def main():
    df = pd.read_csv("data/dados_imoveis.csv")
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = [
        "area_privativa", "num_quartos", "num_suites", "num_vagas",
        "idade_imovel", "latitude", "longitude",
        "dist_centro_comercial", "dist_transporte_publico",
        "idh_setor_censitario", "score_seguranca",
        "valor_condominio"
    ]
    ordinal_features = ["estado_conservacao"]
    binary_features = ["lazer_completo"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("ord", ordinal_transformer, ordinal_features),
        ("bin", binary_transformer, binary_features),
    ])

    reg = LGBMRegressor(
        n_estimators=800, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", reg)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    erro_rel = (y_val - y_pred) / y_val
    p5 = np.percentile(erro_rel, 5)
    p95 = np.percentile(erro_rel, 95)

    joblib.dump(pipeline, "models/modelo_precificacao.pkl")
    joblib.dump({"p5": float(p5), "p95": float(p95), "features": FEATURES}, "models/ic_params.pkl")

    print("Treinamento concluído.")
    print("IC params:", p5, p95)

if __name__ == "__main__":
    main()
