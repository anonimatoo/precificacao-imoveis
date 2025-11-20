import joblib
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from math import radians, sin, cos, sqrt, atan2

# Funções de geocodificação, distância (como no seu design)
def geocode_address(address):
    geolocator = Nominatim(user_agent="precificacao-imoveis")
    loc = geolocator.geocode(address)
    if loc is None:
        raise ValueError("Endereço não encontrado")
    return loc.latitude, loc.longitude

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def inferir(input_dict):
    model = joblib.load("models/modelo_precificacao.pkl")
    ic = joblib.load("models/ic_params.pkl")
    FEATURES = ic["features"]
    p5 = ic["p5"]
    p95 = ic["p95"]

    lat, lon = geocode_address(input_dict["endereco"])
    dist_centro, dist_metro = haversine_km(lat, lon, -23.561414, -46.655881), haversine_km(lat, lon, -23.561, -46.65)

    x = {
        "area_privativa": input_dict["area_privativa"],
        "num_quartos": input_dict["num_quartos"],
        "num_suites": input_dict["num_suites"],
        "num_vagas": input_dict["num_vagas"],
        "idade_imovel": input_dict["idade_imovel"],
        "estado_conservacao": input_dict["estado_conservacao"],
        "latitude": lat,
        "longitude": lon,
        "dist_centro_comercial": dist_centro,
        "dist_transporte_publico": dist_metro,
        "idh_setor_censitario": input_dict["idh_setor_censitario"],
        "score_seguranca": input_dict["score_seguranca"],
        "lazer_completo": input_dict["lazer_completo"],
        "valor_condominio": input_dict["valor_condominio"],
    }

    X = np.array([[x[f] for f in FEATURES]])
    preco = model.predict(X)[0]

    ic_inf = preco * (1 + p5)
    ic_sup = preco * (1 + p95)

    return {"preco": float(preco), "ic_inf": float(ic_inf), "ic_sup": float(ic_sup)}
