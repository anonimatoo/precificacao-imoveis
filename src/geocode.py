"""
geocode.py

Funções para geocodificação com cache simples em disco.
Suporta:
 - Nominatim (OpenStreetMap) por padrão
 - Google Maps Geocoding se a variável de ambiente GOOGLE_MAPS_API_KEY estiver definida

Uso:
    from src.geocode import geocode_address
    lat, lon = geocode_address("Avenida Paulista, 1000, São Paulo, SP")
"""

import os
import json
import threading
from typing import Tuple, Optional
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# caminho do arquivo de cache local (json)
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "geocode_cache.json")
CACHE_LOCK = threading.Lock()

# Garantir que a pasta data exista
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

def _load_cache() -> dict:
    try:
        with CACHE_LOCK:
            if not os.path.exists(CACHE_PATH):
                return {}
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}

def _save_cache(cache: dict):
    try:
        with CACHE_LOCK:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _geocode_nominatim(address: str, timeout: int = 10) -> Optional[Tuple[float, float]]:
    geolocator = Nominatim(user_agent="precificacao-imoveis")
    try:
        loc = geolocator.geocode(address, timeout=timeout)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except (GeocoderTimedOut, GeocoderServiceError):
        return None
    except Exception:
        return None
    return None

def _geocode_google(address: str, api_key: str, timeout: int = 10) -> Optional[Tuple[float, float]]:
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": address, "key": api_key}
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        j = resp.json()
        if j.get("status") == "OK" and j.get("results"):
            loc = j["results"][0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"])
    except Exception:
        return None
    return None

def geocode_address(address: str, use_cache: bool = True) -> Tuple[float, float]:
    """
    Geocodifica um endereço para (latitude, longitude).

    Estratégia:
      1. Verifica cache local (data/geocode_cache.json)
      2. Se existir GOOGLE_MAPS_API_KEY, tenta Google Maps
      3. Senão, tenta Nominatim (OSM)
      4. Se tudo falhar, lança ValueError

    Nota:
      - Em produção, use cache compartilhado (Redis) e respeite rate limits.
      - Para Nominatim: respeite os termos de uso (User-Agent, throttle).
    """
    if not address or not address.strip():
        raise ValueError("Endereço vazio")

    address_key = address.strip().lower()

    if use_cache:
        cache = _load_cache()
        if address_key in cache:
            lat, lon = cache[address_key]
            return float(lat), float(lon)

    # Tentar Google Maps se chave estiver disponível
    google_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    coords = None
    if google_key:
        coords = _geocode_google(address, google_key)
    if coords is None:
        coords = _geocode_nominatim(address)

    if coords is None:
        raise ValueError(f"Não foi possível geocodificar o endereço: {address}")

    # Salvando no cache local
    if use_cache:
        cache = _load_cache()
        cache[address_key] = [coords[0], coords[1]]
        _save_cache(cache)

    return coords
