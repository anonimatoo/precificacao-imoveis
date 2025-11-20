from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import inferir

class Imovel(BaseModel):
    endereco: str
    area_privativa: float
    num_quartos: int
    num_suites: int
    num_vagas: int
    idade_imovel: int
    estado_conservacao: int
    idh_setor_censitario: float
    score_seguranca: float
    lazer_completo: int
    valor_condominio: float

app = FastAPI()

@app.post("/avaliar")
def avaliar(imovel: Imovel):
    result = inferir(imovel.dict())
    return result
