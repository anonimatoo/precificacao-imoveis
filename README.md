# Precificação de Imóveis

Motor de precificação de imóveis usando Machine Learning (LightGBM) + explicabilidade (SHAP).

## Estrutura do Projeto

- `data/`: dados de treino ou template de dados  
- `models/`: modelos treinados e artefatos (pickle, parâmetros)  
- `src/`: código principal de treino, inferência, utilitários  
- `api/`: API REST (FastAPI) para usar o motor de precificação  
- `config/`: arquivos de configuração (ex: chaves, parâmetros)  
- `docs/`: documentação do projeto (fluxo, arquitetura, decisões)  

## Como usar

1. Criar ambiente virtuais (venv)  
2. Instalar dependências: `pip install -r requirements.txt`  
3. Treinar modelo: rodar `src/train_model.py`  
4. Fazer inferência: usar `src/inference.py` ou subir API com `api/app.py`  
5. Documentação: ver `docs/`  

## Licença

Aqui você pode colocar informação de licença, se for open source ou interno.

