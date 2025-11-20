#!/bin/bash

# Cria virtualenv
python3 -m venv venv
source venv/bin/activate

# Atualiza pip
pip install --upgrade pip setuptools wheel

# Instala dependências
pip install -r requirements.txt

echo "Ambiente configurado com sucesso."
