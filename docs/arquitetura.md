# Arquitetura do Motor de Precificação

1. **Interface / Front-end**  
   -> Corretor envia dados do imóvel ou integração API.

2. **Backend / API**  
   -> API FastAPI expõe endpoint `/avaliar`  
   -> Recebe os dados do imóvel + endereço

3. **Pipeline de Inferência**  
   - Geocodificação: transforma endereço em lat/lon  
   - Cálculo de distâncias (centro comercial, transporte)  
   - Montagem do vetor de features  
   - Predição pelo modelo treinado  
   - Cálculo de intervalo de confiança  
   - Retorno do resultado

4. **Explicabilidade (SHAP)**  
   - (Opcional / futuro) calcular shap values e devolver top variáveis

5. **Persistência de modelo**  
   - Modelos treinados salvos em `models/`  
   - Meta de versionamento

6. **Configuração**  
   - Arquivo de configuração `config/config_example.yaml`

