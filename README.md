
# Predição de Chuva com Redes Neurais e Dados de Radar 3D

Este projeto realiza o processamento de dados volumétricos de radar e treina modelos de redes neurais para prever chuva.

## Pipeline de Processamento

1. `data_process.py` — Processa os dados brutos.
2. `data_preprocess.py` — Pré-processa os dados para uso nos modelos.
3. `data_resize.py` — Redimensiona os dados conforme necessário.
4. `model_train.py` — Treina o modelo de rede neural.
5. `model_evaluation.ipynb` — Avalia o desempenho do modelo.

## Estrutura de Pastas

- Dados processados: `output_data`
- Resultados dos modelos: `outputs`

## Requisitos

Consulte `requirements.txt` para instalar as dependências necessárias.
Esse projeto funciona bem com uv.

## Como executar

Execute os scripts na ordem do pipeline para processar os dados e treinar o modelo.
