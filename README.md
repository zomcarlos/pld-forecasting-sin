# Previsão do PLD por Submercado do SIN

Comparação de modelos econométricos e de aprendizado de máquina para previsão diária do Preço de Liquidação das Diferenças (PLD) nos quatro submercados do Sistema Interligado Nacional (SIN) brasileiro.

## Modelos avaliados

| Classe | Modelo |
|--------|--------|
| Econométrico | SARIMAX, TAR |
| Machine Learning | Ridge Regression, Random Forest, LightGBM |
| Deep Learning | LSTM, GRU |

## Variáveis

- **PLD** — Preço de Liquidação das Diferenças (horário → média diária)
- **ENA** — Energia Natural Afluente (MWmed)
- **EAR** — Energia Armazenada (% do reservatório)
- **Carga** — Demanda verificada (MWmed)

Todos os dados são públicos e disponibilizados pelo Operador Nacional do Sistema Elétrico (ONS).

## Estrutura do repositório

```
├── src/                       # Código-fonte dos modelos e utilitários
│   ├── data_loader.py         # Ingestão e pré-processamento dos dados
│   ├── reproducibility.py     # Rastreabilidade de parâmetros e execuções
│   ├── diagnostico_estatistico.py
│   ├── sarimax_diario.py
│   ├── tar_diario.py
│   ├── ridge_diario.py
│   ├── rf_diario.py
│   ├── ml_diario.py           # LightGBM
│   ├── lstm_diario.py
│   ├── gru_diario.py
│   └── run_pipeline.py        # Execução sequencial de todos os modelos
├── output/                    # Resultados experimentais (métricas, gráficos)
├── download_dados_ons.py      # Obtenção dos dados brutos do ONS
└── README.md
```

## Reprodução do experimento

### 1. Obtenção dos dados

```bash
python download_dados_ons.py
```

Os arquivos CSV serão salvos em `dados/`.

### 2. Execução do pipeline

```bash
python src/run_pipeline.py
```

Os resultados serão gerados em `output/`.

## Dependências

- Python ≥ 3.10
- pandas, numpy, matplotlib, statsmodels, scikit-learn, lightgbm, torch

## Referências

Dados: [ONS — Dados Abertos](https://dados.ons.org.br/)
