# Previsão do PLD por Submercado do SIN

Comparação de modelos econométricos e de aprendizado de máquina para previsão diária do Preço de Liquidação das Diferenças (PLD) nos quatro submercados do Sistema Interligado Nacional (SIN) brasileiro.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

## Modelos avaliados

| Classe | Modelo |
|--------|--------|
| Econométrico | SARIMAX, TAR |
| Machine Learning | Ridge Regression, Random Forest, LightGBM |
| Deep Learning | LSTM, GRU |

## Sementes de Aleatoriedade

A reprodutibilidade do experimento é garantida através da fixação de sementes aleatórias (seed = 42) nos parâmetros de todos os algoritmos estocásticos empregados (Random Forest, LightGBM, inicialização de pesos LSTM/GRU, divisões de base).

## Variáveis

Os dados ingeridos abrangem o intervalo de 2021 até 2026. A variável PLD é transformada para média diária, mantendo consistência com as grandezas diárias de demanda e suprimento.

- **PLD** — Preço de Liquidação das Diferenças (horário → média diária)
- **ENA** — Energia Natural Afluente (MWmed)
- **EAR** — Energia Armazenada (% do reservatório)
- **Carga** — Demanda de energia verificada (MWmed)

Todos os dados são de fonte pública, obtidos da plataforma de Dados Abertos do Operador Nacional do Sistema Elétrico (ONS).

## Estrutura do repositório

```
├── src/                       # Código-fonte
│   ├── data_loader.py         # Ingestão e engenharia de features
│   ├── run_pipeline.py        # Pipeline de execução sequencial
│   ├── diagnostico_*.py       # Análises estatísticas estruturais e de resíduos
│   ├── *_diario.py            # Scripts de ajuste para os 7 modelos
│   ├── metricas_por_fold.py   # Validação walk-forward detalhada
│   ├── diebold_mariano.py     # Testes de significância de diferença
│   ├── intervalos_previsao.py # Geração de ICs (95%)
│   └── analise_sensibilidade.py
├── output/                    # Saídas do pipeline
│   ├── predictions/           # CSVs com (data, real, previsto) por fold
│   ├── diagnosticos/          # Resultados de ARCH, Ljung-Box e QQ-plots
│   ├── reprodutibilidade/     # Logs de parâmetros (JSON) e sumários (Markdown)
│   └── *.png, *.csv           # Métricas agregadas e visualizações
├── download_dados_ons.py      # Script utilitário para download dos CSVs
├── requirements.txt           # Configuração de dependências
├── LICENSE                    # Termos de uso (MIT)
└── README.md
```

## Reprodução do experimento

### 1. Obtenção dos dados e preparação

O experimento requer os arquivos brutos do ONS, que não estão diretamente no repositório.

```bash
pip install -r requirements.txt
python download_dados_ons.py
```

Os arquivos CSV históricos serão estruturados em um diretório `dados/` na raiz do projeto.

### 2. Execução do pipeline

O pipeline central orquestra todo o processo analítico:

```bash
python src/run_pipeline.py
```

Isto executará os diagnósticos iniciais, seguido do treinamento dos 7 modelos avaliados, salvando todas as previsões localmente e disparando a suíte de análises estatísticas post-hoc. O runtime esperado é de aproximadamente 15 minutos em um ambiente convencional sem GPU.

## Dependências

Ver `requirements.txt` para versões exatas das bibliotecas (compatíveis com Python ≥ 3.10).
Bibliotecas principais empregadas no projeto:
- Pandas, Numpy e Scipy
- Statsmodels e pmdarima
- Scikit-Learn e LightGBM
- PyTorch
- Matplotlib

## Referências

Dados: [ONS — Dados Abertos](https://dados.ons.org.br/)
