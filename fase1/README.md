# pos-tech-fiap-ia-dev-fase-1

Tech Challenge da Fase 1 — dois projetos independentes de classificação.

| | Projeto | Pasta | Problema |
|---|---|---|---|
| **1** | PCOS | `src/`, `base_dados/` | Prever síndrome dos ovários policísticos a partir de dados clínicos tabulares |
| **2** | Raio-X de tórax | `case_imagens_raiox/` | Classificar radiografias em `NORMAL` ou `PNEUMONIA` |

Relatório técnico dos dois: [`documentacao/documentacao-fase1-pos-tech.pdf`](documentacao/documentacao-fase1-pos-tech.pdf).

Não há `Dockerfile` — a execução é local, via ambiente virtual.

---

## Ambiente (vale para os dois projetos)

Requer **Python 3.9+**. Validado em Python 3.12.13, macOS arm64.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ **Ative o ambiente — não basta chamar `.venv/bin/python` pelo caminho.** O kernel do
> Jupyter é um processo separado que resolve `python` pelo `PATH`. Sem
> `source .venv/bin/activate`, os notebooks podem rodar em outro Python da máquina, com
> versões de biblioteca diferentes — e os resultados deixam de bater com o relatório.

Conferir a instalação:

```bash
python -c "import pandas, sklearn, torch, cv2; print(pandas.__version__, sklearn.__version__, torch.__version__, cv2.__version__)"
```

> `torch` e `opencv-python` são só do projeto 2 e ocupam ~650 MB em disco. Para rodar
> apenas o projeto 1, instale só as oito primeiras linhas do `requirements.txt`
> (tudo menos `torch` e `opencv-python`).

---

# Projeto 1 — PCOS

## Resumo

Classificar pacientes de uma clínica de fertilidade em `PCOS (Y/N)` = `0` ou `1`, a partir de
colunas clínicas, hormonais e de ultrassom. O caminho é: consolidar os dois arquivos de origem,
tratar os dados, treinar cinco classificadores clássicos e comparar.

Base final com 541 linhas e 49 colunas; split 80/20 estratificado (432 treino / 109 teste).
**Random Forest venceu** — acurácia 0,9174, F1 0,8657, AUC 0,9365.

Os dados de entrada já estão no repositório, então este projeto roda sem nenhum download.

## Arquivos

| Arquivo | O que é |
|---|---|
| `base_dados/PCOS_infertility.csv` | Arquivo de origem 1 (dados de infertilidade) |
| `base_dados/PCOS_data_without_infertility.xlsx` | Arquivo de origem 2 (dados clínicos) |
| `base_dados_tratada/PCOS_unificado.csv` | Saída do ETL: base unificada e limpa (541 × 49) |
| `src/tratamento_dados/etl_base_pcos.ipynb` | **ETL** — 13 verificações de qualidade, EDA, junção dos dois arquivos e 14 correções |
| `src/modelagem/modelagem_base_pcos.ipynb` | **Modelagem** — split, treino dos 5 modelos e comparação, com gráficos |
| `src/modelagem/treinamento.py` | A mesma modelagem em script, sem notebook |
| `src/run_ml.py` | Ponto de entrada do treinamento pelo terminal |
| `reports/` | Métricas geradas pelo script (`.csv`, `.json`, `.txt`) — não versionado |

## Como executar

**Passo 1 — ETL** (gera `base_dados_tratada/PCOS_unificado.csv`):

```bash
jupyter lab src/tratamento_dados/etl_base_pcos.ipynb
```

Execute as células em ordem. ~20 s.

**Passo 2 — modelagem.** Duas opções equivalentes:

```bash
# opção A — notebook, com gráficos e comparação visual
jupyter lab src/modelagem/modelagem_base_pcos.ipynb

# opção B — script, imprime as métricas e salva em reports/
python -m src.run_ml
```

~10 s. O passo 2 depende do arquivo gerado no passo 1.

### Resultados no teste (109 pacientes)

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0,9174** | 0,9355 | 0,8056 | **0,8657** | **0,9365** |
| SVM (RBF) | 0,8807 | 0,8108 | 0,8333 | 0,8219 | 0,9315 |
| Regressão Logística | 0,8532 | 0,7381 | 0,8611 | 0,7949 | 0,9277 |
| Árvore de Decisão | 0,8624 | 0,8000 | 0,7778 | 0,7887 | 0,8809 |
| KNN (k=5) | 0,8624 | 0,8889 | 0,6667 | 0,7619 | 0,9007 |

---

# Projeto 2 — Raio-X de tórax

## Resumo

Classificar radiografias de tórax de pacientes pediátricos em `NORMAL` (0) ou `PNEUMONIA` (1),
com rede convolucional em PyTorch. O caminho é: carregar as imagens, tratar (128×128 em tons de
cinza, normalização, CLAHE, aumento de dados só no treino), separar treino/validação/teste,
treinar **dois** modelos e comparar.

5.232 imagens em `train/`, divididas 80/20 em treino e validação; batch 32, 12 épocas. Usa `mps` em Apple Silicon, senão `cpu`.
~8 min em `mps`.

**A CNN simples venceu a CNN profunda** — a rede mais bem construída no papel (BatchNorm, dropout
progressivo) sofre com o deslocamento de distribuição entre `train/` e `test/` e responde
PNEUMONIA para quase tudo. A escolha foi feita na validação, e o teste confirmou.

| | CNN simples | CNN profunda | linha de base |
|---|---|---|---|
| acurácia balanceada na **validação** (critério de escolha) | **0,9574** | 0,8537 | — |
| acurácia no teste | **0,8478** | 0,6699 | 0,6250 |
| acurácia balanceada no teste | **0,8021** | 0,5598 | 0,5000 |
| AUC no teste | 0,9517 | 0,9418 | 0,5000 |

> Os números variam um pouco entre execuções (`mps`/`cpu` não são determinísticos), mas a ordem
> entre os dois modelos se mantém.

## Arquivos

| Arquivo | O que é |
|---|---|
| `case_imagens_raiox/cnn-raiox-torax.ipynb` | **O notebook do projeto** — todo o pipeline, do carregamento à comparação final |
| `case_imagens_raiox/imagens_exemplos/` | **Amostra do dataset versionada no repositório** — 16 imagens no mesmo layout do original (`train/` e `test/`, cada um com `NORMAL/` e `PNEUMONIA/`). É o que o notebook lê por padrão |
| `case_imagens_raiox/.gitignore` | Mantém o dataset (~1,2 GB) fora do repositório |

## Como executar

> ### ⚠️ As imagens não cabem no Git
>
> O dataset completo tem ~1,2 GB e **não está no repositório**. O que está versionado é uma
> amostra de 16 imagens em `case_imagens_raiox/imagens_exemplos/`, no mesmo layout do dataset
> original.
>
> O notebook lê essa amostra **por padrão**, então ele abre e roda em qualquer máquina, sem
> configurar nada, até o fim do pré-processamento — dá para conferir o formato dos dados, o
> inventário, os exemplos por classe e o CLAHE. **Da célula de treino em diante ele para com
> erro**: o treino usa lotes de 32 com `drop_last=True`, e 16 imagens não formam nenhum lote.
>
> **Para reproduzir os resultados deste README é obrigatório baixar o dataset completo** e definir
> `PASTA_RAIOX` (passo 2). Os gráficos e números já salvos no notebook vêm de uma execução com o
> dataset completo.

**Passo 1 — baixar o dataset.** Baixe o *Chest X-Ray Images (Pneumonia)* no Kaggle e descompacte
em qualquer lugar da máquina:

<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

A pasta descompactada tem esta forma:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
```

O notebook usa `train/` (dividido em treino e validação) e `test/` como holdout. É exatamente
esse layout que `imagens_exemplos/` reproduz em miniatura.

**Passo 2 — apontar o notebook para o dataset.** O caminho vem da variável de ambiente
`PASTA_RAIOX`. Sem ela, o notebook usa a amostra do repositório e avisa isso ao rodar.
Exporte antes de abrir o Jupyter:

```bash
export PASTA_RAIOX="/caminho/para/chest_xray"
jupyter lab case_imagens_raiox/cnn-raiox-torax.ipynb
```

**Passo 3 — executar as células em ordem.** ~8 min em `mps` (Apple Silicon); em `cpu` é bem
mais lento. Não depende do projeto 1.

Todos os caminhos do notebook são relativos ao próprio repositório — não há nenhum caminho fixo
de máquina no código.
