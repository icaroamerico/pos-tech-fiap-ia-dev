# POS TECH FIAP — IA para Devs · Tech Challenges

**Aluno:** Icaro Minervini Americo — RM376755 · **Turma:** 10IADT

Repositório único com as duas fases do Tech Challenge. Cada fase é **autossuficiente**: tem seu
próprio `requirements.txt`, seu próprio ambiente virtual e roda sem depender da outra.

| Fase | Tema | Entrega |
|---|---|---|
| [**`fase1/`**](fase1/) | Diagnóstico de SOP (PCOS) com modelos clássicos + CNN de raio-X de tórax | Pipeline de ETL, 5 classificadores, 2 CNNs, documentação em PDF |
| [**`fase2/`**](fase2/) | **Otimização** desses modelos por algoritmo genético + camada de LLM | AG do zero, 22 execuções gravadas, 221 testes, relatório técnico |

**A ligação entre as duas é de dados e de baseline, não de código.** A Fase 2 copiou
`treinamento.py` e `run_ml.py` da Fase 1 **byte a byte** e nunca os editou — é o que permite
afirmar que a diferença entre "antes" e "depois" vem do algoritmo genético, e não de uma
refatoração pelo caminho. A justificativa está no
[ADR 009](fase2/documentacao/decisoes/009-scripts-da-fase-1-intocados.md).

---

## Por que as estruturas internas não foram renomeadas

Era tentador padronizar as duas fases numa única nomenclatura de pastas. **Não fiz, de
propósito:** tanto a Fase 1 quanto a Fase 2 resolvem caminhos como
`Path(__file__).parent.parent.parent`, ou seja, a raiz do projeto é a pasta que contém `src/`.
Aninhar cada repositório sob `fase1/` e `fase2/` mantém todo o código executável sem uma única
alteração. Renomear `base_dados/` para algo como `dados/entrada/` quebraria `treinamento.py` —
exatamente o arquivo que o ADR 009 proíbe editar e de que a Fase 2 depende para reproduzir o
baseline.

**O preço disso:** as duas fases nomeiam as coisas de forma um pouco diferente
(`base_dados/` × `base_dados_tratada/`, `documentacao/` × `docs/`). A tabela abaixo é o mapa.

---

## Mapa: onde está cada tipo de arquivo

| Tipo | Fase 1 | Fase 2 |
|---|---|---|
| **Entrada — dados brutos** | `fase1/base_dados/` (`.xlsx` + `.csv` originais do Kaggle) | — (herda a base tratada) |
| **Entrada — dados tratados** | `fase1/base_dados_tratada/PCOS_unificado.csv` (541 × 49) | `fase2/base_dados_tratada/PCOS_unificado.csv` (mesma cópia) |
| **Entrada — imagens** | `fase1/case_imagens_raiox/imagens_exemplos/` (amostra de 16; o dataset de ~1,2 GB **não** é versionado) | — (CNN fora do escopo) |
| **Código** | `fase1/src/` | `fase2/src/` (17 módulos) |
| **Notebooks** | `fase1/src/tratamento_dados/`, `fase1/src/modelagem/`, `fase1/case_imagens_raiox/` | `fase2/notebooks/` (3, já executados) |
| **Testes** | — (a Fase 1 não tinha) | `fase2/tests/` (**221 testes**) |
| **Prompts de LLM** | — | `fase2/prompts/` (3 templates versionados) |
| **Documentação** | `fase1/documentacao/` (PDF entregue) | `fase2/documentacao/` (relatório técnico, arquitetura, **9 ADRs**) |
| **Saída — métricas** | `fase1/reports/metricas_modelos.{csv,json,txt}` | `fase2/arquivos_saida/arquivos_analiticos/metricas_modelos_baseline/metricas_modelos.{json,txt}` |
| **Saída — execuções do AG** | — | `fase2/arquivos_saida/arquivos_analiticos/historico_execucoes_ag/` (22 execuções: `config`, `history.jsonl`, `best_individual`, `population_final`, `resumo`, `metrics_holdout`) |
| **Saída — logs** | — | `fase2/logs/` |
| **Saída — avaliação da LLM** | — | `fase2/avaliacao_llm/` (45 saídas em JSON, Markdown e CSV) |

> Tudo sob `fase2/arquivos_saida/` — métricas baseline, histórico das execuções do AG, logs e
> `avaliacao_llm/` — **está versionado de propósito.** São os
> artefatos que sustentam cada número do relatório técnico, e são o que faz os notebooks abrirem
> legíveis no GitHub sem reexecutar nada.

---

## Como rodar

Cada fase tem seu próprio ambiente. **Não compartilhe a venv entre elas** — a Fase 1 precisa de
`torch` e `opencv-python` para a CNN, a Fase 2 não os tem.

### Fase 1

```bash
cd fase1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_ml                    # treina os 5 modelos, grava reports/
jupyter lab                             # notebooks de ETL, modelagem e CNN
```

### Fase 2

#### ⭐ Rápido: Notebook Orquestrador (recomendado)

```bash
cd fase2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

jupyter lab notebooks/00_execucao_completa.ipynb
# Execute: Cell → Run All (tudo acontece em um lugar)
```

**Tempo:** ~15 min (simulado) ou ~3h (Ollama real)  
**O que faz:** baseline + AG + análise + LLM com barra de progresso

#### Manual: Comandos individuais

```bash
# Reproduz o baseline da Fase 1
python -m src.run_ml                                     # ~10 s

# Testes (opcional)
python -m pytest tests/ -q                               # ~2 min

# Algoritmo Genético
python -m src.experimentos.rodar_ag --experimento todos  # E1,E2,E3,D × 3 sementes (~3 min)

# Notebooks para exploração
jupyter lab notebooks/
  # 00_execucao_completa.ipynb    ← Orquestrador (executa tudo)
  # 01_ag_hiperparametros_pcos.ipynb
  # 02_experimentos_ag_comparativo.ipynb  ← 5 gráficos comparativos
  # 03_llm_interpretacao.ipynb
```

> ⚠️ **Ative o ambiente — não basta chamar `.venv/bin/python` pelo caminho.** O kernel do Jupyter
> é um processo separado que resolve `python` pelo `PATH`. Sem `source .venv/bin/activate`, os
> notebooks podem rodar em outro Python da máquina e os `import src.*` falham.

---

## Os resultados, em uma tabela

Conjunto de teste de 109 pacientes (36 positivas), tocado **uma única vez** em cada fase.

| | Melhor da Fase 1 (Random Forest) | Melhor da Fase 2 (Regressão Logística, via AG) | Δ |
|---|---|---|---|
| Acurácia | 0,9174 | 0,9174 | 0,0000 |
| Precisão | **0,9355** | 0,8462 | −0,0893 |
| **Recall** | 0,8056 | **0,9167** | **+0,1111** |
| F1 | 0,8657 | **0,8800** | +0,0143 |
| AUC | 0,9365 | **0,9532** | +0,0167 |
| **Score composto** | 0,8618 | **0,9056** | **+0,0438** |

Das 36 pacientes positivas, a Fase 1 identificava 29 e o modelo da Fase 2 identifica **33** —
**4 diagnósticos a mais**, ao custo de 4 exames a mais, com a acurácia global inalterada. O
ganho está concentrado em recall porque **foi para lá que a função de fitness foi apontada**: em
diagnóstico, falso negativo e falso positivo não custam a mesma coisa.

> 🔴 **E a ressalva que dá sentido ao número:** o algoritmo genético **empatou com busca
> aleatória** de mesmo número de candidatos (0,8765 contra 0,8759 em validação cruzada). O ganho
> acima é real, mas **não é atribuível ao algoritmo genético em particular** — qualquer busca
> com esse orçamento teria chegado perto. O AG se distingue por **custo**: empatou gastando ~28%
> menos treinos de modelo. Essa comparação está no
> [relatório técnico §4.2](fase2/documentacao/relatorio-tecnico.md), e omiti-la faria o trabalho parecer
> melhor e valer menos.

---

## Documentação

| Documento | O que é |
|---|---|
| [`fase1/README.md`](fase1/README.md) | Como rodar a Fase 1, decisões de ETL e modelagem |
| [`fase1/documentacao/`](fase1/documentacao/) | PDF entregue da Fase 1 |
| [`fase2/README.md`](fase2/README.md) | Rastreabilidade requisito → código, como rodar, resultados |
| [**`fase2/documentacao/relatorio-tecnico.md`**](fase2/documentacao/relatorio-tecnico.md) | **Relatório técnico da Fase 2** — 9 seções, todas as tabelas geradas a partir de `arquivos_saida/arquivos_analiticos/historico_execucoes_ag/` |
| [`fase2/documentacao/arquitetura.md`](fase2/documentacao/arquitetura.md) | Diagramas Mermaid de componentes e de sequência |
| [`fase2/documentacao/decisoes/`](fase2/documentacao/decisoes/) | 9 ADRs, no formato contexto → decisão → consequências |

---

## A regra que organiza a Fase 2

> 🔴 **O fitness nunca toca o conjunto de teste.**

Não é convenção — é estrutura. O `dataclass DadosTreino` que o AG recebe **não tem nenhum
campo** `X_teste`/`y_teste`; não existe atributo para um bug futuro acessar. O único módulo que
pode ler o holdout (`fase2/src/fitness/avaliacao_final.py`) não é importado por nenhum código do
AG. Verificado por dois testes que substituem o conjunto de teste por um objeto que lança
exceção em qualquer acesso — atributo, `len()`, indexação, iteração —, um para uma chamada
isolada de fitness e outro para o laço evolutivo completo.
