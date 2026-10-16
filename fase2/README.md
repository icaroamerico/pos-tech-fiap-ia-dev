# Tech Challenge Fase 2 — Projeto 1: Otimização de Modelos de Diagnóstico

**Aluno:** Icaro Minervini Americo — RM376755 · **Turma:** 10IADT · POS TECH FIAP, *IA para Devs*

Algoritmo genético para otimizar os hiperparâmetros dos modelos de diagnóstico de SOP (PCOS)
desenvolvidos na Fase 1, mais uma camada de LLM para traduzir as saídas numéricas em
interpretações para profissionais de saúde.

**Base:** [`pos-tech-fiap-ia-dev-fase-1`](../pos-tech-fiap-ia-dev-fase-1) — os scripts do PCOS
foram copiados **sem alteração** (ver [ADR 009](docs/decisoes/009-scripts-da-fase-1-intocados.md)).

---

## Rastreabilidade: requisito do enunciado → onde está

| # | Requisito | Onde |
|---|---|---|
| **1** | AG para otimização de hiperparâmetros dos modelos do Módulo 1 | `src/ag/`, [NB 01](notebooks/01_ag_hiperparametros_pcos.ipynb) |
| 1a | Codificação adequada (representação de genes) | `src/ag/cromossomo.py` · NB 01 §3 · [ADR 002](docs/decisoes/002-codificacao-hibrida.md) |
| 1b | Operadores de seleção, cruzamento e mutação | `src/ag/operadores.py` · NB 01 §6 · [ADR 005](docs/decisoes/005-torneio-em-vez-de-roleta.md) |
| 1c | Função fitness baseada em métricas (accuracy, recall, F1…) | `src/fitness/fitness_pcos.py` · NB 01 §4 · [ADR 003](docs/decisoes/003-fitness-composto-cv-no-treino.md) |
| 1d | **Comparar otimizado vs. original** | [NB 02 §6](notebooks/02_experimentos_ag_comparativo.ipynb) |
| 1e | **≥3 experimentos com configurações diferentes** | NB 02 §2–3 (E1/E2/E3 × 3 sementes) |
| **2** | Escalabilidade para variação de demanda | [`docs/arquitetura.md` §5](docs/arquitetura.md) |
| 2a | Monitoramento e logging de desempenho | `src/monitoramento/registro.py` · `runs/` · `logs/` |
| 2b | Documentar arquitetura e decisões | [`docs/arquitetura.md`](docs/arquitetura.md) · [9 ADRs](docs/decisoes/) |
| **3** | Integrar LLM pré-treinada (GPT/Falcon/LLaMA) | `src/llm/` · [NB 03](notebooks/03_llm_interpretacao.ipynb) |
| 3a | Explicações em linguagem natural dos diagnósticos | NB 03 §6 |
| 3b | Dados numéricos → insights acionáveis | NB 03 §8 |
| 3c | Base para dados textuais no Módulo 3 | NB 03 §10 (interface fina + `prompts/`) |
| 3d | Técnicas de prompt engineering | `prompts/*.md` · NB 03 §4 — ✅ 3 templates **comparados com LLM real** ([relatório §5.5](docs/relatorio-tecnico.md)) |
| 3e | **Avaliar a qualidade das interpretações** | NB 03 §7 · `avaliacao_llm/` — ⚠️ métricas automáticas medidas; 3 critérios qualitativos da rubrica pendentes ([relatório §5.5](docs/relatorio-tecnico.md)) |
| 4a | Projeto Python estruturado com ambiente virtual | `venv` + `requirements.txt` |
| 4b | Documentação detalhada com diagramas de arquitetura | [`docs/arquitetura.md`](docs/arquitetura.md) (Mermaid) |
| 4c | **Testes automatizados** | `tests/` — **215 testes** |
| 4d | IaC (só se nuvem) | **fora de escopo** — nuvem é opcional no enunciado |

---

## Ambiente

Requer **Python 3.9+**. Validado em Python 3.12.13, macOS arm64.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ **Ative o ambiente — não basta chamar `.venv/bin/python` pelo caminho.** O kernel do Jupyter
> é um processo separado que resolve `python` pelo `PATH`. Sem `source .venv/bin/activate`, os
> notebooks podem rodar em outro Python da máquina e os `import src.*` falham.

Conferir a instalação:

```bash
python -c "import pandas, sklearn, joblib; print(pandas.__version__, sklearn.__version__)"
```

Não há `torch` nem `opencv-python` — a CNN de raio-X está **fora do escopo** desta fase
([ADR 001](docs/decisoes/001-escopo-restrito-ao-pcos.md)).

---

## Como executar

### Passo 1 — reproduzir o baseline da Fase 1

```bash
python -m src.run_ml
```

~10 s. Executa o script **copiado intacto** da Fase 1 e grava `reports/`. Os cinco modelos devem
reproduzir exatamente a tabela abaixo. **Faça isso antes de qualquer outra coisa:** sem baseline
verificado, todo ganho reportado depois é ficção.

### Passo 2 — rodar os experimentos

```bash
python -m src.experimentos.rodar_ag --experimento todos   # E1, E2, E3, D × 3 sementes (~3 min)
python -m src.experimentos.rodar_ag --etapa-a             # AG por família, 5 execuções (~2 min)
python -m src.experimentos.rodar_ag --etapa-c             # baseline de busca aleatória (~1,5 min)
```

Cada execução grava em `runs/<experimento>_seed<n>/`: `config.json` (parâmetros + versões de
biblioteca), `history.jsonl` (18 métricas por geração), `best_individual.json`,
`population_final.json`, `resumo.json` e `metrics_holdout.json`.

### Passo 3 — a análise

```bash
jupyter lab
```

| Notebook | O que contém |
|---|---|
| [`01_ag_hiperparametros_pcos.ipynb`](notebooks/01_ag_hiperparametros_pcos.ipynb) | O AG: codificação, fitness, hotstart, operadores, uma execução narrada |
| [`02_experimentos_ag_comparativo.ipynb`](notebooks/02_experimentos_ag_comparativo.ipynb) | Os 3 experimentos, ablação torneio × roleta, original vs. otimizado, AG vs. busca aleatória |
| [`03_llm_interpretacao.ipynb`](notebooks/03_llm_interpretacao.ipynb) | Camada de LLM: 3 templates, guardrails, rubrica de avaliação, ética |

Os notebooks **leem os artefatos de `runs/`** em vez de reexecutar o AG — a análise reproduz em
segundos. Já estão executados, com saídas e gráficos salvos, então abrem e são legíveis no GitHub
sem rodar nada.

### Passo 4 — testes

```bash
python -m pytest tests/ -q        # 215 testes, ~3 min
python -m pytest tests/ -q -k "not motor"   # ~15 s, sem as execuções de AG reais
```

### LLM com modelo real

**O Notebook 03 já está executado com LLM real** — `PROVEDOR = "ollama"`, modelo
`llama3.2:3b`, 45 saídas em `avaliacao_llm/`. Para reproduzir:

```bash
brew install ollama && brew services start ollama
ollama pull llama3.2:3b   # 3B e não 8B: esta máquina tem 8 GB de RAM (relatório §5.5)
```

e reexecutar o Notebook 03 (~2 h: 50 chamadas a ~29 s cada).

Para abrir o notebook em qualquer máquina sem Ollama, troque `PROVEDOR` por `"simulado"` na
célula 6 — determinístico, sem rede. **Mas atenção:** o `simulado` ignora o template e produz
saídas idênticas nos três, então ele serve de *controle* do verificador, não de comparação de
prompts. A execução de controle está preservada em `avaliacao_llm_simulado/`.

#### O que a execução real mediu

| Template | Alucinação | Aderência | Termos proibidos | Números citados | Caracteres |
|---|---|---|---|---|---|
| V1 — zero-shot | **0,0%** | **100%** | 0 | 10,5 | 1.804 |
| V2 — few-shot | **0,0%** | **100%** | 0 | **15,9** | 1.432 |
| V3 — few-shot estruturado | **0,0%** | **100%** | 0 | 13,0 | **1.333** |

**0% de alucinação com um modelo de 3B é resultado da arquitetura, não do modelo:** os números
são pré-escritos no prompt e o modelo só os organiza em prosa.

**O few-shot troca ancoragem por calibração.** V2 cita 51% mais números em 20% menos texto — mas
nenhum template passa de **67%** na regra de sinalizar baixa confiança só entre 40% e 60%, e o
few-shot *piora* esse número (60%): o modelo copiou a expressão do exemplar sem a condição que a
justifica, chamando uma probabilidade de 3% de "caso de baixa confiança". Detalhes no
[relatório §5.5](docs/relatorio-tecnico.md).

---

## Resultados

### Baseline da Fase 1 (conjunto de teste, 109 pacientes)

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0,9174** | 0,9355 | 0,8056 | **0,8657** | **0,9365** |
| SVM (RBF) | 0,8807 | 0,8108 | 0,8333 | 0,8219 | 0,9315 |
| Regressão Logística | 0,8532 | 0,7381 | 0,8611 | 0,7949 | 0,9277 |
| Árvore de Decisão | 0,8624 | 0,8000 | 0,7778 | 0,7887 | 0,8809 |
| KNN (k=5) | 0,8624 | 0,8889 | 0,6667 | 0,7619 | 0,9007 |

Reproduzido integralmente, até a quarta casa decimal.

### Os 3 experimentos + a ablação (média de 3 sementes)

| Experimento | População | Mutação | Fitness | Gerações rodadas | Tempo | Cache |
|---|---|---|---|---|---|---|
| E1 — baseline | 30 | 0,15 | 0,8765 ± 0,0039 | 12,0 | 32 s | 14% |
| **E2 — exploração** | 60 | 0,35 | **0,8804 ± 0,0012** | 11,3 | 78 s | 5% |
| E3 — aproveitamento | 15 | 0,05 | 0,8715 ± 0,0032 | 12,0 | **19 s** | **61%** |
| D — roleta (ablação) | 30 | 0,15 | 0,8746 ± 0,0037 | **6,7** | 20 s | 15% |

> ⚠️ **Diversidade só é comparável em geração casada.** Os experimentos param em gerações
> diferentes (E1 rodou 6/15/15; D rodou 6/8/6), então "entropia final" compara pontos
> distintos da mesma trajetória. Na **geração 5** — a última presente em todas as 12
> execuções:

| Em geração 5 | Entropia (família) | Soluções únicas | Gap melhor − média | Freq. `balanced` |
|---|---|---|---|---|
| E1 — baseline | 1,016 | 29,0 / 30 | 0,0678 | 0,606 |
| E2 — exploração | 1,147 | **60,0 / 60** | 0,0749 | 0,512 |
| **E3 — aproveitamento** | **0,000** | **6,0 / 15** | **0,0152** | **0,956** |
| D — roleta | **1,509** | 28,3 / 30 | 0,0632 | 0,473 |

### 🔴 O resultado principal: o AG empata com busca aleatória

| Método | Fitness CV | Score no holdout | **Treinos de modelo reais** |
|---|---|---|---|
| **AG (E1)** | **0,8765 ± 0,0039** | 0,8897 ± 0,0133 | **309** |
| **Busca aleatória** (mesmos 450 candidatos) | 0,8759 ± 0,0015 | **0,8945 ± 0,0152** | 428 |

Diferença de **0,0006** em CV — um sexto do desvio padrão. No holdout, a busca aleatória fica
ligeiramente à frente. Na semente 42, as duas chegaram ao **mesmo fitness** (0,8745), à
**mesma família** (Regressão Logística) e ao **mesmo score de holdout** (0,8731).

**O orçamento é igual em candidatos amostrados, não em treinos executados** — e a diferença
favorece o AG: por parada antecipada e cache de fenótipo, ele empatou gastando **~28% menos
ajustes de modelo** (na semente 42, 158 contra 430). Em qualidade de solução, empate; em
custo para chegar lá, vantagem do AG.

**A conclusão: neste espaço de busca, o AG não paga seu overhead.** Três razões estruturais,
verificáveis nos dados: dimensão modesta (3–7 genes por família), hotstart forte (a Fase 1 não
usava defaults arbitrários) e paisagem plana perto do ótimo. Isso era previsto como cenário de
probabilidade média-alta no plano de implementação, e a literatura de otimização de
hiperparâmetros documenta busca aleatória como baseline difícil de superar em baixa dimensão.

**Omitir essa comparação faria o trabalho parecer melhor e valer menos.** O requisito avaliado é
implementar e analisar o AG corretamente — e é a Etapa C que transforma "o AG deu um número" em
"sei o que esse número significa".

### Original vs. otimizado — o padrão: recall sobe, F1 cai

Recall melhorou em **4 de 5** modelos (até +0,1389 na árvore de decisão); F1 caiu em 4 de 5.
Não é contradição — é o trade-off precisão/recall aparecendo onde deveria, dado que o fitness tem
peso explícito em recall. Na métrica que o AG realmente otimizou (o score composto), houve melhora
em **3 de 5**, com ganho médio de **+0,009**.

### Três previsões, dois resultados contra mim

| Previsão declarada antes de rodar | Resultado |
|---|---|
| E3 colapsaria a diversidade | ✅ confirmada — entropia 0,000 nas 3 sementes, 6/15 fenótipos, cache 61% |
| E2 convergiria **mais tarde** que E1 | ❌ convergiu **mais cedo** (população maior já nasce com mais bons sorteios); a previsão só vale no tempo de parede, que foi 2,4× |
| Roleta deixaria a média "colada no melhor" | ⚠️ tese certa, **observável invertida** — "colada no melhor" descreve pressão *forte*; pressão fraca dá gap *maior* e entropia *mais alta*, que é o que os dados mostram |
| Alelo `class_weight='balanced'` se fixaria | ⚠️ **depende da pressão seletiva** — sobe com torneio `k=4` (0,66 → **0,95**, nas 3 sementes) e com `k=2` (0,67 → 0,70); **cai** com roleta (0,67 → 0,44) e com mutação 0,35 (0,61 → 0,54) |

As previsões estavam versionadas em `src/experimentos/configuracoes.py` antes da primeira
execução. Ajustá-las depois transformaria os experimentos em confirmação decorativa.

O alelo `class_weight` acabou virando um **medidor independente de pressão seletiva**, e com
isso uma segunda observável para a ablação torneio × roleta — ver
[relatório técnico §3.5](docs/relatorio-tecnico.md).

---

## Decisões de projeto

Nove ADRs em [`docs/decisoes/`](docs/decisoes/), no formato contexto → decisão → consequências:

| ADR | Decisão |
|---|---|
| [001](docs/decisoes/001-escopo-restrito-ao-pcos.md) | Escopo restrito ao PCOS; CNN fora da Fase 2 |
| [002](docs/decisoes/002-codificacao-hibrida.md) | Codificação híbrida com gene de família e genes latentes |
| [003](docs/decisoes/003-fitness-composto-cv-no-treino.md) | Fitness composto por CV no treino; holdout intocado |
| [004](docs/decisoes/004-ag-do-zero.md) | AG do zero, sem DEAP/PyGAD |
| [005](docs/decisoes/005-torneio-em-vez-de-roleta.md) | Torneio em vez de roleta — desvio do material, justificado |
| [006](docs/decisoes/006-mutacao-normalizada-e-familia-separada.md) | Mutação normalizada; troca de família desacoplada |
| [007](docs/decisoes/007-cache-por-fenotipo.md) | Cache e diversidade por fenótipo, não genótipo |
| [008](docs/decisoes/008-llm-local-e-allow-list.md) | LLM local por padrão; anonimização por allow-list |
| [009](docs/decisoes/009-scripts-da-fase-1-intocados.md) | Scripts da Fase 1 copiados sem alteração |

---

## A regra que organiza o projeto

> 🔴 **O fitness nunca toca o conjunto de teste.**

Não é convenção — é estrutura. O `dataclass DadosTreino` que o AG recebe **não tem nenhum campo**
`X_teste`/`y_teste`; não existe atributo para um bug futuro acessar. O módulo que pode ler o
holdout (`src/fitness/avaliacao_final.py`) não é importado por nenhum código do AG.

Verificado por teste que substitui o conjunto de teste por um objeto que lança exceção em qualquer
acesso — atributo, `len()`, indexação, iteração — cobrindo tanto uma chamada isolada de fitness
quanto o laço evolutivo completo.

---

## Limitações declaradas

1. **O ganho possível é estruturalmente pequeno.** Hotstart já bom, 541 registros, espaço de busca
   de 3–7 genes por família. Nenhum ajuste de hiperparâmetro contorna essas três coisas.
2. **Ajustar hiperparâmetros não corrige viés de amostra.** A base vem de **uma única clínica de
   fertilidade**. Um modelo melhor ajustado sobre dados enviesados é um modelo que erra com mais
   confiança — o AG pode até amplificar o viés, porque o fitness premia acerto na distribuição de
   treino. Tratado no NB 03 §9.
3. **A avaliação da LLM não é validação clínica.** Avaliador único, não-médico, n=15 pacientes
   (45 saídas). É avaliação sistemática e reprodutível no escopo de um trabalho de faculdade.
4. **O fitness usa as mesmas 5 dobras para todos os indivíduos.** Remove ruído de amostragem como
   fator de confusão, mas permite que o AG se ajuste àquela partição específica — o que explica
   parte do ganho de CV não transferido ao holdout.

5. **Nenhum template calibra a incerteza corretamente** (≤67%), e o few-shot piora isso — ver
   [relatório §5.5](docs/relatorio-tecnico.md). A correção certa é pré-computar a faixa de
   confiança em código, como já se faz com os números.
6. **Quatro achados da auditoria final permanecem abertos** — nenhum invalida um resultado
   reportado, e os quatro estão descritos com impacto e correção proposta no
   [relatório técnico §7](docs/relatorio-tecnico.md):
   a penalidade de custo do plano nunca foi implementada; o terceiro critério de parada
   (`piso_entropia=0.0`) nunca dispara; o hotstart não cobre as cinco famílias; e o termo de
   AUC do fitness deixa de medir ordenação quando o KNN tem `n_neighbors=1` — foi o que
   produziu a única regressão estrutural da tabela acima (AUC do KNN 0,9007 → 0,7924).

---

## Relatório técnico

[`docs/relatorio-tecnico.md`](docs/relatorio-tecnico.md) — as nove seções que o enunciado
pede, com todas as tabelas geradas a partir de `runs/` por `src/experimentos/analise.py`.

## Pendências

- Vídeo de demonstração (≤ 15 min) — entregável 3 do enunciado.
- Rubrica manual da LLM: os 3 critérios qualitativos (fidelidade às features, acionabilidade,
  clareza para médico) nas 45 saídas de `avaliacao_llm/`. Os outros 3 já têm medição objetiva
  no [relatório §5.5](docs/relatorio-tecnico.md).
- Extensões opcionais não implementadas: seleção de features no cromossomo (a que mais
  mudaria a conclusão do empate com busca aleatória), frente de Pareto (recall × precisão) e
  família `MLP`.
