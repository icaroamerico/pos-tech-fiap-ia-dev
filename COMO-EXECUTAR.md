# Como Executar o Projeto (Fase 1 + Fase 2)

**Aluno:** Icaro Minervini Americo — RM376755  
**Turma:** 10IADT  
**Documento:** Guia de execução para reprodução

---

## Pré-requisitos

- **Python 3.12+**
- **8 GB de RAM** (mínimo; 16 GB recomendado para execução completa com LLM)
- **Ollama** (opcional; necessário apenas para gerar saídas reais com LLM)

### Instalação de dependências

```bash
# Verificar Python
python3 --version

# Se não tiver Ollama e quiser LLM real (opcional):
brew install ollama
ollama pull llama3.2:3b
ollama serve  # Em outro terminal, deixar rodando
```

---

## Estrutura do repositório

```
pos-tech-fiap-ia-dev/
├── fase1/                    # Diagnóstico de SOP (baseline)
│   ├── src/
│   ├── requirements.txt
│   └── [mais arquivos]
│
├── fase2/                    # Otimização com AG + LLM
│   ├── src/
│   ├── tests/
│   ├── notebooks/
│   ├── prompts/
│   ├── requirements.txt
│   └── [mais arquivos]
│
└── COMO-EXECUTAR.md         # Este arquivo
```

**Importante:** Cada fase tem seu próprio ambiente virtual e dependências. Não compartilhe `.venv` entre elas.

---

## Execução da Fase 1 (Baseline)

Se quiser reproduzir o baseline dos modelos clássicos:

```bash
cd fase1

# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Treinar os 5 modelos
python -m src.run_ml
# Saída: fase1/reports/metricas_modelos.json

# Ver notebooks (ETL, modelagem, CNN)
jupyter lab
# Acesse: src/tratamento_dados/ e src/modelagem/
```

**Tempo estimado:** ~10 min

---

## Execução da Fase 2 (Algoritmo Genético + LLM)

### Setup inicial

```bash
cd fase2

# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

---

## ⭐ OPÇÃO RECOMENDADA: Notebook Orquestrador (Mais fácil!)

**Tudo em um único notebook, com barra de progresso:**

```bash
jupyter lab notebooks/00_execucao_completa.ipynb
```

Depois execute `Cell → Run All`. O notebook:
- ✅ Roda baseline (Fase 1)
- ✅ Roda AG (12 execuções) com progresso
- ✅ Roda análise (tabelas comparativas)
- ✅ Roda LLM com **barra de progresso**
- ✅ Verifica saídas
- ✅ Mostra resumo final

**Tempo:** ~15 min (simulado) ou ~3h (Ollama real)

**Mude PROVEDOR na célula 4 para:**
- `"simulado"` — rápido, sem Ollama (padrão)
- `"ollama"` — LLM real, requer Ollama rodando

---

## Execução manual (passo a passo)

Se preferir rodar cada etapa separadamente:

### Passo 1: Reproduzir o baseline (da Fase 1)

```bash
python -m src.run_ml
```

**Tempo:** ~10 s  
**Saída:** Confirma que os 5 modelos treinam igual ao baseline

### Passo 2: Executar o Algoritmo Genético

#### Opção A: Experimentos completos (E1, E2, E3, D × 3 sementes = 12 rodadas)

```bash
python -m src.experimentos.rodar_ag --experimento todos
```

**Tempo:** ~3 min  
**Saída:** 12 pastas em `arquivos_saida/arquivos_analiticos/historico_execucoes_ag/` com histórico, métricas e melhores indivíduos

#### Opção B: Apenas um experimento (e.g., E1)

```bash
python -m src.experimentos.rodar_ag --experimento E1
```

**Tempo:** ~30 s

#### Opção C: Otimização por família (Etapa A)

```bash
python -m src.experimentos.rodar_ag --etapa-a
```

**Tempo:** ~2 min

---

## Execução do LLM (Duas opções)

### Opção 1: Amostra rápida (recomendada — padrão da entrega)

**9 saídas (3 pacientes × 3 templates), sem Ollama instalado**

```bash
jupyter lab
```

1. Abra `notebooks/03_llm_interpretacao.ipynb`
2. **Célula 6:** Certifique-se de que `PROVEDOR = "simulado"`
3. Execute as células até a **célula 23**
4. Resultado: 9 saídas em `avaliacao_llm_simulado/`

**Tempo:** ~5 min  
**Observação:** Usa um provedor determinístico (sem rede). Bom para demonstração, não é generativo.

### Opção 2: Execução completa com modelo real (requer Ollama)

**45 saídas (15 pacientes × 3 templates) com `llama3.2:3b` real**

#### Pré-requisito

```bash
# Em outro terminal, deixar Ollama servindo
ollama serve
```

#### Execução

```bash
# No terminal da fase2, com .venv ativo
jupyter lab
```

1. Abra `notebooks/03_llm_interpretacao.ipynb`
2. **Célula 6:** Mude para `PROVEDOR = "ollama"`
3. Execute até a **célula 23**
4. Resultado: 45 saídas em `avaliacao_llm/avaliacao_ollama_saidas.json`

**Tempo:** ~2h (executa em swap; com 16 GB de RAM: ~20 min)  
**Observação:** Gera textos reais com modelo local. Nenhum dado sai da máquina.

#### Se RAM for limitada

Para máquinas com exatamente 8 GB, o Ollama pode entrar em swap:

```bash
# Aumentar memória virtual (macOS)
sudo sysctl -w vm.max_map_count=262144

# Fechar outras aplicações para liberar RAM
pkill -f "Chrome\|Slack\|Discord"

# Deixar o Jupyter rodar (vai demorar ~2h, mas termina)
```

---

## Ver os resultados

### Métricas do AG

```bash
python -m src.experimentos.analise
```

Gera tabelas em Markdown de:
- Comparação baseline vs. melhor indivíduo por experimento
- Evolução de fitness ao longo das gerações
- Métricas de diversidade genética

### Notebooks com visualizações

```bash
jupyter lab
# Abra notebooks/01_ag_hiperparametros_pcos.ipynb (AG, fitness, evolução)
# Abra notebooks/02_experimentos_ag_comparativo.ipynb (experimentos, diversidade, comparativo)
# Abra notebooks/03_llm_interpretacao.ipynb (LLM, Templates)
```

### Relatório técnico

```bash
# Ver em editor de texto ou markdown
cat documentacao/relatorio-tecnico.md

# Ou converter para PDF (requer pandoc)
pandoc documentacao/relatorio-tecnico.md -o relatorio-tecnico.pdf
```

---

## Leitura recomendada (em ordem)

1. **`README.md`** — Visão geral e resultados em tabela
2. **`fase2/README.md`** — Rastreabilidade requisito → código
3. **`fase2/documentacao/relatorio-tecnico.md`** — Relatório técnico completo (§1-§9)
4. **`fase2/documentacao/decisoes/`** — 9 ADRs explicando cada decisão
5. **`fase2/documentacao/arquitetura.md`** — Diagramas de componentes e sequência

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Causa:** O Jupyter não está usando o venv correto.

**Solução:**
```bash
# Feche o Jupyter
# No terminal, com .venv ativado:
source .venv/bin/activate
jupyter lab

# OU, reabra o Jupyter e selecione o kernel correto:
# Kernel → Select Kernel → Python (venv)
```

### "ConnectionError: Ollama não está rodando"

**Causa:** Ollama deve estar servindo em `http://localhost:11434`

**Solução:**
```bash
# Em outro terminal
ollama serve

# Testar conexão
curl http://localhost:11434/api/tags
```

### "MemoryError: Swap insuficiente"

**Causa:** Machine com < 8 GB tentando rodar modelo 8B

**Solução:**
- Use `llama3.2:3b` (2 GB) — já é o padrão
- Ou use amostra de 9 saídas
- Ou aumente RAM virtual:
```bash
sudo sysctl -w vm.max_map_count=262144
```

### "pytest: command not found"

**Causa:** pytest não está no venv

**Solução:**
```bash
source .venv/bin/activate
pip install pytest pytest-cov
python -m pytest tests/ -v
```

---

## Resumo de tempos

| Operação | Tempo | Requer Ollama? |
|---|---|---|
| Baseline (Fase 1) | ~10 s | Não |
| Testes (221 testes) | ~2 min | Não |
| AG (E1-E3-D × 3 seeds) | ~3 min | Não |
| LLM amostra (9 saídas, simulado) | ~5 min | Não |
| LLM completo (45 saídas, Ollama) | ~20 min (com 16GB) / ~2h (com swap) | **Sim** |
| Relatório gerado automaticamente | — | Não (lê `arquivos_saida/`) |

**Total mínimo para reprodução completa:** ~20 min (sem LLM real)  
**Total máximo:** ~2,5h (com LLM real em swap)

---

## Entregáveis finais

Após executar tudo, você terá:

- ✅ `fase1/reports/` — Métricas dos 5 modelos baseline
- ✅ `fase2/arquivos_saida/arquivos_analiticos/historico_execucoes_ag/` — 12 execuções do AG (config, história, melhores indivíduos, métricas)
- ✅ `fase2/arquivos_saida/arquivos_analiticos/avaliacao_llm/` — 45 saídas LLM reais (ou simuladas, conforme opção escolhida)
- ✅ `fase2/documentacao/relatorio-tecnico.md` — Relatório técnico com todas as tabelas geradas
- ✅ `fase2/tests/` — 221 testes cobrindo anti-leakage, guardrails, modularização

---

## Dúvidas?

Consulte:
- **Arquitetura:** `fase2/documentacao/arquitetura.md`
- **Decisões:** `fase2/documentacao/decisoes/` (9 ADRs)
- **Código:** módulos em `src/` (cada um comenta seu propósito)
- **Testes:** exemplos em `tests/` de como cada módulo funciona
