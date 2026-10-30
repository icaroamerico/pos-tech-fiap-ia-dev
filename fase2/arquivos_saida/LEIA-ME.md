# arquivos_saida/

Tudo o que o código **grava**. Nada aqui é entrada — as entradas ficam em
`base_dados_tratada/` (dataset) e `prompts/` (templates de prompt do LLM).

```
arquivos_saida/
├── logs/                                  # log por execução do AG (.log) e stdout de lotes (.out)
└── arquivos_analiticos/
    ├── historico_execucoes_ag/            # uma pasta por execução do AG
    │   ├── <EXPERIMENTO>_seed<N>/
    │   │   ├── config.json                # os parâmetros usados naquela execução
    │   │   ├── history.jsonl              # uma linha por geração (fonte dos gráficos)
    │   │   ├── resumo.json                # melhor_fitness, motivo de parada, tempos
    │   │   ├── best_individual.json       # o melhor cromossomo encontrado
    │   │   ├── population_final.json      # população da última geração
    │   │   └── metrics_holdout.json       # métricas no holdout (só com --com-holdout)
    │   └── resumo_execucoes.json          # índice de todas as execuções
    ├── metricas_modelos_baseline/         # baseline da Fase 1 reproduzido
    │   ├── metricas_modelos.json          # F1, recall, AUC dos 5 modelos clássicos
    │   └── metricas_modelos.txt           # a mesma tabela em texto
    ├── avaliacao_llm/                     # avaliação do LLM — saída corrente
    │   ├── avaliacao_<provedor>_saidas.json   # texto gerado por paciente × template
    │   ├── avaliacao_<provedor>_saidas.md     # o mesmo, legível
    │   └── avaliacao_<provedor>_rubrica.csv   # alucinação, aderência, latência por saída
    └── avaliacao_llm_simulado/            # rodada anterior do provedor simulado, preservada
```

## Quem grava o quê

| Caminho | Gravado por |
|---|---|
| `logs/` | `src/monitoramento/registro.py` (`DIR_LOGS`) |
| `arquivos_analiticos/historico_execucoes_ag/` | `src/monitoramento/registro.py` (`DIR_RUNS`) |
| `arquivos_analiticos/metricas_modelos_baseline/` | `src/modelagem/treinamento.py` (`REPORTS_DIR`) |
| `arquivos_analiticos/avaliacao_llm/` | `src/llm/avaliacao.py` (`DIR_AVALIACAO`) |

Para mudar um destino, edite a constante — os pontos de uso a importam.
