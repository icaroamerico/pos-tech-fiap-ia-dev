# ADR 007 — Cache e diversidade por fenótipo, não por genótipo

**Status:** aceita · **Data:** 09/10/2026 (após auditoria)

## Contexto

A codificação híbrida (ADR 002) faz todo indivíduo carregar os genes de **todas** as cinco
famílias, mas só a família ativa afeta o modelo treinado. A primeira implementação usava uma
única chave — `chave_individuo` — construída sobre o **genótipo completo**, e ela servia a dois
propósitos: cache de fitness e contagem de soluções únicas.

Isso estava errado nos dois casos.

## Decisão

Separar em duas funções:

- **`chave_genotipo`** — cromossomo completo, incluindo genes latentes;
- **`chave_fenotipo`** — só a família ativa e os genes que ela expressa, respeitando as regras
  condicionais (`l1_ratio` só com elasticnet, `p` só com minkowski).

O **cache de fitness** e a métrica **`numero_solucoes_unicas`** usam `chave_fenotipo`. Foi
acrescentada `numero_genotipos_unicos` como métrica separada.

## Justificativa

### Problema 1 — o cache era quase inoperante

O fitness depende exclusivamente do fenótipo. Com chave de genótipo, dois indivíduos de fitness
**comprovadamente idêntico** recebiam chaves diferentes — e como os genes latentes derivam a cada
mutação, a colisão quase nunca acontecia.

Medido em 15 gerações × população 30 = 450 avaliações:

| chave | acertos | treinos efetivos |
|---|---|---|
| genótipo | 29 = **6,4%** | 421 |
| fenótipo | 121 = **26,9%** | 329 |

**92 treinos economizados por execução** — cada um com 5 folds, ou seja ~460 ajustes de modelo a
menos. A otimização existia no papel e era inócua na prática.

### Problema 2 — a métrica de diversidade estava cega

`numero_solucoes_unicas` contava genótipos, então uma população **já clonal no fenótipo**
continuava reportando diversidade alta. A métrica existe justamente para detectar
**convergência prematura** — e estava cega a ela.

## Consequências

**Positivas**
- Cache 4× mais eficaz.
- A métrica passa a medir diversidade **efetiva**. Foi com ela que o E3 revelou entropia → 0 e
  cache em 61% — a assinatura numérica da convergência prematura.
- Comparar `numero_solucoes_unicas` com `numero_genotipos_unicos` mostra quanta diversidade é
  apenas latente — material de análise sobre o papel do reservatório recessivo.

**Negativas**
- Duas funções onde havia uma; risco de usar a errada. Mitigado por docstrings que explicam
  explicitamente qual serve para quê, e por testes de regressão nos dois sentidos (fenótipos
  idênticos devem colidir; fenótipos diferentes não devem).

## Nota: colisões de fenótipo na geração 0 são legítimas

Um teste falhou inicialmente por assumir que a geração 0 não teria acertos de cache. Investigando:
dois indivíduos KNN sorteados aleatoriamente caíram no **mesmo fenótipo**
(`metric=euclidean, n_neighbors=7, weights=distance`). O espaço de fenótipos do KNN é totalmente
discreto e pequeno, então colisões são esperadas — e o cache corretamente deduplicou. O teste
estava errado, não o código.
