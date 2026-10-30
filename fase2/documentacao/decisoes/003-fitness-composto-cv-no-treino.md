# ADR 003 — Fitness composto por CV no treino; holdout intocado

**Status:** aceita · **Data:** 08/10/2026 · **Criticidade:** 🔴 a mais alta do projeto

## Contexto

Um AG é um otimizador agressivo. Com 450 avaliações sobre 109 linhas de teste, ele *vai*
encontrar a configuração que explora o ruído daquelas linhas específicas — e produzir um número
bonito e sem valor. Este é o erro mais provável de qualquer trabalho desse tipo.

Separadamente: o objetivo é diagnóstico médico, onde falso negativo e falso positivo não custam
o mesmo. A função de fitness é específica do problema e tem de refletir os objetivos e os
critérios de desempenho desejados — aqui, errar menos o diagnóstico positivo.

## Decisão

**1. Fitness composto:** `0,5·F1 + 0,3·recall + 0,2·AUC`, média sobre `StratifiedKFold(k=5)`
aplicado **exclusivamente às 432 linhas de treino**.

**2. Garantia estrutural:** o `dataclass DadosTreino` que o AG recebe **não tem nenhum campo**
`X_teste`/`y_teste`. O módulo que pode ler o holdout (`avaliacao_final.py`) está em arquivo
separado e não é importado por nenhum código do AG.

## Justificativa

**Por que não acurácia.** Trata falso negativo e falso positivo como equivalentes — logo é a
métrica errada para diagnóstico.

**Como cheguei aos pesos.** Da evidência da Fase 1: o Random Forest vencedor tinha **recall
0,8056 como métrica mais fraca**, apesar de precisão 0,9355. O modelo escolhido pela Fase 1 erra
justamente no tipo de erro que mais importa clinicamente. Daí F1 (0,5) como objetivo balanceado,
recall com peso extra (0,3) empurrando na direção da fraqueza conhecida, e AUC (0,2) para
estabilizar o fitness quando dois indivíduos empatam em F1.

**Por que garantia estrutural e não convenção.** Uma regra escrita em comentário pode ser
violada por um bug futuro. Um atributo que não existe, não.

**Por que CV e não split único de validação.** Com 432 linhas, um split de validação teria ~86
linhas — variância altíssima. O AG, otimizando 450 vezes contra alvo tão ruidoso, otimizaria o
ruído daquele split. O custo são 5 treinos por avaliação, pagável porque o alvo é tabular.

**Por que as mesmas 5 dobras para todos.** Reembaralhar introduziria ruído de amostragem como
fator de confusão: dois indivíduos idênticos receberiam fitness diferentes.

## Consequências

**Positivas**
- Teste automatizado substitui o holdout por um objeto que lança exceção em qualquer acesso —
  cobrindo tanto uma chamada isolada quanto o laço evolutivo completo.
- A comparação "antes × depois" é sobre o mesmo split e a mesma semente da Fase 1.

**Negativas, confirmadas pelos resultados**
- O AG melhorou o fitness de CV nas 5 famílias (+0,023 a +0,072), mas o ganho transferiu para o
  holdout em apenas 3 de 5 no score composto. **Fixar as dobras permite que o AG se ajuste
  àquela partição** — risco previsto, declarado no Notebook 02.
- Comparar F1 isolado no holdout contra o AG é injusto: ele maximizou o composto. A comparação
  correta usa a mesma função nos dois lados, e está no Notebook 02 §6.

## Predição que esta decisão gerou — e o que os dados mostraram

O plano previa que o alelo `class_weight='balanced'` se fixaria na população, por causa do peso
em recall. **O resultado não é "fixou" nem "caiu": a frequência do alelo acompanha a pressão
seletiva**, e o fitness é idêntico nos quatro experimentos.

| Experimento | Geração 0 | Geração 5 (casada) | Fim | Δ no fim |
|---|---|---|---|---|
| **E3** (torneio `k=4`) | 0,657 | **0,956** | **0,954** | **+0,297** — nas 3 sementes |
| **E1** (torneio `k=2`) | 0,674 | 0,606 | 0,700 | +0,026 |
| **E2** (mutação 0,35) | 0,614 | 0,512 | 0,540 | −0,074 |
| **D** (roleta) | 0,674 | 0,473 | 0,440 | −0,234 |

> ⚠️ **Correção a uma versão anterior deste ADR.** O texto anterior dizia que o alelo "caiu
> (0,67 → 0,61 no E1)" e atribuía isso ao peso 0,5 do F1 penalizando a perda de precisão.
> O número 0,61 é real, mas é a frequência na **geração 5**, lida num gráfico truncado ao
> comprimento da execução mais curta; no E1 a frequência **volta a 0,700 até a geração 15**,
> acima dos 0,674 iniciais. E a explicação é contrariada pelo E3, onde o alelo se fixa em 0,95
> sob exatamente o mesmo fitness. O fator que separa os quatro experimentos não é o fitness —
> é a **pressão seletiva**.

**Consequência útil:** o alelo passa a funcionar como **medidor independente de pressão
seletiva**, e dá à ablação do ADR 005 uma segunda observável, que não sofre do problema de
sinal invertido da primeira. Com `k=4` o alelo vai a 0,95; com roleta, cai a 0,44.

**Ressalva:** em E3 a população de 15 ficou clonal (6 fenótipos na geração 5), então a
magnitude de +0,297 tem componente de deriva genética. A evidência que sustento é a **direção,
consistente nas três sementes**. A evidência mais limpa é a da roleta: −0,234 em apenas 6,7
gerações, numa população que permaneceu diversa.
