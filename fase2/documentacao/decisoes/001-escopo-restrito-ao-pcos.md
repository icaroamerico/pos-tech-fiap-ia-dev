# ADR 001 — Escopo restrito ao PCOS; CNN fora da Fase 2

**Status:** aceita · **Data:** 08/10/2026

## Contexto

O enunciado pede otimizar "os **modelos** de diagnóstico desenvolvidos no Módulo 1" — plural. A
Fase 1 entregou dois projetos: PCOS (5 classificadores tabulares) e uma CNN de raio-X de tórax.

Um AG avalia `população × gerações` indivíduos, cada um um treino completo:

| Alvo | Custo/treino | Config | Total |
|---|---|---|---|
| PCOS (sklearn) | ~0,3–3 s | pop 30 × 15 ger | **~10–20 min** |
| CNN, 12 épocas, dataset cheio | ~8 min | pop 20 × 10 ger | **~27 h** |
| CNN, proxy de 3 épocas em 30% dos dados | ~36 s | pop 8 × 5 ger | ~25 min |

## Decisão

O AG otimiza **exclusivamente** as cinco famílias de modelo do PCOS. A CNN fica integralmente
fora da Fase 2 — inclusive a variante de fidelidade reduzida.

## Justificativa

**1. O plural do enunciado é satisfeito sem a CNN.** O PCOS sozinho tem cinco famílias de
modelo, com comparação individual antes/depois. Otimizar as cinco *é* otimizar "os modelos".

**2. O proxy de baixa fidelidade mediria a coisa errada.** A terceira linha da tabela era
tecnicamente viável e ainda assim foi descartada: fitness com 3 épocas sobre 30% dos dados é um
*substituto* do desempenho real. Arquiteturas com BatchNorm e dropout alto convergem mais
devagar e seriam **sistematicamente penalizadas** — o AG escolheria a arquitetura errada pelo
motivo errado. O resultado exigiria tantas ressalvas que o valor informativo tenderia a zero.

**3. Dois alvos diluiriam a análise.** Com um alvo só, cabe fazer os 3 experimentos com
repetição estatística, instrumentar todas as métricas de diversidade do AG e comparar contra busca
aleatória — tudo bem feito. Com dois, as duas metades ficam superficiais.

**4. O ganho marginal é baixo.** Os cinco modelos PCOS já dão cinco linhas na tabela
comparativa, com espaços de busca heterogêneos (real em escala log, inteiro, categórico,
booleano) que exercitam a codificação híbrida de forma mais completa que a CNN faria.

## Consequências

**Positivas**
- Profundidade em vez de amplitude: 3 experimentos × 3 sementes, ablação de seleção e baseline
  de busca aleatória — nenhum deles caberia com dois alvos.
- O orçamento liberado virou a Etapa D (ablação torneio × roleta), que mede uma alegação de
  projeto em vez de apenas afirmá-la.

**Negativas, declaradas**
- Perde-se a pergunta *"o AG redescobre que a CNN simples ganha?"* — fica em "próximos passos".
- A otimização de hiperparâmetros de rede neural perde aplicação direta e passa a ser
  fundamento conceitual. Recuperável via `MLPClassifier` como sexta família, registrado como
  extensão opcional.
- O argumento de custo computacional do requisito 2 fica menos dramático — ainda válido (450
  avaliações × 5 folds × 5 famílias × 3 sementes), mas menos impressionante.

## Lição generalizável

Quando o fitness é caro, a tentação é baratear o fitness. Mas **baratear o fitness muda o que
está sendo otimizado**. Antes de adotar um proxy, a pergunta é se ele preserva a *ordem* entre
indivíduos — se não preserva, o AG otimiza o proxy, não o objetivo. Às vezes a resposta certa é
reduzir o escopo, não a fidelidade.
