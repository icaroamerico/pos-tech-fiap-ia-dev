# ADR 006 — Mutação normalizada por família; troca de família desacoplada

**Status:** aceita · **Data:** 09/10/2026 (após auditoria)

## Contexto

A primeira implementação de `mutar()` aplicava a probabilidade de mutação **gene a gene nas 24
posições** do cromossomo (incluindo os genes latentes das famílias inativas) e usava a **mesma
taxa** para o gene de família.

Medindo o comportamento real, dois problemas apareceram.

**Problema 1 — pressão desigual entre famílias.** Só os genes da família ativa afetam o fitness,
e as famílias têm números diferentes de genes:

| família | genes | mutações *ativas* esperadas (p=0,15) |
|---|---|---|
| random_forest | 7 | 1,05 |
| decision_tree | 6 | 0,90 |
| logistic_regression | 4 | 0,60 |
| knn | 4 | 0,60 |
| **svm_rbf** | **3** | **0,45** |

Um indivíduo Random Forest sofria **2,3× mais mutação efetiva** que um SVM à mesma
`probabilidade` — não por razão de projeto, só porque tem mais genes.

**Problema 2 — a família virava passeio aleatório.** Com a taxa unificada, a chance de uma
linhagem **manter** sua família ao longo de 15 gerações era:

| p | persistência da família |
|---|---|
| 0,05 | 46,3% |
| 0,15 | **8,7%** |
| 0,35 | **0,1%** |

No E2 (p=0,35), a escolha do modelo era essencialmente aleatória — o AG nunca conseguia se fixar
numa família e refiná-la, que é o oposto do que a seleção deveria produzir.

## Decisão

**1.** Parâmetro `prob_troca_familia` **separado** de `probabilidade`, com default **0,05**.

**2.** Flag `normalizar_por_familia=True` (default): a probabilidade por gene é escalada para
que o número **esperado** de mutações nos genes expressos seja o mesmo em qualquer família,
calibrado pela família de referência (a maior, com 7 genes).

## Justificativa

**Por que separar a troca de família.** Trocar a família é mudança de **natureza diferente** de
ajustar um hiperparâmetro: descarta todo o refinamento acumulado naquela família. Tratar as duas
coisas com o mesmo parâmetro significa que não é possível explorar entre modelos *e* refinar
dentro de um deles de forma independente. Com 0,05, a persistência sobe para 46%.

**Por que normalizar em vez de mutar só a família ativa.** Mutar apenas os genes expressos
igualaria a pressão, mas abandonaria o **reservatório latente** — que é um argumento explícito
da codificação (ADR 002). A normalização preserva o reservatório *e* iguala a pressão.

**Por que calibrar pela maior família.** Mantém `probabilidade` com ordem de grandeza de
significado próxima à que tinha antes, para que os valores dos experimentos (0,05 / 0,15 / 0,35)
continuem interpretáveis.

## Consequências

**Positivas**
- `probabilidade` passa a significar a mesma coisa para todas as famílias — os experimentos do
  plano §4.6 ficam interpretáveis como pretendido.
- Verificado por teste: com normalização, RF e SVM sofrem mutações ativas estatisticamente
  equivalentes (diferença < 0,35 em 3000 amostras). Há **contraprova** que documenta a assimetria
  quando a normalização é desligada.

**Negativas**
- Mais um parâmetro de configuração. Mitigado por default sensato e por `normalizar_por_familia`
  poder ser desligado para reproduzir o comportamento antigo.
- A probabilidade efetiva por gene do SVM chega a 0,35 quando `probabilidade=0,15` — o clamp em
  1,0 impede exceder, mas em taxas altas a normalização satura.

## Alternativas consideradas

| Opção | Por que não |
|---|---|
| Mutar só os genes da família ativa | Abandona o reservatório latente (ADR 002) |
| Manter taxa unificada | Mantém os dois problemas medidos acima |
