# ADR 002 — Codificação híbrida com gene de família e genes latentes

**Status:** aceita · **Data:** 08/10/2026

## Contexto

Hiperparâmetros de modelos de ML são heterogêneos: `n_estimators` é inteiro, `C` é real em
escala multiplicativa, `criterion` é categórico, `bootstrap` é booleano. A Aula 3 de AG
apresenta quatro codificações (binária, real, combinatória, híbrida) e a regra para escolher:
o operador tem de respeitar **validade da solução** e **custo computacional**.

## Decisão

**Codificação híbrida** com `SCHEMA` declarativo por tipo de gene (`int`, `int_odd`,
`int_or_none`, `real`, `real_log`, `categorical`), onde:

- o **gene 0** escolhe a família de modelo entre as cinco;
- todo indivíduo carrega os genes de **todas** as famílias simultaneamente — as inativas ficam
  **latentes**.

## Justificativa

**Por que não binária.** Forçar tudo em bits exigiria uma camada de decodificação artificial e
jogaria fora a estrutura do problema. A aula é explícita: *"cada trecho da codificação híbrida
representa um fenômeno da solução, com restrições personalizadas"*.

**Por que o gene de família.** Transforma o AG em algo mais forte que busca de hiperparâmetros:
ele faz **seleção de modelo e ajuste simultaneamente**. A pressão seletiva decide qual família
merece orçamento, em vez de eu decidir a priori — e isso responde uma pergunta que a Fase 1 não
podia: *o Random Forest continua ganhando quando todas as famílias são bem ajustadas?*

**Por que genes latentes e não cromossomo de tamanho variável.** Tamanho fixo mantém crossover
e mutação triviais e sempre válidos. E os genes latentes funcionam como **reservatório de
diversidade**: se o gene de família mutar, os hiperparâmetros daquela família já estão
carregados, em vez de serem sorteados do zero. É o paralelo de material genético recessivo.

**Por que `C` e `gamma` em escala logarítmica.** A sensibilidade do SVM a eles é
**multiplicativa**: a diferença entre 0,01 e 0,1 pesa tanto quanto entre 10 e 100. Em escala
linear sobre `[0,01; 1000]`, a mutação gaussiana quase nunca sortearia valores abaixo de 1 —
metade do espaço útil ficaria inacessível. O gene guarda `log10(valor)`; crossover aritmético em
log-space equivale a **média geométrica** no espaço real.

## Consequências

**Positivas**
- Validade **por construção**: `clamp_individuo` após cada operador, pares `penalty`/`solver`
  atômicos, `n_neighbors` sempre ímpar. Nenhum `try/except` em volta do treino.
- Genes latentes dão material para discutir diversidade efetiva vs. latente no relatório.

**Negativas**
- O cromossomo tem 24 genes, mas só 3 a 7 afetam o fitness de cada indivíduo. Isso exigiu
  cuidado na mutação (ver ADR 006) e na chave de cache (ver ADR 007) — **dois bugs reais
  nasceram de ignorar essa assimetria**.

## Bug que esta decisão causou

Guardar `log10` internamente significa que **todo valor em unidades reais precisa ser codificado
antes de entrar no cromossomo**. A tabela de hotstart guardava `C=1.0` cru; a decodificação
calculava `10¹ = 10`, e `gamma=0.1` virava `1,26`. O SVM resultante previa tudo negativo
(**F1 = 0,000**) e passava desapercebido porque os testes só checavam `0 ≤ fitness ≤ 1`.

Correção: `codificar_genes()`, a inversa explícita da decodificação, com teste que fecha o ciclo
nas cinco famílias.
