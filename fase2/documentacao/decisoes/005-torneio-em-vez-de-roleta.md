# ADR 005 — Torneio em vez de roleta: desvio deliberado do material

**Status:** aceita · **Data:** 08/10/2026

## Contexto

A formulação canônica do AG usa **seleção probabilística tipo roleta** — sorteio aleatório
ponderado pelo valor de fitness —, implementada como

```python
probability = 1 / np.array(population_fitness)   # TSP é minimização
parent1, parent2 = random.choices(population, weights=probability, k=2)
```

## Decisão

Usar **seleção por torneio** (sem reposição) como operador padrão, mantendo a roleta
implementada para uma ablação.

## Justificativa

**A roleta depende da escala absoluta do fitness; o torneio não.**

No TSP, caso clássico da roleta, o fitness é distância — varia de centenas a milhares de
pixels, e `1/distância`
gera pesos genuinamente diferentes. **Neste problema, o fitness é uma métrica de classificação
confinada a ~[0,75; 0,92].** A razão entre melhor e pior indivíduo é ~1,2: a roleta daria a
praticamente todos a mesma probabilidade, e a pressão seletiva seria quase nula.

**Outras razões:** o torneio é invariante a transformações monotônicas do fitness; `k` dá
controle direto de pressão seletiva (usado no E3 com k=4); custo O(1) por seleção contra O(n) da
roleta.

**Como tornei isso verificável.** Implementei os dois operadores e rodei a configuração E1 com
cada um, 3 sementes (Etapa D).

## Consequências

### A previsão que eu escrevi estava com a observável invertida

O plano previa: *"média da população colada no melhor indivíduo desde as primeiras gerações"*.
**Isso descreve pressão seletiva forte, não fraca.** Raciocinando corretamente:

| | pressão forte | pressão fraca |
|---|---|---|
| População converge para o melhor? | sim | não |
| Gap melhor − média | pequeno | **grande** |
| Entropia | cai rápido | **permanece alta** |

**O que os dados mostraram:** roleta com gap **maior** e entropia **mais alta**. A população da
roleta **não converge** para o melhor — o que **confirma a tese** (pressão fraca nesta faixa de
fitness) pela observável oposta à que eu escrevi.

> ⚠️ **A comparação tem de ser em geração casada.** O E1 rodou 12 gerações em média e a roleta
> apenas 6,7 (parou por estagnação nas gerações 6, 8 e 6) — "entropia final" compararia pontos
> distintos da mesma trajetória, e uma população que parou mais cedo simplesmente teve menos
> tempo para perder diversidade. Em **geração 5**, a última presente em todas as 12 execuções,
> e também em todas as gerações 1 a 4:

```
entropia média, gerações 0→5:   torneio  2,245 1,981 1,510 1,219 1,238 1,016
                                 roleta  2,245 2,086 1,834 1,661 1,668 1,509
gap melhor−média, gerações 0→5: torneio  0,1372 0,0882 0,0636 0,0374 0,0393 0,0678
                                 roleta  0,1372 0,1092 0,1002 0,0634 0,0667 0,0632
```

A entropia da roleta é mais alta em **todas** as gerações casadas, e o gap é maior nas gerações
1 a 4 (na geração 5 os dois praticamente empatam). *(A versão anterior deste ADR citava
"1,509 vs 0,614" e "0,0865 vs 0,0547", que são os valores **no fim de cada execução** — não
comparáveis entre si, porque as execuções têm comprimentos diferentes.)*

**A observável mais limpa da ablação é outra:** a roleta **parou por estagnação nas três
sementes**, nas gerações 6, 8 e 6, enquanto o torneio rodou até a geração 15 em duas das três.
Sem pressão seletiva, o melhor indivíduo para de melhorar; o elitismo o preserva, a estagnação
dispara, e a execução encerra. E a frequência do alelo `class_weight='balanced'` cai 0,234 na
roleta contra uma subida de 0,026 no torneio — ver ADR 003.

### O efeito no resultado é pequeno

Fitness: roleta 0,8746 vs torneio 0,8765 — diferença de 0,0019, dentro do desvio padrão
(0,0039). **Empate estatístico.** Com pressão fraca e elitismo preservando o melhor, a roleta
mantém diversidade e chega quase no mesmo lugar — num espaço de busca raso, onde diversidade não
custa muito. É mais um indício de que o espaço é pequeno (ver ADR 003 e o empate com busca
aleatória) que de que a roleta seja boa.

**Por que mantive a previsão errada no relatório.** Ela estava versionada antes da execução.
Ajustá-la depois transformaria o experimento em confirmação decorativa — o valor de declarar
hipóteses antes é poder ser contrariado por elas.

## Bug corrigido durante a implementação

A primeira versão do torneio sorteava **com reposição** (`rng.integers`), podendo colocar o mesmo
indivíduo duas vezes no mesmo torneio e nunca comparar o segundo candidato — pressão seletiva
inconsistente em populações pequenas. Corrigido para `rng.choice(..., replace=False)`.
