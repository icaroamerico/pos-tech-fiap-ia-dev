# ADR 004 — AG implementado do zero, sem DEAP/PyGAD

**Status:** aceita · **Data:** 08/10/2026

## Contexto

Existem bibliotecas maduras de AG em Python (DEAP, PyGAD) e de otimização de hiperparâmetros
(Optuna, Hyperopt). O enunciado pede "implementar um algoritmo genético" com codificação,
operadores de seleção/cruzamento/mutação e função fitness definidos pelo aluno.

## Decisão

Implementar o AG do zero com `numpy` + Python padrão. Sem DEAP, PyGAD, Optuna ou similares.

## Justificativa

**O requisito avaliado é a implementação.** A Aula 4 de AG é inteiramente sobre *"programar um
algoritmo genético do zero"*, e o relatório precisa descrever meus operadores. Usar biblioteca
pronta esvaziaria exatamente a parte avaliada.

**Controle sobre o que o enunciado exige explicitar.** O enunciado pede "definir uma codificação
adequada" e "implementar operadores de seleção, cruzamento e mutação". Com DEAP eu configuraria
operadores existentes; aqui eu os escrevo e posso justificar cada escolha — inclusive divergir
do material quando houver razão (ver ADR 005).

**Instrumentação sob medida.** As métricas de AG da Aula 3 (entropia genética, distância
euclidiana entre soluções, nº de soluções únicas) precisam enxergar a estrutura do cromossomo
híbrido. Com biblioteca, eu estaria adaptando a instrumentação à representação dela.

## Consequências

**Positivas**
- 205 testes cobrindo invariantes dos operadores — possível porque o código é meu e pequeno.
- Dois bugs reais encontrados e corrigidos (torneio com reposição; chave de cache por genótipo)
  que eu não teria autoridade para encontrar numa biblioteca.

**Negativas**
- Risco de reimplementar errado. Mitigado por: validar entropia contra o exemplo numérico da
  própria Aula 3 (H ≈ 1,585), testar invariantes (população constante, elitismo preserva o
  melhor, torneio prefere fitness maior) e comparar contra **busca aleatória** — se o AG
  estivesse quebrado, perderia feio para ela, não empataria.
- Sem NSGA-II pronto, a frente de Pareto (extensão opcional) daria mais trabalho.
