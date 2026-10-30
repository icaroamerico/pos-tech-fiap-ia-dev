# ADR 004 — AG implementado do zero, sem DEAP/PyGAD

**Status:** aceita · **Data:** 08/10/2026

## Contexto

Existem bibliotecas maduras de AG em Python (DEAP, PyGAD) e de otimização de hiperparâmetros
(Optuna, Hyperopt). O enunciado pede "implementar um algoritmo genético" com codificação,
operadores de seleção/cruzamento/mutação e função fitness definidos pelo aluno.

## Decisão

Implementar o AG do zero com `numpy` + Python padrão. Sem DEAP, PyGAD, Optuna ou similares.

## Justificativa

**O requisito avaliado é a implementação.** O enunciado pede um algoritmo genético programado
do zero, e o relatório precisa descrever os operadores. Usar biblioteca pronta esvaziaria
exatamente a parte avaliada.

**Controle sobre o que o enunciado exige explicitar.** O enunciado pede "definir uma codificação
adequada" e "implementar operadores de seleção, cruzamento e mutação". Com DEAP eu configuraria
operadores existentes; aqui eu os escrevo e posso justificar cada escolha — inclusive divergir
da formulação canônica quando houver razão (ver ADR 005).

**Instrumentação sob medida.** As métricas de diversidade (entropia genética, distância
euclidiana entre soluções, nº de soluções únicas) precisam enxergar a estrutura do cromossomo
híbrido. Com biblioteca, eu estaria adaptando a instrumentação à representação dela.

## Consequências

**Positivas**
- 205 testes cobrindo invariantes dos operadores — possível porque o código é meu e pequeno.
- Dois bugs reais encontrados e corrigidos (torneio com reposição; chave de cache por genótipo)
  que eu não teria autoridade para encontrar numa biblioteca.

**Negativas**
- Risco de reimplementar errado. Mitigado por: validar entropia contra um valor fechado
  analiticamente (3 alelos equiprováveis → H = log2(3) ≈ 1,585), testar invariantes
  (população constante, elitismo preserva o
  melhor, torneio prefere fitness maior) e comparar contra **busca aleatória** — se o AG
  estivesse quebrado, perderia feio para ela, não empataria.
- Sem NSGA-II pronto, a frente de Pareto (extensão opcional) daria mais trabalho.
