"""Inicialização da população e elitismo (plano §4.2 e §4.4)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from src.ag.cromossomo import FAMILIAS, gerar_individuo_aleatorio, gerar_individuo_hotstart

Individuo = dict[str, Any]


def inicializar_populacao(
    tamanho: int, rng: np.random.Generator, fracao_hotstart: float = 0.1
) -> list[Individuo]:
    """Gera a população inicial: ~90% aleatória + ~10% hotstart com os hiperparâmetros da
    Fase 1 (plano §4.2).

    Os indivíduos de hotstart são distribuídos entre as 5 famílias (um por família, até o
    número de vagas de hotstart permitir) — o melhor fitness da geração 0 fica, por
    construção, ≥ o desempenho da Fase 1 **para as famílias efetivamente cobertas**.

    ⚠️ LIMITAÇÃO CONHECIDA (relatório técnico §2.3 e §7, achado nº 9): com
    `n_hotstart = round(tamanho * fracao_hotstart)`, uma população de 30 cobre apenas as 3
    primeiras famílias de FAMILIAS, e uma de 15 cobre 2. Pior: quando `ConfigAG.familia_fixa`
    está definida, `executar_ag` sobrescreve `individuo["familia"]` DEPOIS desta chamada, então
    os hiperparâmetros de hotstart expressos passam a ser os de outra família — e as execuções
    da Etapa A para logistic_regression, decision_tree e knn não têm nenhum indivíduo com a
    configuração da Fase 1. Nas execuções gravadas a geração 0 superou o hotstart em todas as
    5 famílias de qualquer forma, então nenhum resultado é inválido; mas a garantia valeu por
    sorte, não por construção. Correção: aceitar a família a semear como parâmetro e priorizá-la
    quando `familia_fixa` estiver definida.
    """
    n_hotstart = round(tamanho * fracao_hotstart)
    n_aleatorio = tamanho - n_hotstart

    populacao = [gerar_individuo_aleatorio(rng) for _ in range(n_aleatorio)]
    for i in range(n_hotstart):
        familia = FAMILIAS[i % len(FAMILIAS)]
        populacao.append(gerar_individuo_hotstart(familia, rng))

    rng.shuffle(populacao)  # evita que a ordem revele quem é hotstart
    return populacao


def aplicar_elitismo(
    populacao_ordenada: list[Individuo], fitnesses_ordenados: list[float], n_elite: int
) -> list[Individuo]:
    """Retorna cópias dos `n_elite` melhores indivíduos de uma população já ordenada por
    fitness decrescente (maior primeiro).

    Chamar sempre com a população ORDENADA — esta função não ordena, só recorta. É a mesma
    lógica de `new_population = [population[0]]` da Aula 4 de AG, generalizada para n_elite
    indivíduos: garante que o melhor fitness nunca piora entre gerações (plano §4.4).
    """
    if n_elite <= 0:
        return []
    return [copy.deepcopy(ind) for ind in populacao_ordenada[:n_elite]]
