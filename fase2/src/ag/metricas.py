"""Métricas de desempenho do AG.

Estas métricas medem o ALGORITMO, não a solução — são o que permite explicar *por que* o AG
convergiu (ou não), em vez de só reportar o fitness final.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from src.ag.cromossomo import FAMILIAS, SCHEMA, chave_fenotipo, chave_genotipo

Individuo = dict[str, Any]


# ---------------------------------------------------------------------------
# Convergência (melhor / média / desvio padrão do fitness)
# ---------------------------------------------------------------------------


def melhor_fitness(fitnesses: list[float]) -> float:
    return max(fitnesses)


def media_fitness(fitnesses: list[float]) -> float:
    return float(np.mean(fitnesses))


def desvio_padrao_fitness(fitnesses: list[float]) -> float:
    return float(np.std(fitnesses))


# ---------------------------------------------------------------------------
# Diversidade — entropia genética
# ---------------------------------------------------------------------------


def entropia_alelo(valores: list[Any]) -> float:
    """Entropia genética de Shannon sobre uma lista de valores categóricos (um "gene" da
    população): H = -Σ pᵢ·log2(pᵢ), onde pᵢ é a proporção de cada valor distinto.

    H=0 quando todos os valores são iguais (população clonal nesse gene); H=log2(n) quando
    há n valores distintos igualmente frequentes (diversidade máxima).

    Validado no teste contra um caso fechado: 3 valores distintos e equiprováveis
    → H = log2(3) ≈ 1,585.
    """
    if not valores:
        return 0.0
    contagem = Counter(valores)
    n = len(valores)
    probabilidades = [c / n for c in contagem.values()]
    return -sum(p * math.log2(p) for p in probabilidades if p > 0)


def entropia_genetica_familia(populacao: list[Individuo]) -> float:
    """Entropia genética do gene `familia` (o "gene 0") — a métrica mais direta de o quanto
    o AG já convergiu para um subconjunto restrito de famílias de modelo.
    """
    return entropia_alelo([ind["familia"] for ind in populacao])


def entropia_genetica_media_categorica(populacao: list[Individuo]) -> float:
    """Média da entropia genética sobre TODOS os genes categóricos do esquema (incluindo o
    gene `familia`), calculada apenas entre os indivíduos cuja família ativa é a dona
    daquele gene — cobre a diversidade latente além do gene 0 (entropia e
    distância euclidiana olham partes diferentes do cromossomo híbrido).
    """
    entropias = [entropia_genetica_familia(populacao)]
    for familia, genes in SCHEMA.items():
        individuos_da_familia = [ind for ind in populacao if ind["familia"] == familia]
        if len(individuos_da_familia) < 2:
            continue
        for nome, spec in genes.items():
            if spec["tipo"] == "categorical":
                valores = [ind[familia][nome] for ind in individuos_da_familia]
                entropias.append(entropia_alelo(valores))
    return float(np.mean(entropias)) if entropias else 0.0


# ---------------------------------------------------------------------------
# Diversidade — distância euclidiana nos genes reais
# ---------------------------------------------------------------------------


def _genes_numericos_normalizados(individuo: Individuo, familia: str) -> np.ndarray:
    """Vetor dos genes numéricos (int/real/real_log) da família ativa, normalizados para
    [0, 1] usando os limites do SCHEMA. `int_or_none` com valor None é tratado como o
    extremo inferior da faixa (convenção documentada, não um valor "verdadeiro").
    """
    valores = []
    for nome, spec in SCHEMA[familia].items():
        tipo = spec["tipo"]
        if tipo not in ("int", "int_odd", "real", "real_log", "int_or_none"):
            continue
        valor = individuo[familia][nome]
        if valor is None:
            valor = spec["low"]
        if tipo == "real_log":
            low, high = math.log10(spec["low"]), math.log10(spec["high"])
        else:
            low, high = spec["low"], spec["high"]
        normalizado = (valor - low) / (high - low) if high > low else 0.0
        valores.append(normalizado)
    return np.array(valores, dtype=float)


def distancia_euclidiana_media(populacao: list[Individuo]) -> float:
    """Distância euclidiana média entre pares de indivíduos DA MESMA família ativa, nos
    genes numéricos normalizados.

    Por quê restringir a pares da mesma família: não há escala comum entre, por exemplo,
    `n_estimators` do Random Forest e `C` do SVM — comparar indivíduos de famílias
    diferentes produziria uma distância sem significado. Agrega pela média simples sobre
    todos os pares elegíveis de todas as famílias presentes.
    """
    por_familia: dict[str, list[Individuo]] = {}
    for ind in populacao:
        por_familia.setdefault(ind["familia"], []).append(ind)

    distancias: list[float] = []
    for familia, individuos in por_familia.items():
        if len(individuos) < 2:
            continue
        vetores = [_genes_numericos_normalizados(ind, familia) for ind in individuos]
        for v1, v2 in combinations(vetores, 2):
            distancias.append(float(np.linalg.norm(v1 - v2)))

    return float(np.mean(distancias)) if distancias else 0.0


# ---------------------------------------------------------------------------
# Diversidade — soluções únicas
# ---------------------------------------------------------------------------


def numero_solucoes_unicas(populacao: list[Individuo]) -> int:
    """Conta quantos FENÓTIPOS distintos existem na população — ou seja, quantos modelos
    efetivamente diferentes ela representa. Um declínio ao longo das gerações indica
    convergência para um grupo restrito de soluções.

    Usa `chave_fenotipo`, não o genótipo completo: dois indivíduos que diferem apenas em
    genes latentes treinam exatamente o mesmo modelo e têm o mesmo fitness, então contá-los
    como duas soluções distintas superestimaria a diversidade e esconderia justamente o
    sinal de convergência prematura que esta métrica existe para detectar.
    """
    return len({chave_fenotipo(ind) for ind in populacao})


def numero_genotipos_unicos(populacao: list[Individuo]) -> int:
    """Conta genótipos completos distintos (incluindo genes latentes).

    Comparar este número com `numero_solucoes_unicas` mostra quanta diversidade da população
    é apenas latente — informativo para discutir o papel do reservatório de genes recessivos
   , mas não é a métrica de convergência.
    """
    return len({chave_genotipo(ind) for ind in populacao})


# ---------------------------------------------------------------------------
# Rastreamento específico: o alelo que a fitness deveria favorecer
# ---------------------------------------------------------------------------


def frequencia_class_weight_balanced(populacao: list[Individuo]) -> float:
    """Fração da população — entre os indivíduos cuja família ativa TEM o gene
    `class_weight` (todas exceto KNN) — cujo valor é `"balanced"`.

    Esta é a predição verificável: como o fitness pesa recall, este alelo
    deveria se fixar na população ao longo das gerações. Rastrear isso é o que permite
    mostrar que o AG faz o que se previu que faria, em vez de apenas afirmar.
    """
    elegiveis = [ind for ind in populacao if "class_weight" in SCHEMA[ind["familia"]]]
    if not elegiveis:
        return 0.0
    balanceados = sum(
        1 for ind in elegiveis if ind[ind["familia"]]["class_weight"] == "balanced"
    )
    return balanceados / len(elegiveis)


def distribuicao_familias(populacao: list[Individuo]) -> dict[str, float]:
    """Proporção da população em cada família de modelo — para visualizar se o AG converge
    para Random Forest (como a Fase 1 sugeriria) ou se outra família ultrapassa quando
    todas são bem ajustadas.
    """
    contagem = Counter(ind["familia"] for ind in populacao)
    n = len(populacao)
    return {familia: contagem.get(familia, 0) / n for familia in FAMILIAS}
