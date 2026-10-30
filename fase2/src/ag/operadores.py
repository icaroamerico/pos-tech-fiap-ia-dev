"""Operadores genéticos — seleção, cruzamento e mutação.

Cruzamento e mutação são despachados POR TIPO DE GENE, não por família: cada gene sabe como
se cruzar e mutar consigo mesmo, independentemente de pertencer ao Random Forest ou ao SVM.
Ou seja: a codificação híbrida também tem crossover híbrido.

Dois operadores de seleção estão implementados, não só um: o AG usa **torneio** como
operador principal (justificativa: o fitness vive numa faixa estreita, ~0.75–0.92, e a
roleta proporcional dá pressão seletiva quase nula nessa faixa), mas mantém **roleta** para a
Etapa D — uma ablação que mede essa alegação em vez de só afirmá-la.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from src.ag.cromossomo import FAMILIAS, SCHEMA, clamp_individuo

Individuo = dict[str, Any]


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------


def selecao_torneio(
    populacao: list[Individuo], fitnesses: list[float], k: int, rng: np.random.Generator
) -> Individuo:
    """Sorteia k indivíduos e retorna o de maior fitness (maximização).

    Invariante a escalas do fitness — só compara, nunca pondera por magnitude. É por isso
    que funciona bem mesmo com o fitness confinado a uma faixa estreita.

    Os k participantes são sorteados SEM reposição (o padrão em seleção por torneio): com
    reposição, um torneio k=2 sobre uma população pequena poderia sortear o mesmo indivíduo
    duas vezes e nunca colocar o segundo candidato em disputa.
    """
    k = min(k, len(populacao))  # nunca pede mais participantes do que a população tem
    indices = rng.choice(len(populacao), size=k, replace=False)
    melhor_idx = max(indices, key=lambda i: fitnesses[i])
    return populacao[melhor_idx]


def selecao_roleta(
    populacao: list[Individuo], fitnesses: list[float], rng: np.random.Generator
) -> Individuo:
    """Seleção proporcional ao fitness — roleta clássica, adaptada para maximização.
    Na formulação usual de minimização o peso é `1/custo`; aqui o fitness já é "quanto
    maior, melhor", então os pesos são o próprio vetor de fitness.

    Mantida apenas para a ablação da Etapa D — não é o operador padrão do AG.
    """
    pesos = np.asarray(fitnesses, dtype=float)
    pesos = np.clip(pesos, a_min=1e-12, a_max=None)  # evita soma zero
    probabilidades = pesos / pesos.sum()
    idx = rng.choice(len(populacao), p=probabilidades)
    return populacao[idx]


# ---------------------------------------------------------------------------
# Cruzamento
# ---------------------------------------------------------------------------


def _cruzar_valor_numerico(v1: Any, v2: Any, rng: np.random.Generator) -> tuple[Any, Any]:
    """Arithmetic crossover: filho = alpha*pai1 + (1-alpha)*pai2,
    com alpha sorteado por gene. Cobre int, int_odd, real e real_log — para estes últimos,
    o valor já está em log-space, então a média aritmética em log corresponde à média
    geométrica no espaço real, que é a escala correta para C e gamma.
    """
    alpha = rng.uniform(0.0, 1.0)
    filho1 = alpha * v1 + (1 - alpha) * v2
    filho2 = alpha * v2 + (1 - alpha) * v1
    return filho1, filho2


def _cruzar_gene(v1: Any, v2: Any, spec: dict[str, Any], rng: np.random.Generator) -> tuple[Any, Any]:
    tipo = spec["tipo"]
    if tipo in ("int", "int_odd", "real", "real_log"):
        return _cruzar_valor_numerico(v1, v2, rng)
    if tipo == "int_or_none":
        if v1 is None or v2 is None:
            # Não há média aritmética sensata com None — uniform crossover (herda de um pai)
            return (v1, v2) if rng.random() < 0.5 else (v2, v1)
        return _cruzar_valor_numerico(v1, v2, rng)
    if tipo == "categorical":
        # Uniform crossover: cada filho herda, por gene, de um dos dois pais
        return (v1, v2) if rng.random() < 0.5 else (v2, v1)
    raise ValueError(f"Tipo de gene desconhecido: {tipo!r}")


def cruzar(
    pai1: Individuo, pai2: Individuo, taxa_crossover: float, rng: np.random.Generator
) -> tuple[Individuo, Individuo]:
    """Cruza dois indivíduos, gene a gene, despachando o operador certo por tipo.

    Com probabilidade `1 - taxa_crossover`, os filhos são cópias exatas dos pais — a
    taxa de crossover é lida como "probabilidade de o cruzamento ocorrer", não como
    fração do cromossomo trocada.
    """
    if rng.random() >= taxa_crossover:
        return copy.deepcopy(pai1), copy.deepcopy(pai2)

    # A família ativa (gene 0) é tratada como categórico: cada filho herda de um pai.
    familia_f1, familia_f2 = (
        (pai1["familia"], pai2["familia"])
        if rng.random() < 0.5
        else (pai2["familia"], pai1["familia"])
    )
    filho1: Individuo = {"familia": familia_f1}
    filho2: Individuo = {"familia": familia_f2}

    for familia, genes in SCHEMA.items():
        filho1[familia] = {}
        filho2[familia] = {}
        for nome, spec in genes.items():
            v1, v2 = pai1[familia][nome], pai2[familia][nome]
            f1, f2 = _cruzar_gene(v1, v2, spec, rng)
            filho1[familia][nome] = f1
            filho2[familia][nome] = f2

    return clamp_individuo(filho1), clamp_individuo(filho2)


# ---------------------------------------------------------------------------
# Mutação
# ---------------------------------------------------------------------------


def _mutar_gene(valor: Any, spec: dict[str, Any], intensidade: float, rng: np.random.Generator) -> Any:
    tipo = spec["tipo"]
    if tipo in ("int", "int_odd"):
        amplitude = spec["high"] - spec["low"]
        return valor + rng.normal(0, intensidade * amplitude)
    if tipo == "real":
        amplitude = spec["high"] - spec["low"]
        return valor + rng.normal(0, intensidade * amplitude)
    if tipo == "real_log":
        log_low, log_high = math.log10(spec["low"]), math.log10(spec["high"])
        amplitude = log_high - log_low
        return valor + rng.normal(0, intensidade * amplitude)
    if tipo == "int_or_none":
        if valor is None:
            # Reabre a possibilidade de voltar a ser um número — resorteio como na
            # inicialização (30% de chance de permanecer None).
            if rng.random() < 0.3:
                return None
            return rng.integers(spec["low"], spec["high"] + 1)
        amplitude = spec["high"] - spec["low"]
        return valor + rng.normal(0, intensidade * amplitude)
    if tipo == "categorical":
        # Resorteio uniforme entre os valores válidos DIFERENTES do atual.
        outros = [c for c in spec["choices"] if c != valor]
        if not outros:
            return valor
        idx = rng.integers(0, len(outros))
        return outros[idx]
    raise ValueError(f"Tipo de gene desconhecido: {tipo!r}")


#: Taxa de troca de família padrão — desacoplada da taxa de mutação dos hiperparâmetros.
#: Ver a justificativa em `mutar()`.
PROB_TROCA_FAMILIA_PADRAO = 0.05


def mutar(
    individuo: Individuo,
    probabilidade: float,
    intensidade: float,
    rng: np.random.Generator,
    prob_troca_familia: float | None = None,
    normalizar_por_familia: bool = True,
) -> Individuo:
    """Aplica mutação ao indivíduo.

    `probabilidade` e `intensidade` são independentes: o primeiro é a chance de um gene
    mutar, o segundo é a força da mutação gaussiana aplicada quando ele muta.

    Duas decisões de design, tomadas após revisão dos primeiros experimentos:

    **1. `prob_troca_familia` é separada de `probabilidade`.**
    Trocar a família ativa é uma mudança de natureza completamente diferente de ajustar um
    hiperparâmetro: troca o modelo inteiro, descartando todo o refinamento acumulado naquela
    família. Com as duas taxas unificadas, a chance de uma linhagem manter sua família ao
    longo de 15 gerações caía para 8,7% (p=0,15) ou 0,1% (p=0,35) — a escolha do modelo virava
    passeio aleatório e o AG nunca conseguia se fixar numa família e refiná-la. Mantendo a
    troca em 0,05 por padrão, a persistência sobe para 46%, o que permite refinamento sem
    congelar a exploração entre famílias. Default `None` significa `PROB_TROCA_FAMILIA_PADRAO`.

    **2. `normalizar_por_familia` iguala a pressão de mutação entre famílias.**
    A mutação é aplicada a todos os genes, inclusive os latentes das famílias inativas (o
    "reservatório recessivo"). Mas só os genes da família ATIVA afetam o
    fitness, e as famílias têm números diferentes de genes (3 no SVM, 7 no Random Forest).
    Sem normalização, um indivíduo Random Forest sofria 2,3x mais mutação efetiva que um SVM
    à mesma `probabilidade` — uma assimetria de pressão evolutiva que não foi escolhida por
    ninguém. Com `normalizar_por_familia=True`, a probabilidade por gene é escalada para que
    o número ESPERADO de mutações nos genes expressos seja o mesmo em qualquer família.
    """
    novo = copy.deepcopy(individuo)

    if prob_troca_familia is None:
        prob_troca_familia = PROB_TROCA_FAMILIA_PADRAO

    if rng.random() < prob_troca_familia:
        outras_familias = [f for f in FAMILIAS if f != novo["familia"]]
        novo["familia"] = outras_familias[rng.integers(0, len(outras_familias))]

    # Alvo de mutações esperadas nos genes expressos, calibrado pela família de referência
    # (a maior, Random Forest). Mantém `probabilidade` com a mesma ordem de grandeza de
    # significado que tinha antes, só equaliza entre famílias.
    n_genes_referencia = max(len(genes) for genes in SCHEMA.values())

    for familia, genes in SCHEMA.items():
        if normalizar_por_familia:
            # p_efetiva * n_genes_da_familia == probabilidade * n_genes_referencia
            p_efetiva = min(1.0, probabilidade * n_genes_referencia / len(genes))
        else:
            p_efetiva = probabilidade
        for nome, spec in genes.items():
            if rng.random() < p_efetiva:
                novo[familia][nome] = _mutar_gene(novo[familia][nome], spec, intensidade, rng)

    return clamp_individuo(novo)
