"""Codificação híbrida dos indivíduos do AG.

Um indivíduo é um dicionário com:
    {"familia": <uma das FAMILIAS>, "random_forest": {...}, "svm_rbf": {...}, ...}

Todas as famílias carregam genes simultaneamente (latentes se não ativas) para manter o
indivíduo de tamanho fixo — crossover e mutação não precisam saber qual família está ativa.
Funciona como "material genético recessivo": se o gene `familia` mutar, os
hiperparâmetros da nova família já estão lá, prontos para serem expressos.

Cada gene tem um `tipo` que determina como ele é amostrado, mutado e cruzado
(ver src/ag/operadores.py para os operadores):

    int        — inteiro uniforme em [low, high]
    int_odd    — inteiro ímpar em [low, high] (KNN n_neighbors; evita empate binário)
    int_or_none— inteiro em [low, high], ou None (max_depth irrestrito)
    real       — real uniforme em [low, high]
    real_log   — real cujo valor INTERNO é log10(valor real); decodificado via 10**valor.
                 Crossover aritmético em log-space equivale a média geométrica no espaço real,
                 que é a escala certa para C e gamma.
    categorical— um dos valores em `choices` (pode ser string, número, None, ou tupla)
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

FAMILIAS = [
    "random_forest",
    "svm_rbf",
    "logistic_regression",
    "decision_tree",
    "knn",
]

# Pares (penalty, solver) válidos para a Regressão Logística — herdados como unidade atômica
# no crossover, nunca recombinados entre si (evita indivíduo inválido por
# construção).
PENALTY_SOLVER_CHOICES = [("l2", "lbfgs"), ("l1", "liblinear"), ("elasticnet", "saga")]

SCHEMA: dict[str, dict[str, dict[str, Any]]] = {
    "random_forest": {
        "n_estimators": {"tipo": "int", "low": 50, "high": 300},
        "max_depth": {"tipo": "int_or_none", "low": 2, "high": 20},
        "min_samples_split": {"tipo": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"tipo": "int", "low": 1, "high": 10},
        "max_features": {"tipo": "categorical", "choices": ["sqrt", "log2", 0.3, 0.5, 0.8]},
        "criterion": {"tipo": "categorical", "choices": ["gini", "entropy"]},
        "class_weight": {"tipo": "categorical", "choices": [None, "balanced"]},
    },
    "svm_rbf": {
        "C": {"tipo": "real_log", "low": 1e-2, "high": 1e3},
        "gamma": {"tipo": "real_log", "low": 1e-4, "high": 1e1},
        "class_weight": {"tipo": "categorical", "choices": [None, "balanced"]},
    },
    "logistic_regression": {
        "C": {"tipo": "real_log", "low": 1e-3, "high": 1e2},
        "penalty_solver": {"tipo": "categorical", "choices": PENALTY_SOLVER_CHOICES},
        "l1_ratio": {"tipo": "real", "low": 0.0, "high": 1.0},
        "class_weight": {"tipo": "categorical", "choices": [None, "balanced"]},
    },
    "decision_tree": {
        "max_depth": {"tipo": "int_or_none", "low": 2, "high": 20},
        "min_samples_split": {"tipo": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"tipo": "int", "low": 1, "high": 10},
        "criterion": {"tipo": "categorical", "choices": ["gini", "entropy"]},
        "ccp_alpha": {"tipo": "real", "low": 0.0, "high": 0.05},
        "class_weight": {"tipo": "categorical", "choices": [None, "balanced"]},
    },
    "knn": {
        "n_neighbors": {"tipo": "int_odd", "low": 1, "high": 31},
        "weights": {"tipo": "categorical", "choices": ["uniform", "distance"]},
        "metric": {"tipo": "categorical", "choices": ["euclidean", "manhattan", "minkowski"]},
        "p": {"tipo": "int", "low": 1, "high": 5},
    },
}

# Hiperparâmetros REAIS usados pela Fase 1 em cada família (extraídos de
# src/modelagem/treinamento.py, SEED=42) — a base dos indivíduos de hotstart.
#
# ⚠️ IMPORTANTE: os valores aqui estão em UNIDADES DO MUNDO REAL (C=1.0 significa C=1.0),
# propositalmente, para que esta tabela possa ser conferida linha a linha contra o código da
# Fase 1. A conversão para a representação INTERNA do cromossomo (log10, no caso dos genes
# `real_log`) é feita por `_codificar_gene`, chamada em `gerar_individuo_hotstart`.
# Guardar valores crus aqui e esquecer de codificá-los foi um bug real: C=1.0 era decodificado
# como 10**1.0 = 10, e gamma=0.1 como 10**0.1 = 1.26 — este último deixava o SVM com F1=0.
#
# Sobre `gamma`: o SVC da Fase 1 usa `gamma='scale'`, que não é um número fixo — o sklearn o
# calcula como 1/(n_features * X.var()). Medido sobre esta base, após o pré-processamento
# (53 features, variância 0.8676), 'scale' equivale a 0.021748. É esse o valor usado abaixo,
# e a derivação está registrada no teste `test_gamma_hotstart_equivale_ao_scale_do_sklearn`.
HOTSTART_FASE1: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "criterion": "gini",
        "class_weight": "balanced",
    },
    "svm_rbf": {
        "C": 1.0,  # default do SVC
        "gamma": 0.021748,  # equivalente numérico de gamma='scale' nesta base — ver nota acima
        "class_weight": "balanced",
    },
    "logistic_regression": {
        "C": 1.0,
        "penalty_solver": ("l2", "lbfgs"),
        "l1_ratio": 0.5,  # não expresso (penalty != elasticnet), valor é irrelevante
        "class_weight": "balanced",
    },
    "decision_tree": {
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": "gini",
        "ccp_alpha": 0.0,
        "class_weight": "balanced",
    },
    "knn": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
        "p": 2,
    },
}


def _amostrar_gene(spec: dict[str, Any], rng: np.random.Generator) -> Any:
    """Amostra um valor aleatório válido para um gene, conforme seu tipo."""
    tipo = spec["tipo"]
    if tipo == "int":
        return int(rng.integers(spec["low"], spec["high"] + 1))
    if tipo == "int_odd":
        low, high = spec["low"], spec["high"]
        n_impares = (high - low) // 2 + 1
        return int(low + 2 * rng.integers(0, n_impares))
    if tipo == "int_or_none":
        if rng.random() < 0.3:
            return None
        return int(rng.integers(spec["low"], spec["high"] + 1))
    if tipo == "real":
        return float(rng.uniform(spec["low"], spec["high"]))
    if tipo == "real_log":
        log_low, log_high = math.log10(spec["low"]), math.log10(spec["high"])
        return float(rng.uniform(log_low, log_high))
    if tipo == "categorical":
        idx = rng.integers(0, len(spec["choices"]))
        return spec["choices"][idx]
    raise ValueError(f"Tipo de gene desconhecido: {tipo!r}")


def gerar_individuo_aleatorio(rng: np.random.Generator) -> dict[str, Any]:
    """Gera um indivíduo com família e todos os genes (de todas as famílias) aleatórios."""
    individuo: dict[str, Any] = {"familia": FAMILIAS[rng.integers(0, len(FAMILIAS))]}
    for familia, genes in SCHEMA.items():
        individuo[familia] = {nome: _amostrar_gene(spec, rng) for nome, spec in genes.items()}
    return individuo


def _codificar_gene(valor: Any, spec: dict[str, Any]) -> Any:
    """Converte um valor em UNIDADES DO MUNDO REAL para a representação INTERNA do gene.

    É a inversa exata da etapa correspondente em `decodificar()`: hoje só os genes
    `real_log` têm representação interna diferente do valor real (guardam log10), mas manter
    esta função explícita garante que qualquer tipo futuro com codificação própria seja
    tratado nos dois sentidos.
    """
    if spec["tipo"] == "real_log":
        return math.log10(valor)
    return valor


def codificar_genes(familia: str, valores_reais: dict[str, Any]) -> dict[str, Any]:
    """Codifica um dicionário de hiperparâmetros em unidades reais para a representação
    interna do cromossomo, conforme o SCHEMA da família.
    """
    return {
        nome: _codificar_gene(valores_reais[nome], SCHEMA[familia][nome])
        for nome in SCHEMA[familia]
    }


def gerar_individuo_hotstart(familia: str, rng: np.random.Generator) -> dict[str, Any]:
    """Gera um indivíduo de hotstart: família fixa com os hiperparâmetros da Fase 1; as
    demais famílias recebem genes aleatórios (servem de reservatório de diversidade caso a
    família mute).

    Os valores de `HOTSTART_FASE1` estão em unidades do mundo real e são convertidos para a
    representação interna por `codificar_genes` — sem essa conversão, genes `real_log` são
    decodificados com erro de ordem de magnitude (ver nota em HOTSTART_FASE1).
    """
    if familia not in FAMILIAS:
        raise ValueError(f"Família desconhecida: {familia!r}")
    individuo = gerar_individuo_aleatorio(rng)
    individuo["familia"] = familia
    individuo[familia] = codificar_genes(familia, HOTSTART_FASE1[familia])
    return individuo


def _clamp_gene(valor: Any, spec: dict[str, Any]) -> Any:
    """Força um valor de gene a respeitar os limites/validade do seu tipo."""
    tipo = spec["tipo"]
    if tipo == "int":
        return int(np.clip(round(valor), spec["low"], spec["high"]))
    if tipo == "int_odd":
        v = int(np.clip(round(valor), spec["low"], spec["high"]))
        if v % 2 == 0:
            # empurra para o ímpar mais próximo sem sair da faixa
            v = v + 1 if v < spec["high"] else v - 1
        return v
    if tipo == "int_or_none":
        if valor is None:
            return None
        return int(np.clip(round(valor), spec["low"], spec["high"]))
    if tipo == "real":
        return float(np.clip(valor, spec["low"], spec["high"]))
    if tipo == "real_log":
        log_low, log_high = math.log10(spec["low"]), math.log10(spec["high"])
        return float(np.clip(valor, log_low, log_high))
    if tipo == "categorical":
        if valor in spec["choices"]:
            return valor
        # valor fora do conjunto (não deveria ocorrer por construção) — cai no primeiro válido
        return spec["choices"][0]
    raise ValueError(f"Tipo de gene desconhecido: {tipo!r}")


def clamp_individuo(individuo: dict[str, Any]) -> dict[str, Any]:
    """Retorna uma cópia do indivíduo com todos os genes de todas as famílias dentro dos
    limites válidos. Chamado sempre após crossover e mutação — é o que garante que o AG
    nunca produza um indivíduo inválido (validade por construção).
    """
    novo: dict[str, Any] = {"familia": individuo["familia"]}
    for familia, genes in SCHEMA.items():
        novo[familia] = {
            nome: _clamp_gene(individuo[familia][nome], spec) for nome, spec in genes.items()
        }
    return novo


def decodificar(individuo: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Traduz um indivíduo nos kwargs prontos para o construtor do sklearn da família ativa.

    Aplica as regras condicionais:
      - `l1_ratio` só é incluído quando `penalty == "elasticnet"`;
      - `p` (Minkowski) só é incluído quando `metric == "minkowski"`;
      - `penalty_solver` é desempacotado em `penalty` e `solver` separados.
    """
    familia = individuo["familia"]
    genes = individuo[familia]
    kwargs: dict[str, Any] = {}

    if familia == "random_forest":
        kwargs = {
            "n_estimators": genes["n_estimators"],
            "max_depth": genes["max_depth"],
            "min_samples_split": genes["min_samples_split"],
            "min_samples_leaf": genes["min_samples_leaf"],
            "max_features": genes["max_features"],
            "criterion": genes["criterion"],
            "class_weight": genes["class_weight"],
        }
    elif familia == "svm_rbf":
        kwargs = {
            "C": 10 ** genes["C"],
            "gamma": 10 ** genes["gamma"],
            "class_weight": genes["class_weight"],
            "kernel": "rbf",
            "probability": True,  # necessário para AUC
        }
    elif familia == "logistic_regression":
        penalty, solver = genes["penalty_solver"]
        kwargs = {
            "C": 10 ** genes["C"],
            "penalty": penalty,
            "solver": solver,
            "class_weight": genes["class_weight"],
            "max_iter": 2000,
        }
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = genes["l1_ratio"]
    elif familia == "decision_tree":
        kwargs = {
            "max_depth": genes["max_depth"],
            "min_samples_split": genes["min_samples_split"],
            "min_samples_leaf": genes["min_samples_leaf"],
            "criterion": genes["criterion"],
            "ccp_alpha": genes["ccp_alpha"],
            "class_weight": genes["class_weight"],
        }
    elif familia == "knn":
        kwargs = {
            "n_neighbors": genes["n_neighbors"],
            "weights": genes["weights"],
            "metric": genes["metric"],
        }
        if genes["metric"] == "minkowski":
            kwargs["p"] = genes["p"]
    else:
        raise ValueError(f"Família desconhecida: {familia!r}")

    return familia, kwargs


def chave_genotipo(individuo: dict[str, Any]) -> tuple:
    """Chave hashável do GENÓTIPO COMPLETO — inclui os genes latentes das famílias inativas.

    Dois indivíduos com a mesma chave de genótipo são idênticos em todo o cromossomo. Útil
    para detectar clones exatos, mas **não serve para cache de fitness nem para contagem de
    diversidade efetiva**: genes latentes não afetam o fitness, então dois indivíduos com
    genótipos diferentes podem ter exatamente o mesmo desempenho (use `chave_fenotipo`).
    """
    partes: list[Any] = [individuo["familia"]]
    for familia in FAMILIAS:
        genes = individuo[familia]
        partes.append(familia)
        partes.extend(genes[nome] for nome in sorted(genes))
    return tuple(partes)


def chave_fenotipo(individuo: dict[str, Any]) -> tuple:
    """Chave hashável do FENÓTIPO — só a família ativa e os genes que ela expressa.

    É esta a chave correta para:
      - **cache de fitness**: o fitness depende exclusivamente do fenótipo, então
        usar o genótipo completo como chave faz o cache perder quase todos os acertos, porque
        os genes latentes derivam livremente a cada mutação;
      - **contagem de soluções únicas**: a métrica deve medir diversidade
        *efetiva*. Com a chave de genótipo, uma população já convergida no fenótipo continua
        parecendo diversa, e o sinal de convergência prematura se perde.

    Respeita as mesmas regras condicionais de `decodificar()` — genes não expressos
    (`l1_ratio` sem elasticnet, `p` sem minkowski) ficam fora da chave, porque de fato não
    influenciam o modelo treinado.
    """
    familia, kwargs = decodificar(individuo)
    return (familia, *((nome, kwargs[nome]) for nome in sorted(kwargs)))


def validar_individuo(individuo: dict[str, Any]) -> bool:
    """Confere que todo gene de todo indivíduo está dentro do espaço de busca definido no
    SCHEMA. Usado nos testes para garantir que crossover e mutação nunca escapam da faixa.
    """
    if individuo["familia"] not in FAMILIAS:
        return False
    for familia, genes in SCHEMA.items():
        if familia not in individuo:
            return False
        for nome, spec in genes.items():
            if nome not in individuo[familia]:
                return False
            valor = individuo[familia][nome]
            tipo = spec["tipo"]
            if tipo in ("int", "int_odd"):
                if not isinstance(valor, int) or not (spec["low"] <= valor <= spec["high"]):
                    return False
                if tipo == "int_odd" and valor % 2 == 0:
                    return False
            elif tipo == "int_or_none":
                if valor is not None and (
                    not isinstance(valor, int) or not (spec["low"] <= valor <= spec["high"])
                ):
                    return False
            elif tipo == "real":
                if not (spec["low"] <= valor <= spec["high"]):
                    return False
            elif tipo == "real_log":
                log_low, log_high = math.log10(spec["low"]), math.log10(spec["high"])
                if not (log_low <= valor <= log_high):
                    return False
            elif tipo == "categorical":
                if valor not in spec["choices"]:
                    return False
    return True
