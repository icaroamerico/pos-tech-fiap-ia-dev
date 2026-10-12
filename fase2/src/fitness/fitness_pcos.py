"""Fitness do AG para o PCOS (plano §4.3).

GARANTIA ANTI-VAZAMENTO (plano §3 — a regra mais importante do projeto): este módulo NUNCA
recebe nem acessa o conjunto de TESTE. `DadosTreino` não tem nenhum campo que carregue
X_teste/y_teste — essa ausência é estrutural, não uma convenção de código que alguém possa
violar por engano. A validação é feita inteiramente por StratifiedKFold (k=5) sobre as 432
linhas de treino. O teste só é tocado, uma única vez, fora deste módulo, na avaliação final.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.ag.cromossomo import decodificar
from src.modelagem.treinamento import (
    SEED,
    carregar_dados,
    criar_preprocessador,
    dividir_dados,
    identificar_colunas_tipo,
    preparar_features,
)

CONSTRUTORES = {
    "random_forest": RandomForestClassifier,
    "svm_rbf": SVC,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
}

# Pesos do fitness composto (plano §4.3): F1 como objetivo balanceado principal, recall com
# peso extra para empurrar na direção da fraqueza conhecida do baseline da Fase 1 (o Random
# Forest vencedor teve recall 0,8056 — sua métrica mais fraca), e AUC para desempatar
# indivíduos com F1 parecido.
PESO_F1 = 0.5
PESO_RECALL = 0.3
PESO_AUC = 0.2
N_FOLDS = 5


@dataclass(frozen=True)
class DadosTreino:
    """Carrega SOMENTE o que o AG tem permissão de ver. Não existe campo `X_teste` ou
    `y_teste` nesta classe — por construção, não há atributo para acessar por engano.
    """

    X_treino: pd.DataFrame
    y_treino: pd.Series
    colunas_numericas: list[str]
    colunas_categoricas: list[str]


def carregar_dados_treino() -> DadosTreino:
    """Reaproveita o pipeline de dados da Fase 1 (src/modelagem/treinamento.py, não
    modificado) para obter EXATAMENTE o mesmo split 80/20 e a mesma semente (SEED=42) que
    produziram os números documentados — a comparação "antes x depois" fica sobre o mesmo
    recorte de dados.

    As variáveis de teste retornadas por `dividir_dados` são imediatamente descartadas
    (nomeadas com `_NAO_USAR` para deixar isso explícito a quem ler o código) e nunca saem
    desta função.
    """
    df = carregar_dados()
    X, y = preparar_features(df)
    colunas_numericas, colunas_categoricas = identificar_colunas_tipo(X)
    X_treino, _X_teste_NAO_USAR, y_treino, _y_teste_NAO_USAR = dividir_dados(X, y)
    return DadosTreino(
        X_treino=X_treino,
        y_treino=y_treino,
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
    )


def construir_pipeline(
    individuo: dict[str, Any], dados: DadosTreino, n_jobs_modelo: int = 1
) -> Pipeline:
    """Decodifica o indivíduo e monta um Pipeline (pré-processador + modelo).

    `n_jobs_modelo=1` por padrão: a paralelização do AG acontece no NÍVEL DO INDIVÍDUO —
    vários indivíduos avaliados em paralelo via joblib (em motor.py), não dentro de cada
    Random Forest. Combinar os dois níveis de paralelismo causa *oversubscription* de
    threads e fica MAIS LENTO (plano §5.2); é por isso que o default aqui é 1, não -1.
    """
    familia, kwargs = decodificar(individuo)
    construtor = CONSTRUTORES[familia]

    kwargs_finais = dict(kwargs)
    if familia == "random_forest":
        kwargs_finais["n_jobs"] = n_jobs_modelo
    if familia != "knn":
        kwargs_finais["random_state"] = SEED

    preprocessador = criar_preprocessador(dados.colunas_numericas, dados.colunas_categoricas)
    return Pipeline(
        steps=[("preprocessador", preprocessador), ("modelo", construtor(**kwargs_finais))]
    )


def calcular_fitness(
    individuo: dict[str, Any], dados: DadosTreino, n_jobs_modelo: int = 1
) -> dict[str, float]:
    """Calcula o fitness de um indivíduo por validação cruzada estratificada (k=5) SOMENTE
    sobre `dados.X_treino` / `dados.y_treino`.

    fitness = 0,5·F1 + 0,3·recall + 0,2·AUC, média sobre os 5 folds.

    Determinístico dado o mesmo indivíduo: `StratifiedKFold` usa `random_state=SEED` fixo
    (os mesmos 5 folds para todo indivíduo, toda geração — remove ruído de amostragem como
    fator de confusão ao comparar fitness entre indivíduos) e cada modelo recebe
    `random_state=SEED` (exceto KNN, que não tem componente estocástico).
    """
    inicio = time.perf_counter()

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    f1_por_fold: list[float] = []
    recall_por_fold: list[float] = []
    auc_por_fold: list[float] = []

    for idx_treino, idx_val in cv.split(dados.X_treino, dados.y_treino):
        X_tr = dados.X_treino.iloc[idx_treino]
        X_val = dados.X_treino.iloc[idx_val]
        y_tr = dados.y_treino.iloc[idx_treino]
        y_val = dados.y_treino.iloc[idx_val]

        pipeline = construir_pipeline(individuo, dados, n_jobs_modelo=n_jobs_modelo)
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)

        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_val)[:, 1]
        else:
            y_proba = pipeline.decision_function(X_val)

        f1_por_fold.append(f1_score(y_val, y_pred, zero_division=0))
        recall_por_fold.append(recall_score(y_val, y_pred, zero_division=0))
        try:
            auc_por_fold.append(roc_auc_score(y_val, y_proba))
        except ValueError:
            # fold degenerado (y_val com uma classe só) — AUC não é informativa; 0.5 é o
            # valor de um classificador aleatório, neutro no cálculo da média.
            auc_por_fold.append(0.5)

    f1_medio = float(np.mean(f1_por_fold))
    recall_medio = float(np.mean(recall_por_fold))
    auc_medio = float(np.mean(auc_por_fold))
    fitness = PESO_F1 * f1_medio + PESO_RECALL * recall_medio + PESO_AUC * auc_medio

    return {
        "fitness": fitness,
        "f1_medio": f1_medio,
        "recall_medio": recall_medio,
        "auc_medio": auc_medio,
        "tempo_s": time.perf_counter() - inicio,
    }


def calcular_fitness_com_cache(
    individuo: dict[str, Any],
    dados: DadosTreino,
    cache: dict[tuple, dict[str, float]],
    n_jobs_modelo: int = 1,
) -> tuple[dict[str, float], bool]:
    """Memoiza o fitness por `chave_fenotipo` (plano §5.2) — o análogo, para hiperparâmetros,
    da matriz de distâncias pré-calculada da Aula 4 de AG: evita retreinar um indivíduo que
    o elitismo ou a mutação fraca já fizeram repetir.

    A chave é o FENÓTIPO (família ativa + genes expressos), não o genótipo completo. Usar o
    genótipo faria o cache perder quase todo acerto: os genes latentes das famílias inativas
    derivam a cada mutação sem alterar o modelo treinado, então dois indivíduos com fitness
    comprovadamente idêntico teriam chaves diferentes.

    Só funciona porque `calcular_fitness` é determinístico (ver docstring acima) — um cache
    sobre uma função não-determinística devolveria valores enganosos.

    Retorna `(resultado, veio_do_cache)` — o segundo valor alimenta a métrica de eficiência
    `n_cache_hits` do logging (plano §5.1).
    """
    from src.ag.cromossomo import chave_fenotipo  # import local evita ciclo de módulos

    chave = chave_fenotipo(individuo)
    if chave in cache:
        return cache[chave], True

    resultado = calcular_fitness(individuo, dados, n_jobs_modelo=n_jobs_modelo)
    cache[chave] = resultado
    return resultado, False
