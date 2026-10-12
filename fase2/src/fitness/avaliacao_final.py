"""Avaliação final no conjunto de teste (plano §3) e baseline de busca aleatória (§4.8 C).

⚠️ Este é o ÚNICO módulo do projeto autorizado a tocar o conjunto de teste, e só depois que o
AG terminou. Está deliberadamente separado de `fitness_pcos.py` para que a fronteira seja
visível na estrutura de arquivos: o módulo que o AG usa não tem acesso ao holdout, e este,
que tem, não é importado por nenhum código do AG.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.fitness.fitness_pcos import DadosTreino, construir_pipeline
from src.modelagem.treinamento import (
    carregar_dados,
    dividir_dados,
    identificar_colunas_tipo,
    preparar_features,
)


def avaliar_no_holdout(individuo: dict[str, Any]) -> dict[str, Any]:
    """Treina o indivíduo no conjunto de treino COMPLETO e avalia nas 109 linhas de teste.

    Chamado uma única vez por modelo, no fim de tudo — é a "abertura do envelope" do plano §3.
    Reproduz exatamente o mesmo split da Fase 1 (SEED=42), então os números são diretamente
    comparáveis com a tabela do baseline.
    """
    df = carregar_dados()
    X, y = preparar_features(df)
    colunas_numericas, colunas_categoricas = identificar_colunas_tipo(X)
    X_treino, X_teste, y_treino, y_teste = dividir_dados(X, y)

    dados = DadosTreino(
        X_treino=X_treino,
        y_treino=y_treino,
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
    )

    pipeline = construir_pipeline(individuo, dados)
    pipeline.fit(X_treino, y_treino)
    y_pred = pipeline.predict(X_teste)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_teste)[:, 1]
    else:
        y_proba = pipeline.decision_function(X_teste)

    return {
        "familia": individuo["familia"],
        "acuracia": float(accuracy_score(y_teste, y_pred)),
        "precisao": float(precision_score(y_teste, y_pred, zero_division=0)),
        "recall": float(recall_score(y_teste, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_teste, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_teste, y_proba)),
        "matriz_confusao": confusion_matrix(y_teste, y_pred).tolist(),
        "n_teste": int(len(y_teste)),
    }


def busca_aleatoria(
    n_avaliacoes: int,
    dados: DadosTreino,
    semente: int = 42,
    familia_fixa: str | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Baseline da Etapa C (plano §4.8): sorteia `n_avaliacoes` indivíduos aleatórios e
    devolve o melhor, usando o MESMO fitness, o MESMO espaço de busca e o MESMO orçamento de
    avaliações que o AG.

    Sem esta comparação, "o AG melhorou o modelo" é ambíguo: melhorou por ser algoritmo
    genético, ou porque *qualquer* busca com N avaliações bateria o baseline? Só esta linha
    permite responder.
    """
    from joblib import Parallel, delayed

    from src.ag.cromossomo import chave_fenotipo, gerar_individuo_aleatorio
    from src.fitness.fitness_pcos import calcular_fitness

    rng = np.random.default_rng(semente)
    candidatos: list[dict[str, Any]] = []
    for _ in range(n_avaliacoes):
        individuo = gerar_individuo_aleatorio(rng)
        if familia_fixa is not None:
            individuo["familia"] = familia_fixa
        candidatos.append(individuo)

    # deduplicação por fenótipo, pelo mesmo motivo do cache do AG (plano §5.2)
    unicos: dict[tuple, dict[str, Any]] = {}
    for individuo in candidatos:
        unicos.setdefault(chave_fenotipo(individuo), individuo)

    chaves = list(unicos)
    resultados = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(calcular_fitness)(unicos[chave], dados, 1) for chave in chaves
    )

    melhor_idx = max(range(len(resultados)), key=lambda i: resultados[i]["fitness"])
    melhor_chave = chaves[melhor_idx]

    return {
        "melhor_individuo": unicos[melhor_chave],
        "melhor_fitness": resultados[melhor_idx]["fitness"],
        "melhor_metricas": resultados[melhor_idx],
        "n_avaliacoes_pedidas": n_avaliacoes,
        "n_avaliacoes_efetivas": len(unicos),
        "historico_fitness": [r["fitness"] for r in resultados],
    }
