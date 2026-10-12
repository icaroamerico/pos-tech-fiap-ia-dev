"""Funções de análise e gráficos (plano §4.9).

Consomem exclusivamente os artefatos de `runs/` — nunca reexecutam o AG. É o que torna a
análise reprodutível sem pagar o custo da busca de novo, e o que permite reabrir os notebooks
para o relatório e para o vídeo sem esperar 20 minutos (plano §7.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.monitoramento.registro import DIR_RUNS, carregar_historico

#: Métricas da Fase 1 no conjunto de teste — o baseline congelado (plano §1.1).
BASELINE_FASE1: dict[str, dict[str, float]] = {
    "random_forest": {
        "acuracia": 0.9174, "precisao": 0.9355, "recall": 0.8056,
        "f1_score": 0.8657, "auc_roc": 0.9365,
    },
    "svm_rbf": {
        "acuracia": 0.8807, "precisao": 0.8108, "recall": 0.8333,
        "f1_score": 0.8219, "auc_roc": 0.9315,
    },
    "logistic_regression": {
        "acuracia": 0.8532, "precisao": 0.7381, "recall": 0.8611,
        "f1_score": 0.7949, "auc_roc": 0.9277,
    },
    "decision_tree": {
        "acuracia": 0.8624, "precisao": 0.8000, "recall": 0.7778,
        "f1_score": 0.7887, "auc_roc": 0.8809,
    },
    "knn": {
        "acuracia": 0.8624, "precisao": 0.8889, "recall": 0.6667,
        "f1_score": 0.7619, "auc_roc": 0.9007,
    },
}

NOMES_AMIGAVEIS = {
    "random_forest": "Random Forest",
    "svm_rbf": "SVM (RBF)",
    "logistic_regression": "Regressão Logística",
    "decision_tree": "Árvore de Decisão",
    "knn": "KNN",
}

#: Pesos do fitness composto, repetidos aqui para calcular o score no holdout.
PESO_F1, PESO_RECALL, PESO_AUC = 0.5, 0.3, 0.2


def score_composto(f1: float, recall: float, auc: float) -> float:
    """O mesmo score que o AG maximizou, aplicado a métricas de holdout.

    Comparar F1 de holdout contra o fitness de CV é comparar coisas diferentes. Para saber se
    o ganho do AG transferiu, a comparação tem de usar a MESMA função objetivo nos dois lados.
    """
    return PESO_F1 * f1 + PESO_RECALL * recall + PESO_AUC * auc


def carregar_resumo(nome_execucao: str) -> dict[str, Any]:
    return json.loads((DIR_RUNS / nome_execucao / "resumo.json").read_text(encoding="utf-8"))


def carregar_melhor(nome_execucao: str) -> dict[str, Any]:
    caminho = DIR_RUNS / nome_execucao / "best_individual.json"
    return json.loads(caminho.read_text(encoding="utf-8"))


def carregar_holdout(nome_execucao: str) -> dict[str, Any] | None:
    caminho = DIR_RUNS / nome_execucao / "metrics_holdout.json"
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def execucoes_disponiveis(prefixo: str = "") -> list[str]:
    if not DIR_RUNS.exists():
        return []
    return sorted(
        p.name for p in DIR_RUNS.iterdir() if p.is_dir() and p.name.startswith(prefixo)
    )


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------


def tabela_original_vs_otimizado(semente: int = 42) -> pd.DataFrame:
    """A tabela que o enunciado pede: cada modelo da Fase 1, antes e depois (Etapa A)."""
    linhas = []
    for familia, base in BASELINE_FASE1.items():
        holdout = carregar_holdout(f"A_{familia}_seed{semente}")
        if holdout is None:
            continue
        linhas.append(
            {
                "Modelo": NOMES_AMIGAVEIS[familia],
                "F1 orig": base["f1_score"],
                "F1 otim": holdout["f1_score"],
                "Δ F1": holdout["f1_score"] - base["f1_score"],
                "Recall orig": base["recall"],
                "Recall otim": holdout["recall"],
                "Δ Recall": holdout["recall"] - base["recall"],
                "AUC orig": base["auc_roc"],
                "AUC otim": holdout["auc_roc"],
                "Δ AUC": holdout["auc_roc"] - base["auc_roc"],
                "Score orig": score_composto(base["f1_score"], base["recall"], base["auc_roc"]),
                "Score otim": score_composto(
                    holdout["f1_score"], holdout["recall"], holdout["auc_roc"]
                ),
            }
        )
    df = pd.DataFrame(linhas)
    if not df.empty:
        df["Δ Score"] = df["Score otim"] - df["Score orig"]
    return df


def tabela_experimentos(
    experimentos: tuple[str, ...] = (
        "E1_baseline", "E2_exploracao", "E3_aproveitamento", "D_roleta"
    ),
    sementes: tuple[int, ...] = (42, 7, 2024),
) -> pd.DataFrame:
    """Comparação dos experimentos, com média ± desvio sobre as sementes (plano §4.6).

    Reportar média ± desvio não é formalidade: AG é estocástico, e comparar configurações por
    uma execução cada é comparar ruído.
    """
    linhas = []
    for experimento in experimentos:
        fitnesses, tempos, caches, geracoes, convergencias = [], [], [], [], []
        entropias_finais, gaps = [], []
        for semente in sementes:
            nome = f"{experimento}_seed{semente}"
            if not (DIR_RUNS / nome).exists():
                continue
            resumo = carregar_resumo(nome)
            historico = carregar_historico(nome)

            fitnesses.append(resumo["melhor_fitness"])
            tempos.append(resumo["tempo_total_s"])
            caches.append(resumo["taxa_cache_hit"])
            geracoes.append(resumo["geracoes_executadas"])
            entropias_finais.append(historico["entropia_familia"].iloc[-1])
            gaps.append(
                float((historico["melhor_fitness"] - historico["fitness_medio"]).mean())
            )
            alvo = 0.99 * historico["melhor_fitness"].max()
            convergencias.append(int(historico[historico["melhor_fitness"] >= alvo]["geracao"].min()))

        if not fitnesses:
            continue
        linhas.append(
            {
                "Experimento": experimento,
                "Fitness (média)": np.mean(fitnesses),
                "Fitness (desvio)": np.std(fitnesses),
                "Ger. p/ 99% do máx": np.mean(convergencias),
                "Gerações rodadas": np.mean(geracoes),
                "Entropia final": np.mean(entropias_finais),
                "Gap melhor-média": np.mean(gaps),
                "Tempo (s)": np.mean(tempos),
                "Cache hit": np.mean(caches),
            }
        )
    return pd.DataFrame(linhas)


def tabela_ag_vs_busca_aleatoria(sementes: tuple[int, ...] = (42, 7, 2024)) -> pd.DataFrame:
    """Etapa C (plano §4.8): o AG contra busca aleatória com o MESMO orçamento.

    Sem esta linha, "o AG melhorou o modelo" é ambíguo — melhorou por ser algoritmo
    genético, ou porque qualquer busca com 450 avaliações bateria o baseline?
    """
    linhas = []
    for metodo, prefixo in (("AG (E1)", "E1_baseline"), ("Busca aleatória", "C_busca_aleatoria")):
        fitnesses, scores = [], []
        for semente in sementes:
            nome = f"{prefixo}_seed{semente}"
            if not (DIR_RUNS / nome).exists():
                continue
            melhor = carregar_melhor(nome)
            fitnesses.append(melhor["fitness"])
            holdout = carregar_holdout(nome)
            if holdout:
                scores.append(
                    score_composto(
                        holdout["f1_score"], holdout["recall"], holdout["auc_roc"]
                    )
                )
        if not fitnesses:
            continue
        linhas.append(
            {
                "Método": metodo,
                "Fitness CV (média)": np.mean(fitnesses),
                "Fitness CV (desvio)": np.std(fitnesses),
                "Score holdout (média)": np.mean(scores) if scores else np.nan,
            }
        )
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Gráficos (plano §4.9)
# ---------------------------------------------------------------------------


def _estilo():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def grafico_convergencia(
    experimentos: tuple[str, ...] = ("E1_baseline", "E2_exploracao", "E3_aproveitamento"),
    sementes: tuple[int, ...] = (42, 7, 2024),
    fitness_hotstart: float | None = None,
):
    """Gráfico 1 do plano §4.9 — a imagem-resumo do trabalho.

    Mostra melhor e média ± desvio das sementes, com linha horizontal no baseline. Num só
    lugar: se o AG superou o ponto de partida, quanto, e quão rápido.
    """
    import matplotlib.pyplot as plt

    _estilo()
    figura, eixos = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for experimento in experimentos:
        curvas_melhor, curvas_media = [], []
        for semente in sementes:
            nome = f"{experimento}_seed{semente}"
            if not (DIR_RUNS / nome).exists():
                continue
            historico = carregar_historico(nome)
            curvas_melhor.append(historico["melhor_fitness"].values)
            curvas_media.append(historico["fitness_medio"].values)
        if not curvas_melhor:
            continue

        n = min(len(c) for c in curvas_melhor)
        melhor = np.array([c[:n] for c in curvas_melhor])
        media = np.array([c[:n] for c in curvas_media])
        geracoes = np.arange(n)

        eixos[0].plot(geracoes, melhor.mean(axis=0), label=experimento, linewidth=2)
        eixos[0].fill_between(
            geracoes,
            melhor.mean(axis=0) - melhor.std(axis=0),
            melhor.mean(axis=0) + melhor.std(axis=0),
            alpha=0.15,
        )
        eixos[1].plot(geracoes, media.mean(axis=0), label=experimento, linewidth=2)
        eixos[1].fill_between(
            geracoes,
            media.mean(axis=0) - media.std(axis=0),
            media.mean(axis=0) + media.std(axis=0),
            alpha=0.15,
        )

    for eixo, titulo in zip(eixos, ("Melhor fitness da população", "Fitness médio da população")):
        if fitness_hotstart is not None:
            eixo.axhline(
                fitness_hotstart,
                linestyle="--",
                color="crimson",
                linewidth=1.2,
                label="hotstart (Fase 1)",
            )
        eixo.set_title(titulo)
        eixo.set_xlabel("geração")
        eixo.legend(fontsize=8)
    eixos[0].set_ylabel("fitness (CV no treino)")
    figura.suptitle(
        "Convergência do AG — média ± desvio sobre 3 sementes", fontsize=12, y=1.02
    )
    figura.tight_layout()
    return figura


def grafico_diversidade(
    experimentos: tuple[str, ...] = ("E1_baseline", "E2_exploracao", "E3_aproveitamento"),
    sementes: tuple[int, ...] = (42, 7, 2024),
):
    """Gráfico 2 do plano §4.9 — onde a convergência prematura do E3 deve aparecer."""
    import matplotlib.pyplot as plt

    _estilo()
    figura, eixos = plt.subplots(1, 3, figsize=(15, 4))
    metricas = [
        ("entropia_familia", "Entropia genética (gene família)"),
        ("n_solucoes_unicas", "Soluções únicas (fenótipos)"),
        ("distancia_euclidiana_media", "Distância euclidiana média"),
    ]

    for eixo, (coluna, titulo) in zip(eixos, metricas):
        for experimento in experimentos:
            curvas = []
            for semente in sementes:
                nome = f"{experimento}_seed{semente}"
                if not (DIR_RUNS / nome).exists():
                    continue
                curvas.append(carregar_historico(nome)[coluna].values)
            if not curvas:
                continue
            n = min(len(c) for c in curvas)
            dados = np.array([c[:n] for c in curvas])
            eixo.plot(np.arange(n), dados.mean(axis=0), label=experimento, linewidth=2)
        eixo.set_title(titulo, fontsize=10)
        eixo.set_xlabel("geração")
        eixo.legend(fontsize=8)
    figura.suptitle("Diversidade da população ao longo das gerações", fontsize=12, y=1.03)
    figura.tight_layout()
    return figura


def grafico_ablacao_selecao(sementes: tuple[int, ...] = (42, 7, 2024)):
    """Gráfico 8 do plano §4.9 (Etapa D) — torneio vs. roleta.

    O painel-chave é o da DISTÂNCIA entre melhor e média: ele mede pressão seletiva
    diretamente. Gap grande = população não converge para o melhor = pressão fraca.
    """
    import matplotlib.pyplot as plt

    _estilo()
    figura, eixos = plt.subplots(1, 3, figsize=(15, 4))

    for experimento, rotulo in (("E1_baseline", "torneio (k=2)"), ("D_roleta", "roleta")):
        melhores, medias, entropias = [], [], []
        for semente in sementes:
            nome = f"{experimento}_seed{semente}"
            if not (DIR_RUNS / nome).exists():
                continue
            historico = carregar_historico(nome)
            melhores.append(historico["melhor_fitness"].values)
            medias.append(historico["fitness_medio"].values)
            entropias.append(historico["entropia_familia"].values)
        if not melhores:
            continue

        n = min(min(len(c) for c in melhores), min(len(c) for c in medias))
        melhor = np.array([c[:n] for c in melhores]).mean(axis=0)
        media = np.array([c[:n] for c in medias]).mean(axis=0)
        entropia = np.array([c[:n] for c in entropias]).mean(axis=0)
        geracoes = np.arange(n)

        eixos[0].plot(geracoes, melhor, label=f"{rotulo} — melhor", linewidth=2)
        eixos[0].plot(geracoes, media, linestyle=":", label=f"{rotulo} — média", linewidth=1.6)
        eixos[1].plot(geracoes, melhor - media, label=rotulo, linewidth=2)
        eixos[2].plot(geracoes, entropia, label=rotulo, linewidth=2)

    eixos[0].set_title("Melhor e média do fitness", fontsize=10)
    eixos[1].set_title("Gap melhor − média\n(proxy de pressão seletiva)", fontsize=10)
    eixos[2].set_title("Entropia genética", fontsize=10)
    for eixo in eixos:
        eixo.set_xlabel("geração")
        eixo.legend(fontsize=8)
    figura.suptitle("Etapa D — ablação do operador de seleção", fontsize=12, y=1.03)
    figura.tight_layout()
    return figura


def grafico_comparativo_modelos(semente: int = 42):
    """Gráfico 3 do plano §4.9 — barras agrupadas original vs. otimizado."""
    import matplotlib.pyplot as plt

    _estilo()
    tabela = tabela_original_vs_otimizado(semente)
    if tabela.empty:
        raise RuntimeError("Nenhum resultado da Etapa A encontrado em runs/")

    metricas = [("F1", "F1 orig", "F1 otim"), ("Recall", "Recall orig", "Recall otim"),
                ("AUC", "AUC orig", "AUC otim")]
    figura, eixos = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)

    for eixo, (titulo, col_orig, col_otim) in zip(eixos, metricas):
        x = np.arange(len(tabela))
        largura = 0.38
        eixo.bar(x - largura / 2, tabela[col_orig], largura, label="Fase 1 (original)")
        eixo.bar(x + largura / 2, tabela[col_otim], largura, label="otimizado (AG)")
        eixo.set_xticks(x)
        eixo.set_xticklabels(tabela["Modelo"], rotation=30, ha="right", fontsize=8)
        eixo.set_title(titulo, fontsize=11)
        eixo.set_ylim(0, 1.0)
        eixo.legend(fontsize=8)
    eixos[0].set_ylabel("valor no conjunto de teste")
    figura.suptitle(
        "Original vs. otimizado — 5 modelos do PCOS, conjunto de teste (109 pacientes)",
        fontsize=12,
        y=1.03,
    )
    figura.tight_layout()
    return figura


def grafico_alelo_e_familias(
    experimentos: tuple[str, ...] = ("E1_baseline", "E2_exploracao"),
    sementes: tuple[int, ...] = (42, 7, 2024),
):
    """Gráficos 6 e 9 do plano §4.9 — frequência do alelo `class_weight='balanced'` e
    distribuição das famílias por geração.

    O primeiro painel testa a predição do plano §4.3: como o fitness pesa recall, esse alelo
    *deveria* se fixar na população.
    """
    import matplotlib.pyplot as plt

    _estilo()
    figura, eixos = plt.subplots(1, 2, figsize=(13, 4.2))

    for experimento in experimentos:
        curvas = []
        for semente in sementes:
            nome = f"{experimento}_seed{semente}"
            if not (DIR_RUNS / nome).exists():
                continue
            curvas.append(carregar_historico(nome)["freq_class_weight_balanced"].values)
        if not curvas:
            continue
        n = min(len(c) for c in curvas)
        dados = np.array([c[:n] for c in curvas])
        eixos[0].plot(np.arange(n), dados.mean(axis=0), label=experimento, linewidth=2)

    eixos[0].axhline(0.5, linestyle=":", color="gray", linewidth=1)
    eixos[0].set_title("Frequência do alelo class_weight='balanced'", fontsize=10)
    eixos[0].set_xlabel("geração")
    eixos[0].set_ylabel("fração da população elegível")
    eixos[0].set_ylim(0, 1)
    eixos[0].legend(fontsize=8)

    nome_ref = f"{experimentos[0]}_seed{sementes[0]}"
    if (DIR_RUNS / nome_ref).exists():
        historico = carregar_historico(nome_ref)
        familias = pd.DataFrame(list(historico["distribuicao_familias"]))
        eixos[1].stackplot(
            historico["geracao"],
            *[familias[c].values for c in familias.columns],
            labels=[NOMES_AMIGAVEIS.get(c, c) for c in familias.columns],
            alpha=0.85,
        )
        eixos[1].set_title(f"Distribuição das famílias — {nome_ref}", fontsize=10)
        eixos[1].set_xlabel("geração")
        eixos[1].set_ylabel("fração da população")
        eixos[1].set_ylim(0, 1)
        eixos[1].legend(fontsize=7, loc="upper left")

    figura.tight_layout()
    return figura
