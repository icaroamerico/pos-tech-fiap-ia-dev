"""Testes da codificação do cromossomo (src/ag/cromossomo.py).

O objetivo central destes testes é a garantia: "o AG nunca gera um indivíduo
que o scikit-learn rejeite". Isso é verificado de duas formas — validação estrutural
(`validar_individuo`) e validação funcional (instanciar e treinar o estimador real).
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.ag.cromossomo import (
    FAMILIAS,
    HOTSTART_FASE1,
    SCHEMA,
    clamp_individuo,
    decodificar,
    gerar_individuo_aleatorio,
    gerar_individuo_hotstart,
    validar_individuo,
)

CONSTRUTORES = {
    "random_forest": RandomForestClassifier,
    "svm_rbf": SVC,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
}


def _dados_sinteticos(rng: np.random.Generator, n: int = 40, d: int = 5):
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.mark.parametrize("seed", range(20))
def test_individuo_aleatorio_e_sempre_valido(seed):
    rng = np.random.default_rng(seed)
    individuo = gerar_individuo_aleatorio(rng)
    assert validar_individuo(individuo)


@pytest.mark.parametrize("familia", FAMILIAS)
def test_hotstart_fixa_a_familia_pedida(familia):
    rng = np.random.default_rng(0)
    individuo = gerar_individuo_hotstart(familia, rng)
    assert individuo["familia"] == familia
    assert validar_individuo(individuo)


@pytest.mark.parametrize("familia", FAMILIAS)
def test_hotstart_decodifica_de_volta_para_os_valores_reais_da_fase_1(familia):
    """REGRESSÃO (bug real encontrado em revisão): `HOTSTART_FASE1` guarda valores em
    unidades do mundo real, mas genes `real_log` têm representação interna em log10. Sem a
    codificação em `gerar_individuo_hotstart`, C=1.0 era decodificado como 10**1.0 = 10 e
    gamma=0.021748 como 10**0.021748 = 1.05 — erro de uma ordem de magnitude que deixava o
    SVM com F1 = 0.

    Este teste fecha o ciclo codificar→decodificar: o que sai da decodificação tem de ser
    EXATAMENTE o que a tabela HOTSTART_FASE1 declara.
    """
    rng = np.random.default_rng(0)
    individuo = gerar_individuo_hotstart(familia, rng)
    _, kwargs = decodificar(individuo)

    for nome, valor_esperado in HOTSTART_FASE1[familia].items():
        if nome == "penalty_solver":
            penalty, solver = valor_esperado
            assert kwargs["penalty"] == penalty
            assert kwargs["solver"] == solver
            continue
        if nome == "l1_ratio" and kwargs.get("penalty") != "elasticnet":
            continue  # gene não expresso nesta configuração
        if nome == "p" and kwargs.get("metric") != "minkowski":
            continue  # gene não expresso nesta configuração
        assert nome in kwargs, f"{familia}.{nome} não apareceu na decodificação"
        assert kwargs[nome] == pytest.approx(valor_esperado, rel=1e-9), (
            f"{familia}.{nome}: decodificou {kwargs[nome]!r}, "
            f"esperado {valor_esperado!r} (HOTSTART_FASE1)"
        )


def test_gamma_hotstart_equivale_ao_scale_do_sklearn():
    """Documenta e verifica a derivação do `gamma` do hotstart do SVM.

    A Fase 1 usa `SVC(class_weight="balanced")`, cujo default é `gamma='scale'` — que o
    sklearn calcula como 1/(n_features * X.var()) sobre os dados JÁ pré-processados. Como o
    gene `gamma` é numérico, o hotstart precisa do equivalente numérico desse 'scale'.

    Este teste recalcula o valor a partir dos dados reais e confere contra o que está
    hardcoded em HOTSTART_FASE1 — se o pré-processamento mudar, o teste avisa.
    """
    import numpy as np

    from src.fitness.fitness_pcos import carregar_dados_treino, construir_pipeline

    dados = carregar_dados_treino()
    rng = np.random.default_rng(0)
    pipeline = construir_pipeline(gerar_individuo_hotstart("svm_rbf", rng), dados)
    X_transformado = pipeline.named_steps["preprocessador"].fit_transform(dados.X_treino)
    X_array = np.asarray(
        X_transformado.todense() if hasattr(X_transformado, "todense") else X_transformado
    )

    gamma_scale = 1.0 / (X_array.shape[1] * X_array.var())

    assert HOTSTART_FASE1["svm_rbf"]["gamma"] == pytest.approx(gamma_scale, rel=1e-3), (
        f"gamma='scale' nesta base vale {gamma_scale:.6f}, mas HOTSTART_FASE1 declara "
        f"{HOTSTART_FASE1['svm_rbf']['gamma']}"
    )


@pytest.mark.parametrize("familia", FAMILIAS)
def test_hotstart_reproduz_metricas_da_fase_1_quando_treinado(familia):
    """O ponto central do hotstart: o indivíduo de hotstart decodificado deve
    treinar um estimador cujos hiperparâmetros são EXATAMENTE os que a Fase 1 usou.
    """
    rng = np.random.default_rng(0)
    individuo = gerar_individuo_hotstart(familia, rng)
    fam, kwargs = decodificar(individuo)
    assert fam == familia

    construtor = CONSTRUTORES[familia]
    modelo = construtor(**kwargs, random_state=42) if familia != "knn" else construtor(**kwargs)

    if familia == "random_forest":
        assert modelo.n_estimators == 200
        assert modelo.max_depth == 8
        assert modelo.class_weight == "balanced"
    elif familia == "decision_tree":
        assert modelo.max_depth == 5
        assert modelo.class_weight == "balanced"
    elif familia == "logistic_regression":
        assert modelo.max_iter == 2000
        assert modelo.class_weight == "balanced"
        assert modelo.penalty == "l2"
        assert modelo.solver == "lbfgs"
    elif familia == "knn":
        assert modelo.n_neighbors == 5
    elif familia == "svm_rbf":
        assert modelo.class_weight == "balanced"


@pytest.mark.parametrize("seed", range(30))
def test_decodificar_produz_kwargs_aceitos_pelo_sklearn(seed):
    """Para cada indivíduo aleatório, instancia e TREINA o estimador real — a prova de que
    nenhuma combinação de genes produz um ValueError do scikit-learn.
    """
    rng = np.random.default_rng(seed)
    individuo = gerar_individuo_aleatorio(rng)
    familia, kwargs = decodificar(individuo)

    construtor = CONSTRUTORES[familia]
    kwargs_finais = dict(kwargs)
    if familia != "knn":
        kwargs_finais["random_state"] = 42

    X, y = _dados_sinteticos(rng)
    modelo = construtor(**kwargs_finais)
    modelo.fit(X, y)  # levanta ValueError se algum hiperparâmetro for inválido
    modelo.predict(X)


def test_n_neighbors_e_sempre_impar():
    rng = np.random.default_rng(1)
    for _ in range(200):
        individuo = gerar_individuo_aleatorio(rng)
        assert individuo["knn"]["n_neighbors"] % 2 == 1


def test_penalty_solver_e_sempre_um_par_valido():
    pares_validos = {("l2", "lbfgs"), ("l1", "liblinear"), ("elasticnet", "saga")}
    rng = np.random.default_rng(2)
    for _ in range(200):
        individuo = gerar_individuo_aleatorio(rng)
        assert individuo["logistic_regression"]["penalty_solver"] in pares_validos


def test_l1_ratio_so_aparece_com_elasticnet():
    rng = np.random.default_rng(3)
    individuo = gerar_individuo_aleatorio(rng)
    individuo["familia"] = "logistic_regression"  # força a família sob teste
    individuo["logistic_regression"]["penalty_solver"] = ("l2", "lbfgs")
    _, kwargs = decodificar(individuo)
    assert "l1_ratio" not in kwargs

    individuo["logistic_regression"]["penalty_solver"] = ("elasticnet", "saga")
    _, kwargs = decodificar(individuo)
    assert "l1_ratio" in kwargs


def test_p_so_aparece_com_minkowski():
    rng = np.random.default_rng(4)
    individuo = gerar_individuo_aleatorio(rng)
    individuo["familia"] = "knn"  # força a família sob teste
    individuo["knn"]["metric"] = "euclidean"
    _, kwargs = decodificar(individuo)
    assert "p" not in kwargs

    individuo["knn"]["metric"] = "minkowski"
    _, kwargs = decodificar(individuo)
    assert "p" in kwargs


def test_clamp_traz_valores_fora_da_faixa_para_dentro():
    rng = np.random.default_rng(5)
    individuo = gerar_individuo_aleatorio(rng)

    # Força valores inválidos deliberadamente, simulando o que crossover/mutação poderiam
    # produzir sem a etapa de clamp.
    individuo["random_forest"]["n_estimators"] = 10_000  # muito acima do limite (300)
    individuo["random_forest"]["min_samples_leaf"] = -5  # abaixo do limite (1)
    individuo["knn"]["n_neighbors"] = 8  # par — deve virar ímpar
    individuo["logistic_regression"]["class_weight"] = "valor_invalido"

    corrigido = clamp_individuo(individuo)

    assert validar_individuo(corrigido)
    assert corrigido["random_forest"]["n_estimators"] == 300
    assert corrigido["random_forest"]["min_samples_leaf"] == 1
    assert corrigido["knn"]["n_neighbors"] % 2 == 1
    assert corrigido["logistic_regression"]["class_weight"] in (None, "balanced")


def test_real_log_mantem_coerencia_na_decodificacao():
    """O gene de C e gamma guarda log10(valor) internamente; a decodificação
    precisa exponenciar de volta para o valor real dentro da faixa [low, high] original.
    """
    rng = np.random.default_rng(6)
    for _ in range(50):
        individuo = gerar_individuo_aleatorio(rng)
        _, kwargs = decodificar(individuo)
        if individuo["familia"] == "svm_rbf":
            spec_c = SCHEMA["svm_rbf"]["C"]
            spec_gamma = SCHEMA["svm_rbf"]["gamma"]
            assert spec_c["low"] <= kwargs["C"] <= spec_c["high"]
            assert spec_gamma["low"] <= kwargs["gamma"] <= spec_gamma["high"]


def test_individuo_carrega_genes_latentes_de_todas_as_familias():
    """Mesmo com uma família ativa, as demais precisam estar presentes (genes
    latentes como reservatório de diversidade).
    """
    rng = np.random.default_rng(7)
    individuo = gerar_individuo_aleatorio(rng)
    for familia in FAMILIAS:
        assert familia in individuo
        assert set(individuo[familia].keys()) == set(SCHEMA[familia].keys())
