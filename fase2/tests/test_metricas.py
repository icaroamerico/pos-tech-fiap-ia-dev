"""Testes das métricas de AG.

O teste mais importante deste arquivo valida a implementação de entropia genética contra
um valor fechado analiticamente — testar contra o número que a definição exige, não contra
a minha própria expectativa.
"""

import math

import numpy as np

from src.ag.cromossomo import gerar_individuo_aleatorio, gerar_individuo_hotstart
from src.ag.metricas import (
    desvio_padrao_fitness,
    distancia_euclidiana_media,
    distribuicao_familias,
    entropia_alelo,
    entropia_genetica_familia,
    frequencia_class_weight_balanced,
    media_fitness,
    melhor_fitness,
    numero_genotipos_unicos,
    numero_solucoes_unicas,
)


def test_entropia_de_populacao_clonal_e_zero():
    assert entropia_alelo(["A", "A", "A", "A"]) == 0.0


def test_entropia_de_tres_alelos_equiprovaveis():
    """3 alelos distintos e equiprováveis → H = log2(3) ≈ 1,585, pela definição de Shannon."""
    valores = ["rota_A", "rota_B", "rota_C"]  # 3 alelos distintos, cada um com 1 ocorrência
    h = entropia_alelo(valores)
    assert math.isclose(h, math.log2(3), abs_tol=1e-9)
    assert math.isclose(h, 1.585, abs_tol=0.001)


def test_entropia_de_populacao_totalmente_diversa_e_log2_n():
    valores = [f"v{i}" for i in range(8)]
    h = entropia_alelo(valores)
    assert math.isclose(h, math.log2(8), abs_tol=1e-9)


def test_entropia_e_sensivel_a_desbalanceamento():
    """Uma população desbalanceada (1 alelo dominante) tem entropia menor que uma
    equilibrada com o mesmo número de alelos distintos.
    """
    equilibrada = ["A", "A", "B", "B"]
    desbalanceada = ["A", "A", "A", "B"]
    assert entropia_alelo(desbalanceada) < entropia_alelo(equilibrada)


def test_entropia_genetica_familia_usa_o_gene_familia():
    rng = np.random.default_rng(0)
    populacao = [gerar_individuo_hotstart(f, rng) for f in
                 ["random_forest", "svm_rbf", "logistic_regression"]]
    h = entropia_genetica_familia(populacao)
    assert math.isclose(h, math.log2(3), abs_tol=1e-9)


def test_distancia_euclidiana_de_populacao_clonal_e_zero():
    rng = np.random.default_rng(1)
    individuo = gerar_individuo_aleatorio(rng)
    populacao = [individuo, individuo, individuo]
    assert distancia_euclidiana_media(populacao) == 0.0


def test_distancia_euclidiana_e_positiva_para_populacao_diversa():
    rng = np.random.default_rng(2)
    populacao = []
    for _ in range(10):
        ind = gerar_individuo_aleatorio(rng)
        ind["familia"] = "random_forest"  # força a mesma família para garantir pares comparáveis
        populacao.append(ind)
    assert distancia_euclidiana_media(populacao) > 0.0


def test_distancia_euclidiana_fica_entre_zero_e_raiz_do_numero_de_genes():
    """Com genes normalizados em [0,1], a distância euclidiana máxima possível entre dois
    indivíduos de uma família com g genes numéricos é sqrt(g).
    """
    rng = np.random.default_rng(3)
    populacao = []
    for _ in range(20):
        ind = gerar_individuo_aleatorio(rng)
        ind["familia"] = "svm_rbf"  # 2 genes numéricos (C, gamma) -> máximo = sqrt(2)
        populacao.append(ind)
    d = distancia_euclidiana_media(populacao)
    assert 0.0 <= d <= math.sqrt(2) + 1e-9


def test_numero_solucoes_unicas_conta_corretamente():
    rng = np.random.default_rng(4)
    individuo = gerar_individuo_aleatorio(rng)
    populacao_clonal = [individuo] * 5
    assert numero_solucoes_unicas(populacao_clonal) == 1

    populacao_diversa = [gerar_individuo_aleatorio(rng) for _ in range(20)]
    # com espaço de busca contínuo, 20 sorteios independentes quase certamente são distintos
    assert numero_solucoes_unicas(populacao_diversa) >= 15


def test_solucoes_unicas_ignora_diferencas_apenas_latentes():
    """REGRESSÃO (bug real encontrado em revisão): a métrica usava a chave do genótipo
    completo, então dois indivíduos que treinam EXATAMENTE o mesmo modelo (mesma família
    ativa, mesmos genes expressos) contavam como duas soluções distintas só por diferirem em
    genes latentes.

    O efeito era esconder convergência prematura — justamente o que esta métrica
    existe para detectar: a população poderia estar clonal no fenótipo e a métrica
    continuaria reportando diversidade alta.
    """
    import copy

    rng = np.random.default_rng(40)
    a = gerar_individuo_hotstart("svm_rbf", rng)
    b = copy.deepcopy(a)
    b["random_forest"]["n_estimators"] = 299  # latente
    b["knn"]["n_neighbors"] = 29  # latente
    b["decision_tree"]["max_depth"] = 17  # latente

    assert numero_solucoes_unicas([a, b]) == 1, (
        "fenótipos idênticos contados como soluções distintas — chave provavelmente voltou a "
        "usar o genótipo completo"
    )
    # o genótipo, por outro lado, é de fato diferente — e a outra métrica deve mostrar isso
    assert numero_genotipos_unicos([a, b]) == 2


def test_solucoes_unicas_distingue_fenotipos_diferentes():
    """Contraprova: a métrica não pode ser tão permissiva que deixe de ver modelos distintos."""
    import copy

    rng = np.random.default_rng(41)
    a = gerar_individuo_hotstart("svm_rbf", rng)
    b = copy.deepcopy(a)
    b["svm_rbf"]["C"] = a["svm_rbf"]["C"] + 1.0  # gene ATIVO
    assert numero_solucoes_unicas([a, b]) == 2


def test_genes_nao_expressos_nao_contam_como_diferenca_de_fenotipo():
    """`l1_ratio` só é usado quando penalty='elasticnet'; `p` só quando metric='minkowski'.
    Variar um gene não expresso não muda o modelo treinado, logo não deve contar como
    solução distinta.
    """
    import copy

    rng = np.random.default_rng(42)
    a = gerar_individuo_hotstart("logistic_regression", rng)
    a["logistic_regression"]["penalty_solver"] = ("l2", "lbfgs")  # l1_ratio não expresso
    b = copy.deepcopy(a)
    b["logistic_regression"]["l1_ratio"] = 0.99
    assert numero_solucoes_unicas([a, b]) == 1


def test_frequencia_class_weight_balanced_exclui_knn():
    rng = np.random.default_rng(5)
    knn_individuo = gerar_individuo_hotstart("knn", rng)
    rf_balanceado = gerar_individuo_hotstart("random_forest", rng)  # Fase 1 usa "balanced"
    populacao = [knn_individuo, rf_balanceado]

    freq = frequencia_class_weight_balanced(populacao)
    # só o RF é elegível (KNN não tem class_weight), e ele é "balanced" -> frequência = 1.0
    assert freq == 1.0


def test_frequencia_class_weight_balanced_populacao_vazia_ou_so_knn():
    rng = np.random.default_rng(6)
    so_knn = [gerar_individuo_hotstart("knn", rng) for _ in range(5)]
    assert frequencia_class_weight_balanced(so_knn) == 0.0


def test_distribuicao_familias_soma_um():
    rng = np.random.default_rng(7)
    populacao = [gerar_individuo_aleatorio(rng) for _ in range(50)]
    dist = distribuicao_familias(populacao)
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-9)
    assert all(v >= 0 for v in dist.values())


def test_convergencia_melhor_media_desvio():
    fitnesses = [0.5, 0.6, 0.7, 0.8, 0.9]
    assert melhor_fitness(fitnesses) == 0.9
    assert math.isclose(media_fitness(fitnesses), 0.7, abs_tol=1e-9)
    assert desvio_padrao_fitness(fitnesses) > 0.0
    assert desvio_padrao_fitness([0.5, 0.5, 0.5]) == 0.0
