"""Testes do fitness do PCOS (plano §7.2).

O teste `test_fitness_nunca_acessa_o_conjunto_de_teste` é o mais importante de todo o
projeto (plano §3): um AG que otimiza contra o teste produz um número bonito sem valor.
Este teste prova estruturalmente que isso não pode acontecer aqui.
"""

import dataclasses

import numpy as np
import pytest

from src.ag.cromossomo import gerar_individuo_hotstart
from src.fitness.fitness_pcos import (
    DadosTreino,
    calcular_fitness,
    calcular_fitness_com_cache,
    carregar_dados_treino,
)


@pytest.fixture(scope="module")
def dados() -> DadosTreino:
    return carregar_dados_treino()


def test_dados_treino_tem_432_linhas_como_a_fase_1(dados):
    """432 = 80% de 541 (base PCOS unificada) — o mesmo split documentado na Fase 1."""
    assert len(dados.X_treino) == 432
    assert len(dados.y_treino) == 432


def test_dataclass_dados_treino_nao_tem_nenhum_campo_de_teste():
    """Garantia ESTRUTURAL (plano §3): não existe atributo `X_teste`/`y_teste` nesta classe
    — não há nada para um bug futuro acessar por engano, mesmo que alguém tente.
    """
    nomes_dos_campos = {f.name for f in dataclasses.fields(DadosTreino)}
    assert nomes_dos_campos == {"X_treino", "y_treino", "colunas_numericas", "colunas_categoricas"}
    for nome in nomes_dos_campos:
        assert "test" not in nome.lower()
        assert "teste" not in nome.lower()


def test_split_de_treino_e_disjunto_do_split_de_teste_real(dados):
    """Reconstrói o split completo (treino E teste) de forma independente e confirma que
    os índices do que o AG recebe (`dados.X_treino`) são disjuntos dos índices do conjunto
    de teste real — ou seja, nenhuma linha de teste vazou para dentro do treino do AG.
    """
    from src.modelagem.treinamento import carregar_dados, dividir_dados, preparar_features

    df = carregar_dados()
    X, y = preparar_features(df)
    _X_treino_independente, X_teste_real, _y_treino_independente, _y_teste_real = dividir_dados(X, y)

    indices_treino_do_ag = set(dados.X_treino.index)
    indices_teste_real = set(X_teste_real.index)

    assert indices_treino_do_ag.isdisjoint(indices_teste_real)
    assert len(indices_teste_real) == 109  # 20% de 541


def test_fitness_nunca_acessa_o_conjunto_de_teste(monkeypatch, dados):
    """O teste mais importante do projeto (plano §3 e §7.2).

    Substitui `dividir_dados` por uma versão que devolve, no lugar de X_teste/y_teste, um
    objeto "espião" que LEVANTA EXCEÇÃO em qualquer tentativa de acesso (atributo, item,
    len, iteração). Se qualquer código no caminho de `carregar_dados_treino` +
    `calcular_fitness` tocar no "teste" — mesmo um `len()` ou um `.shape` —, o espião
    explode e o teste falha. O teste passar significa que o teste genuinamente nunca foi
    tocado, não apenas que não deveria ter sido.
    """
    import src.fitness.fitness_pcos as fitness_pcos

    class EspiaoNuncaDeveSerAcessado:
        def __getattr__(self, nome):
            raise AssertionError(
                f"VAZAMENTO DE DADOS: código acessou '.{nome}' do conjunto de TESTE "
                "durante o cálculo do fitness."
            )

        def __len__(self):
            raise AssertionError("VAZAMENTO DE DADOS: código chamou len() no conjunto de TESTE.")

        def __getitem__(self, _chave):
            raise AssertionError("VAZAMENTO DE DADOS: código indexou o conjunto de TESTE.")

        def __iter__(self):
            raise AssertionError("VAZAMENTO DE DADOS: código iterou sobre o conjunto de TESTE.")

    def dividir_dados_espiao(X, y, test_size=0.2, random_state=42):
        from src.modelagem.treinamento import dividir_dados as dividir_dados_original

        X_treino_real, _X_teste_original, y_treino_real, _y_teste_original = (
            dividir_dados_original(X, y, test_size=test_size, random_state=random_state)
        )
        return X_treino_real, EspiaoNuncaDeveSerAcessado(), y_treino_real, EspiaoNuncaDeveSerAcessado()

    monkeypatch.setattr(fitness_pcos, "dividir_dados", dividir_dados_espiao)

    # Se qualquer coisa no caminho tocar no "teste", a linha abaixo levanta AssertionError
    # e o teste falha — é exatamente esse o comportamento que queremos verificar.
    dados_instrumentados = fitness_pcos.carregar_dados_treino()

    rng = np.random.default_rng(0)
    individuo = gerar_individuo_hotstart("random_forest", rng)
    resultado = calcular_fitness(individuo, dados_instrumentados)

    assert 0.0 <= resultado["fitness"] <= 1.0


@pytest.mark.parametrize("familia", ["random_forest", "svm_rbf", "logistic_regression",
                                      "decision_tree", "knn"])
def test_fitness_esta_sempre_entre_zero_e_um(dados, familia):
    rng = np.random.default_rng(0)
    individuo = gerar_individuo_hotstart(familia, rng)
    resultado = calcular_fitness(individuo, dados)
    assert 0.0 <= resultado["fitness"] <= 1.0
    assert 0.0 <= resultado["f1_medio"] <= 1.0
    assert 0.0 <= resultado["recall_medio"] <= 1.0
    assert 0.0 <= resultado["auc_medio"] <= 1.0


def test_fitness_e_deterministico_para_o_mesmo_individuo(dados):
    """Pré-requisito do cache (plano §5.2): o mesmo indivíduo precisa produzir exatamente o
    mesmo fitness em duas chamadas independentes.
    """
    rng = np.random.default_rng(0)
    individuo = gerar_individuo_hotstart("random_forest", rng)

    resultado_1 = calcular_fitness(individuo, dados)
    resultado_2 = calcular_fitness(individuo, dados)

    assert resultado_1["fitness"] == resultado_2["fitness"]
    assert resultado_1["f1_medio"] == resultado_2["f1_medio"]
    assert resultado_1["recall_medio"] == resultado_2["recall_medio"]
    assert resultado_1["auc_medio"] == resultado_2["auc_medio"]


def test_cache_acerta_quando_so_genes_latentes_diferem(dados):
    """REGRESSÃO (bug real encontrado em auditoria): o cache usava a chave do GENÓTIPO
    completo, incluindo genes latentes das famílias inativas. Como esses genes derivam a
    cada mutação sem afetar o modelo treinado, dois indivíduos com fitness comprovadamente
    idêntico tinham chaves diferentes — o cache perdia quase todos os acertos e a otimização
    do plano §5.2 ficava inoperante na prática.

    Este teste prova as duas metades: o fitness é mesmo idêntico, e o cache reconhece isso.
    """
    import copy

    rng = np.random.default_rng(0)
    individuo_a = gerar_individuo_hotstart("svm_rbf", rng)
    individuo_b = copy.deepcopy(individuo_a)
    # altera SOMENTE genes de famílias inativas (a família ativa é svm_rbf)
    individuo_b["random_forest"]["n_estimators"] = 299
    individuo_b["knn"]["n_neighbors"] = 29
    individuo_b["decision_tree"]["max_depth"] = 17

    # premissa do teste: o fitness realmente não depende dos genes latentes
    assert calcular_fitness(individuo_a, dados)["fitness"] == pytest.approx(
        calcular_fitness(individuo_b, dados)["fitness"], rel=1e-12
    )

    cache: dict = {}
    _, veio_do_cache_a = calcular_fitness_com_cache(individuo_a, dados, cache)
    _, veio_do_cache_b = calcular_fitness_com_cache(individuo_b, dados, cache)

    assert veio_do_cache_a is False
    assert veio_do_cache_b is True, (
        "cache não reconheceu dois indivíduos de fenótipo idêntico — a chave provavelmente "
        "voltou a incluir genes latentes"
    )


def test_cache_nao_acerta_quando_o_fenotipo_difere(dados):
    """Contraprova do teste acima: o cache não pode ser permissivo a ponto de confundir
    indivíduos que treinam modelos diferentes.
    """
    import copy

    rng = np.random.default_rng(0)
    individuo_a = gerar_individuo_hotstart("svm_rbf", rng)
    individuo_b = copy.deepcopy(individuo_a)
    individuo_b["svm_rbf"]["C"] = individuo_a["svm_rbf"]["C"] + 1.0  # gene ATIVO

    cache: dict = {}
    calcular_fitness_com_cache(individuo_a, dados, cache)
    _, veio_do_cache_b = calcular_fitness_com_cache(individuo_b, dados, cache)
    assert veio_do_cache_b is False


def test_cache_devolve_o_mesmo_valor_que_o_calculo_direto(dados):
    rng = np.random.default_rng(0)
    individuo = gerar_individuo_hotstart("svm_rbf", rng)

    direto = calcular_fitness(individuo, dados)

    cache: dict = {}
    via_cache_1, veio_do_cache_1 = calcular_fitness_com_cache(individuo, dados, cache)
    via_cache_2, veio_do_cache_2 = calcular_fitness_com_cache(individuo, dados, cache)

    assert veio_do_cache_1 is False  # primeira chamada: miss, calcula de verdade
    assert veio_do_cache_2 is True  # segunda chamada: hit, vem do cache
    assert via_cache_1["fitness"] == direto["fitness"]
    assert via_cache_2["fitness"] == direto["fitness"]


# F1 documentado pela Fase 1 no conjunto de teste, por família. O fitness aqui é outro
# número (score composto por CV no treino), então a comparação é de ORDEM DE GRANDEZA: o
# hotstart tem de ficar na vizinhança do baseline, não idêntico a ele.
F1_FASE1 = {
    "random_forest": 0.8657,
    "svm_rbf": 0.8219,
    "logistic_regression": 0.7949,
    "decision_tree": 0.7887,
    "knn": 0.7619,
}
TOLERANCIA_F1 = 0.12  # folga para a diferença CV-no-treino vs. holdout


@pytest.mark.parametrize("familia", list(F1_FASE1))
def test_hotstart_de_toda_familia_reproduz_a_vizinhanca_do_f1_da_fase_1(dados, familia):
    """REGRESSÃO (bug real encontrado em auditoria): este teste antes existia só para o
    Random Forest, e foi exatamente nessa lacuna que o bug de codificação `real_log` se
    esconde — o hotstart do SVM produzia **F1 = 0,0000** (modelo prevendo tudo negativo,
    por gamma decodificado 58x maior que o pretendido) e nenhum teste reclamava, porque
    `test_fitness_esta_sempre_entre_zero_e_um` aceita 0,069 como "válido".

    A garantia do plano §4.2 — "o melhor fitness da geração 0 é, por construção, ≥ o
    desempenho da Fase 1" — depende de TODO hotstart estar correto, não só de um.
    """
    rng = np.random.default_rng(0)
    individuo = gerar_individuo_hotstart(familia, rng)
    resultado = calcular_fitness(individuo, dados)

    f1_esperado = F1_FASE1[familia]
    assert resultado["f1_medio"] > f1_esperado - TOLERANCIA_F1, (
        f"hotstart de {familia} produziu F1={resultado['f1_medio']:.4f}, muito abaixo do "
        f"F1={f1_esperado:.4f} documentado na Fase 1 — provável erro de codificação de gene"
    )


@pytest.mark.parametrize("familia", list(F1_FASE1))
def test_nenhum_hotstart_produz_modelo_degenerado(dados, familia):
    """Guarda explícita contra o modo de falha mais silencioso: um modelo que prevê sempre
    a mesma classe tem F1=0 ou recall=0 e ainda assim passa por qualquer verificação de
    "fitness está entre 0 e 1".
    """
    rng = np.random.default_rng(0)
    individuo = gerar_individuo_hotstart(familia, rng)
    resultado = calcular_fitness(individuo, dados)
    assert resultado["f1_medio"] > 0.0, f"{familia}: F1=0 indica modelo degenerado"
    assert resultado["recall_medio"] > 0.0, f"{familia}: recall=0 indica modelo degenerado"
