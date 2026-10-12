"""Testes dos operadores genéticos e da população (plano §7.2).

Cobrem os invariantes que o plano lista como críticos:
  - crossover de pais válidos produz filhos válidos e dentro das faixas;
  - crossover de pais idênticos produz filho idêntico;
  - mutação respeita o espaço de busca;
  - mutação com probabilidade 0 não altera nada; com probabilidade 1, altera;
  - torneio prefere fitness maior em >50% dos sorteios;
  - população mantém tamanho constante ao longo das gerações;
  - elitismo preserva o melhor indivíduo.
"""

import copy

import numpy as np
import pytest

from src.ag.cromossomo import gerar_individuo_aleatorio, validar_individuo
from src.ag.operadores import cruzar, mutar, selecao_roleta, selecao_torneio
from src.ag.populacao import aplicar_elitismo, inicializar_populacao


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(15))
def test_crossover_de_pais_validos_produz_filhos_validos(seed):
    rng = np.random.default_rng(seed)
    pai1 = gerar_individuo_aleatorio(rng)
    pai2 = gerar_individuo_aleatorio(rng)
    filho1, filho2 = cruzar(pai1, pai2, taxa_crossover=1.0, rng=rng)
    assert validar_individuo(filho1)
    assert validar_individuo(filho2)


def test_crossover_de_pais_identicos_produz_filho_identico():
    rng = np.random.default_rng(42)
    pai = gerar_individuo_aleatorio(rng)
    filho1, filho2 = cruzar(pai, copy.deepcopy(pai), taxa_crossover=1.0, rng=rng)
    assert filho1 == pai
    assert filho2 == pai


def test_taxa_crossover_zero_retorna_copias_dos_pais():
    rng = np.random.default_rng(1)
    pai1 = gerar_individuo_aleatorio(rng)
    pai2 = gerar_individuo_aleatorio(rng)
    filho1, filho2 = cruzar(pai1, pai2, taxa_crossover=0.0, rng=rng)
    assert filho1 == pai1
    assert filho2 == pai2
    # confirma que são cópias, não o mesmo objeto (mutar um não afeta o outro)
    assert filho1 is not pai1


# ---------------------------------------------------------------------------
# Mutação
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(15))
def test_mutacao_mantem_individuo_dentro_do_espaco_de_busca(seed):
    rng = np.random.default_rng(seed)
    individuo = gerar_individuo_aleatorio(rng)
    mutado = mutar(individuo, probabilidade=0.5, intensidade=0.3, rng=rng)
    assert validar_individuo(mutado)


def test_mutacao_com_probabilidade_zero_nao_altera_nada():
    rng = np.random.default_rng(2)
    individuo = gerar_individuo_aleatorio(rng)
    mutado = mutar(
        individuo, probabilidade=0.0, intensidade=0.5, rng=rng, prob_troca_familia=0.0
    )
    assert mutado == individuo


def test_prob_troca_familia_zero_preserva_a_familia():
    """A troca de família é desacoplada da mutação dos genes (decisão pós-auditoria): com
    `prob_troca_familia=0`, a família nunca muda, mesmo com mutação de genes em 100%.
    """
    rng = np.random.default_rng(50)
    for _ in range(50):
        individuo = gerar_individuo_aleatorio(rng)
        mutado = mutar(
            individuo, probabilidade=1.0, intensidade=0.3, rng=rng, prob_troca_familia=0.0
        )
        assert mutado["familia"] == individuo["familia"]


def test_prob_troca_familia_um_sempre_troca_a_familia():
    rng = np.random.default_rng(51)
    for _ in range(50):
        individuo = gerar_individuo_aleatorio(rng)
        mutado = mutar(
            individuo, probabilidade=0.0, intensidade=0.3, rng=rng, prob_troca_familia=1.0
        )
        assert mutado["familia"] != individuo["familia"]


def test_normalizacao_iguala_mutacoes_esperadas_entre_familias():
    """Com `normalizar_por_familia=True`, o número esperado de mutações nos genes EXPRESSOS
    deve ser estatisticamente o mesmo em qualquer família — corrige a assimetria medida na
    auditoria (RF sofria 2,3x mais mutação efetiva que SVM à mesma probabilidade).
    """
    from src.ag.cromossomo import SCHEMA

    rng = np.random.default_rng(52)
    n_amostras = 3000
    probabilidade = 0.15
    medias: dict[str, float] = {}

    for familia in ("random_forest", "svm_rbf", "knn"):
        mutacoes_ativas = []
        for _ in range(n_amostras):
            individuo = gerar_individuo_aleatorio(rng)
            individuo["familia"] = familia
            mutado = mutar(
                individuo,
                probabilidade=probabilidade,
                intensidade=0.1,
                rng=rng,
                prob_troca_familia=0.0,
                normalizar_por_familia=True,
            )
            n_alterados = sum(
                1
                for nome in SCHEMA[familia]
                if mutado[familia][nome] != individuo[familia][nome]
            )
            mutacoes_ativas.append(n_alterados)
        medias[familia] = float(np.mean(mutacoes_ativas))

    valores = list(medias.values())
    # as médias devem ficar próximas entre si (tolerância generosa para ruído amostral e
    # para mutações que, por sorte, caem no mesmo valor do gene)
    assert max(valores) - min(valores) < 0.35, f"assimetria persistente entre famílias: {medias}"


def test_sem_normalizacao_a_assimetria_reaparece():
    """Contraprova: documenta o comportamento que a normalização corrige. Sem ela, famílias
    com mais genes sofrem mais mutações efetivas.
    """
    from src.ag.cromossomo import SCHEMA

    rng = np.random.default_rng(53)
    n_amostras = 2000
    probabilidade = 0.3
    medias: dict[str, float] = {}

    for familia in ("random_forest", "svm_rbf"):
        contagens = []
        for _ in range(n_amostras):
            individuo = gerar_individuo_aleatorio(rng)
            individuo["familia"] = familia
            mutado = mutar(
                individuo,
                probabilidade=probabilidade,
                intensidade=0.1,
                rng=rng,
                prob_troca_familia=0.0,
                normalizar_por_familia=False,
            )
            contagens.append(
                sum(
                    1
                    for nome in SCHEMA[familia]
                    if mutado[familia][nome] != individuo[familia][nome]
                )
            )
        medias[familia] = float(np.mean(contagens))

    assert medias["random_forest"] > medias["svm_rbf"] * 1.5, (
        f"esperava assimetria sem normalização, obtive {medias}"
    )


def test_mutacao_com_probabilidade_um_altera_o_individuo():
    rng = np.random.default_rng(3)
    individuo = gerar_individuo_aleatorio(rng)
    mutado = mutar(individuo, probabilidade=1.0, intensidade=0.3, rng=rng)
    assert mutado != individuo
    assert validar_individuo(mutado)


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------


def test_torneio_prefere_fitness_maior_na_maioria_dos_sorteios():
    """Teste explícito do plano §7.2: torneio prefere fitness maior em >50% dos sorteios."""
    rng = np.random.default_rng(10)
    populacao = [gerar_individuo_aleatorio(rng) for _ in range(30)]
    # fitnesses bem distintos para que o resultado do torneio seja decidido pelo valor,
    # não por sorte de empate
    fitnesses = list(np.linspace(0.1, 0.9, len(populacao)))

    n_rodadas = 500
    vitorias_do_melhor_do_par = 0
    for _ in range(n_rodadas):
        idx_a, idx_b = rng.choice(len(populacao), size=2, replace=False)
        # roda um torneio k=2 restrito a este par específico, construindo uma mini-população
        par_pop = [populacao[idx_a], populacao[idx_b]]
        par_fit = [fitnesses[idx_a], fitnesses[idx_b]]
        selecionado = selecao_torneio(par_pop, par_fit, k=2, rng=rng)
        melhor_do_par = par_pop[0] if par_fit[0] > par_fit[1] else par_pop[1]
        if selecionado == melhor_do_par:
            vitorias_do_melhor_do_par += 1

    # com k=2 sobre apenas 2 candidatos, o melhor SEMPRE ganha — este teste checa que a
    # implementação não inverte a lógica de "maior fitness vence"
    assert vitorias_do_melhor_do_par == n_rodadas


def test_torneio_com_populacao_inteira_favorece_fitness_alto_estatisticamente():
    """Com k=3 sobre uma população de fitness variado, indivíduos de fitness alto devem ser
    selecionados mais que proporcionalmente à sua fração na população.
    """
    rng = np.random.default_rng(11)
    populacao = [gerar_individuo_aleatorio(rng) for _ in range(20)]
    fitnesses = list(np.linspace(0.1, 0.9, len(populacao)))

    selecoes = [selecao_torneio(populacao, fitnesses, k=3, rng=rng) for _ in range(2000)]
    indices_selecionados = [populacao.index(s) for s in selecoes]
    media_fitness_selecionado = np.mean([fitnesses[i] for i in indices_selecionados])

    assert media_fitness_selecionado > np.mean(fitnesses)


def test_roleta_e_executavel_e_retorna_individuo_da_populacao():
    """A roleta é mantida só para a ablação da Etapa D — este teste confere que ela
    funciona, não que tem a mesma pressão seletiva do torneio (o plano prevê que NÃO tem).
    """
    rng = np.random.default_rng(12)
    populacao = [gerar_individuo_aleatorio(rng) for _ in range(10)]
    fitnesses = list(np.linspace(0.75, 0.92, len(populacao)))  # faixa estreita, como no PCOS
    selecionado = selecao_roleta(populacao, fitnesses, rng)
    assert selecionado in populacao


# ---------------------------------------------------------------------------
# População e elitismo
# ---------------------------------------------------------------------------


def test_inicializar_populacao_tem_o_tamanho_pedido():
    rng = np.random.default_rng(20)
    pop = inicializar_populacao(tamanho=30, rng=rng, fracao_hotstart=0.1)
    assert len(pop) == 30
    assert all(validar_individuo(ind) for ind in pop)


def test_elitismo_preserva_os_melhores():
    rng = np.random.default_rng(21)
    pop = inicializar_populacao(tamanho=10, rng=rng)
    fitnesses = list(rng.uniform(0.5, 0.9, size=10))
    ordem = sorted(range(10), key=lambda i: fitnesses[i], reverse=True)
    pop_ordenada = [pop[i] for i in ordem]
    fit_ordenado = [fitnesses[i] for i in ordem]

    elite = aplicar_elitismo(pop_ordenada, fit_ordenado, n_elite=2)

    assert len(elite) == 2
    assert elite[0] == pop_ordenada[0]
    assert elite[1] == pop_ordenada[1]
    # são cópias — mutar a elite não deve afetar a população original
    elite[0]["familia"] = "__mutado__"
    assert pop_ordenada[0]["familia"] != "__mutado__"


def test_populacao_mantem_tamanho_constante_ao_longo_de_geracoes():
    """Teste explícito do plano §7.2: simula algumas gerações completas (seleção + crossover
    + mutação + elitismo) e confere que o tamanho da população nunca muda — pega o bug
    clássico de off-by-one no laço geracional.
    """
    rng = np.random.default_rng(30)
    tamanho_pop = 20
    n_elite = 2
    pop = inicializar_populacao(tamanho=tamanho_pop, rng=rng)

    for _geracao in range(5):
        fitnesses = list(rng.uniform(0.1, 0.9, size=len(pop)))
        ordem = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)
        pop_ordenada = [pop[i] for i in ordem]
        fit_ordenado = [fitnesses[i] for i in ordem]

        nova_pop = aplicar_elitismo(pop_ordenada, fit_ordenado, n_elite)
        while len(nova_pop) < tamanho_pop:
            p1 = selecao_torneio(pop_ordenada, fit_ordenado, k=2, rng=rng)
            p2 = selecao_torneio(pop_ordenada, fit_ordenado, k=2, rng=rng)
            f1, f2 = cruzar(p1, p2, taxa_crossover=0.8, rng=rng)
            f1 = mutar(f1, probabilidade=0.15, intensidade=0.1, rng=rng)
            nova_pop.append(f1)
            if len(nova_pop) < tamanho_pop:
                f2 = mutar(f2, probabilidade=0.15, intensidade=0.1, rng=rng)
                nova_pop.append(f2)

        assert len(nova_pop) == tamanho_pop
        assert all(validar_individuo(ind) for ind in nova_pop)
        pop = nova_pop
