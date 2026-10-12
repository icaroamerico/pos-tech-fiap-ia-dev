"""As configurações de experimento do plano §4.6 e §4.8.

Os três experimentos obrigatórios são três pontos do eixo **exploração ↔ aproveitamento** da
Aula 3 de AG. As hipóteses estão declaradas aqui, ANTES de rodar — é o que diferencia "rodei
três configurações e reportei" de "usei o AG para testar o que a aula afirma".
"""

from __future__ import annotations

from src.ag.motor import ConfigAG

#: Sementes para repetição. AG é estocástico — comparar configurações por uma execução cada
#: é comparar ruído (plano §4.6).
SEMENTES = (42, 7, 2024)

HIPOTESES: dict[str, str] = {
    "E1_baseline": (
        "Ponto de referência. Sem ele, E2 e E3 não têm contra o quê comparar."
    ),
    "E2_exploracao": (
        "População 2x + mutação forte. A Aula 3 prevê 'maior diversidade genética; exploração "
        "eficiente de diferentes regiões', com a desvantagem de 'convergência mais lenta, pois "
        "a cada geração há mais indivíduos para avaliar'. PREVISÃO: fitness final igual ou "
        "levemente superior ao E1, atingido em geração mais tardia, com ~2x o tempo de parede."
    ),
    "E3_aproveitamento": (
        "População pequena + mutação fraca + torneio k=4. A Aula 3 prevê 'convergência "
        "potencialmente mais rápida' contra 'pode ficar presa em ótimos locais; possibilidade "
        "de convergência prematura'. PREVISÃO: convergência rápida nas primeiras gerações, "
        "platô mais baixo que E1, e entropia genética caindo visivelmente mais rápido."
    ),
    "D_roleta": (
        "Ablação do operador de seleção (plano §4.8 Etapa D). O §4.4 afirma que a roleta é "
        "inadequada aqui porque o fitness vive numa faixa estreita (~0,75-0,92) e a seleção "
        "proporcional dá pressão seletiva quase nula. PREVISÃO: (a) fitness final igual ou "
        "inferior ao E1-torneio; (b) média da população colada no melhor indivíduo desde as "
        "primeiras gerações; (c) convergência mais lenta ou ausente."
    ),
}


def e1_baseline(semente: int) -> ConfigAG:
    return ConfigAG(
        nome="E1_baseline",
        tamanho_populacao=30,
        n_geracoes=15,
        prob_mutacao=0.15,
        intensidade_mutacao=0.10,
        taxa_crossover=0.8,
        torneio_k=2,
        n_elite=2,
        selecao="torneio",
        semente=semente,
    )


def e2_exploracao(semente: int) -> ConfigAG:
    return ConfigAG(
        nome="E2_exploracao",
        tamanho_populacao=60,
        n_geracoes=15,
        prob_mutacao=0.35,
        intensidade_mutacao=0.25,
        taxa_crossover=0.9,
        torneio_k=2,
        n_elite=2,
        selecao="torneio",
        semente=semente,
    )


def e3_aproveitamento(semente: int) -> ConfigAG:
    return ConfigAG(
        nome="E3_aproveitamento",
        tamanho_populacao=15,
        n_geracoes=15,
        prob_mutacao=0.05,
        intensidade_mutacao=0.05,
        taxa_crossover=0.6,
        torneio_k=4,
        n_elite=2,
        selecao="torneio",
        semente=semente,
    )


def d_roleta(semente: int) -> ConfigAG:
    """Idêntico ao E1 em tudo, exceto o operador de seleção — é o que torna a comparação uma
    ablação limpa (uma variável de cada vez).
    """
    config = e1_baseline(semente)
    config.nome = "D_roleta"
    config.selecao = "roleta"
    return config


def a_familia_fixa(familia: str, semente: int) -> ConfigAG:
    """Etapa A do plano §4.8: AG restrito a uma família, para produzir a tabela
    "original vs. otimizado" linha a linha, com os 5 modelos da Fase 1.

    Orçamento menor que o E1 (pop 20 x 10 ger = 200 avaliações) porque o espaço de busca de
    uma família só é muito menor que o espaço conjunto das cinco.
    """
    return ConfigAG(
        nome=f"A_{familia}",
        tamanho_populacao=20,
        n_geracoes=10,
        prob_mutacao=0.15,
        intensidade_mutacao=0.10,
        taxa_crossover=0.8,
        torneio_k=2,
        n_elite=2,
        selecao="torneio",
        familia_fixa=familia,
        semente=semente,
    )


EXPERIMENTOS_PRINCIPAIS = {
    "E1_baseline": e1_baseline,
    "E2_exploracao": e2_exploracao,
    "E3_aproveitamento": e3_aproveitamento,
    "D_roleta": d_roleta,
}
