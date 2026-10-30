"""Motor do algoritmo genético — o laço evolutivo.

Segue o laço evolutivo canônico:

    inicialização -> avaliação -> critério de parada? -> seleção -> cruzamento -> mutação -+
                          ^                                                                |
                          +----------------------------------------------------------------+
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from joblib import Parallel, delayed

from src.ag.cromossomo import FAMILIAS
from src.ag.metricas import (
    desvio_padrao_fitness,
    distancia_euclidiana_media,
    distribuicao_familias,
    entropia_genetica_familia,
    entropia_genetica_media_categorica,
    frequencia_class_weight_balanced,
    media_fitness,
    melhor_fitness as metrica_melhor_fitness,
    numero_genotipos_unicos,
    numero_solucoes_unicas,
)
from src.ag.operadores import (
    PROB_TROCA_FAMILIA_PADRAO,
    cruzar,
    mutar,
    selecao_roleta,
    selecao_torneio,
)
from src.ag.populacao import aplicar_elitismo, inicializar_populacao
from src.fitness.fitness_pcos import DadosTreino, calcular_fitness
from src.monitoramento.registro import (
    Cronometro,
    EscritorExecucao,
    RegistroGeracao,
    ResultadoExecucao,
    configurar_logger,
)

Individuo = dict[str, Any]
TipoSelecao = Literal["torneio", "roleta"]


@dataclass
class ConfigAG:
    """Parâmetros de uma execução do AG.

    Os valores default correspondem ao experimento **E1 (baseline)**.
    """

    nome: str = "E1_baseline"
    tamanho_populacao: int = 30
    n_geracoes: int = 15
    prob_mutacao: float = 0.15
    intensidade_mutacao: float = 0.10
    taxa_crossover: float = 0.8
    torneio_k: int = 2
    n_elite: int = 2
    selecao: TipoSelecao = "torneio"
    prob_troca_familia: float = PROB_TROCA_FAMILIA_PADRAO
    normalizar_mutacao_por_familia: bool = True
    fracao_hotstart: float = 0.1
    semente: int = 42

    # Família fixa: quando definida, o AG otimiza SÓ aquela família (Etapa A).
    # Com `None`, o gene `familia` é livre e o AG faz seleção de modelo junto com o ajuste
    # de hiperparâmetros (Etapa B).
    familia_fixa: str | None = None

    # Critérios de parada — o primeiro que disparar encerra a execução.
    paciencia_estagnacao: int = 5
    tolerancia_melhoria: float = 1e-4
    piso_entropia: float = 0.0

    n_jobs: int = 1  # Evita overhead de multiprocessing. CV=5 já paraleliza dentro de cada modelo.

    def __post_init__(self) -> None:
        if self.familia_fixa is not None and self.familia_fixa not in FAMILIAS:
            raise ValueError(f"familia_fixa inválida: {self.familia_fixa!r}")
        if self.n_elite >= self.tamanho_populacao:
            raise ValueError("n_elite tem de ser menor que tamanho_populacao")


def _avaliar_populacao(
    populacao: list[Individuo],
    dados: DadosTreino,
    cache: dict[tuple, dict[str, float]],
    n_jobs: int,
) -> tuple[list[dict[str, float]], int]:
    """Avalia a população, usando o cache para o que já foi visto e paralelizando o resto.

    Estratégia: primeiro resolve os acertos de cache em série (custo zero),
    depois manda só os indivíduos inéditos para o `joblib`. Isso evita o custo de serializar
    para os workers indivíduos cuja resposta já se conhece.

    O paralelismo é no nível do INDIVÍDUO, com `n_jobs=1` dentro de cada modelo — combinar
    os dois níveis causa *oversubscription* de threads e fica mais lento.
    """
    from src.ag.cromossomo import chave_fenotipo

    chaves = [chave_fenotipo(ind) for ind in populacao]
    resultados: list[dict[str, float] | None] = [None] * len(populacao)
    n_cache_hits = 0

    indices_pendentes: list[int] = []
    chaves_pendentes_vistas: dict[tuple, int] = {}
    for i, chave in enumerate(chaves):
        if chave in cache:
            resultados[i] = cache[chave]
            n_cache_hits += 1
        elif chave in chaves_pendentes_vistas:
            # duplicata DENTRO da mesma geração — avalia uma vez e replica
            n_cache_hits += 1
            indices_pendentes.append(i)
        else:
            chaves_pendentes_vistas[chave] = i
            indices_pendentes.append(i)

    indices_para_calcular = [i for i in indices_pendentes if chaves_pendentes_vistas.get(chaves[i]) == i]

    if indices_para_calcular:
        calculados = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(calcular_fitness)(populacao[i], dados, 1) for i in indices_para_calcular
        )
        for i, resultado in zip(indices_para_calcular, calculados):
            cache[chaves[i]] = resultado

    for i, chave in enumerate(chaves):
        if resultados[i] is None:
            resultados[i] = cache[chave]

    return [r for r in resultados if r is not None], n_cache_hits


def _checar_parada(
    config: ConfigAG,
    geracao: int,
    historico_melhor: list[float],
    entropia_atual: float,
) -> str | None:
    """Avalia os três critérios de parada. Retorna o motivo ou `None`."""
    if geracao >= config.n_geracoes - 1:
        return "maximo_de_geracoes"

    if len(historico_melhor) > config.paciencia_estagnacao:
        melhor_recente = max(historico_melhor[-config.paciencia_estagnacao :])
        melhor_anterior = max(historico_melhor[: -config.paciencia_estagnacao])
        if melhor_recente - melhor_anterior < config.tolerancia_melhoria:
            return f"estagnacao_{config.paciencia_estagnacao}_geracoes"

    if config.familia_fixa is None and entropia_atual < config.piso_entropia:
        return "colapso_de_diversidade"

    return None


def executar_ag(
    config: ConfigAG,
    dados: DadosTreino,
    escritor: EscritorExecucao | None = None,
    logger: logging.Logger | None = None,
    verboso: bool = True,
) -> ResultadoExecucao:
    """Executa o algoritmo genético completo e devolve o resultado + histórico.

    Não toca o conjunto de teste em nenhum momento — `dados` é um `DadosTreino`, que
    estruturalmente não carrega o holdout.
    """
    nome_execucao = f"{config.nome}_seed{config.semente}"
    if logger is None:
        logger = configurar_logger(
            nome_execucao, nivel_console=logging.INFO if verboso else logging.WARNING
        )
    if escritor is None:
        escritor = EscritorExecucao(nome_execucao)

    escritor.escrever_config(asdict(config))
    rng = np.random.default_rng(config.semente)
    cache: dict[tuple, dict[str, float]] = {}

    logger.info(
        "AG %s | pop=%d ger=%d mut=%.2f(int=%.2f) cx=%.2f k=%d elite=%d sel=%s familia=%s",
        config.nome,
        config.tamanho_populacao,
        config.n_geracoes,
        config.prob_mutacao,
        config.intensidade_mutacao,
        config.taxa_crossover,
        config.torneio_k,
        config.n_elite,
        config.selecao,
        config.familia_fixa or "livre",
    )

    populacao = inicializar_populacao(
        config.tamanho_populacao, rng, fracao_hotstart=config.fracao_hotstart
    )
    if config.familia_fixa is not None:
        for individuo in populacao:
            individuo["familia"] = config.familia_fixa

    resultado = ResultadoExecucao(config=asdict(config))
    historico_melhor: list[float] = []
    cronometro_total = Cronometro()
    cronometro_total.__enter__()

    for geracao in range(config.n_geracoes):
        with Cronometro() as cronometro_geracao:
            with Cronometro() as cronometro_avaliacao:
                metricas_por_individuo, n_cache_hits = _avaliar_populacao(
                    populacao, dados, cache, config.n_jobs
                )
            fitnesses = [m["fitness"] for m in metricas_por_individuo]

            ordem = sorted(range(len(populacao)), key=lambda i: fitnesses[i], reverse=True)
            populacao = [populacao[i] for i in ordem]
            fitnesses = [fitnesses[i] for i in ordem]
            metricas_por_individuo = [metricas_por_individuo[i] for i in ordem]

            if fitnesses[0] > resultado.melhor_fitness:
                resultado.melhor_fitness = fitnesses[0]
                resultado.melhor_individuo = populacao[0]
                resultado.melhor_metricas = metricas_por_individuo[0]

            historico_melhor.append(fitnesses[0])
            resultado.total_avaliacoes += len(populacao)
            resultado.total_cache_hits += n_cache_hits

            entropia_familia = entropia_genetica_familia(populacao)

            with Cronometro() as cronometro_operadores:
                nova_populacao = aplicar_elitismo(populacao, fitnesses, config.n_elite)
                while len(nova_populacao) < config.tamanho_populacao:
                    if config.selecao == "torneio":
                        pai1 = selecao_torneio(populacao, fitnesses, config.torneio_k, rng)
                        pai2 = selecao_torneio(populacao, fitnesses, config.torneio_k, rng)
                    else:
                        pai1 = selecao_roleta(populacao, fitnesses, rng)
                        pai2 = selecao_roleta(populacao, fitnesses, rng)

                    filho1, filho2 = cruzar(pai1, pai2, config.taxa_crossover, rng)
                    for filho in (filho1, filho2):
                        if len(nova_populacao) >= config.tamanho_populacao:
                            break
                        mutado = mutar(
                            filho,
                            probabilidade=config.prob_mutacao,
                            intensidade=config.intensidade_mutacao,
                            rng=rng,
                            prob_troca_familia=(
                                0.0 if config.familia_fixa is not None
                                else config.prob_troca_familia
                            ),
                            normalizar_por_familia=config.normalizar_mutacao_por_familia,
                        )
                        if config.familia_fixa is not None:
                            mutado["familia"] = config.familia_fixa
                        nova_populacao.append(mutado)

        registro = RegistroGeracao(
            experimento=config.nome,
            semente=config.semente,
            geracao=geracao,
            melhor_fitness=metrica_melhor_fitness(fitnesses),
            fitness_medio=media_fitness(fitnesses),
            desvio_padrao=desvio_padrao_fitness(fitnesses),
            entropia_familia=entropia_familia,
            entropia_media_categorica=entropia_genetica_media_categorica(populacao),
            distancia_euclidiana_media=distancia_euclidiana_media(populacao),
            n_solucoes_unicas=numero_solucoes_unicas(populacao),
            n_genotipos_unicos=numero_genotipos_unicos(populacao),
            freq_class_weight_balanced=frequencia_class_weight_balanced(populacao),
            distribuicao_familias=distribuicao_familias(populacao),
            familia_do_melhor=populacao[0]["familia"],
            n_avaliacoes=len(populacao),
            n_cache_hits=n_cache_hits,
            tempo_geracao_s=cronometro_geracao.decorrido,
            tempo_avaliacao_s=cronometro_avaliacao.decorrido,
            tempo_operadores_s=cronometro_operadores.decorrido,
        )
        resultado.historico.append(registro)
        escritor.anexar_geracao(registro)

        logger.info(
            "ger %2d | melhor=%.4f medio=%.4f sd=%.4f | H=%.3f unicos=%2d | %s | "
            "cache=%2d/%2d | %.1fs",
            geracao,
            registro.melhor_fitness,
            registro.fitness_medio,
            registro.desvio_padrao,
            registro.entropia_familia,
            registro.n_solucoes_unicas,
            registro.familia_do_melhor,
            registro.n_cache_hits,
            registro.n_avaliacoes,
            registro.tempo_geracao_s,
        )

        motivo = _checar_parada(config, geracao, historico_melhor, entropia_familia)
        if motivo:
            resultado.motivo_parada = motivo
            resultado.geracoes_executadas = geracao + 1
            break

        populacao = nova_populacao
    else:
        resultado.motivo_parada = "maximo_de_geracoes"
        resultado.geracoes_executadas = config.n_geracoes

    cronometro_total.__exit__()
    resultado.tempo_total_s = cronometro_total.decorrido
    resultado.populacao_final = populacao

    escritor.escrever_resultado(resultado)
    logger.info(
        "FIM %s | melhor=%.4f (%s) | %d gerações | parada=%s | %.1fs | cache %.0f%%",
        config.nome,
        resultado.melhor_fitness,
        resultado.melhor_individuo["familia"] if resultado.melhor_individuo else "?",
        resultado.geracoes_executadas,
        resultado.motivo_parada,
        resultado.tempo_total_s,
        100 * resultado.total_cache_hits / max(resultado.total_avaliacoes, 1),
    )
    return resultado
