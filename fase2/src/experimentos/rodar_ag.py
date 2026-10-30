"""CLI para rodar os experimentos do AG.

Separar a execução (script) da análise (notebook) é deliberado: execuções longas num notebook
são frágeis (kernel morre, estado se perde, diff do .ipynb fica ilegível com outputs). O
script roda e persiste em `arquivos_saida/arquivos_analiticos/historico_execucoes_ag/`; o notebook lê e explica.

Uso:
    python -m src.experimentos.rodar_ag --experimento E1_baseline
    python -m src.experimentos.rodar_ag --experimento todos
    python -m src.experimentos.rodar_ag --etapa-a            # AG por família (5 execuções)
    python -m src.experimentos.rodar_ag --etapa-c            # baseline de busca aleatória
    python -m src.experimentos.rodar_ag --experimento E1_baseline --sementes 42
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

from src.ag.cromossomo import FAMILIAS
from src.ag.motor import executar_ag
from src.experimentos.configuracoes import (
    EXPERIMENTOS_PRINCIPAIS,
    HIPOTESES,
    SEMENTES,
    a_familia_fixa,
)
from src.fitness.avaliacao_final import avaliar_no_holdout, busca_aleatoria
from src.fitness.fitness_pcos import carregar_dados_treino
from src.monitoramento.registro import DIR_RUNS, EscritorExecucao


def _rodar_um(
    construtor, semente: int, dados, avaliar_holdout: bool, sufixo: str = "",
    n_jobs: int | None = None,
) -> dict:
    config = construtor(semente)
    if n_jobs is not None:
        config = replace(config, n_jobs=n_jobs)
    nome_execucao = f"{config.nome}_seed{semente}{sufixo}"
    escritor = EscritorExecucao(nome_execucao)
    resultado = executar_ag(config, dados, escritor=escritor)

    linha = {
        "experimento": config.nome,
        "semente": semente,
        "melhor_fitness": resultado.melhor_fitness,
        "familia_vencedora": resultado.melhor_individuo["familia"],
        "geracoes": resultado.geracoes_executadas,
        "motivo_parada": resultado.motivo_parada,
        "tempo_s": resultado.tempo_total_s,
        "taxa_cache": resultado.total_cache_hits / max(resultado.total_avaliacoes, 1),
    }

    if avaliar_holdout and resultado.melhor_individuo is not None:
        metricas = avaliar_no_holdout(resultado.melhor_individuo)
        escritor.escrever_metricas_holdout(metricas)
        linha["holdout"] = metricas

    return linha


def rodar_experimentos(
    nomes: list[str],
    sementes: tuple[int, ...],
    avaliar_holdout: bool,
    sufixo: str = "",
    n_jobs: int | None = None,
) -> list[dict]:
    dados = carregar_dados_treino()
    resumo: list[dict] = []
    for nome in nomes:
        print(f"\n{'=' * 78}\n{nome}\n{'-' * 78}\n{HIPOTESES.get(nome, '')}\n{'=' * 78}")
        for semente in sementes:
            resumo.append(
                _rodar_um(
                    EXPERIMENTOS_PRINCIPAIS[nome], semente, dados, avaliar_holdout,
                    sufixo, n_jobs,
                )
            )
    return resumo


def rodar_etapa_a(
    sementes: tuple[int, ...], sufixo: str = "", n_jobs: int | None = None
) -> list[dict]:
    """Etapa A: um AG por família, para a tabela original vs. otimizado."""
    dados = carregar_dados_treino()
    resumo: list[dict] = []
    for familia in FAMILIAS:
        print(f"\n{'=' * 78}\nEtapa A — família {familia}\n{'=' * 78}")
        for semente in sementes:
            resumo.append(
                _rodar_um(
                    lambda s, f=familia: a_familia_fixa(f, s),
                    semente,
                    dados,
                    avaliar_holdout=True,
                    sufixo=sufixo,
                    n_jobs=n_jobs,
                )
            )
    return resumo


def rodar_etapa_c(n_avaliacoes: int, sementes: tuple[int, ...], sufixo: str = "") -> list[dict]:
    """Etapa C: busca aleatória com o mesmo orçamento do AG."""
    dados = carregar_dados_treino()
    resumo: list[dict] = []
    for semente in sementes:
        print(f"\n=== Busca aleatória | {n_avaliacoes} avaliações | semente {semente}")
        inicio = time.perf_counter()
        resultado = busca_aleatoria(n_avaliacoes, dados, semente=semente)
        tempo = time.perf_counter() - inicio

        escritor = EscritorExecucao(f"C_busca_aleatoria_seed{semente}{sufixo}")
        escritor.escrever_config(
            {"metodo": "busca_aleatoria", "n_avaliacoes": n_avaliacoes, "semente": semente}
        )
        metricas = avaliar_no_holdout(resultado["melhor_individuo"])
        escritor.escrever_metricas_holdout(metricas)
        escritor.escrever_artefato_json(
            "best_individual.json",
            {
                "individuo": resultado["melhor_individuo"],
                "fitness": resultado["melhor_fitness"],
                "metricas_cv": resultado["melhor_metricas"],
            },
        )

        # Histórico real da busca aleatória. Busca aleatória não tem gerações, então a
        # linha é por avaliação e os campos são só os que existem de verdade: o fitness
        # sorteado e o melhor até ali. Não há entropia nem distância média a reportar, e
        # inventar esses campos para igualar o esquema do AG seria pior que omiti-los.
        melhor_ate_agora = float("-inf")
        for indice, fitness in enumerate(resultado["historico_fitness"]):
            melhor_ate_agora = max(melhor_ate_agora, fitness)
            escritor.anexar_linha_historico(
                {
                    "metodo": "busca_aleatoria",
                    "avaliacao": indice,
                    "fitness": fitness,
                    "melhor_fitness_ate_agora": melhor_ate_agora,
                }
            )

        escritor.escrever_artefato_json(
            "resumo.json",
            {
                "metodo": "busca_aleatoria",
                "semente": semente,
                "n_avaliacoes": n_avaliacoes,
                "n_avaliacoes_efetivas": resultado["n_avaliacoes_efetivas"],
                "melhor_fitness": resultado["melhor_fitness"],
                "familia_vencedora": resultado["melhor_individuo"]["familia"],
                "tempo_total_s": tempo,
                "holdout": metricas,
            },
        )

        print(
            f"  melhor fitness = {resultado['melhor_fitness']:.4f} "
            f"({resultado['melhor_individuo']['familia']}) | "
            f"{resultado['n_avaliacoes_efetivas']} avaliações efetivas | {tempo:.1f}s"
        )
        resumo.append(
            {
                "metodo": "busca_aleatoria",
                "semente": semente,
                "melhor_fitness": resultado["melhor_fitness"],
                "familia_vencedora": resultado["melhor_individuo"]["familia"],
                "n_avaliacoes_efetivas": resultado["n_avaliacoes_efetivas"],
                "tempo_s": tempo,
                "holdout": metricas,
            }
        )
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa os experimentos de AG do Projeto 1.")
    parser.add_argument(
        "--experimento",
        choices=[*EXPERIMENTOS_PRINCIPAIS, "todos"],
        help="Experimento a rodar (ou 'todos' para E1, E2, E3 e D).",
    )
    parser.add_argument("--etapa-a", action="store_true", help="AG por família (5 execuções).")
    parser.add_argument("--etapa-c", action="store_true", help="Baseline de busca aleatória.")
    parser.add_argument(
        "--sementes",
        type=int,
        nargs="+",
        default=list(SEMENTES),
        help=f"Sementes a usar (default: {list(SEMENTES)}).",
    )
    parser.add_argument(
        "--n-avaliacoes-aleatoria",
        type=int,
        default=450,
        help="Orçamento da busca aleatória (default 450 = pop 30 x 15 ger do E1).",
    )
    parser.add_argument(
        "--sufixo",
        default="",
        help=(
            "Sufixo no nome da pasta de cada execução (ex.: --sufixo _demo grava em "
            "E1_baseline_seed42_demo). Sem sufixo, a execução sobrescreve os artefatos "
            "oficiais que sustentam os números do relatório."
        ),
    )
    parser.add_argument(
        "--sem-holdout",
        action="store_true",
        help="Não avaliar no conjunto de teste (útil para execuções exploratórias).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Paralelismo da validação cruzada do fitness, sobrescrevendo o valor de "
            "cada ConfigAG. É só velocidade: o resultado por geração não muda, só o "
            "tempo de parede e o campo n_jobs gravado em config.json. As execuções "
            "oficiais do relatório usaram -1 (todos os núcleos)."
        ),
    )
    args = parser.parse_args()

    sementes = tuple(args.sementes)
    resumo: list[dict] = []

    if args.experimento:
        nomes = list(EXPERIMENTOS_PRINCIPAIS) if args.experimento == "todos" else [args.experimento]
        resumo += rodar_experimentos(
            nomes, sementes, avaliar_holdout=not args.sem_holdout, sufixo=args.sufixo,
            n_jobs=args.n_jobs,
        )
    if args.etapa_a:
        resumo += rodar_etapa_a(sementes, sufixo=args.sufixo, n_jobs=args.n_jobs)
    if args.etapa_c:
        resumo += rodar_etapa_c(args.n_avaliacoes_aleatoria, sementes, sufixo=args.sufixo)

    if not resumo:
        parser.error("escolha --experimento, --etapa-a ou --etapa-c")

    # O sufixo entra aqui também: sem isso uma execução de demonstração anexaria
    # suas linhas ao resumo das execuções oficiais, indistinguíveis delas.
    caminho = DIR_RUNS / f"resumo_execucoes{args.sufixo}.json"
    anteriores = []
    if caminho.exists():
        anteriores = json.loads(caminho.read_text(encoding="utf-8"))

    # Acumula as execuções que este comando não rodou (outras etapas, outras sementes) e
    # substitui as que rodou. Anexar sem substituir deixaria duas linhas para a mesma
    # execução a cada reexecução, e a pasta dela guarda só o resultado mais recente: o
    # resumo passaria a descrever um estado que não está mais no disco.
    def _chave(linha: dict) -> tuple:
        return (linha.get("experimento") or linha.get("metodo"), linha.get("semente"))

    linhas = {_chave(linha): linha for linha in anteriores}
    linhas.update({_chave(linha): linha for linha in resumo})
    caminho.write_text(
        json.dumps(list(linhas.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResumo acumulado em {caminho}")


if __name__ == "__main__":
    main()
