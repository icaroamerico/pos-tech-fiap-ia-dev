"""CLI para rodar os experimentos do AG (plano §7.4).

Separar a execução (script) da análise (notebook) é deliberado: execuções longas num notebook
são frágeis (kernel morre, estado se perde, diff do .ipynb fica ilegível com outputs). O
script roda e persiste em `runs/`; o notebook lê e explica.

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


def _rodar_um(construtor, semente: int, dados, avaliar_holdout: bool) -> dict:
    config = construtor(semente)
    nome_execucao = f"{config.nome}_seed{semente}"
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


def rodar_experimentos(nomes: list[str], sementes: tuple[int, ...], avaliar_holdout: bool) -> list[dict]:
    dados = carregar_dados_treino()
    resumo: list[dict] = []
    for nome in nomes:
        print(f"\n{'=' * 78}\n{nome}\n{'-' * 78}\n{HIPOTESES.get(nome, '')}\n{'=' * 78}")
        for semente in sementes:
            resumo.append(
                _rodar_um(EXPERIMENTOS_PRINCIPAIS[nome], semente, dados, avaliar_holdout)
            )
    return resumo


def rodar_etapa_a(sementes: tuple[int, ...]) -> list[dict]:
    """Etapa A (plano §4.8): um AG por família, para a tabela original vs. otimizado."""
    dados = carregar_dados_treino()
    resumo: list[dict] = []
    for familia in FAMILIAS:
        print(f"\n{'=' * 78}\nEtapa A — família {familia}\n{'=' * 78}")
        for semente in sementes:
            resumo.append(
                _rodar_um(
                    lambda s, f=familia: a_familia_fixa(f, s), semente, dados, avaliar_holdout=True
                )
            )
    return resumo


def rodar_etapa_c(n_avaliacoes: int, sementes: tuple[int, ...]) -> list[dict]:
    """Etapa C (plano §4.8): busca aleatória com o mesmo orçamento do AG."""
    dados = carregar_dados_treino()
    resumo: list[dict] = []
    for semente in sementes:
        print(f"\n=== Busca aleatória | {n_avaliacoes} avaliações | semente {semente}")
        inicio = time.perf_counter()
        resultado = busca_aleatoria(n_avaliacoes, dados, semente=semente)
        tempo = time.perf_counter() - inicio

        escritor = EscritorExecucao(f"C_busca_aleatoria_seed{semente}")
        escritor.escrever_config(
            {"metodo": "busca_aleatoria", "n_avaliacoes": n_avaliacoes, "semente": semente}
        )
        metricas = avaliar_no_holdout(resultado["melhor_individuo"])
        escritor.escrever_metricas_holdout(metricas)
        escritor._escrever_json(
            "best_individual.json",
            {
                "individuo": resultado["melhor_individuo"],
                "fitness": resultado["melhor_fitness"],
                "metricas_cv": resultado["melhor_metricas"],
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
        "--sem-holdout",
        action="store_true",
        help="Não avaliar no conjunto de teste (útil para execuções exploratórias).",
    )
    args = parser.parse_args()

    sementes = tuple(args.sementes)
    resumo: list[dict] = []

    if args.experimento:
        nomes = list(EXPERIMENTOS_PRINCIPAIS) if args.experimento == "todos" else [args.experimento]
        resumo += rodar_experimentos(nomes, sementes, avaliar_holdout=not args.sem_holdout)
    if args.etapa_a:
        resumo += rodar_etapa_a(sementes)
    if args.etapa_c:
        resumo += rodar_etapa_c(args.n_avaliacoes_aleatoria, sementes)

    if not resumo:
        parser.error("escolha --experimento, --etapa-a ou --etapa-c")

    caminho = DIR_RUNS / "resumo_execucoes.json"
    anteriores = []
    if caminho.exists():
        anteriores = json.loads(caminho.read_text(encoding="utf-8"))
    caminho.write_text(
        json.dumps(anteriores + resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResumo acumulado em {caminho}")


if __name__ == "__main__":
    main()
