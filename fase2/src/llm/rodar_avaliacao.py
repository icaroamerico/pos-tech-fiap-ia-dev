"""Executa a avaliação de LLM de ponta a ponta e grava os artefatos.

Este módulo é para a avaliação de LLM o que `src/experimentos/rodar_ag.py` é para o AG:
o caminho único de execução. O notebook 03 explica cada etapa célula por célula; o
notebook 00 chama este módulo. Os dois produzem os mesmos arquivos no mesmo lugar,
porque executam o mesmo código — a diferença está só na forma de apresentar.

O pipeline é o do §5 do relatório, nesta ordem:

1. recupera o melhor indivíduo do Random Forest da Etapa A (é a família que expõe
   `feature_importances_`, necessárias para o contexto da LLM);
2. treina o pipeline no conjunto de treino e prediz no conjunto de teste. O holdout é
   lido aqui e na avaliação final de `avaliar_no_holdout`, nunca pelo fitness: a busca
   do AG é decidida só pela validação cruzada dentro do treino;
3. monta a amostra estratificada: 5 positivas de alta confiança, 5 negativas de alta
   confiança e 5 no limiar;
4. monta o contexto de cada paciente, sempre pela allow-list;
5. gera e avalia as 45 saídas (15 pacientes × 3 templates);
6. grava JSON e Markdown.

O CSV da rubrica **não** é regravado: ele é preenchido à mão e nenhuma execução
reproduz as notas. Use `--regravar-csv` para emitir uma planilha em branco, ciente de
que isso apaga as notas existentes.

Uso:

    python -m src.llm.rodar_avaliacao                      # ollama, llama3.2:3b
    python -m src.llm.rodar_avaliacao --provedor simulado  # controle do verificador
    python -m src.llm.rodar_avaliacao --sufixo _teste      # não toca os artefatos oficiais
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.ag.cromossomo import decodificar
from src.experimentos.analise import carregar_melhor
from src.fitness.fitness_pcos import DadosTreino, construir_pipeline
from src.llm.avaliacao import (
    avaliar_uma_saida,
    persistir_avaliacao,
    resumir_metricas_automaticas,
)
from src.llm.cliente import ConfigLLM
from src.llm.contexto import montar_contexto_paciente
from src.modelagem.treinamento import (
    carregar_dados,
    dividir_dados,
    identificar_colunas_tipo,
    preparar_features,
)

TEMPLATES = ["v1_zero_shot", "v2_few_shot", "v3_few_shot_estruturado"]
EXECUCAO_INTERPRETACAO = "A_random_forest_seed42"


def preparar_predicoes(execucao: str = EXECUCAO_INTERPRETACAO):
    """Treina o melhor Random Forest da Etapa A e prediz no conjunto de teste."""
    melhor = carregar_melhor(execucao)
    individuo = melhor["individuo"]

    df = carregar_dados()
    X, y = preparar_features(df)
    col_num, col_cat = identificar_colunas_tipo(X)
    X_treino, X_teste, y_treino, y_teste = dividir_dados(X, y)

    pipeline = construir_pipeline(individuo, DadosTreino(X_treino, y_treino, col_num, col_cat))
    pipeline.fit(X_treino, y_treino)

    probabilidades = pipeline.predict_proba(X_teste)[:, 1]
    predicoes = pipeline.predict(X_teste)

    familia, _ = decodificar(individuo)
    return pipeline, X_teste, probabilidades, predicoes, familia, melhor["fitness"]


def mapear_importancias(pipeline, X_teste) -> dict[str, float]:
    """Soma a importância das colunas transformadas de volta na coluna original."""
    importancias_brutas = pipeline.named_steps["modelo"].feature_importances_
    nomes_transformados = pipeline.named_steps["preprocessador"].get_feature_names_out()

    # Casa o prefixo MAIS LONGO, não o primeiro. Com `break` na primeira coluna que
    # casava, toda dummy `Blood Group_*` caía em `Blood Group` se essa coluna viesse
    # antes na ordem do DataFrame: 5 atribuições erradas. Ordenar por tamanho
    # decrescente resolve porque nomes de coluna não se repetem.
    colunas_por_tamanho = sorted(X_teste.columns, key=len, reverse=True)

    importancias: dict[str, float] = {}
    for nome_transformado, peso in zip(nomes_transformados, importancias_brutas):
        nome_base = nome_transformado.split("__", 1)[-1]
        for coluna in colunas_por_tamanho:
            if nome_base.startswith(coluna):
                importancias[coluna] = importancias.get(coluna, 0.0) + float(peso)
                break
    return importancias


def amostrar_estratificado(
    probabilidades, predicoes, n_por_estrato: int = 5, semente: int = 42
) -> dict[str, np.ndarray]:
    """5 positivas de alta confiança, 5 negativas de alta confiança, 5 no limiar.

    Os modos de falha da LLM são diferentes por faixa de probabilidade: num caso no
    limiar, exagerar a certeza é o erro provável; num caso de alta confiança, o erro
    provável é inventar um número. Estratificar é o que deixa os dois visíveis.
    """
    rng = np.random.default_rng(semente)
    idx_todos = np.arange(len(probabilidades))

    limiar = idx_todos[(probabilidades >= 0.4) & (probabilidades <= 0.6)]
    pos_alta = idx_todos[(predicoes == 1) & (probabilidades > 0.6)]
    neg_alta = idx_todos[(predicoes == 0) & (probabilidades < 0.4)]

    def amostrar(pool, n):
        return rng.choice(pool, size=min(n, len(pool)), replace=False)

    return {
        "positivo_alta_confianca": amostrar(pos_alta, n_por_estrato),
        "negativo_alta_confianca": amostrar(neg_alta, n_por_estrato),
        "limiar": amostrar(limiar, n_por_estrato),
    }


def montar_contextos(X_teste, probabilidades, predicoes, importancias, amostra) -> dict:
    contextos = {}
    for estrato, indices in amostra.items():
        for i in indices:
            contextos[f"{estrato}_{i}"] = montar_contexto_paciente(
                X_teste.iloc[i],
                predicao=int(predicoes[i]),
                probabilidade=float(probabilidades[i]),
                importancias=importancias,
            )
    return contextos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provedor",
        default="ollama",
        choices=["ollama", "simulado"],
        help=(
            "`ollama` é a LLM real, local — é o que torna os requisitos 3d e 3e "
            "mensuráveis. `simulado` é determinístico e ignora o template: serve de "
            "controle do verificador, não de comparação entre V1/V2/V3."
        ),
    )
    parser.add_argument("--modelo", default="llama3.2:3b", help="Modelo do provedor.")
    parser.add_argument("--temperatura", type=float, default=0.2)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument(
        "--n-por-estrato",
        type=int,
        default=5,
        help="Pacientes por estrato. 5 dá as 15 da amostra oficial.",
    )
    parser.add_argument(
        "--sufixo",
        default="",
        help=(
            "Sufixo no nome dos arquivos (ex.: --sufixo _teste grava "
            "avaliacao_ollama_teste_saidas.json). Sem sufixo, a execução sobrescreve "
            "os artefatos citados no relatório."
        ),
    )
    parser.add_argument(
        "--regravar-csv",
        action="store_true",
        help=(
            "Regravar o CSV da rubrica em branco. As notas da rubrica são preenchidas à "
            "mão e nenhuma execução as reproduz: sem esta opção o CSV não é tocado."
        ),
    )
    args = parser.parse_args()

    config_llm = ConfigLLM(
        provedor=args.provedor,
        modelo=args.modelo,
        temperatura=args.temperatura,
        semente=args.semente,
    )

    print(f"{'=' * 78}\nAvaliação de LLM — provedor {args.provedor} | modelo {args.modelo}\n{'=' * 78}")

    pipeline, X_teste, probabilidades, predicoes, familia, fitness = preparar_predicoes()
    print(f"modelo das interpretações: {familia} (fitness CV {fitness:.4f})")
    print(f"predições geradas para {len(X_teste)} pacientes do conjunto de teste")

    importancias = mapear_importancias(pipeline, X_teste)
    amostra = amostrar_estratificado(
        probabilidades, predicoes, args.n_por_estrato, args.semente
    )
    for estrato, indices in amostra.items():
        print(
            f"  {estrato:26} {len(indices)} pacientes "
            f"(prob {probabilidades[indices].min():.2f}–{probabilidades[indices].max():.2f})"
        )

    contextos = montar_contextos(X_teste, probabilidades, predicoes, importancias, amostra)
    total = len(contextos) * len(TEMPLATES)
    print(f"\n{len(contextos)} contextos montados (todos pela allow-list)")
    print(f"gerando {total} saídas ({len(contextos)} pacientes × {len(TEMPLATES)} templates)\n")

    inicio = time.time()
    saidas = []
    for template in TEMPLATES:
        for id_amostra, contexto in contextos.items():
            saidas.append(avaliar_uma_saida(id_amostra, contexto, template, config_llm))
            print(
                f"  [{len(saidas):2d}/{total}] {template:24} {id_amostra:28} "
                f"{saidas[-1].latencia_s:5.1f}s",
                flush=True,
            )

    # Uma falha aqui precisa aparecer: contar menos de 45 saídas e seguir em frente
    # invalidaria todas as médias do §5.4 sem nenhum aviso.
    if len(saidas) != total:
        raise RuntimeError(f"esperava {total} saídas, obtive {len(saidas)}")

    nome = f"avaliacao_{args.provedor}{args.sufixo}"
    caminho = persistir_avaliacao(saidas, nome=nome, escrever_csv=args.regravar_csv)

    print(f"\n{'=' * 78}\n{len(saidas)} saídas em {time.time() - inicio:.0f}s\n{'=' * 78}")
    print(json.dumps(resumir_metricas_automaticas(saidas), ensure_ascii=False, indent=2))
    print(f"\nartefatos: {caminho.parent}/{nome}_saidas.json e {nome}_saidas.md")
    if not args.regravar_csv:
        print(f"{nome}_rubrica.csv não foi tocado (notas preenchidas à mão).")


if __name__ == "__main__":
    main()
