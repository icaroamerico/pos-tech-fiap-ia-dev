"""Avaliação da qualidade das interpretações geradas (plano §6.4).

O enunciado pede "avaliar a qualidade das interpretações geradas" — e esse é o sub-requisito
mais fácil de deixar virar "as respostas ficaram boas". Ele pede **avaliação**, então precisa
de instrumento.

Duas camadas:

  - **métricas automáticas** (este módulo as calcula) — objetivas e reprodutíveis: taxa de
    alucinação numérica, aderência estrutural, comprimento, latência, estabilidade;
  - **rubrica humana** (este módulo monta a planilha para preencher) — 6 critérios × 0-2
    pontos, aplicada por mim com justificativa escrita por amostra.

A taxa de alucinação numérica é a métrica mais importante do conjunto: é a única totalmente
objetiva e a que mede o risco mais grave.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.llm.cliente import ConfigLLM, gerar_explicacao
from src.llm.verificador import verificar_saida

RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
DIR_AVALIACAO = RAIZ_PROJETO / "avaliacao_llm"

SECOES_OBRIGATORIAS = (
    "Resumo da indicação do modelo",
    "Variáveis que mais pesaram",
    "Grau de confiança e limitações",
    "Próximo passo sugerido",
)

#: A rubrica do plano §6.4 — 6 critérios, 0 a 2 pontos cada.
RUBRICA: dict[str, dict[int, str]] = {
    "fidelidade_numerica": {
        0: "Cita número errado",
        1: "Omite números relevantes",
        2: "Todos os números batem com o modelo",
    },
    "fidelidade_features": {
        0: "Cita variável ausente ou irrelevante",
        1: "Genérico",
        2: "Cita as features de maior peso real",
    },
    "calibracao_incerteza": {
        0: "Afirma certeza",
        1: "Menciona vagamente",
        2: "Declara probabilidade e limitação",
    },
    "nao_diagnostico": {
        0: "Emite diagnóstico",
        1: "Ambíguo",
        2: "Fica claramente no apoio à interpretação",
    },
    "acionabilidade": {
        0: "Sem indicação de próximo passo",
        1: "Vago",
        2: "Próximo passo claro e apropriado",
    },
    "clareza_medico": {
        0: "Confuso ou jargão de ML cru",
        1: "Aceitável",
        2: "Direto e adequado ao público",
    },
}


@dataclass
class SaidaAvaliada:
    """Uma saída da LLM com suas métricas automáticas."""

    id_amostra: str
    template: str
    estrato: str
    texto: str
    contexto: dict[str, Any]
    # métricas automáticas
    n_numeros_citados: int
    n_numeros_nao_justificados: int
    taxa_alucinacao_numerica: float
    secoes_ausentes: list[str]
    aderencia_estrutural: float
    #: Os números concretos que não encontraram respaldo no contexto — guardados, e não só
    #: contados, porque a inspeção humana precisa saber QUAL número foi inventado.
    numeros_problematicos: list[float] = field(default_factory=list)
    termos_proibidos: list[str] = field(default_factory=list)
    n_caracteres: int = 0
    latencia_s: float = 0.0
    # rubrica humana — preenchida depois, à mão
    rubrica: dict[str, int | None] = field(default_factory=dict)
    justificativa: str = ""


def _estratificar(probabilidade: float, predicao: int) -> str:
    """Classifica a amostra em um dos três estratos do plano §6.4.

    A estratificação importa porque os modos de falha são diferentes por faixa: num caso de
    alta confiança o risco é excesso de assertividade; num caso de limiar, o risco é a LLM
    não sinalizar a incerteza.
    """
    if 0.4 <= probabilidade <= 0.6:
        return "limiar"
    return "positivo_alta_confianca" if predicao == 1 else "negativo_alta_confianca"


def avaliar_uma_saida(
    id_amostra: str,
    contexto: dict[str, Any],
    template: str,
    config: ConfigLLM,
) -> SaidaAvaliada:
    """Gera uma interpretação e calcula suas métricas automáticas."""
    inicio = time.perf_counter()
    texto = gerar_explicacao(contexto, template=template, config=config)
    latencia = time.perf_counter() - inicio

    verificacao = verificar_saida(texto, contexto, secoes_obrigatorias=SECOES_OBRIGATORIAS)
    n_secoes_presentes = len(SECOES_OBRIGATORIAS) - len(verificacao.secoes_ausentes)

    return SaidaAvaliada(
        id_amostra=id_amostra,
        template=template,
        estrato=_estratificar(contexto["probabilidade"], contexto["predicao"]),
        texto=texto,
        contexto=contexto,
        n_numeros_citados=len(verificacao.numeros_encontrados),
        n_numeros_nao_justificados=len(verificacao.numeros_nao_justificados),
        taxa_alucinacao_numerica=verificacao.taxa_alucinacao_numerica,
        secoes_ausentes=verificacao.secoes_ausentes,
        aderencia_estrutural=n_secoes_presentes / len(SECOES_OBRIGATORIAS),
        numeros_problematicos=verificacao.numeros_nao_justificados,
        termos_proibidos=verificacao.contem_termo_proibido,
        n_caracteres=len(texto),
        latencia_s=latencia,
        rubrica={criterio: None for criterio in RUBRICA},
    )


def medir_estabilidade(
    contexto: dict[str, Any], template: str, config: ConfigLLM, n_repeticoes: int = 3
) -> dict[str, Any]:
    """Gera a mesma entrada N vezes e mede a divergência entre as saídas.

    Divergência numérica entre repetições da MESMA entrada é sinal de temperatura alta demais
    (plano §6.4) — e é um diagnóstico que não depende de julgamento humano.
    """
    from src.llm.verificador import extrair_numeros

    textos = [
        gerar_explicacao(contexto, template=template, config=config)
        for _ in range(n_repeticoes)
    ]
    conjuntos_de_numeros = [set(extrair_numeros(t)) for t in textos]

    uniao = set().union(*conjuntos_de_numeros) if conjuntos_de_numeros else set()
    intersecao = (
        set.intersection(*conjuntos_de_numeros) if conjuntos_de_numeros else set()
    )

    return {
        "n_repeticoes": n_repeticoes,
        "textos_identicos": len(set(textos)) == 1,
        "n_numeros_na_uniao": len(uniao),
        "n_numeros_em_todas": len(intersecao),
        "estabilidade_numerica": len(intersecao) / len(uniao) if uniao else 1.0,
        "variacao_comprimento": (
            max(len(t) for t in textos) - min(len(t) for t in textos) if textos else 0
        ),
    }


def resumir_metricas_automaticas(saidas: list[SaidaAvaliada]) -> dict[str, Any]:
    """Agrega as métricas automáticas por template — a tabela do plano §6.4."""
    import numpy as np

    por_template: dict[str, list[SaidaAvaliada]] = {}
    for saida in saidas:
        por_template.setdefault(saida.template, []).append(saida)

    resumo: dict[str, Any] = {}
    for template, grupo in sorted(por_template.items()):
        resumo[template] = {
            "n_amostras": len(grupo),
            "taxa_alucinacao_numerica_media": float(
                np.mean([s.taxa_alucinacao_numerica for s in grupo])
            ),
            "n_saidas_com_alucinacao": sum(
                1 for s in grupo if s.n_numeros_nao_justificados > 0
            ),
            "aderencia_estrutural_media": float(
                np.mean([s.aderencia_estrutural for s in grupo])
            ),
            "n_saidas_com_termo_proibido": sum(1 for s in grupo if s.termos_proibidos),
            "comprimento_medio_caracteres": float(np.mean([s.n_caracteres for s in grupo])),
            "latencia_media_s": float(np.mean([s.latencia_s for s in grupo])),
        }
    return resumo


def persistir_avaliacao(saidas: list[SaidaAvaliada], nome: str = "avaliacao") -> Path:
    """Grava as saídas brutas e a planilha da rubrica em `avaliacao_llm/`.

    As saídas brutas são versionadas de propósito (plano §6.4): o avaliador precisa poder
    conferir a nota que eu dei, e para isso precisa ver o texto que gerou a nota.
    """
    DIR_AVALIACAO.mkdir(parents=True, exist_ok=True)

    caminho_json = DIR_AVALIACAO / f"{nome}_saidas.json"
    caminho_json.write_text(
        json.dumps([asdict(s) for s in saidas], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # planilha da rubrica, em CSV, pronta para preencher à mão
    import csv

    caminho_csv = DIR_AVALIACAO / f"{nome}_rubrica.csv"
    with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo:
        colunas = ["id_amostra", "template", "estrato", *RUBRICA, "justificativa"]
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        for saida in saidas:
            escritor.writerow(
                {
                    "id_amostra": saida.id_amostra,
                    "template": saida.template,
                    "estrato": saida.estrato,
                    **{criterio: "" for criterio in RUBRICA},
                    "justificativa": "",
                }
            )

    # as saídas em markdown, legíveis, para a leitura humana que a rubrica exige
    caminho_md = DIR_AVALIACAO / f"{nome}_saidas.md"
    linhas = [f"# Saídas da LLM para avaliação — {nome}\n"]
    for saida in saidas:
        linhas.append(
            f"\n---\n\n## {saida.id_amostra} · template `{saida.template}` · "
            f"estrato `{saida.estrato}`\n\n"
            f"**Métricas automáticas:** alucinação numérica "
            f"{saida.taxa_alucinacao_numerica:.0%} "
            f"({saida.n_numeros_nao_justificados}/{saida.n_numeros_citados} números) · "
            f"aderência estrutural {saida.aderencia_estrutural:.0%} · "
            f"{saida.n_caracteres} caracteres · {saida.latencia_s:.2f}s\n"
        )
        if saida.numeros_problematicos:
            linhas.append(
                f"\n⚠️ **Números sem respaldo no contexto:** {saida.numeros_problematicos}\n"
            )
        linhas.append(f"\n**Probabilidade do modelo:** {saida.contexto['probabilidade_pct']}\n")
        linhas.append(f"\n### Texto gerado\n\n{saida.texto}\n")
    caminho_md.write_text("".join(linhas), encoding="utf-8")

    return caminho_json
