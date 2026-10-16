"""Verificação programática da saída da LLM (plano §6.3).

Esta é a única defesa contra alucinação que **não depende da boa vontade do modelo**. A
estratégia tem duas metades:

  1. **Prevenção** — os números vão para o texto por formatação programática, não pedidos à
     LLM (`formatar_numeros_do_contexto`);
  2. **Detecção** — todo número que aparece na saída é conferido contra o conjunto de valores
     que o modelo tinha autorização de citar (`verificar_numeros`).

A Aula 2 de LLMs lista alucinação entre os três desafios centrais do campo. Em saúde, um
número errado num laudo é o pior modo de falha possível — e a Aula 4 (teoria CASA) explica
por que não basta avisar o leitor: a fluência do texto produz confiança que o conteúdo não
necessariamente merece.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Captura números em formato brasileiro ou internacional: 73, 73.4, 73,4, 1.234,5
PADRAO_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")

#: Tolerância relativa ao comparar um número do texto com um valor autorizado. Cobre
#: arredondamento legítimo (0,7342 citado como "73,4%") sem aceitar invenção.
TOLERANCIA_RELATIVA = 0.02


@dataclass
class ResultadoVerificacao:
    """Diagnóstico de uma saída da LLM."""

    aprovado: bool
    numeros_encontrados: list[float] = field(default_factory=list)
    numeros_nao_justificados: list[float] = field(default_factory=list)
    secoes_ausentes: list[str] = field(default_factory=list)
    contem_termo_proibido: list[str] = field(default_factory=list)

    @property
    def taxa_alucinacao_numerica(self) -> float:
        """Fração dos números citados que não encontram respaldo no contexto.

        É a métrica mais importante da avaliação do plano §6.4 — a única totalmente objetiva,
        e a que mede o risco mais grave.
        """
        if not self.numeros_encontrados:
            return 0.0
        return len(self.numeros_nao_justificados) / len(self.numeros_encontrados)


def _normalizar_numero(texto: str) -> float | None:
    """Converte um número textual (pt-BR ou internacional) em float."""
    bruto = texto.strip()
    if bruto.count(",") and bruto.count("."):
        # formato 1.234,5 — ponto é separador de milhar
        bruto = bruto.replace(".", "").replace(",", ".")
    else:
        bruto = bruto.replace(",", ".")
    try:
        return float(bruto)
    except ValueError:
        return None


def extrair_numeros(texto: str) -> list[float]:
    """Todos os números citados num texto."""
    numeros = []
    for achado in PADRAO_NUMERO.findall(texto):
        valor = _normalizar_numero(achado)
        if valor is not None:
            numeros.append(valor)
    return numeros


def coletar_valores_autorizados(contexto: dict[str, Any]) -> set[float]:
    """Todos os valores numéricos que a LLM tem respaldo para citar, derivados do contexto.

    Inclui as variantes de apresentação do mesmo valor — uma probabilidade de 0,7342 pode
    legitimamente aparecer como "0,73", "73,4" ou "73" no texto, e nenhuma dessas é
    alucinação.
    """
    autorizados: set[float] = set()

    def registrar(valor: Any) -> None:
        if isinstance(valor, bool) or valor is None:
            return
        if isinstance(valor, (int, float)):
            numero = float(valor)
            autorizados.add(numero)
            autorizados.add(round(numero, 2))
            autorizados.add(round(numero, 1))
            autorizados.add(round(numero))
            # versões percentuais
            autorizados.add(round(numero * 100, 1))
            autorizados.add(round(numero * 100))
        elif isinstance(valor, str):
            for extraido in extrair_numeros(valor):
                autorizados.add(extraido)
                autorizados.add(round(extraido, 1))
                autorizados.add(round(extraido))
        elif isinstance(valor, dict):
            for item in valor.values():
                registrar(item)
        elif isinstance(valor, (list, tuple)):
            for item in valor:
                registrar(item)

    registrar(contexto)
    return autorizados


def verificar_numeros(
    texto: str, contexto: dict[str, Any], ignorar_ate: float = 10.0
) -> tuple[list[float], list[float]]:
    """Confere cada número do texto contra os valores autorizados pelo contexto.

    `ignorar_ate` dispensa números pequenos (default: ≤ 10), porque eles aparecem
    legitimamente em prosa — "3 fatores", "2 exames", numeração de itens — sem serem
    afirmações sobre os dados. Elevar esse piso reduz falso positivo e aumenta falso
    negativo; 10 é o ponto onde a contagem de itens acaba e as medidas clínicas começam.

    Retorna `(todos_os_numeros, nao_justificados)`.
    """
    autorizados = coletar_valores_autorizados(contexto)
    encontrados = extrair_numeros(texto)
    nao_justificados: list[float] = []

    for numero in encontrados:
        if abs(numero) <= ignorar_ate:
            continue
        if any(
            abs(numero - permitido) <= max(TOLERANCIA_RELATIVA * abs(permitido), 1e-9)
            for permitido in autorizados
        ):
            continue
        nao_justificados.append(numero)

    return encontrados, nao_justificados


#: Termos que a LLM não deve usar porque constituem ato diagnóstico, não interpretação
#: (plano §6.2: "a LLM não diagnostica, ela traduz a saída do modelo").
TERMOS_PROIBIDOS = (
    "diagnostico definitivo",
    "diagnóstico definitivo",
    "com certeza absoluta",
    "certamente tem",
    "definitivamente tem",
    "está curada",
    "esta curada",
    "prescrevo",
    "receito",
)

#: Marcadores de negação que, imediatamente antes de um termo proibido, invertem o seu
#: sentido — "não é um diagnóstico definitivo" é o CUMPRIMENTO da regra, não a violação dela.
#: Descoberto na primeira execução com LLM real: as 10 "violações" detectadas eram todas
#: frases de ressalva, porque a verificação era `termo in texto` (relatório §7, achado nº 12).
MARCADORES_DE_NEGACAO = (
    "não",
    "nao",
    "nunca",
    "sem",
    "longe de",
    "em vez de",
    "ao invés de",
    "ao inves de",
)

#: Janela, em caracteres, olhada à esquerda do termo em busca de negação. 60 cobre
#: "não constitui um", "não deve ser interpretado como", "isto não é um" — e para antes
#: de alcançar a frase anterior, onde uma negação não governaria mais o termo.
JANELA_DE_NEGACAO = 60


def _posicoes(texto: str, termo: str) -> list[int]:
    """Todas as posições de `termo` em `texto` (não só a primeira)."""
    posicoes: list[int] = []
    inicio = texto.find(termo)
    while inicio != -1:
        posicoes.append(inicio)
        inicio = texto.find(termo, inicio + 1)
    return posicoes


def _esta_negado(texto: str, posicao: int) -> bool:
    """Diz se o termo em `posicao` está sob escopo de negação.

    Olha para a esquerda até `JANELA_DE_NEGACAO` caracteres, parando na fronteira de frase
    (`.`, `;`, `!`, `?` ou quebra de linha) — uma negação na frase anterior não governa este
    termo. Heurística, não análise sintática: erra em subordinação complexa, e isso está
    declarado. O custo dos dois erros é assimétrico e justifica a escolha: um falso positivo
    (o que havia antes) desqualifica uma ressalva correta e **inverte a conclusão do
    experimento**; um falso negativo deixa passar uma violação que a rubrica manual ainda
    pode pegar.
    """
    trecho = texto[max(0, posicao - JANELA_DE_NEGACAO) : posicao]
    for fronteira in (".", ";", "!", "?", "\n"):
        corte = trecho.rfind(fronteira)
        if corte != -1:
            trecho = trecho[corte + 1 :]
    return any(marcador in trecho for marcador in MARCADORES_DE_NEGACAO)


def verificar_saida(
    texto: str,
    contexto: dict[str, Any],
    secoes_obrigatorias: tuple[str, ...] = (),
) -> ResultadoVerificacao:
    """Verificação completa: números, aderência estrutural e termos proibidos."""
    encontrados, nao_justificados = verificar_numeros(texto, contexto)

    texto_minusculo = texto.lower()
    secoes_ausentes = [
        secao for secao in secoes_obrigatorias if secao.lower() not in texto_minusculo
    ]
    proibidos = [
        termo
        for termo in TERMOS_PROIBIDOS
        for pos in _posicoes(texto_minusculo, termo)
        if not _esta_negado(texto_minusculo, pos)
    ]

    return ResultadoVerificacao(
        aprovado=not nao_justificados and not secoes_ausentes and not proibidos,
        numeros_encontrados=encontrados,
        numeros_nao_justificados=nao_justificados,
        secoes_ausentes=secoes_ausentes,
        contem_termo_proibido=proibidos,
    )


def formatar_numeros_do_contexto(contexto: dict[str, Any]) -> str:
    """Pré-formata os números do contexto num bloco de texto pronto para ser injetado no
    prompt.

    Metade **preventiva** da estratégia: a LLM recebe os números já escritos e só precisa
    organizá-los em prosa, em vez de recalculá-los ou lembrá-los. Reduz drasticamente a
    chance de alucinação numérica na origem, em vez de depender da detecção a posteriori.
    """
    linhas: list[str] = []
    if "probabilidade_pct" in contexto:
        linhas.append(f"- Probabilidade estimada pelo modelo: {contexto['probabilidade_pct']}")
    if "predicao_rotulo" in contexto:
        linhas.append(f"- Classificação do modelo: {contexto['predicao_rotulo']}")

    features = contexto.get("features_mais_relevantes") or {}
    if features:
        linhas.append("- Variáveis de maior peso no modelo, com os valores desta paciente:")
        for nome, info in features.items():
            valor = info.get("valor_da_paciente")
            peso = info.get("peso_no_modelo")
            linhas.append(f"    - {nome.strip()}: {valor} (peso no modelo: {peso})")

    return "\n".join(linhas)
