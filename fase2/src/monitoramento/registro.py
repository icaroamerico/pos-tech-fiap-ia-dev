"""Monitoramento, logging e persistência de artefatos.

Duas saídas, com propósitos distintos:

  - **log de console/arquivo** (`logging` da stdlib) — acompanhamento humano da execução;
  - **histórico JSONL** (uma linha por geração) — a fonte de dados dos gráficos.

A separação é deliberada: os notebooks leem `history.jsonl` em DataFrame e plotam a partir
dele, sem reexecutar o AG. Isso é o que torna a análise reprodutível sem pagar o custo da
busca de novo, e o que permite gravar o vídeo de demonstração sem esperar 20 minutos ao vivo
.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
DIR_SAIDA = RAIZ_PROJETO / "arquivos_saida"
DIR_ANALITICOS = DIR_SAIDA / "arquivos_analiticos"
DIR_RUNS = DIR_ANALITICOS / "historico_execucoes_ag"
DIR_LOGS = DIR_SAIDA / "logs"


def _serializavel(valor: Any) -> Any:
    """Converte valores para algo que `json.dumps` aceite.

    Necessário porque os cromossomos carregam tipos que não são JSON nativos: tuplas (o par
    `penalty_solver`), `None` em `max_depth`, e escalares do numpy que escapam de operações
    vetorizadas.
    """
    import numpy as np

    if isinstance(valor, dict):
        return {str(k): _serializavel(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_serializavel(v) for v in valor]
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    if is_dataclass(valor) and not isinstance(valor, type):
        return _serializavel(asdict(valor))
    return valor


@dataclass
class RegistroGeracao:
    """Uma linha do histórico — todas as métricas de uma geração."""

    experimento: str
    semente: int
    geracao: int
    melhor_fitness: float
    fitness_medio: float
    desvio_padrao: float
    entropia_familia: float
    entropia_media_categorica: float
    distancia_euclidiana_media: float
    n_solucoes_unicas: int
    n_genotipos_unicos: int
    freq_class_weight_balanced: float
    distribuicao_familias: dict[str, float]
    familia_do_melhor: str
    n_avaliacoes: int
    n_cache_hits: int
    tempo_geracao_s: float
    tempo_avaliacao_s: float
    tempo_operadores_s: float


@dataclass
class ResultadoExecucao:
    """O que uma execução completa do AG devolve e persiste."""

    config: dict[str, Any]
    historico: list[RegistroGeracao] = field(default_factory=list)
    melhor_individuo: dict[str, Any] | None = None
    melhor_fitness: float = float("-inf")
    melhor_metricas: dict[str, float] = field(default_factory=dict)
    populacao_final: list[dict[str, Any]] = field(default_factory=list)
    motivo_parada: str = ""
    geracoes_executadas: int = 0
    tempo_total_s: float = 0.0
    total_avaliacoes: int = 0
    total_cache_hits: int = 0


def configurar_logger(nome_execucao: str, nivel_console: int = logging.INFO) -> logging.Logger:
    """Logger com duas saídas: console no nível pedido e arquivo em
    `arquivos_saida/logs/` sempre em DEBUG.

    Idempotente — chamar duas vezes para a mesma execução não duplica handlers (evita linhas
    repetidas quando o AG é invocado várias vezes no mesmo processo, como nos notebooks).
    """
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ag.{nome_execucao}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setLevel(nivel_console)
    handler_console.setFormatter(formato)
    logger.addHandler(handler_console)

    handler_arquivo = logging.FileHandler(
        DIR_LOGS / f"{nome_execucao}.log", mode="w", encoding="utf-8"
    )
    handler_arquivo.setLevel(logging.DEBUG)
    handler_arquivo.setFormatter(formato)
    logger.addHandler(handler_arquivo)

    return logger


def coletar_ambiente() -> dict[str, Any]:
    """Versões e plataforma — sem isso, um número do relatório não é rastreável.

    A Fase 1 registra que resultados em `mps` não são determinísticos entre máquinas; aqui
    tudo roda em CPU e é determinístico, mas as versões de biblioteca ainda importam para
    reproduzir.
    """
    import numpy
    import pandas
    import sklearn

    return {
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "processador": platform.processor() or platform.machine(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }


class EscritorExecucao:
    """Gerencia o diretório `arquivos_saida/arquivos_analiticos/historico_execucoes_ag/<nome>/` e escreve os artefatos da execução."""

    def __init__(self, nome_execucao: str, dir_base: Path | None = None) -> None:
        self.nome = nome_execucao
        self.dir_execucao = (dir_base or DIR_RUNS) / nome_execucao
        self.dir_execucao.mkdir(parents=True, exist_ok=True)
        self.caminho_historico = self.dir_execucao / "history.jsonl"
        # trunca o histórico de uma execução anterior com o mesmo nome
        self.caminho_historico.write_text("", encoding="utf-8")

    def escrever_config(self, config: dict[str, Any]) -> None:
        self._escrever_json("config.json", {**config, "ambiente": coletar_ambiente()})

    def anexar_geracao(self, registro: RegistroGeracao) -> None:
        """Anexa uma linha ao JSONL — escrita incremental, para que uma execução interrompida
        preserve o que já foi medido.
        """
        self.anexar_linha_historico(_serializavel(asdict(registro)))

    def anexar_linha_historico(self, registro: dict[str, Any]) -> None:
        """Anexa um registro arbitrário ao JSONL.

        Existe para a busca aleatória da etapa C, que não tem gerações e portanto não
        preenche um `RegistroGeracao`. Sem isto, `__init__` criava o `history.jsonl`, nada
        era anexado, e os três `C_busca_aleatoria_seed*/history.jsonl` ficavam com 0 bytes —
        indistinguíveis de uma execução perdida.
        """
        linha = json.dumps(_serializavel(registro), ensure_ascii=False)
        with self.caminho_historico.open("a", encoding="utf-8") as arquivo:
            arquivo.write(linha + "\n")

    def escrever_resultado(self, resultado: ResultadoExecucao) -> None:
        self._escrever_json(
            "best_individual.json",
            {
                "individuo": resultado.melhor_individuo,
                "fitness": resultado.melhor_fitness,
                "metricas_cv": resultado.melhor_metricas,
            },
        )
        self._escrever_json("population_final.json", resultado.populacao_final)
        self._escrever_json(
            "resumo.json",
            {
                "motivo_parada": resultado.motivo_parada,
                "geracoes_executadas": resultado.geracoes_executadas,
                "melhor_fitness": resultado.melhor_fitness,
                "tempo_total_s": resultado.tempo_total_s,
                "total_avaliacoes": resultado.total_avaliacoes,
                "total_cache_hits": resultado.total_cache_hits,
                "taxa_cache_hit": (
                    resultado.total_cache_hits / resultado.total_avaliacoes
                    if resultado.total_avaliacoes
                    else 0.0
                ),
            },
        )

    def escrever_artefato_json(self, nome_arquivo: str, conteudo: Any) -> None:
        """Grava um artefato JSON avulso na pasta da execução.

        É a porta pública para `_escrever_json`: a etapa C chamava o privado de fora da
        classe, o que acoplava o CLI ao detalhe interno do escritor.
        """
        self._escrever_json(nome_arquivo, conteudo)

    def escrever_metricas_holdout(self, metricas: dict[str, Any]) -> None:
        """Métricas no conjunto de teste — escritas SÓ no fim, uma vez."""
        self._escrever_json("metrics_holdout.json", metricas)

    def _escrever_json(self, nome_arquivo: str, conteudo: Any) -> None:
        caminho = self.dir_execucao / nome_arquivo
        caminho.write_text(
            json.dumps(_serializavel(conteudo), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def carregar_historico(nome_execucao: str, dir_base: Path | None = None):
    """Lê o `history.jsonl` de uma execução em DataFrame — o ponto de entrada dos notebooks.

    É esta função que torna a análise independente da execução: os gráficos
    saem daqui, não de variáveis em memória.
    """
    import pandas as pd

    caminho = (dir_base or DIR_RUNS) / nome_execucao / "history.jsonl"
    if not caminho.exists():
        raise FileNotFoundError(f"Histórico não encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        registros = [json.loads(linha) for linha in arquivo if linha.strip()]
    return pd.DataFrame(registros)


class Cronometro:
    """Cronômetro de contexto para medir etapas separadamente (tempo por operador).

        with Cronometro() as c:
            ...
        c.decorrido  # segundos
    """

    def __enter__(self) -> Cronometro:
        self._inicio = time.perf_counter()
        self.decorrido = 0.0
        return self

    def __exit__(self, *_excecao: object) -> None:
        self.decorrido = time.perf_counter() - self._inicio
