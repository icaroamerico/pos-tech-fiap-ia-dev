"""Cliente de LLM — interface fina sobre os provedores.

**Interface fina de propósito.** Todo o resto do projeto conversa apenas com
`gerar_explicacao(contexto) -> str`. Trocar o modelo, ou acrescentar RAG no Módulo 3, não
exige tocar o pipeline de ML.

**Três vias, com a local como padrão** — execução local para máxima privacidade, nuvem
para escalabilidade, API para custo:

  - `ollama`  — local, nenhum dado sai da máquina. **Padrão**, porque os dados são clínicos.
  - `openai`  — só com dados agregados ou anonimizados; exige `OPENAI_API_KEY`.
  - `simulado` — determinístico, sem rede. Serve aos testes e permite que o notebook abra em
    qualquer máquina sem Ollama instalado nem chave de API, como a Fase 1 fez versionando uma
    amostra de 16 imagens para o notebook do raio-X rodar em qualquer lugar.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
DIR_PROMPTS = RAIZ_PROJETO / "prompts"

Provedor = Literal["ollama", "openai", "simulado"]


@dataclass
class ConfigLLM:
    """Parâmetros de geração.

    `temperatura` baixa por default: esta é tarefa de **fidelidade**, não de criatividade.
    Temperatura alta produz texto mais agradável e menos fiel aos números — exatamente o
    trade-off errado num laudo médico.
    """

    provedor: Provedor = "ollama"
    #: `llama3.2:3b` e não o 8B: é o modelo que a §5.5 declara, o que está instalado e o
    #: único que caberia na máquina de 8 GB da execução. O default antigo (`llama3.1:8b`)
    #: divergia do default da CLI e levava quem construísse `ConfigLLM()` direto a rodar
    #: um modelo diferente do documentado — e a entrar em swap.
    modelo: str = "llama3.2:3b"
    temperatura: float = 0.2
    semente: int = 42
    max_tokens: int = 900
    url_ollama: str = "http://localhost:11434/api/generate"
    timeout_s: int = 180


class ErroLLM(RuntimeError):
    """Falha ao gerar texto com o provedor configurado."""


class GeradorTexto(Protocol):
    def __call__(
        self,
        prompt_sistema: str,
        prompt_usuario: str,
        config: ConfigLLM,
        dados_formatados: str,
    ) -> str:
        """`dados_formatados` é o bloco de números desta paciente, já montado.

        Provedores reais o ignoram — ele já está embutido em `prompt_usuario`. Existe na
        assinatura para o provedor simulado, que precisa saber quais números são os DESTA
        paciente e não os dos exemplares few-shot (ver `_gerar_simulado`).
        """
        ...


# ---------------------------------------------------------------------------
# Provedores
# ---------------------------------------------------------------------------


def _gerar_ollama(
    prompt_sistema: str, prompt_usuario: str, config: ConfigLLM, dados_formatados: str = ""
) -> str:
    corpo = json.dumps(
        {
            "model": config.modelo,
            "system": prompt_sistema,
            "prompt": prompt_usuario,
            "stream": False,
            "options": {
                "temperature": config.temperatura,
                "seed": config.semente,
                "num_predict": config.max_tokens,
            },
        }
    ).encode("utf-8")

    requisicao = urllib.request.Request(
        config.url_ollama, data=corpo, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=config.timeout_s) as resposta:
            return json.loads(resposta.read()).get("response", "").strip()
    except urllib.error.URLError as erro:
        raise ErroLLM(
            f"Ollama indisponível em {config.url_ollama} ({erro}). "
            "Inicie com `ollama serve` e baixe o modelo com "
            f"`ollama pull {config.modelo}`, ou use provedor='simulado'."
        ) from erro


def _gerar_openai(
    prompt_sistema: str, prompt_usuario: str, config: ConfigLLM, dados_formatados: str = ""
) -> str:
    chave = os.environ.get("OPENAI_API_KEY")
    if not chave:
        raise ErroLLM("OPENAI_API_KEY não definida no ambiente.")

    corpo = json.dumps(
        {
            "model": config.modelo,
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario},
            ],
            "temperature": config.temperatura,
            "max_tokens": config.max_tokens,
            "seed": config.semente,
        }
    ).encode("utf-8")

    requisicao = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=corpo,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {chave}"},
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=config.timeout_s) as resposta:
            dados = json.loads(resposta.read())
            return dados["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as erro:
        raise ErroLLM(f"Falha ao chamar a API da OpenAI: {erro}") from erro


def _gerar_simulado(
    prompt_sistema: str, prompt_usuario: str, config: ConfigLLM, dados_formatados: str = ""
) -> str:
    """Gerador determinístico, sem rede.

    Produz um texto estruturado a partir de `dados_formatados` — o bloco de números DESTA
    paciente. Não é uma LLM: é o substituto que permite testar a camada inteira (montagem de
    contexto, template, verificação, rubrica) sem depender de serviço externo, e deixa o
    notebook executável por qualquer avaliador.

    Como só repete números que recebeu, por construção tem taxa de alucinação numérica zero —
    o que o torna o **controle** da avaliação do §6.4: qualquer taxa de alucinação medida num
    provedor real é atribuível ao modelo, não ao instrumento de medida.

    ⚠️ Usa `dados_formatados`, NUNCA varre `prompt_usuario`. A primeira versão raspava todas
    as linhas iniciadas por "-" do prompt e acabava capturando os números dos **exemplares
    few-shot** (91,2% / 18 folículos / 52,7%) em vez dos da paciente — o que fazia o
    "controle de alucinação zero" alucinar 19% dos números. Um instrumento de medida
    contaminado pelo próprio prompt não serve como controle.
    """
    bloco_dados = dados_formatados.strip() or "- (sem dados numéricos)"

    return (
        "## Resumo da indicação do modelo\n"
        f"{bloco_dados}\n\n"
        "## Variáveis que mais pesaram\n"
        "As variáveis listadas acima foram as de maior peso na decisão do modelo para este "
        "caso, apresentadas com os valores registrados para esta paciente.\n\n"
        "## Grau de confiança e limitações\n"
        "A probabilidade acima é uma estimativa estatística produzida por um modelo treinado "
        "em dados de uma única clínica de fertilidade, e não constitui diagnóstico. O modelo "
        "não considera histórico clínico completo, exames de imagem nem avaliação presencial.\n\n"
        "## Próximo passo sugerido\n"
        "Recomenda-se correlacionar estes achados com a avaliação clínica e os exames "
        "complementares pertinentes antes de qualquer conduta.\n"
    )


PROVEDORES: dict[Provedor, GeradorTexto] = {
    "ollama": _gerar_ollama,
    "openai": _gerar_openai,
    "simulado": _gerar_simulado,
}


# ---------------------------------------------------------------------------
# Templates de prompt
# ---------------------------------------------------------------------------


def carregar_template(nome: str) -> tuple[str, str]:
    """Carrega um template de `prompts/<nome>.md`.

    O arquivo é dividido em duas partes pelo marcador `---USUARIO---`: antes é o *system
    prompt*, depois é o molde da mensagem do usuário (com `{contexto}` e `{dados}` a
    preencher).

    Templates ficam em ARQUIVOS versionados, não em strings no código, porque prompt é
    artefato experimental: o enunciado pede "prompts utilizados" no relatório, e o diff do
    Git é o que mostra a evolução entre as versões.
    """
    caminho = DIR_PROMPTS / f"{nome}.md"
    if not caminho.exists():
        disponiveis = sorted(p.stem for p in DIR_PROMPTS.glob("*.md"))
        raise FileNotFoundError(f"Template {nome!r} não existe. Disponíveis: {disponiveis}")

    conteudo = caminho.read_text(encoding="utf-8")
    if "---USUARIO---" not in conteudo:
        raise ValueError(f"Template {nome!r} não tem o marcador ---USUARIO---")

    sistema, usuario = conteudo.split("---USUARIO---", 1)
    return sistema.strip(), usuario.strip()


def templates_disponiveis() -> list[str]:
    return sorted(p.stem for p in DIR_PROMPTS.glob("*.md"))


# ---------------------------------------------------------------------------
# A interface que o resto do projeto usa
# ---------------------------------------------------------------------------


def gerar_explicacao(
    contexto: dict[str, Any],
    template: str = "v3_few_shot_estruturado",
    config: ConfigLLM | None = None,
    incluir_disclaimer: bool = True,
) -> str:
    """Gera a explicação em linguagem natural para um contexto.

    **É esta a interface fina** — a única função que o pipeline de ML conhece.
    Acrescentar RAG no Módulo 3 significa enriquecer `contexto` antes desta chamada, sem
    tocar em nada do AG.

    O disclaimer é concatenado **em código**, não gerado pelo modelo: um aviso que a LLM
    pudesse omitir não é um aviso.
    """
    from src.llm.verificador import formatar_numeros_do_contexto

    config = config or ConfigLLM()
    prompt_sistema, molde_usuario = carregar_template(template)

    dados_formatados = formatar_numeros_do_contexto(contexto)
    contexto_json = json.dumps(contexto, ensure_ascii=False, indent=2)
    prompt_usuario = molde_usuario.format(dados=dados_formatados, contexto=contexto_json)

    gerador = PROVEDORES[config.provedor]
    texto = gerador(prompt_sistema, prompt_usuario, config, dados_formatados)

    if incluir_disclaimer:
        texto += (
            "\n\n---\n"
            "**Aviso automático (não gerado pelo modelo de linguagem):** este texto é um apoio "
            "à interpretação de uma saída estatística e **não constitui diagnóstico médico**. "
            "A decisão clínica cabe ao profissional de saúde responsável."
        )
    return texto
