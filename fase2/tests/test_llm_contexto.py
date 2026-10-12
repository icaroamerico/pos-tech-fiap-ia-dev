"""Testes da camada de LLM (plano §7.2).

O foco é nas partes DETERMINÍSTICAS — montagem de contexto, allow-list de anonimização,
verificação de alucinação numérica e carregamento de templates. A saída da LLM em si não é
testada por asserção de conteúdo (é não determinística; a avaliação dela é a rubrica do §6.4).

O teste mais importante aqui é `test_contexto_nunca_inclui_campo_fora_da_allow_list`: é o que
transforma a garantia de privacidade do plano §6.1 em algo verificável.
"""

import numpy as np
import pandas as pd
import pytest

from src.llm.cliente import (
    ConfigLLM,
    carregar_template,
    gerar_explicacao,
    templates_disponiveis,
)
from src.llm.contexto import (
    CAMPOS_CLINICOS_AUTORIZADOS,
    CAMPOS_NUNCA_PERMITIDOS,
    CampoNaoAutorizadoError,
    filtrar_campos_autorizados,
    montar_contexto_coorte,
    montar_contexto_paciente,
)
from src.llm.verificador import (
    coletar_valores_autorizados,
    extrair_numeros,
    formatar_numeros_do_contexto,
    verificar_numeros,
    verificar_saida,
)


# ---------------------------------------------------------------------------
# Allow-list / anonimização
# ---------------------------------------------------------------------------


def test_contexto_nunca_inclui_campo_fora_da_allow_list():
    """O teste central de privacidade (plano §6.1): um campo novo e desconhecido na base NÃO
    deve entrar no contexto só por existir. É a diferença entre allow-list e deny-list.
    """
    registro = {
        "BMI": 24.5,
        "FSH(mIU/mL)": 5.1,
        # campos inventados, simulando colunas novas que apareceram na base
        "nome_da_paciente": "Maria Silva",
        "cpf": "123.456.789-00",
        "telefone": "11999998888",
        "endereco": "Rua X, 123",
        "coluna_nova_qualquer": 42,
    }
    filtrado = filtrar_campos_autorizados(registro)

    assert "BMI" in filtrado
    assert "FSH(mIU/mL)" in filtrado
    for campo_sensivel in ("nome_da_paciente", "cpf", "telefone", "endereco", "coluna_nova_qualquer"):
        assert campo_sensivel not in filtrado


@pytest.mark.parametrize("campo_proibido", sorted(CAMPOS_NUNCA_PERMITIDOS))
def test_campos_proibidos_levantam_excecao(campo_proibido):
    """Identificadores e o próprio rótulo (`PCOS (Y/N)`) não são apenas descartados — eles
    levantam exceção, porque a presença deles significa que alguém os passou de propósito.
    """
    with pytest.raises(CampoNaoAutorizadoError):
        filtrar_campos_autorizados({campo_proibido: 1, "BMI": 24.0})


def test_rotulo_verdadeiro_nunca_chega_ao_prompt():
    """`PCOS (Y/N)` é a resposta que o modelo tenta prever — se vazar para o prompt, a LLM
    poderia "explicar" o diagnóstico usando o gabarito em vez da saída do modelo.
    """
    assert "PCOS (Y/N)" in CAMPOS_NUNCA_PERMITIDOS
    assert "PCOS (Y/N)" not in CAMPOS_CLINICOS_AUTORIZADOS


def test_montar_contexto_paciente_nao_vaza_identificadores():
    paciente = pd.Series(
        {
            "BMI": 27.3,
            "Follicle No. (R)": 12,
            "Follicle No. (L)": 10,
            "Sl. No": 999,  # identificador
        }
    )
    # `Sl. No` está na lista de proibidos -> deve levantar, não silenciosamente passar
    with pytest.raises(CampoNaoAutorizadoError):
        montar_contexto_paciente(paciente, predicao=1, probabilidade=0.73)


def test_montar_contexto_paciente_estrutura_esperada():
    paciente = pd.Series({"BMI": 27.3, "Follicle No. (R)": 12, "Follicle No. (L)": 10})
    contexto = montar_contexto_paciente(
        paciente,
        predicao=1,
        probabilidade=0.7342,
        importancias={"Follicle No. (R)": 0.21, "Follicle No. (L)": 0.19, "BMI": 0.04},
    )

    assert contexto["predicao"] == 1
    assert contexto["probabilidade"] == pytest.approx(0.7342)
    assert contexto["probabilidade_pct"] == "73.4%"
    assert "indicativo de SOP" in contexto["predicao_rotulo"]
    assert "Follicle No. (R)" in contexto["features_mais_relevantes"]
    assert contexto["features_mais_relevantes"]["Follicle No. (R)"]["valor_da_paciente"] == 12


def test_contexto_coorte_so_tem_agregados():
    predicoes = [
        {"predicao": 1, "probabilidade": 0.9},
        {"predicao": 0, "probabilidade": 0.1},
        {"predicao": 1, "probabilidade": 0.52},
        {"predicao": 0, "probabilidade": 0.45},
    ]
    contexto = montar_contexto_coorte(predicoes)
    assert contexto["n_pacientes"] == 4
    assert contexto["n_indicativo_sop"] == 2
    assert contexto["n_zona_incerteza"] == 2  # 0.52 e 0.45 estão entre 0,4 e 0,6
    texto = str(contexto)
    assert "0.9" not in texto or "probabilidade_media" in texto  # nenhum caso individual exposto


# ---------------------------------------------------------------------------
# Verificador de alucinação numérica
# ---------------------------------------------------------------------------


def test_extrair_numeros_cobre_formatos_pt_br_e_internacional():
    assert 73.4 in extrair_numeros("probabilidade de 73,4%")
    assert 73.4 in extrair_numeros("probability of 73.4%")
    assert 1234.5 in extrair_numeros("valor de 1.234,5 unidades")


def test_verificador_detecta_numero_inventado():
    """O caso que a verificação existe para pegar: a LLM cita um número plausível que não
    está em nenhum lugar do contexto.
    """
    contexto = {"probabilidade": 0.7342, "probabilidade_pct": "73.4%"}
    texto = "A probabilidade é de 73,4%, com AMH de 8,9 ng/mL e FSH de 412 mIU/mL."
    _, nao_justificados = verificar_numeros(texto, contexto)
    assert 412.0 in nao_justificados  # inventado
    assert 73.4 not in nao_justificados  # legítimo


def test_verificador_aceita_arredondamento_legitimo():
    """0,7342 citado como "73,4%" ou "73%" não é alucinação — é apresentação."""
    contexto = {"probabilidade": 0.7342, "probabilidade_pct": "73.4%"}
    for texto in ("probabilidade de 73,4%", "probabilidade de 73%", "cerca de 0,73"):
        _, nao_justificados = verificar_numeros(texto, contexto)
        assert not nao_justificados, f"falso positivo em: {texto!r}"


def test_verificador_ignora_numeros_pequenos_de_prosa():
    """"3 fatores", "2 exames" — contagens em prosa não são afirmações sobre os dados."""
    contexto = {"probabilidade": 0.73}
    texto = "Foram considerados 3 fatores principais e 2 exames complementares."
    _, nao_justificados = verificar_numeros(texto, contexto)
    assert not nao_justificados


def test_verificador_detecta_secao_ausente():
    contexto = {"probabilidade": 0.73}
    texto = "## Resumo da indicação do modelo\nTexto qualquer."
    resultado = verificar_saida(
        texto,
        contexto,
        secoes_obrigatorias=("Resumo da indicação do modelo", "Próximo passo sugerido"),
    )
    assert not resultado.aprovado
    assert "Próximo passo sugerido" in resultado.secoes_ausentes


def test_verificador_detecta_termo_de_diagnostico():
    """A fronteira do plano §6.2: a LLM traduz, não diagnostica."""
    contexto = {"probabilidade": 0.95}
    resultado = verificar_saida("A paciente definitivamente tem SOP.", contexto)
    assert not resultado.aprovado
    assert resultado.contem_termo_proibido


def test_taxa_alucinacao_numerica_e_calculada():
    contexto = {"probabilidade": 0.73, "probabilidade_pct": "73.0%"}
    texto = "Probabilidade de 73%, AMH 450, FSH 820."
    resultado = verificar_saida(texto, contexto)
    assert resultado.taxa_alucinacao_numerica > 0
    assert len(resultado.numeros_nao_justificados) == 2


def test_coletar_valores_autorizados_percorre_estrutura_aninhada():
    contexto = {
        "probabilidade": 0.73,
        "features_mais_relevantes": {
            "Follicle No. (R)": {"valor_da_paciente": 12, "peso_no_modelo": 0.214}
        },
        "dados_clinicos": {"BMI": 27.3},
    }
    autorizados = coletar_valores_autorizados(contexto)
    assert 12 in autorizados
    assert 27.3 in autorizados
    assert 73 in autorizados  # versão percentual de 0,73


def test_formatar_numeros_injeta_valores_prontos():
    """A metade PREVENTIVA da defesa (plano §6.3): a LLM recebe os números já escritos."""
    contexto = {
        "probabilidade_pct": "73.4%",
        "predicao_rotulo": "indicativo de SOP",
        "features_mais_relevantes": {
            "Follicle No. (R)": {"valor_da_paciente": 12, "peso_no_modelo": 0.214}
        },
    }
    bloco = formatar_numeros_do_contexto(contexto)
    assert "73.4%" in bloco
    assert "indicativo de SOP" in bloco
    assert "12" in bloco


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_os_tres_templates_do_plano_existem():
    disponiveis = templates_disponiveis()
    for esperado in ("v1_zero_shot", "v2_few_shot", "v3_few_shot_estruturado"):
        assert esperado in disponiveis


@pytest.mark.parametrize(
    "nome", ["v1_zero_shot", "v2_few_shot", "v3_few_shot_estruturado"]
)
def test_template_carrega_sistema_e_usuario(nome):
    sistema, usuario = carregar_template(nome)
    assert sistema and usuario
    assert "{dados}" in usuario
    assert "{contexto}" in usuario
    # a proibição de diagnosticar tem de estar em TODOS os templates (plano §6.2)
    assert "não** emite diagnóstico" in sistema or "não emite diagnóstico" in sistema


@pytest.mark.parametrize("nome", ["v2_few_shot", "v3_few_shot_estruturado"])
def test_templates_few_shot_tem_exemplares_diversos_e_consistentes(nome):
    """Achados de ICL da Aula 3 de LLMs: exemplares duplicados degradam, inconsistentes causam
    falha severa. Os três exemplos devem cobrir casos DISTINTOS com estrutura IDÊNTICA.
    """
    _, usuario = carregar_template(nome)
    assert usuario.count("### EXEMPLO") == 3
    # casos distintos: alta, baixa e limiar
    assert "alta probabilidade" in usuario
    assert "baixa probabilidade" in usuario
    assert "limiar" in usuario
    # estrutura idêntica: as 4 seções repetidas em cada exemplo
    assert usuario.count("## Resumo da indicação do modelo") == 3
    assert usuario.count("## Próximo passo sugerido") == 3


def test_template_v3_tem_percurso_de_raciocinio_estruturado():
    """O que distingue o V3 do V2 (plano §6.3): raciocínio com campos obrigatórios, em vez de
    cadeia livre — que em contexto médico convidaria a LLM a inventar fisiopatologia.
    """
    sistema, _ = carregar_template("v3_few_shot_estruturado")
    assert "PERCURSO DE RACIOCÍNIO OBRIGATÓRIO" in sistema
    assert "Não explique mecanismo fisiopatológico" in sistema


def test_template_inexistente_da_erro_util():
    with pytest.raises(FileNotFoundError, match="Disponíveis"):
        carregar_template("template_que_nao_existe")


# ---------------------------------------------------------------------------
# Geração ponta a ponta com o provedor simulado
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template", ["v1_zero_shot", "v2_few_shot", "v3_few_shot_estruturado"]
)
def test_geracao_simulada_produz_as_quatro_secoes(template):
    """O provedor simulado permite testar a camada inteira sem Ollama nem chave de API — e
    deixa o notebook executável por qualquer avaliador (plano §6.1).
    """
    contexto = montar_contexto_paciente(
        pd.Series({"BMI": 27.3, "Follicle No. (R)": 12}),
        predicao=1,
        probabilidade=0.7342,
        importancias={"Follicle No. (R)": 0.21},
    )
    texto = gerar_explicacao(contexto, template=template, config=ConfigLLM(provedor="simulado"))

    for secao in (
        "Resumo da indicação do modelo",
        "Variáveis que mais pesaram",
        "Grau de confiança e limitações",
        "Próximo passo sugerido",
    ):
        assert secao in texto


def test_disclaimer_e_concatenado_em_codigo():
    """Um aviso que a LLM pudesse omitir não é um aviso (plano §6.3)."""
    contexto = montar_contexto_paciente(
        pd.Series({"BMI": 27.3}), predicao=0, probabilidade=0.12
    )
    texto = gerar_explicacao(contexto, config=ConfigLLM(provedor="simulado"))
    assert "não constitui diagnóstico médico" in texto
    assert "não gerado pelo modelo de linguagem" in texto

    sem = gerar_explicacao(
        contexto, config=ConfigLLM(provedor="simulado"), incluir_disclaimer=False
    )
    assert "não gerado pelo modelo de linguagem" not in sem


def test_provedor_simulado_tem_taxa_de_alucinacao_zero():
    """O simulado é o CONTROLE da avaliação do §6.4: por construção só repete números que o
    prompt forneceu. Qualquer alucinação medida num provedor real é atribuível ao modelo, não
    ao instrumento de medida.
    """
    contexto = montar_contexto_paciente(
        pd.Series({"BMI": 27.3, "Follicle No. (R)": 12}),
        predicao=1,
        probabilidade=0.7342,
        importancias={"Follicle No. (R)": 0.21, "BMI": 0.04},
    )
    texto = gerar_explicacao(contexto, config=ConfigLLM(provedor="simulado"))
    resultado = verificar_saida(texto, contexto)
    assert resultado.taxa_alucinacao_numerica == 0.0, (
        f"simulado alucinou: {resultado.numeros_nao_justificados}"
    )


def test_config_llm_tem_temperatura_baixa_por_padrao():
    """Tarefa de fidelidade, não de criatividade (plano §6.3)."""
    assert ConfigLLM().temperatura <= 0.3
    assert ConfigLLM().provedor == "ollama"  # local por padrão, por privacidade
