"""Montagem do contexto enviado à LLM, com anonimização por construção (plano §6.1).

A decisão central deste módulo: **allow-list, não deny-list**.

Uma deny-list ("remova `nome`, `cpf`, `email`") vaza por omissão — basta uma coluna nova
aparecer na base para ela ir ao prompt sem ninguém decidir isso. Uma allow-list falha de
forma segura: o campo que não foi explicitamente autorizado simplesmente não existe no
contexto. Com dados clínicos de pacientes, essa é a diferença entre um bug e um incidente.

Contexto da decisão (IA Generativa, Aula 5): Amazon, Apple, Samsung e JP Morgan proibiram o
ChatGPT internamente porque o que é digitado no prompt vai para os servidores do provedor.
Aqui o modelo roda local por padrão, mas a allow-list vale de todo modo — defesa em
profundidade.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

#: Campos clínicos autorizados a compor o contexto da LLM.
#:
#: São variáveis clínicas e hormonais — nenhum identificador. A base PCOS unificada não tem
#: nome/CPF/endereço, mas a allow-list é a garantia de que, se algum dia tiver, esse campo
#: não entra no prompt por acidente.
#:
#: Para acrescentar um campo aqui, a pergunta a responder é: "este dado, isolado ou combinado
#: com os outros desta lista, permite reidentificar a paciente?"
CAMPOS_CLINICOS_AUTORIZADOS: frozenset[str] = frozenset(
    {
        " Age (yrs)",
        "Age (yrs)",
        "Weight (Kg)",
        "Height(Cm) ",
        "Height(Cm)",
        "BMI",
        "Blood Group",
        "Pulse rate(bpm) ",
        "Pulse rate(bpm)",
        "RR (breaths/min)",
        "Hb(g/dl)",
        "Cycle(R/I)",
        "Cycle length(days)",
        "Marriage Status (Yrs)",
        "Pregnant(Y/N)",
        "No. of absortions",
        "  I   beta-HCG(mIU/mL)",
        "II    beta-HCG(mIU/mL)",
        "FSH(mIU/mL)",
        "LH(mIU/mL)",
        "FSH/LH",
        "Hip(inch)",
        "Waist(inch)",
        "Waist:Hip Ratio",
        "TSH (mIU/L)",
        "AMH(ng/mL)",
        "PRL(ng/mL)",
        "Vit D3 (ng/mL)",
        "PRG(ng/mL)",
        "RBS(mg/dl)",
        "Weight gain(Y/N)",
        "hair growth(Y/N)",
        "Skin darkening (Y/N)",
        "Hair loss(Y/N)",
        "Pimples(Y/N)",
        "Fast food (Y/N)",
        "Reg.Exercise(Y/N)",
        "BP _Systolic (mmHg)",
        "BP _Diastolic (mmHg)",
        "Follicle No. (L)",
        "Follicle No. (R)",
        "Avg. F size (L) (mm)",
        "Avg. F size (R) (mm)",
        "Endometrium (mm)",
    }
)

#: Campos explicitamente PROIBIDOS, mesmo que algum dia entrem na allow-list por engano.
#: Redundante com a allow-list por desenho — é a segunda barreira.
CAMPOS_NUNCA_PERMITIDOS: frozenset[str] = frozenset(
    {"Sl. No", "Patient File No.", "PCOS (Y/N)"}
)


class CampoNaoAutorizadoError(RuntimeError):
    """Levantada quando um campo fora da allow-list tenta entrar no contexto."""


def filtrar_campos_autorizados(registro: dict[str, Any]) -> dict[str, Any]:
    """Mantém apenas os campos da allow-list. Campos desconhecidos são descartados em
    silêncio (não são erro — a base pode ter colunas que simplesmente não interessam ao
    prompt), mas campos da lista de proibidos levantam exceção, porque a presença deles
    indica que alguém os passou de propósito.
    """
    for campo in registro:
        if campo.strip() in {c.strip() for c in CAMPOS_NUNCA_PERMITIDOS}:
            raise CampoNaoAutorizadoError(
                f"Campo proibido no contexto da LLM: {campo!r}. "
                "Identificadores e o próprio rótulo nunca entram no prompt."
            )

    autorizados_normalizados = {c.strip() for c in CAMPOS_CLINICOS_AUTORIZADOS}
    return {
        campo: valor
        for campo, valor in registro.items()
        if campo.strip() in autorizados_normalizados
    }


def montar_contexto_paciente(
    X_paciente: pd.Series,
    predicao: int,
    probabilidade: float,
    importancias: dict[str, float] | None = None,
    top_n_features: int = 6,
) -> dict[str, Any]:
    """Monta o contexto de uma paciente para a LLM.

    Recebe features e métricas, nunca identificadores — e passa tudo pela allow-list antes de
    devolver. A `probabilidade` e a `predicao` vêm do modelo já treinado: a LLM traduz essa
    saída, não a produz (plano §6.2).
    """
    registro_bruto = X_paciente.to_dict()
    campos = filtrar_campos_autorizados(registro_bruto)

    features_relevantes: dict[str, Any] = {}
    if importancias:
        ordenadas = sorted(importancias.items(), key=lambda kv: kv[1], reverse=True)
        for nome, peso in ordenadas[:top_n_features]:
            if nome.strip() in {c.strip() for c in CAMPOS_CLINICOS_AUTORIZADOS}:
                features_relevantes[nome] = {
                    "valor_da_paciente": campos.get(nome),
                    "peso_no_modelo": round(float(peso), 4),
                }

    return {
        "predicao": int(predicao),
        "predicao_rotulo": "indicativo de SOP" if predicao == 1 else "sem indicativo de SOP",
        "probabilidade": round(float(probabilidade), 4),
        "probabilidade_pct": f"{100 * float(probabilidade):.1f}%",
        "features_mais_relevantes": features_relevantes,
        "dados_clinicos": campos,
    }


def montar_contexto_experimento(
    resumo_ag: dict[str, Any], comparativo: list[dict[str, Any]]
) -> dict[str, Any]:
    """Contexto para o relatório do experimento de otimização (plano §6.2, artefato 2) —
    "transformar dados numéricos e estatísticos em insights acionáveis", aplicado ao próprio AG.
    """
    return {
        "experimento": resumo_ag.get("experimento"),
        "melhor_fitness_cv": resumo_ag.get("melhor_fitness"),
        "familia_vencedora": resumo_ag.get("familia_vencedora"),
        "geracoes_executadas": resumo_ag.get("geracoes"),
        "motivo_parada": resumo_ag.get("motivo_parada"),
        "taxa_cache": resumo_ag.get("taxa_cache"),
        "comparativo_modelos": comparativo,
    }


def montar_contexto_coorte(predicoes: list[dict[str, Any]]) -> dict[str, Any]:
    """Contexto do resumo de coorte (plano §6.2, artefato 3): agregados sobre as predições do
    conjunto de teste, sem nenhum dado individual.
    """
    import numpy as np

    probabilidades = [p["probabilidade"] for p in predicoes]
    n_positivos = sum(1 for p in predicoes if p["predicao"] == 1)
    # "zona de incerteza": probabilidade entre 0,4 e 0,6 — onde o modelo está menos confiante
    incertos = [p for p in predicoes if 0.4 <= p["probabilidade"] <= 0.6]

    return {
        "n_pacientes": len(predicoes),
        "n_indicativo_sop": n_positivos,
        "pct_indicativo_sop": f"{100 * n_positivos / len(predicoes):.1f}%",
        "probabilidade_media": round(float(np.mean(probabilidades)), 4),
        "probabilidade_mediana": round(float(np.median(probabilidades)), 4),
        "n_zona_incerteza": len(incertos),
        "pct_zona_incerteza": f"{100 * len(incertos) / len(predicoes):.1f}%",
        "faixas": {
            "baixo_risco_ate_30pct": sum(1 for p in probabilidades if p < 0.3),
            "intermediario_30_70pct": sum(1 for p in probabilidades if 0.3 <= p <= 0.7),
            "alto_risco_acima_70pct": sum(1 for p in probabilidades if p > 0.7),
        },
    }
