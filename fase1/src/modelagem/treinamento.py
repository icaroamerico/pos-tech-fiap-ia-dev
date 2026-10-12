"""ETAPA 3 — Treinamento e avaliação de modelos de classificação.

Recebe o dataset já limpo (``base_dados_tratada/``), separa 80% para
treino e 20% para teste, treina vários classificadores e imprime/salva
as principais métricas de cada um.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Constantes do projeto
# ---------------------------------------------------------------------------

# Raiz do repositório: src/modelagem/treinamento.py -> src -> raiz
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "base_dados_tratada" / "PCOS_unificado.csv"
REPORTS_DIR = BASE_DIR / "reports"

# Variável que queremos prever: presença (1) ou ausência (0) de SOP
TARGET = "PCOS (Y/N)"

# Colunas de identificação: não são features, apenas rastreabilidade
COLUNAS_ID = {"Sl. No", "Patient File No."}

# Semente para que o resultado seja reproduzível
SEED = 42


# ---------------------------------------------------------------------------
# 1. Carregar e preparar os dados
# ---------------------------------------------------------------------------

def carregar_dados(caminho: Path = INPUT_PATH) -> pd.DataFrame:
    """Lê a base tratada (CSV unificado) e remove colunas que não são features."""
    df = pd.read_csv(caminho)
    print(f"Base carregada: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) do alvo (y) e descarta IDs."""
    X = df.drop(columns=[TARGET]).drop(columns=COLUNAS_ID, errors="ignore")
    y = df[TARGET]
    return X, y


def identificar_colunas_tipo(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Classifica as colunas em numéricas ou categóricas.

    - Numéricas: todos os valores são de tipo numérico e NÃO são
      identificadores categóricos (como ``Blood Group``).
    - Categóricas: colunas de texto ou códigos que representam categorias.
    """
    colunas_numericas = [
        c for c in X.columns
        if c != "Blood Group" and pd.api.types.is_numeric_dtype(X[c])
    ]
    colunas_categoricas = [
        c for c in X.columns
        if c == "Blood Group" or not pd.api.types.is_numeric_dtype(X[c])
    ]
    return colunas_numericas, colunas_categoricas


# ---------------------------------------------------------------------------
# 2. Separação treino/teste (80% / 20%)
# ---------------------------------------------------------------------------

def dividir_dados(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide os dados em treino e teste, mantendo a proporção do alvo.

    ``stratify=y`` garante que tanto treino quanto teste tenham a mesma
    proporção de positivos/negativos que a base original.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(
        f"Treino: {len(X_train)} registros | "
        f"Teste: {len(X_test)} registros | "
        f"Positivos no teste: {int(y_test.sum())}/{len(y_test)}"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 3. Pré-processamento por tipo de coluna
# ---------------------------------------------------------------------------

def criar_preprocessador(
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
) -> ColumnTransformer:
    """Monta o transformador que escala numéricas e one-hot nas categóricas.

    - StandardScaler: subtrai a média e divide pelo desvio-padrão. Essencial
      para modelos sensíveis a escala (Regressão Logística, KNN, SVM).
    - OneHotEncoder: transforma categorias em colunas 0/1, sem impor ordem.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colunas_numericas),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                colunas_categoricas,
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# 4. Definição dos modelos que serão testados
# ---------------------------------------------------------------------------

def criar_modelos() -> dict[str, object]:
    """Retorna um dicionário com os classificadores a comparar.

    Cada modelo é um Pipeline: pré-processamento + classificador. Dessa
    forma o scaler/encoder se ajusta SOMENTE aos dados de treino e aplica
    a mesma transformação no teste, evitando vazamento de dados.
    """
    return {
        "Regressão Logística": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=SEED,
        ),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "SVM (RBF)": SVC(
            class_weight="balanced",
            random_state=SEED,
        ),
    }


def criar_pipelines(
    modelos: dict[str, object],
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
) -> dict[str, Pipeline]:
    """Envolve cada classificador em um Pipeline com o pré-processador."""
    preprocessador = criar_preprocessador(colunas_numericas, colunas_categoricas)
    return {
        nome: Pipeline(
            steps=[
                ("preprocessador", preprocessador),
                ("modelo", modelo),
            ]
        )
        for nome, modelo in modelos.items()
    }


# ---------------------------------------------------------------------------
# 5. Treinamento, predição e avaliação
# ---------------------------------------------------------------------------

def treinar_e_avaliar(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[dict]:
    """Roda o treino de cada modelo e coleta as métricas na base de teste.

    Para cada modelo, imprime em tela:
      - acurácia, precisão, recall, F1, AUC-ROC
      - matriz de confusão
      - relatório de classificação (precision/recall/f1 por classe)
    """
    resultados: list[dict] = []

    for nome, pipe in pipelines.items():
        print("\n" + "=" * 70)
        print(f"TREINANDO: {nome}")
        print("=" * 70)

        # .fit() ajusta o pré-processador e treina o algoritmo
        pipe.fit(X_train, y_train)

        # .predict() devolve a classe predita (0 ou 1)
        y_pred = pipe.predict(X_test)

        # .predict_proba() devolve a probabilidade; usamos a classe positiva
        # para AUC-ROC. Modelos sem predict_proca usam decision_function.
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            y_proba = pipe.decision_function(X_test)

        # Cálculo das principais métricas de classificação
        metricas = {
            "modelo": nome,
            "acuracia": float(accuracy_score(y_test, y_pred)),
            "precisao": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
        }

        # Matriz de confusão: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_test, y_pred)

        print(f"Acurácia : {metricas['acuracia']:.4f}")
        print(f"Precisão : {metricas['precisao']:.4f}")
        print(f"Recall   : {metricas['recall']:.4f}")
        print(f"F1-Score : {metricas['f1_score']:.4f}")
        print(f"AUC-ROC  : {metricas['auc_roc']:.4f}")
        print(f"\nMatriz de confusão:\n{cm}")
        print("\nRelatório de classificação:")
        print(classification_report(y_test, y_pred, target_names=["Sem SOP", "Com SOP"], zero_division=0))

        # Guarda a matriz como lista para poder serializar em JSON
        metricas["matriz_confusao"] = cm.tolist()
        resultados.append(metricas)

    return resultados


def salvar_relatorio(resultados: list[dict], caminho: Path) -> None:
    """Salva as métricas em um arquivo texto e um JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Relatório legível em texto
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE TREINAMENTO E AVALIAÇÃO\n")
        f.write("=" * 70 + "\n\n")
        for r in resultados:
            f.write(f"Modelo: {r['modelo']}\n")
            f.write(f"  Acurácia : {r['acuracia']:.4f}\n")
            f.write(f"  Precisão : {r['precisao']:.4f}\n")
            f.write(f"  Recall   : {r['recall']:.4f}\n")
            f.write(f"  F1-Score : {r['f1_score']:.4f}\n")
            f.write(f"  AUC-ROC  : {r['auc_roc']:.4f}\n")
            f.write(f"  Matriz de confusão: {r['matriz_confusao']}\n\n")

    # JSON com os dados brutos (útil para análise posterior)
    json_path = caminho.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(f"\nRelatórios salvos em:\n  - {caminho}\n  - {json_path}")


# ---------------------------------------------------------------------------
# 6. Ponto de entrada
# ---------------------------------------------------------------------------

def executar_treinamento() -> list[dict]:
    """Executa o fluxo completo: carrega, divide, treina, avalia, salva."""
    df = carregar_dados()
    X, y = preparar_features(df)

    colunas_numericas, colunas_categoricas = identificar_colunas_tipo(X)
    print(f"Features numéricas ({len(colunas_numericas)}): {colunas_numericas}")
    print(f"Features categóricas ({len(colunas_categoricas)}): {colunas_categoricas}")

    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    modelos = criar_modelos()
    pipelines = criar_pipelines(modelos, colunas_numericas, colunas_categoricas)
    resultados = treinar_e_avaliar(pipelines, X_train, X_test, y_train, y_test)
    salvar_relatorio(resultados, REPORTS_DIR / "metricas_modelos.txt")
    return resultados


if __name__ == "__main__":
    executar_treinamento()
