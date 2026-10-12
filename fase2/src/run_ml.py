"""Ponto de entrada para a ETAPA 3 — Treinamento e avaliação.

Uso:
    python -m src.run_ml

ou, a partir da raiz do projeto:

    python src/run_ml.py
"""

import sys
from pathlib import Path

# Garante que a raiz do projeto esteja em sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modelagem.treinamento import executar_treinamento  # noqa: E402


if __name__ == "__main__":
    executar_treinamento()