# ADR 009 — Scripts da Fase 1 copiados sem alteração

**Status:** aceita · **Data:** 08/10/2026 · **Restrição do usuário**

## Contexto

A Fase 2 é construída sobre o pipeline do PCOS da Fase 1. Havia duas formas de reaproveitá-lo:
editar os scripts para expor o que a Fase 2 precisa, ou copiá-los intactos e construir em volta.

Instrução explícita do usuário: *"nao altere o script de pcos, copie ele e inicia a partir dai"*.

## Decisão

`src/run_ml.py` é **cópia byte a byte** do repositório da Fase 1
(md5 `9709d0fd67559d38115e8aa561759410` nas duas fases).

`src/modelagem/treinamento.py` tem **uma linha alterada**, e só ela — `REPORTS_DIR`, o diretório
de saída:

```diff
-REPORTS_DIR = BASE_DIR / "reports"
+REPORTS_DIR = (
+    BASE_DIR / "arquivos_saida" / "arquivos_analiticos" / "metricas_modelos_baseline"
+)
```

Nenhuma outra diferença: `diff fase1/src/modelagem/treinamento.py fase2/src/modelagem/treinamento.py`
devolve exatamente esse hunk. A mudança existe porque a Fase 2 consolida saídas sob
`arquivos_saida/` e não toca nenhuma decisão de modelagem — divisão, pré-processamento,
hiperparâmetros e semente são os mesmos, e é por isso que `python -m src.run_ml` reproduz a
tabela da Fase 1 até a quarta casa decimal.

Toda lógica nova vive em arquivos novos que os **importam**.

## Justificativa

**Além da instrução do usuário, há razão técnica.** O baseline da Fase 2 é a tabela de métricas
da Fase 1. Se o script fosse editado — mesmo uma refatoração aparentemente inócua — não haveria
como provar que a diferença entre "antes" e "depois" vem do AG e não da edição.

Com a lógica intacta, `python -m src.run_ml` reproduz os cinco modelos **exatamente** como
documentado, até a quarta casa decimal. Isso foi o Passo 1 da implementação, antes de qualquer
linha de AG. A única linha alterada — `REPORTS_DIR` — não entra em nenhum cálculo: ela diz onde
gravar, não o que treinar. O baseline reproduzido é prova disso, não promessa.

## Como a Fase 2 se acopla sem modificar

`fitness_pcos.py` importa as funções do script original e as compõe:

```python
from src.modelagem.treinamento import (
    SEED, carregar_dados, criar_preprocessador, dividir_dados,
    identificar_colunas_tipo, preparar_features,
)
```

`carregar_dados_treino()` chama `dividir_dados()` e **descarta** as variáveis de teste — nomeadas
`_X_teste_NAO_USAR` para deixar explícito a quem ler o código (ver ADR 003).

## Consequências

**Positivas**
- Baseline verificável e verificado.
- Mesmo split, mesma semente (SEED=42), mesmo pré-processamento que a Fase 1 — a comparação
  "original vs. otimizado" é sobre o mesmo recorte de dados.

**Negativas**
- Algumas funções têm assinatura menos conveniente do que eu escreveria hoje. Absorvido em
  `fitness_pcos.py`, sem tocar a origem.
- `treinamento.py` recalcula o split a cada chamada de `carregar_dados_treino()`. Irrelevante no
  fluxo atual (chamada uma vez por execução).

## Descoberta que esta decisão permitiu

Lendo o script em vez de assumir, descobri que o plano estava **errado** ao afirmar que a Fase 1
usava "parâmetros default do scikit-learn". Ela já tinha `class_weight="balanced"` em todos os
modelos, `n_estimators=200`/`max_depth=8` no Random Forest, `max_depth=5` na árvore e
`max_iter=2000` na regressão logística.

Isso **fortalece** o argumento do hotstart — o ponto de partida é uma configuração
conscientemente escolhida, não um default arbitrário — mas significa que a barra a superar é mais
alta do que o plano supunha. Se eu tivesse editado o script, provavelmente teria normalizado
esses valores sem notar.
