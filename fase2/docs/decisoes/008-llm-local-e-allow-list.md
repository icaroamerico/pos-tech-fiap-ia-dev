# ADR 008 — LLM local por padrão; anonimização por allow-list

**Status:** aceita · **Data:** 09/10/2026

## Contexto

Duas aulas da fase dão recomendações **conflitantes**:

- **Aula 5 de LLMs** apresenta as três formas de integração (local / nuvem / API) e recomenda
  **API** como *"a mais viável para a maioria das empresas"*, por custo.
- **Aula 5 de IA Generativa** registra que **Amazon, Apple, Samsung e JP Morgan proibiram o
  ChatGPT** internamente, porque o que é digitado no prompt vai para os servidores do provedor e
  pode ser armazenado ou usado em treinamento.

Os dados aqui são clínicos e hormonais de pacientes de uma clínica de fertilidade, em território
de LGPD.

## Decisão

**1. Provedor padrão: local** (Ollama com LLaMA). API (OpenAI) é opcional e apenas para dados
**agregados ou anonimizados**. Um terceiro provedor, `simulado`, é determinístico e sem rede.

**2. Anonimização por allow-list**, não deny-list: o contexto enviado à LLM é filtrado por uma
lista **explícita** de campos clínicos autorizados. Identificadores e o rótulo verdadeiro estão
numa lista de proibidos que **levanta exceção** em vez de descartar em silêncio.

## Justificativa

**Como resolvi o conflito entre as aulas.** Quando privacidade e custo colidem em dados de
paciente, privacidade ganha. Local para dados identificáveis, API só para agregados.

**Por que allow-list e não deny-list — a decisão mais importante desta camada.** Uma deny-list
("remova `nome`, `cpf`, `email`") **vaza por omissão**: basta uma coluna nova aparecer na base
para ela ir ao prompt sem ninguém decidir isso. Uma allow-list **falha de forma segura** — o campo
não autorizado simplesmente não existe no contexto. Com dados de paciente, é a diferença entre um
bug e um incidente.

**Por que o rótulo verdadeiro (`PCOS (Y/N)`) também é proibido.** Se vazasse, a LLM poderia
"explicar" o diagnóstico usando o gabarito em vez da saída do modelo — e o texto pareceria
excelente justamente quando o modelo estivesse errado.

**Por que o provedor `simulado` existe.** Deixa o notebook executável por qualquer avaliador, sem
Ollama instalado nem chave de API — mesmo princípio que levou a Fase 1 a versionar uma amostra de
16 imagens de raio-X. E, por só repetir números que recebeu, serve de **controle** da avaliação de
alucinação: taxa zero esperada por construção.

## Consequências

**Positivas**
- Nenhum dado de paciente sai da máquina na configuração padrão.
- Testado: colunas novas e desconhecidas não entram no contexto; campos proibidos levantam
  exceção.
- O notebook roda em qualquer máquina.

**Negativas**
- A allow-list precisa de manutenção: coluna clínica nova exige decisão explícita de incluí-la.
  **Isso é a feature, não o bug** — a pergunta a responder ao incluir é "este dado, isolado ou
  combinado, permite reidentificar a paciente?".
- O modelo local é menor que GPT-4, então a qualidade do texto é inferior ao teto possível.

## Bug encontrado no provedor simulado

A primeira versão varria **todas** as linhas iniciadas por `-` do prompt — e capturava os números
dos **exemplares few-shot** (91,2% / 18 folículos / 52,7%) em vez dos da paciente. O "controle de
alucinação zero" alucinava **19%** dos números.

Um instrumento de medida contaminado pelo próprio prompt não serve como controle. Correção: passar
o bloco de dados explicitamente na assinatura do provedor, em vez de raspar o prompt.
