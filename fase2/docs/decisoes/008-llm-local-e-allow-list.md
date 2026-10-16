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

## Execução com LLM real — 14/10/2026

A decisão deste ADR foi finalmente **exercida**: Ollama 0.40.3 local, **`llama3.2:3b`**, 45 saídas
(15 pacientes × 3 templates), ~29 s por geração, ~2 h no total. Nenhum dado de paciente saiu da
máquina — a premissa do ADR valendo na prática, não só no papel.

**Modelo: 3B e não o 8B que este ADR previa.** A máquina de execução tem 8 GB de RAM; o
`llama3.1:8b` quantizado ocupa ~4,7 GB e, com o kernel do Jupyter e o servidor Ollama
concorrendo, entra em swap. O default do código segue `llama3.1:8b`; a execução reportada passou
`modelo="llama3.2:3b"` no `ConfigLLM`, sem editar `cliente.py`. **Continua sendo LLaMA** — a
família que o enunciado nomeia. Ver relatório §5.5 para o que o porte menor custa.

**O que a execução mostrou sobre a decisão deste ADR:**

- **Os guardrails sobrevivem a um modelo pequeno.** 0% de alucinação numérica e 100% de aderência
  estrutural nas 45 saídas. Isso vem da **arquitetura do prompt** (números pré-escritos no bloco
  `DADOS`), não da capacidade do modelo — e é o argumento mais forte a favor de rodar local: um
  3B basta quando o trabalho numérico é feito em código.
- **O porte do modelo custa calibração, não fidelidade.** Nenhum template passou de 67% na regra
  de faixa de confiança, e o few-shot *piorou* esse número (60%) ao ensinar a expressão "baixa
  confiança" sem a condição que a justifica. Um modelo maior provavelmente aplicaria a regra
  melhor; a correção certa, porém, é **pré-computar a faixa em código**, igual aos números.

## Bug encontrado no verificador (achado nº 12)

A execução real expôs um segundo instrumento quebrado, este mais grave que o do `simulado`:
`TERMOS_PROIBIDOS` era testado com `termo in texto`, **sem tratar negação**. A frase
"**não** constitui diagnóstico definitivo" — que o prompt **manda** escrever — era contada como
violação da proibição de diagnosticar.

**O efeito era inverter uma conclusão:** as 10 "violações" detectadas eram todas ressalvas, e o
V1 acusava mais (9 de 10) por ser o template mais verboso, logo o que mais repete a ressalva. A
leitura ingênua seria *"o few-shot ensina o modelo a não diagnosticar"* — falso. O número real é
**0 violações nos três templates**.

Correção: `_esta_negado()` olha 60 caracteres à esquerda do termo e para na fronteira de frase —
uma negação na frase anterior não governa o termo. É heurística, não análise sintática, e a
escolha tem justificativa de custo assimétrico: um **falso positivo** desqualifica uma ressalva
correta e inverte a conclusão do experimento; um **falso negativo** deixa passar uma violação que
a rubrica manual ainda pode pegar. Dez testes novos fixam os dois lados — 5 ressalvas que não
devem acusar e **5 violações reais que devem continuar acusando**, porque sem esse par a correção
passaria num verificador que simplesmente parou de acusar.

**O padrão, pela segunda vez neste projeto:** um agregado que parecia um resultado, e que só se
revelou errado quando fui olhar o dado bruto. O primeiro caso foi a métrica de diversidade do AG
(ADR 007, genótipo × fenótipo); este é o mesmo erro — presença literal de um termo confundida com
a afirmação que o termo faz.
