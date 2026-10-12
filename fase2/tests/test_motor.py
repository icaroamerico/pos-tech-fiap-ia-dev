"""Testes do motor evolutivo (plano §4.5, §5.1, §5.2).

Usam configurações minúsculas (população 6, 3 gerações) — o objetivo é verificar a LÓGICA do
laço, não a qualidade da otimização. Os experimentos reais ficam em src/experimentos/.
"""

import json

import numpy as np
import pytest

from src.ag.cromossomo import FAMILIAS, validar_individuo
from src.ag.motor import ConfigAG, executar_ag
from src.fitness.fitness_pcos import carregar_dados_treino
from src.monitoramento.registro import EscritorExecucao, carregar_historico


@pytest.fixture(scope="module")
def dados():
    return carregar_dados_treino()


@pytest.fixture
def escritor(tmp_path):
    return EscritorExecucao("teste", dir_base=tmp_path)


def _config_minima(**kwargs) -> ConfigAG:
    base = dict(
        nome="TESTE", tamanho_populacao=6, n_geracoes=3, n_elite=1, semente=7, n_jobs=1
    )
    base.update(kwargs)
    return ConfigAG(**base)


def test_executa_e_devolve_resultado_coerente(dados, escritor):
    resultado = executar_ag(_config_minima(), dados, escritor=escritor, verboso=False)

    assert resultado.geracoes_executadas == 3
    assert len(resultado.historico) == 3
    assert resultado.melhor_individuo is not None
    assert validar_individuo(resultado.melhor_individuo)
    assert 0.0 <= resultado.melhor_fitness <= 1.0
    assert resultado.motivo_parada == "maximo_de_geracoes"
    assert resultado.tempo_total_s > 0


def test_melhor_fitness_nunca_piora_entre_geracoes(dados, escritor):
    """Com elitismo (n_elite >= 1), a curva de melhor fitness tem de ser monotonicamente não
    decrescente — é o que torna o gráfico de convergência interpretável (plano §4.4).
    """
    resultado = executar_ag(
        _config_minima(n_geracoes=5, n_elite=1), dados, escritor=escritor, verboso=False
    )
    melhores = [r.melhor_fitness for r in resultado.historico]
    for anterior, atual in zip(melhores, melhores[1:]):
        assert atual >= anterior - 1e-12, f"melhor fitness piorou: {melhores}"


def test_populacao_mantem_tamanho_em_todas_as_geracoes(dados, escritor):
    config = _config_minima(n_geracoes=4, tamanho_populacao=8)
    resultado = executar_ag(config, dados, escritor=escritor, verboso=False)
    for registro in resultado.historico:
        assert registro.n_avaliacoes == config.tamanho_populacao
    assert len(resultado.populacao_final) == config.tamanho_populacao


def test_execucao_e_reprodutivel_com_a_mesma_semente(dados, tmp_path):
    """Mesma semente -> mesmo resultado. Pré-requisito para os experimentos com repetição
    serem comparáveis entre si (plano §4.6).
    """
    r1 = executar_ag(
        _config_minima(semente=123),
        dados,
        escritor=EscritorExecucao("r1", dir_base=tmp_path),
        verboso=False,
    )
    r2 = executar_ag(
        _config_minima(semente=123),
        dados,
        escritor=EscritorExecucao("r2", dir_base=tmp_path),
        verboso=False,
    )
    assert r1.melhor_fitness == r2.melhor_fitness
    assert [r.melhor_fitness for r in r1.historico] == [r.melhor_fitness for r in r2.historico]


def test_sementes_diferentes_produzem_execucoes_diferentes(dados, tmp_path):
    r1 = executar_ag(
        _config_minima(semente=1),
        dados,
        escritor=EscritorExecucao("s1", dir_base=tmp_path),
        verboso=False,
    )
    r2 = executar_ag(
        _config_minima(semente=999),
        dados,
        escritor=EscritorExecucao("s2", dir_base=tmp_path),
        verboso=False,
    )
    trajetoria_1 = [r.fitness_medio for r in r1.historico]
    trajetoria_2 = [r.fitness_medio for r in r2.historico]
    assert trajetoria_1 != trajetoria_2


@pytest.mark.parametrize("familia", FAMILIAS)
def test_familia_fixa_mantem_todos_os_individuos_na_familia(dados, familia, tmp_path):
    """Etapa A do plano §4.8: com `familia_fixa`, o AG otimiza SÓ aquela família — nenhum
    indivíduo pode escapar para outra, nem por mutação nem por crossover.
    """
    resultado = executar_ag(
        _config_minima(familia_fixa=familia, n_geracoes=3),
        dados,
        escritor=EscritorExecucao(f"fix_{familia}", dir_base=tmp_path),
        verboso=False,
    )
    assert resultado.melhor_individuo is not None
    assert resultado.melhor_individuo["familia"] == familia
    for individuo in resultado.populacao_final:
        assert individuo["familia"] == familia
    for registro in resultado.historico:
        assert registro.familia_do_melhor == familia
        assert registro.distribuicao_familias[familia] == pytest.approx(1.0)


def test_parada_por_estagnacao_dispara(dados, escritor):
    """Com mutação e crossover desligados e elitismo ativo, a população congela e o critério
    de estagnação do plano §4.5 tem de encerrar antes do máximo de gerações.
    """
    config = _config_minima(
        n_geracoes=30,
        n_elite=1,
        prob_mutacao=0.0,
        taxa_crossover=0.0,
        prob_troca_familia=0.0,
        paciencia_estagnacao=3,
    )
    resultado = executar_ag(config, dados, escritor=escritor, verboso=False)
    assert resultado.geracoes_executadas < 30
    assert "estagnacao" in resultado.motivo_parada


def test_selecao_roleta_tambem_executa(dados, escritor):
    """A roleta é o operador da Etapa D (ablação, plano §4.8) — precisa rodar de ponta a
    ponta, não só existir.
    """
    resultado = executar_ag(
        _config_minima(selecao="roleta"), dados, escritor=escritor, verboso=False
    )
    assert resultado.melhor_individuo is not None
    assert resultado.config["selecao"] == "roleta"


def test_cache_registra_acertos_ao_longo_das_geracoes(dados, escritor):
    """O cache do plano §5.2 tem de produzir acertos de verdade num laço real — com elitismo,
    pelo menos os indivíduos da elite repetem entre gerações.

    Note que a geração 0 já pode ter acertos, e isso é correto: são duplicatas DENTRO da
    mesma geração. O KNN tem espaço de fenótipos totalmente discreto e pequeno
    (n_neighbors ímpar × weights × metric), então dois indivíduos aleatórios colidirem é
    esperado — e o cache evita treinar o mesmo modelo duas vezes.
    """
    resultado = executar_ag(
        _config_minima(n_geracoes=4, n_elite=2, prob_mutacao=0.05),
        dados,
        escritor=escritor,
        verboso=False,
    )
    assert resultado.total_cache_hits > 0
    # o reaproveitamento ENTRE gerações é o que o elitismo garante
    assert sum(r.n_cache_hits for r in resultado.historico[1:]) > 0
    # nunca pode haver mais acertos que avaliações numa geração
    for registro in resultado.historico:
        assert 0 <= registro.n_cache_hits <= registro.n_avaliacoes


def test_artefatos_sao_escritos_no_disco(dados, tmp_path):
    escritor = EscritorExecucao("artefatos", dir_base=tmp_path)
    resultado = executar_ag(_config_minima(), dados, escritor=escritor, verboso=False)

    dir_execucao = tmp_path / "artefatos"
    for nome in ("config.json", "history.jsonl", "best_individual.json",
                 "population_final.json", "resumo.json"):
        assert (dir_execucao / nome).exists(), f"artefato ausente: {nome}"

    config_salva = json.loads((dir_execucao / "config.json").read_text(encoding="utf-8"))
    assert config_salva["tamanho_populacao"] == 6
    assert "ambiente" in config_salva  # versões de biblioteca, para rastreabilidade

    melhor = json.loads((dir_execucao / "best_individual.json").read_text(encoding="utf-8"))
    assert melhor["fitness"] == pytest.approx(resultado.melhor_fitness)


def test_historico_jsonl_carrega_em_dataframe(dados, tmp_path):
    """O contrato com os notebooks (plano §7.4): a análise lê o JSONL, não reexecuta o AG."""
    escritor = EscritorExecucao("hist", dir_base=tmp_path)
    executar_ag(_config_minima(n_geracoes=3), dados, escritor=escritor, verboso=False)

    df = carregar_historico("hist", dir_base=tmp_path)
    assert len(df) == 3
    for coluna in ("geracao", "melhor_fitness", "fitness_medio", "entropia_familia",
                   "n_solucoes_unicas", "freq_class_weight_balanced", "tempo_geracao_s"):
        assert coluna in df.columns


def test_motor_nunca_acessa_o_conjunto_de_teste(monkeypatch, tmp_path):
    """Extensão do teste anti-vazamento do plano §3 para o LAÇO COMPLETO — não só para uma
    chamada isolada de fitness. Se qualquer parte do motor tocar no holdout ao longo de
    várias gerações, o espião levanta exceção.
    """
    import src.fitness.fitness_pcos as fitness_pcos

    class EspiaoNuncaDeveSerAcessado:
        def __getattr__(self, nome):
            raise AssertionError(f"VAZAMENTO: motor acessou '.{nome}' do conjunto de TESTE")

        def __len__(self):
            raise AssertionError("VAZAMENTO: motor chamou len() no conjunto de TESTE")

        def __getitem__(self, _chave):
            raise AssertionError("VAZAMENTO: motor indexou o conjunto de TESTE")

        def __iter__(self):
            raise AssertionError("VAZAMENTO: motor iterou o conjunto de TESTE")

    def dividir_dados_espiao(X, y, test_size=0.2, random_state=42):
        from src.modelagem.treinamento import dividir_dados as original

        X_tr, _, y_tr, _ = original(X, y, test_size=test_size, random_state=random_state)
        return X_tr, EspiaoNuncaDeveSerAcessado(), y_tr, EspiaoNuncaDeveSerAcessado()

    monkeypatch.setattr(fitness_pcos, "dividir_dados", dividir_dados_espiao)
    dados_instrumentados = fitness_pcos.carregar_dados_treino()

    resultado = executar_ag(
        _config_minima(n_geracoes=3),
        dados_instrumentados,
        escritor=EscritorExecucao("leak", dir_base=tmp_path),
        verboso=False,
    )
    assert resultado.melhor_individuo is not None


def test_config_rejeita_elite_maior_que_populacao():
    with pytest.raises(ValueError, match="n_elite"):
        ConfigAG(tamanho_populacao=5, n_elite=5)


def test_config_rejeita_familia_invalida():
    with pytest.raises(ValueError, match="familia_fixa"):
        ConfigAG(familia_fixa="modelo_inexistente")
