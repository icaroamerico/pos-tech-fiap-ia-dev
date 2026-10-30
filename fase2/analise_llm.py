"""Comparação V1/V2/V3 a partir das saídas do provedor real."""
import json, statistics as st, hashlib
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
p = (
    RAIZ / "arquivos_saida" / "arquivos_analiticos" / "avaliacao_llm"
    / "avaliacao_ollama_saidas.json"
)
d = json.load(open(p))
TPL = ["v1_zero_shot", "v2_few_shot", "v3_few_shot_estruturado"]
by = defaultdict(list)
for a in d:
    by[a["template"]].append(a)

print(f"=== {len(d)} saidas | {len(by)} templates ===\n")
print(f"{'template':26s} {'aluc%':>7s} {'ader%':>7s} {'chars':>7s} {'lat_s':>7s} {'n_num':>6s} {'proib':>6s} {'sec_falt':>9s}")
for t in TPL:
    xs = by[t]
    if not xs: continue
    print(f"{t:26s} "
          f"{100*st.mean(a['taxa_alucinacao_numerica'] for a in xs):7.1f} "
          f"{100*st.mean(a['aderencia_estrutural'] for a in xs):7.1f} "
          f"{st.mean(a['n_caracteres'] for a in xs):7.0f} "
          f"{st.mean(a['latencia_s'] for a in xs):7.1f} "
          f"{st.mean(a['n_numeros_citados'] for a in xs):6.1f} "
          f"{sum(len(a['termos_proibidos']) for a in xs):6d} "
          f"{sum(len(a['secoes_ausentes']) for a in xs):9d}")

print("\n=== textos identicos entre templates? ===")
pac = defaultdict(dict)
for a in d:
    pac[a["id_amostra"]][a["template"]] = hashlib.md5(a["texto"].encode()).hexdigest()[:10]
iguais = sum(1 for v in pac.values() if len(set(v.values())) == 1)
print(f"{iguais}/{len(pac)} pacientes com os 3 templates identicos  "
      f"({'DEGENERADO' if iguais == len(pac) else 'TEMPLATES TEM EFEITO'})")

print("\n=== por estrato (aderencia e alucinacao) ===")
est = defaultdict(list)
for a in d:
    est[(a["template"], a["estrato"])].append(a)
print(f"{'template':26s} {'estrato':26s} {'n':>3s} {'aluc%':>7s} {'ader%':>7s}")
for t in TPL:
    for e in ["positivo_alta_confianca", "negativo_alta_confianca", "limiar"]:
        xs = est.get((t, e), [])
        if not xs: continue
        print(f"{t:26s} {e:26s} {len(xs):3d} "
              f"{100*st.mean(a['taxa_alucinacao_numerica'] for a in xs):7.1f} "
              f"{100*st.mean(a['aderencia_estrutural'] for a in xs):7.1f}")

print("\n=== numeros problematicos (amostras) ===")
n = 0
for a in d:
    if a["numeros_problematicos"]:
        print(f"  {a['id_amostra']:28s} {a['template']:26s} {a['numeros_problematicos'][:6]}")
        n += 1
        if n >= 12: break
if n == 0:
    print("  nenhum")

print("\n=== secoes ausentes (quem falhou) ===")
n = 0
for a in d:
    if a["secoes_ausentes"]:
        print(f"  {a['id_amostra']:28s} {a['template']:26s} faltou: {a['secoes_ausentes']}")
        n += 1
        if n >= 12: break
if n == 0:
    print("  nenhuma")
