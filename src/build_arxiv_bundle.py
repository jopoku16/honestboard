"""Build release/arxiv_source/ and release/arxiv_source.zip from the compiled manuscript.

arXiv wants a flat folder with the .tex, the .bbl (not the .bib) and the figures, with paths that resolve inside
that folder. Run after `latexmk -pdf main.tex` in paper/.
"""
import os
import re
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAP = os.path.join(ROOT, "paper")
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "release", "arxiv_source")
os.makedirs(OUT, exist_ok=True)

tex = open(os.path.join(PAP, "main.tex"), encoding="utf-8").read()
figs = sorted(set(re.findall(r"\.\./figures/([A-Za-z0-9_]+\.pdf)", tex)))
tex = tex.replace("../figures/", "")
tex = re.sub(r"\\bibliography\{refs\}", r"\\bibliography{refs}", tex)
open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(tex)
for f in figs:
    shutil.copy(os.path.join(FIG, f), os.path.join(OUT, f))
support = ("numbers.tex", "refs.bib", "main.bbl") + tuple(f for f in os.listdir(PAP) if f.startswith("tab_") and f.endswith(".tex"))
for name in support:
    shutil.copy(os.path.join(PAP, name), os.path.join(OUT, name))
include = sorted(["main.tex", *figs, *support])          # only source files, never build artefacts
zpath = os.path.join(ROOT, "release", "arxiv_source.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in include:
        z.write(os.path.join(OUT, f), f)
print("bundle:", include)
print("zip size KB:", os.path.getsize(zpath) // 1024)
