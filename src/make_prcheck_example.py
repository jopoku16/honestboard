"""Write the honestboard `check` report for the current SWE-bench Verified leader to paper/prcheck_example.txt, which the
appendix "What a maintainer sees" includes verbatim (the board-wide `report` output is printed to the console only)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board  # noqa: E402

board = os.path.join(ROOT, "data", "boards", "swebench_verified.json")
b = Board.from_json(board)
rs = b.rank_summary(B=4000, top=3)
leader = max(rs, key=lambda r: r["p_best"])["name"]
env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "honestboard", "src"), OPENBLAS_NUM_THREADS="1")
out = subprocess.run([sys.executable, "-m", "honestboard", "check", board, leader], capture_output=True, text=True, env=env).stdout
rep = subprocess.run([sys.executable, "-m", "honestboard", "report", board], capture_output=True, text=True, env=env).stdout
text = f"$ python -m honestboard check swebench_verified.json {leader}\n{out.strip()}\n"
print(f"$ python -m honestboard report swebench_verified.json\n{rep.strip()}\n")
# wrap long lines so the verbatim block fits the page
wrapped = []
for line in text.splitlines():
    while len(line) > 96:
        cut = line.rfind(" ", 0, 96)
        cut = 96 if cut < 40 else cut
        wrapped.append(line[:cut])
        line = "    " + line[cut:].lstrip()
    wrapped.append(line)
with open(os.path.join(ROOT, "paper", "prcheck_example.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(wrapped) + "\n")
print("\n".join(wrapped))
