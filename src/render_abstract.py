"""Print the paper's abstract as plain text with the generated numbers substituted (for release/ARXIV_METADATA.md)."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tex = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
nums = dict(re.findall(r"\\newcommand\{\\(Num\w+)\}\{(.*?)\}\n", open(os.path.join(ROOT, "paper", "numbers.tex"), encoding="utf-8").read()))
abs_ = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
abs_ = re.sub(r"\\(Num\w+)\{\}", lambda m: nums[m.group(1)], abs_)
abs_ = re.sub(r"\\(Num\w+)", lambda m: nums[m.group(1)], abs_)
for a, b in [(r"\%", "%"), ("~", " "), (r"$\pm$", "+/-"), (r"\texttt{honestboard}", "honestboard"), ("$p$", "p"), ("{,}", ","), ("$-$", "-")]:
    abs_ = abs_.replace(a, b)
abs_ = " ".join(line.strip() for line in abs_.strip().splitlines())
print(abs_)
print("WORDS", len(abs_.split()))
