"""Make arXiv identifiers visible in the bibliography: plainnat prints `note` but not `eprint`."""
import os
import re

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "refs.bib")
s = open(p, encoding="utf-8").read()
entries = re.split(r"(?=^@)", s, flags=re.M)
out = []
for e in entries:
    m = re.search(r"eprint\s*=\s*\{([^}]+)\}", e)
    if m and "arXiv:" not in e:
        ident = m.group(1)
        note = re.search(r"note\s*=\s*\{([^}]*)\}", e)
        if note:
            e = e.replace(note.group(0), f"note = {{arXiv:{ident}. {note.group(1)}}}")
        else:
            e = re.sub(r"\n\}\s*$", f",\n  note = {{arXiv:{ident}}}\n}}\n", e.rstrip() + "\n") + "\n"
    out.append(e)
open(p, "w", encoding="utf-8").write("".join(out))
print("updated", sum("arXiv:" in e for e in out), "entries with arXiv notes")
