"""Convert the IEEE-style section sources in paper/ into MDPI (Algorithms) section sources
in paper_mdpi/. Templates (*.tex.in) stay templates, so `eaagn.render` fills both versions
from the same numbers.json.

    python -m eaagn.to_mdpi --src paper --dst paper_mdpi
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

SECTIONS = ["sec_intro.tex.in", "sec_related.tex", "sec_method.tex.in", "sec_experiments.tex.in",
            "sec_results.tex.in", "sec_conclusion.tex.in"]

GENAI_METHODS = r"\input{sec_genai}"  # stage-by-stage AI-use statement (revision round 1, R2.3)

# single-column figures: width as a fraction of the MDPI text width
FIG_WIDTH = {"figures/example.pdf": r"0.72\textwidth", "figures/runtime_numpy.pdf": r"0.8\textwidth"}


def wide_figure(m):
    body = m.group(1)
    inc = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", body).group(1)
    cap_lab = body[body.index(r"\caption"):].strip()
    return ("\\begin{figure}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{inc}}}\n\\end{{adjustwidth}}\n{cap_lab}\n\\end{{figure}}")


def narrow_figure(m):
    body = m.group(1)
    inc = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", body).group(1)
    width = FIG_WIDTH.get(inc, r"\textwidth")
    cap_lab = body[body.index(r"\caption"):].strip()
    return f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width={width}]{{{inc}}}\n{cap_lab}\n\\end{{figure}}"


def convert(s: str) -> str:
    s = s.replace(r"\IEEEPARstart{A}{} digital", "A digital")
    s = s.replace(r"\begin{IEEEproof}", r"\begin{proof}").replace(r"\end{IEEEproof}", r"\end{proof}")
    s = s.replace(r"\begin{proposition}", r"\begin{Proposition}").replace(r"\end{proposition}", r"\end{Proposition}")
    s = re.sub(r"\\begin\{figure\*\}\[!t\](.*?)\\end\{figure\*\}", wide_figure, s, flags=re.S)
    s = re.sub(r"\\begin\{figure\}\[!t\](.*?)\\end\{figure\}", narrow_figure, s, flags=re.S)
    s = s.replace(r"\begin{table}[!t]", r"\begin{table}[H]")
    s = s.replace("Fig.~", "Figure~")
    # MDPI number style: thousands separator only from five digits
    s = re.sub(r"(?<![\d,.])(\d),(\d{3})(?![\d,])", r"\1\2", s)
    # MDPI asks for a Discussion section, which may be combined with the Results
    s = s.replace(r"\section{Results}", r"\section{Results and Discussion}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="paper")
    ap.add_argument("--dst", default="paper_mdpi")
    a = ap.parse_args()
    for name in SECTIONS:
        txt = open(os.path.join(a.src, name), encoding="utf-8").read()
        txt = convert(txt)
        if name == "sec_experiments.tex.in":  # MDPI requires GenAI use to be disclosed in the Methods
            txt = txt.rstrip() + "\n\n" + GENAI_METHODS + "\n"
        open(os.path.join(a.dst, name), "w", encoding="utf-8").write(txt)
    shutil.copy(os.path.join(a.src, "sec_genai.tex.in"), os.path.join(a.dst, "sec_genai.tex.in"))
    for d in ("figures", "tables"):
        shutil.copytree(os.path.join(a.src, d), os.path.join(a.dst, d), dirs_exist_ok=True)
    shutil.copy(os.path.join(a.src, "ggflag.tex"), os.path.join(a.dst, "ggflag.tex"))
    # mdpi.bst has no month macros (ACS style omits months), and warns on volume+number in proceedings
    bib = open(os.path.join(a.src, "refs.bib"), encoding="utf-8").read()
    bib = re.sub(r"^\s*month\s*=\s*[^\n]*\n", "", bib, flags=re.M)
    bib = re.sub(r"(@inproceedings\{pareja2020evolvegcn,.*?)^\s*number\s*=\s*[^\n]*\n", r"\1", bib, flags=re.S | re.M)
    # mdpi.bst prints "In Proceedings of the" itself
    bib = re.sub(r"(booktitle\s*=\s*\{)(?:Proc\.\s+|Proceedings of the\s+)", r"\1", bib)
    # arXiv preprints: ACS style "arXiv YEAR"
    bib = re.sub(r"journal\s*=\s*\{arXiv preprint arXiv:[^}]*\}", "journal = {arXiv}", bib)
    bib = re.sub(r"@misc\{hinton2015distilling,(.*?)\n\}",
                 lambda m: "@article{hinton2015distilling," + re.sub(r"\n\s*(eprint|archivePrefix|primaryClass)\s*=[^\n]*", "", m.group(1))
                 + ",\n  journal       = {arXiv}\n}", bib, flags=re.S)
    # web resource in MDPI form: "Available online: URL (accessed on ...)"
    bib = re.sub(r"@misc\{sprsound2025repo,.*?\n\}",
                 "@www{sprsound2025repo,\n  author  = {{SJTU Yongfu Li Research Group}},\n"
                 "  title   = {{SPRSound}: {BioCAS} 2025 Grand Challenge Test Set (Directory BioCAS2025)},\n"
                 "  url     = {https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound},\n"
                 "  urldate = {14 October 2026},\n  year    = {2025}\n}", bib, flags=re.S)
    open(os.path.join(a.dst, "refs.bib"), "w", encoding="utf-8").write(bib)
    print("converted", len(SECTIONS), "sections")


if __name__ == "__main__":
    main()
