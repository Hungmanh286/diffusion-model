#!/usr/bin/env python3
"""Extract the structure of a research-paper PDF so it can be planned for presentation.

Outputs (in --out directory):
  paper.txt     full text, each page preceded by a "=== Page N ===" marker
  structure.md  headings, Figure/Table captions, equation numbers, abstract, page-1 head lines
  pages/*.png   rendered pages (only with --render)

Usage:
  python3 -I extract_paper.py paper.pdf --out outdir [--render figures | --render 3,5,7]

Needs `pdftotext`/`pdftoppm` (poppler) or the `pypdf` package. Heuristics are approximate:
treat structure.md as a map, and read paper.txt for the real content.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HEADING_WORDS = (
    r"abstract|introduction|related work|background|preliminaries|"
    r"problem (?:formulation|statement|setup|definition)|methods?|methodology|approach|"
    r"proposed (?:method|approach|model)|architecture|framework|"
    r"experiments?|experimental (?:setup|results|settings?)|evaluation|results?|"
    r"ablations?(?: stud(?:y|ies))?|analysis|discussion|limitations?|"
    r"conclusions?|future work|acknowledg(?:e)?ments?|references|appendix"
)
NUMBERED = re.compile(r"^(?P<num>(?:[A-Z]|\d{1,2})(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][^\n]{2,90})$")
NAMED = re.compile(rf"^(?P<title>(?:{HEADING_WORDS}))\.?$", re.I)
CAPTION = re.compile(
    r"^(?P<kind>Figure|Fig\.|Table)\s*(?P<n>[A-Z]?\d+)\s*[.:|–—-]\s*(?P<text>.*)$"
)
EQ_NUMBER = re.compile(r"\((\d{1,3})\)\s*$")
MATH_HINT = re.compile(r"[=∑∏∫≤≥≈∈α-ωΑ-Ω]|\\frac|\bargmax\b|\bargmin\b|\bsoftmax\b")


def have(cmd):
    return shutil.which(cmd) is not None


def page_count(pdf: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf)).pages)
    except Exception:
        pass
    if have("pdfinfo"):
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        m = re.search(r"^Pages:\s+(\d+)", out, re.M)
        if m:
            return int(m.group(1))
    sys.exit("Cannot determine page count: install pypdf or poppler-utils.")


def page_text(pdf: Path, i: int) -> str:
    """1-based page index. pdftotext keeps reading order across columns better than pypdf."""
    if have("pdftotext"):
        r = subprocess.run(
            ["pdftotext", "-f", str(i), "-l", str(i), "-enc", "UTF-8", str(pdf), "-"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return r.stdout
    try:
        from pypdf import PdfReader

        return PdfReader(str(pdf)).pages[i - 1].extract_text() or ""
    except Exception:
        return ""


def find_headings(pages):
    heads = []
    for pno, text in pages:
        for line in text.splitlines():
            s = line.strip()
            if not s or len(s) > 100:
                continue
            m = NUMBERED.match(s)
            # reject author lines ("A. Author, B. Researcher") and sentences: headings have no commas
            if m and not s.endswith((".", ",", ";")) and "," not in s and len(s.split()) <= 12:
                heads.append((pno, f"{m.group('num')} {m.group('title')}"))
                continue
            m = NAMED.match(s)
            if m:
                heads.append((pno, m.group("title").title()))
    # drop consecutive duplicates (e.g. running headers repeated on many pages)
    seen, out = set(), []
    for pno, h in heads:
        key = h.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((pno, h))
    return out


def find_captions(pages):
    caps = []
    for pno, text in pages:
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            m = CAPTION.match(line.strip())
            if not m:
                continue
            body = [m.group("text").strip()]
            for nxt in lines[idx + 1 : idx + 4]:
                if not nxt.strip():
                    break
                body.append(nxt.strip())
            label = f"{'Figure' if m.group('kind').startswith('Fig') else 'Table'} {m.group('n')}"
            caps.append((pno, label, " ".join(body)[:260]))
    return caps


def find_equations(pages):
    eqs = []
    for pno, text in pages:
        for line in text.splitlines():
            s = line.strip()
            m = EQ_NUMBER.search(s)
            if m and len(s) < 160 and MATH_HINT.search(s):
                eqs.append((pno, int(m.group(1))))
    return sorted(set(eqs), key=lambda x: (x[1], x[0]))


def find_abstract(full_text: str) -> str:
    m = re.search(r"\babstract\b[\s:.—-]*", full_text, re.I)
    if not m:
        return ""
    rest = full_text[m.end() : m.end() + 3500]
    end = re.search(r"\n\s*(?:1\.?\s+)?(?:introduction)\b", rest, re.I)
    return re.sub(r"\s+", " ", rest[: end.start() if end else 1800]).strip()


def render_pages(pdf: Path, out_dir: Path, page_numbers):
    if not have("pdftoppm"):
        print("pdftoppm not found: skipping --render", file=sys.stderr)
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for p in page_numbers:
        prefix = out_dir / f"page-{p:02d}"
        subprocess.run(
            ["pdftoppm", "-f", str(p), "-l", str(p), "-r", "100", "-png", "-singlefile", str(pdf), str(prefix)],
            capture_output=True,
        )
        if prefix.with_suffix(".png").exists():
            made.append(prefix.with_suffix(".png"))
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output directory (created if missing)")
    ap.add_argument("--render", default="", help="'figures' (pages with a Figure caption, max 10) or comma list like 3,5")
    args = ap.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"File not found: {args.pdf}")
    args.out.mkdir(parents=True, exist_ok=True)

    n = page_count(args.pdf)
    pages = [(i, page_text(args.pdf, i)) for i in range(1, n + 1)]
    total_chars = sum(len(t.strip()) for _, t in pages)

    (args.out / "paper.txt").write_text(
        "\n".join(f"\n=== Page {p} ===\n{t}" for p, t in pages), encoding="utf-8"
    )

    heads = find_headings(pages)
    caps = find_captions(pages)
    eqs = find_equations(pages)
    full = "\n".join(t for _, t in pages[:3])
    abstract = find_abstract(full)
    head_lines = [l.strip() for l in pages[0][1].splitlines() if l.strip()][:10] if pages else []
    ref_page = next((p for p, h in heads if h.lower().startswith("references")), None)

    lines = [f"# Structure of {args.pdf.name}", ""]
    lines.append(f"- Pages: {n}; extracted characters: {total_chars}")
    if ref_page:
        lines.append(f"- References start on page {ref_page} (body is pages 1-{ref_page}; later pages are references/appendix)")
    if total_chars < 300 * max(n, 1) * 0.3:
        lines.append(
            "- WARNING: very little text was extracted. The PDF is probably scanned; "
            "run OCR (see the pdf skill) before using paper.txt."
        )
    lines += ["", "## First lines of page 1 (title / authors / affiliations)", ""]
    lines += [f"    {l}" for l in head_lines] or ["    (none)"]
    lines += ["", "## Abstract (auto-detected, verify against paper.txt)", "", abstract or "(not detected)"]
    lines += ["", "## Headings (page: heading)", ""]
    lines += [f"- p.{p}: {h}" for p, h in heads] or ["- (none detected; read paper.txt directly)"]
    lines += ["", "## Figure and table captions", ""]
    lines += [f"- p.{p} **{lab}**: {txt}" for p, lab, txt in caps] or ["- (none detected)"]
    lines += ["", "## Numbered equations seen (equation number, page)", ""]
    lines += [f"- Eq. ({num}) p.{p}" for p, num in eqs] or ["- (none detected)"]

    rendered = []
    if args.render:
        if args.render == "figures":
            fig_pages = []
            for p, lab, _ in caps:
                if lab.startswith("Figure") and p not in fig_pages:
                    fig_pages.append(p)
            targets = fig_pages[:10]
        else:
            targets = [int(x) for x in args.render.split(",") if x.strip().isdigit() and 1 <= int(x) <= n]
        rendered = render_pages(args.pdf, args.out / "pages", targets)
    if rendered:
        lines += ["", "## Rendered pages (view with Read)", ""]
        lines += [f"- {p}" for p in rendered]

    (args.out / "structure.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFull text: {args.out / 'paper.txt'}")


if __name__ == "__main__":
    main()
