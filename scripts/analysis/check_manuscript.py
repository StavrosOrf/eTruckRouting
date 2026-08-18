"""Static checks for the manuscript, for use when no LaTeX toolchain exists.

This does not replace compiling. It catches the classes of error that a text
edit can introduce -- unbalanced environments, undefined references, missing
citation keys, missing figure files, unbalanced math mode, malformed tabular
rows -- so that the compile the authors eventually run is unlikely to fail on
anything introduced here.

Exit status is non-zero if any check fails, so it can be wired into CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def strip_comments(text: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.split("\n"))


def brace_group(text: str, index: int) -> tuple[str | None, int]:
    depth = 0
    position = index
    while position < len(text):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[index + 1 : position], position + 1
        position += 1
    return None, len(text)


def check_environments(body: str, problems: list[str]) -> None:
    """Environments must nest, not merely balance in count."""
    stack: list[tuple[str, int]] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]*)\}", body):
        kind, name = match.group(1), match.group(2)
        line = body.count("\n", 0, match.start()) + 1
        if kind == "begin":
            stack.append((name, line))
        else:
            if not stack:
                problems.append(f"line {line}: \\end{{{name}}} with nothing open")
            elif stack[-1][0] != name:
                opened, opened_line = stack[-1]
                problems.append(
                    f"line {line}: \\end{{{name}}} closes \\begin{{{opened}}} "
                    f"opened at line {opened_line}"
                )
                stack.pop()
            else:
                stack.pop()
    for name, line in stack:
        problems.append(f"line {line}: \\begin{{{name}}} never closed")


def check_references(body: str, problems: list[str]) -> None:
    labels = set(re.findall(r"\\label\{([^}]*)\}", body))
    refs = set(re.findall(r"\\(?:ref|eqref|autoref|cref)\{([^}]*)\}", body))
    for name in sorted(refs - labels):
        problems.append(f"\\ref to undefined label: {name}")
    duplicates = [
        name
        for name in labels
        if len(re.findall(r"\\label\{" + re.escape(name) + r"\}", body)) > 1
    ]
    for name in sorted(duplicates):
        problems.append(f"duplicate \\label: {name}")


def check_citations(body: str, bib: Path, problems: list[str]) -> None:
    if not bib.exists():
        problems.append(f"bibliography not found: {bib}")
        return
    keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib.read_text(errors="ignore")))
    cited: set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\{([^}]*)\}", body):
        cited.update(part.strip() for part in match.group(1).split(","))
    for key in sorted(cited - keys):
        if key:
            problems.append(f"\\cite to key absent from the .bib: {key}")


def check_graphics(body: str, root: Path, problems: list[str]) -> None:
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body):
        name = match.group(1)
        candidates = [root / name] + [
            root / f"{name}{suffix}" for suffix in (".pdf", ".png", ".jpg", ".eps")
        ]
        if not any(path.exists() for path in candidates):
            problems.append(f"\\includegraphics file not found: {name}")


def check_math(body: str, problems: list[str]) -> None:
    """Inline math must balance within each paragraph."""
    for number, block in enumerate(re.split(r"\n\s*\n", body), start=1):
        without_display = block.replace("$$", "")
        singles = len(re.findall(r"(?<!\\)\$", without_display))
        if singles % 2:
            snippet = " ".join(block.split())[:60]
            problems.append(f"paragraph {number}: odd number of $ ... {snippet!r}")


def check_tabulars(body: str, problems: list[str]) -> None:
    for match in re.finditer(r"\\begin\{tabular\}", body):
        index = match.end()
        if index < len(body) and body[index] == "[":
            index = body.index("]", index) + 1
        spec, after = brace_group(body, index)
        if spec is None:
            continue
        end = body.find("\\end{tabular}", after)
        content = body[after:end]
        stripped = re.sub(r"@\{[^}]*\}", "", spec)
        stripped = re.sub(r"[pmb]\{[^}]*\}", "p", stripped)
        columns = len(re.findall(r"[lcrp]", stripped))
        for row in content.split("\\\\"):
            row = row.strip()
            if not row or re.match(r"^\\(top|mid|bottom)rule|^\\hline|^\\cmidrule", row):
                continue
            expanded = re.sub(
                r"\\multicolumn\{(\d+)\}", lambda m: "&" * (int(m.group(1)) - 1), row
            )
            cells = expanded.count("&") + 1
            if cells != columns:
                problems.append(
                    f"tabular({spec.strip()}): expected {columns} cells, got {cells} "
                    f"in {row[:50]!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", default=None)
    parser.add_argument("--bib", default="latex/ref.bib")
    arguments = parser.parse_args()
    sources = arguments.sources or ["latex/main.tex", "latex/response_to_reviewers.tex"]

    failed = False
    for source in sources:
        path = Path(source)
        if not path.exists():
            print(f"{source}: MISSING")
            failed = True
            continue
        body = strip_comments(path.read_text())
        problems: list[str] = []
        check_environments(body, problems)
        check_references(body, problems)
        check_graphics(body, path.parent, problems)
        check_math(body, problems)
        check_tabulars(body, problems)
        if "\\bibliography{" in body or "\\cite" in body:
            check_citations(body, Path(arguments.bib), problems)

        if problems:
            failed = True
            print(f"{source}: {len(problems)} problem(s)")
            for problem in problems:
                print(f"  - {problem}")
        else:
            labels = len(re.findall(r"\\label\{", body))
            refs = len(re.findall(r"\\(?:ref|eqref|autoref)\{", body))
            cites = len(re.findall(r"\\cite", body))
            print(
                f"{source}: OK ({labels} labels, {refs} refs, {cites} citations, "
                "environments nested, math balanced, tables well-formed)"
            )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
