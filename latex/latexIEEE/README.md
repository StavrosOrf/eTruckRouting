# IEEE T-ITS version

Revised manuscript prepared for the *IEEE Transactions on Intelligent
Transportation Systems*, ported from the Elsevier `cas-sc` version in `../`.

## Files

| File | What it is |
| --- | --- |
| `main.tex` → `main.pdf` | The paper. **10 pages**, including the reference list. |
| `supplementary.tex` → `supplementary.pdf` | Material moved out of the main paper to fit the page limit. 9 pages. |
| `IEEEtran.cls` | Official IEEEtran class, v1.8b, from CTAN. |
| `IEEEtran.bst`, `IEEEtranN.bst` | Official IEEE BibTeX styles. `IEEEtranN` is the natbib-compatible one the paper uses. |
| `ref.bib` | `../ref.bib` with `url`, `doi`, `abstract`, `keywords`, `issn`, `isbn` and `urldate` fields stripped — IEEE style does not print them and they cost roughly half a page. |
| `*.pdf` (figures) | Copied unchanged from `../`. |

## Building

```
tectonic -X compile main.tex
tectonic -X compile supplementary.tex
```

Any standard toolchain works too (`pdflatex` → `bibtex` → `pdflatex` ×2). The
class and both `.bst` files are vendored, so the folder is self-contained apart
from ordinary LaTeX packages.

## What changed relative to `../main.tex`

**Class and front matter.** `cas-sc` → `\documentclass[journal]{IEEEtran}`.
`\shorttitle`/`\shortauthors`/`\affiliation`/`\credit`/`\cormark` are replaced by
IEEEtran's `\title`, `\author`, `\markboth`; the abstract and keywords move after
`\maketitle` as `abstract` + `IEEEkeywords`; the bibliography uses
`IEEEtranN.bst`. The author block is anonymized, as in the Elsevier source; the
full block sits commented out in the preamble, ready to uncomment.

**Two-column layout fixes.** The Elsevier version is single-column, so several
displays and tables did not fit an IEEE column. The optimization model, the CCCV
profile, the feasible-charge set, the reward, the node update and the action GCN
were re-broken across lines (quantifiers for the constraint block are now stated
in the sentence that follows it), and the three widest tables became `table*`.
The reference list is wrapped in `\footnotesize`, which is what IEEEtran does on
its own before natbib replaces `thebibliography`.

**Fitting 10 pages.** The Elsevier manuscript is 27 pages single-column, which is
about 21 pages in this class. The reduction was made first by relocating whole
blocks verbatim into `supplementary.tex` — both appendices, the literature
taxonomy and the extended related-work discussion, the size-transfer and 10T3S
studies, the component sweep, the PPO recap and algorithm listing, the
experimental-settings table, the evaluation protocol, and four figures — and then
by deleting duplicated sentences and compressing the remaining prose. No result,
number, claim or citation was changed; surviving passages are largely the
original wording.

**Figures kept in the main paper:** the graph-construction figure and the
actor--critic architecture. **Moved to the supplement:** the eVRP/eTFRP overview,
the truck state machine, the charging-curve comparison, the California network,
and the win-rate bars. Each is pointed to from the main text.
