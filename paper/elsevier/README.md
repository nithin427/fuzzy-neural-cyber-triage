# Knowledge-Based Systems submission folder (Elsevier `elsarticle`)

This folder is self-contained. Upload it to Overleaf as a zip and set `main.tex` as the main document, or compile locally:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or: tectonic main.tex      (tested with tectonic 0.15.0; compiles cleanly)
```

| File | Purpose |
|---|---|
| `main.tex` | Manuscript source (elsarticle `preprint,12pt`, numbered references `elsarticle-num`) |
| `refs.bib` | 20 references (15 Q1 IEEE/Elsevier/MDPI journal articles + 5 dataset/method sources), all cited in order of first appearance; 20/20 DOIs checked against Crossref on 2026-10-24 |
| `figures/fig1_decision_bands.pdf`, `figures/fig2_cost_sensitivity.pdf` | Vector figures used by `main.tex` (300-dpi PNG copies alongside) |
| `figures/graphical_abstract.png` | Optional graphical abstract (upload separately if used) |
| `highlights.txt` | 5 highlights, each ≤ 85 characters (upload as a separate file) |
| `manuscript.docx` | Word version generated from the same content with pandoc (Times New Roman 12 pt, 1.5 spacing, Word equations) |
| `fuzzy-neural-cyber-triage-kbs.pdf` | PDF compiled from this folder |

The compiled `preprint,12pt` PDF is 11 pages including references (references start on p. 10; tectonic build). For double-spaced review, change the class options to `review,12pt`; that layout runs to 15 pages (references start on p. 13). The 1.5-spaced Word version is 11 pages (LibreOffice render).
Fig. 1 and Fig. 2 are drawn by `src/make_figures.py` from the result files (called at the end of `src/experiment.py`); the graphical abstract is drawn by `src/experiment.py`. Rerunning either script refreshes the copies in `figures/`.
