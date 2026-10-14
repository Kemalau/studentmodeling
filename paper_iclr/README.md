# Unified Generative Student Modeling Paper

This directory contains the ICLR-format manuscript for the unified generative
student modeling project.

## Structure

- `main.tex`: paper entry point and draft abstract.
- `sections/introduction.tex`: complete first-pass introduction.
- `sections/related_work.tex`: complete first-pass related work.
- `sections/method.tex`: recurrent causally grounded state method.
- `sections/experiments.tex`: experiment scaffold.
- `sections/conclusion.tex`: conclusion scaffold.
- `references.bib`: references used by the current draft.
- `notes/evidence_status.md`: claims that are ready, preliminary, or unsafe.
- `notes/method_implementation_plan.md`: engineering plan for the new method.

The repository currently vendors the official ICLR 2026 style because the
official ICLR repository does not yet contain an ICLR 2027 template. Replace
the style and bibliography files when the ICLR 2027 author kit is released.

## Build

```bash
latexmk -pdf main.tex
```

On this server, use the installed Tectonic binary:

```bash
tectonic --keep-logs main.tex
```

If only a conventional TeX distribution is available:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
