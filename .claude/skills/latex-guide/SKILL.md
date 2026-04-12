---
name: latex-guide
description: LaTeX and BibTeX troubleshooting guide. Error patterns, fixes, best practices.
---

# /latex-guide -- LaTeX/BibTeX Troubleshooting

> Practical reference for common LaTeX errors, BibTeX workflow, and Beamer specifics.

## When to Use

Consult when encountering LaTeX compilation errors, bibliography issues, or package conflicts. Auto-invocable by writing agents when LaTeX errors arise.

## Common Error Patterns

### Compilation Errors

| Error | Cause | Fix |
|---|---|---|
| `Undefined control sequence` | Unknown command or missing package | Add `\usepackage{package}` or fix typo |
| `Missing $ inserted` | Math symbol outside math mode | Wrap in `$...$` or `\(...\)` |
| `File not found` | Wrong path or missing file | Check `\includegraphics` / `\input` paths |
| `Too many }'s` | Brace mismatch | Count braces; use editor matching |
| `Overfull \hbox` | Content exceeds margins | Rephrase, resize figure, or use `\adjustbox` |
| `Missing \begin{document}` | Preamble error before `\begin{document}` | Check for syntax errors in preamble |
| `Float(s) lost` | Float in restricted context | Move float to top level or use `[H]` placement |

### BibTeX Errors

| Error | Cause | Fix |
|---|---|---|
| `Citation undefined` | Missing bib entry or wrong key | Verify key exists in `.bib` file |
| `Empty bibliography` | Wrong `.bib` path or no `\cite{}` | Check `\bibliography{file}` path |
| `I found no \citation commands` | `\cite{}` not in document | Ensure citations exist before `\bibliography` |

### Compilation Workflow

```bash
# Standard workflow (handles cross-references)
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Or use latexmk (automates the cycle)
latexmk -pdf main.tex
```

Always compile at least twice after adding citations or cross-references.

## Package Conflict Resolution

Common conflicts and fixes:
- `hyperref` should be loaded **last** (with few exceptions like `cleveref`)
- `cleveref` must load **after** `hyperref`
- `biblatex` and `natbib` are mutually exclusive -- choose one
- `inputenc` with `utf8` is default in modern LaTeX; remove if using LuaLaTeX/XeLaTeX
- `geometry` and `fullpage` conflict -- use only `geometry`

## Figure Best Practices

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\columnwidth]{figures/result.pdf}
  \caption{Description of what the figure shows.}
  \label{fig:result}
\end{figure}
```

- Use PDF/EPS for vector graphics, PNG for rasters
- Use relative paths from project root
- Always include `\label` after `\caption`
- Reference with `\ref{fig:result}` or `\autoref{fig:result}`

## Cross-Referencing

- `\label{}` must come **after** `\caption{}` in floats
- Use `\ref{}`, `\eqref{}`, `\autoref{}`, or `\cref{}` (cleveref)
- Run twice to resolve forward references
- Check log for `LaTeX Warning: Reference ... undefined`

## Beamer Specifics

### Frame Structure
```latex
\begin{frame}{Frame Title}
  Content here. Keep it minimal.
\end{frame}
```

### Common Beamer Issues

| Issue | Cause | Fix |
|---|---|---|
| `Overfull` in frames | Too much content | Reduce content, split frame, or use `[shrink]` (last resort) |
| Fragile content errors | Verbatim/listings in frame | Use `\begin{frame}[fragile]` |
| `Token not allowed` | Fragile command in frame title | Move to `\frametitle{}` inside frame body |
| Blank frame after title | Missing content or extra `\end{frame}` | Check brace/environment matching |
| Animations not working | Wrong overlay spec | Use `\pause`, `\only<N>{...}`, `\onslide<N>{...}`, `\uncover<N>{...}` |
| Theme not applying | Load order | Set theme before `\begin{document}`: `\usetheme{...}`, `\usecolortheme{...}` |

### Metropolis Theme

Metropolis (`\usetheme{metropolis}`) is a common modern Beamer theme.

```latex
\usetheme{metropolis}
% Optional customization
\metroset{
  sectionpage=progressbar,  % or: none, simple, progressbar
  numbering=fraction,       % or: none, counter, fraction
  progressbar=frametitle,   % or: none, head, frametitle, foot
}
```

Known issues:
- Requires `fontspec` + XeLaTeX/LuaLaTeX for full font support. With `pdflatex`, use `\usetheme[numbering=fraction]{metropolis}` and avoid `\setsansfont`.
- If using `pdflatex`, add `\usepackage[T1]{fontenc}` to avoid font warnings.
- Metropolis uses `\alert{}` for emphasis (colored text), not bold.

### Speaker Notes

```latex
% In preamble
\usepackage{pgfpages}
\setbeameroption{show notes on second screen=right}

% In frames
\begin{frame}{Title}
  Visible content.
  \note{Speaker-only notes for this frame.}
\end{frame}
```

Compile to PDF normally -- notes appear on the right half. Use a PDF viewer that supports dual-screen (e.g., pdfpc, Présentation).

For handouts without notes: `\setbeameroption{hide notes}` or compile with `\documentclass[handout]{beamer}`.

### Animation and Overlays

```latex
\pause                    % reveal next items one by one
\only<2>{visible on 2}    % takes no space when hidden
\onslide<2->{from 2 on}   % takes space always, visible from slide 2
\uncover<3>{visible on 3}  % takes space always
\alert<2>{red on slide 2}  % highlight on specific overlay
```

For itemize: `\begin{itemize}[<+->]` makes each item appear incrementally.

### TikZ in Frames

```latex
\begin{frame}{Diagram}
  \begin{tikzpicture}
    % TikZ code here
  \end{tikzpicture}
\end{frame}
```

- Use `\begin{frame}[fragile]` only if TikZ code contains verbatim-like constructs.
- For overlays with TikZ: `\only<2>{\node at (0,0) {text};}` or use `\visible<2>`.

### Beamer + BibTeX
```latex
% In preamble
\usepackage[style=authoryear]{biblatex}
\addbibresource{refs.bib}

% At end, before \end{document}
\begin{frame}[allowframebreaks]{References}
  \printbibliography
\end{frame}
```

## Protocol

- When a LaTeX error occurs, check this guide first before searching externally.
- Fix errors immediately after compilation -- don't accumulate.
- Compile frequently: after each section, not just at the end.
- Check the `.log` file for warnings, not just errors.
