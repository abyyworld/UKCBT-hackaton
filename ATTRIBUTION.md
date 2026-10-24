# What we did not write ourselves

Everything in `src/`, `run.py`, `scripts/` and `report/` was written by team Batman and Robin during the hackathon
on 24 October 2026, with help from the AI coding assistants listed below. This file lists everything else.

## Libraries

Installed from `requirements.txt`. None of them are shipped in the ZIP.

| Library | Version | Licence | Used for |
|---|---|---|---|
| pandas | 3.1.0 | BSD 3-Clause | Tables, joins, time windows |
| numpy | 2.5.4 | BSD 3-Clause | Arrays and maths |
| pyarrow | 26.0.0 | Apache 2.0 | Reading the Parquet files |
| scikit-learn | 1.9.1 | BSD 3-Clause | Grouped folds, average precision |
| lightgbm | 4.7.0 | MIT | Gradient-boosted tree model |
| networkx | 3.7 | BSD 3-Clause | Payment graph features |
| scipy | 1.18.1 | BSD 3-Clause | Statistics helpers |

## Organiser material

- **The data** (training, validation and test packages) comes from the Decode & Detect organisers at
  <https://hackathon.ukcbt.org/>. It is synthetic and fictional. We do not ship it; see `SUBMISSION.md` for where to
  put it.
- **`payments.py` and `starter_notebook.ipynb`** are organiser files. We read them to learn the data format.
  None of our code is copied from them.
- **Training decks** in `training/`: the XRPL Commons deck is CC BY-SA and credited to XRPL Commons.

No outside data, labels, pre-trained models or paid services were used.

## AI assistance

The rules require us to disclose every AI tool and model. Prompts are summarised in the report (`report/`).

| Who | Session | Tool | Model | What it did |
|---|---|---|---|---|
| Aby | Main session | Claude Code (Anthropic), cloud | Claude (Anthropic) | Pipeline in `src/` (loading, features, cross-validation, prediction), combining results, uploads, `run.py` |
| Aby | Second session | Claude Code (Anthropic), cloud | Claude (Anthropic) | Stage 2 collective scoring (`src/stage2.py`) |
| Aby | Hunter 1 | Claude Code (Anthropic), cloud | Claude (Anthropic) | Hard relay case features (`src/features/hardrelay.py`) |
| Aby | Hunter 2 | Claude Code (Anthropic), cloud | Claude (Anthropic) | Hub traffic features (`src/features/hubs.py`) |
| Aby | Alternative method session | Claude Code (Anthropic), cloud | Claude (Anthropic) | Whole-scheme model for blending (`scripts/scheme_model.py`) |
| Partner | Main session | Claude Code (Anthropic), cloud | Claude (Anthropic) | Data exploration, experiments in `explore/` (relay hubs, medium cases, squares, group classifier, pseudo-labelling coordination, model settings and blends, train vs test shift), reproducibility checks, `LICENSE`, `TEAM.md`, this file, `SUBMISSION.md`, slides and examples |
| Partner | Helper 1 | Claude Code (Anthropic), cloud | Claude (Anthropic) | Motif shape features (`src/features/shape.py`), medium slow motifs |
| Partner | Helper 2 | Claude Code (Anthropic), cloud | Claude (Anthropic) | Pseudo-labelling check (`explore/pseudo.py`), medium wallet reuse |
| Partner | Helper 3 | Claude Code (Anthropic), cloud | Claude (Anthropic) | Medium value matching |

We reviewed and ran all AI-written code ourselves and can explain how it works. Out-of-fold scores were measured
by running the code, not reported by the AI.
