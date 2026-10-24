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
  TODO: say here whether any of our code copies from them.
- **Training decks** in `training/`: the XRPL Commons deck is CC BY-SA and credited to XRPL Commons.

No outside data, labels, pre-trained models or paid services were used.

## AI assistance

The rules require us to disclose every AI tool and model. Prompts are summarised in the report (`report/`).

| Who | Tool | Model | What it did |
|---|---|---|---|
| Aby | TODO: tool | TODO: model | Wrote most of the pipeline in `src/` (loading, features, cross-validation, prediction) from our instructions |
| Partner | Claude Code (Anthropic), cloud sessions: one main session plus two parallel helper sessions | TODO: model name | Data exploration of the labelled training set, feature experiments in `explore/` (relay hubs, medium cases, motif shape, pseudo-labelling, model settings and blends, train vs test shift), `LICENSE`, this file, `SUBMISSION.md`, the slides, and setup help |

We reviewed and ran all AI-written code ourselves and can explain how it works. Out-of-fold scores were measured
by running the code, not reported by the AI.
