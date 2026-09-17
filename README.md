# AI Support Ticket Intelligence Platform

Classical ML baselines for **queue/category classification** and **priority
classification** from ticket text available when a support request is created.
The package provides reproducible training, evaluation, saved inference
pipelines, and synthetic tests. Performance is exploratory, not production-grade.

## Architecture

```text
Ticket (subject, body)
  |
  +--> combined text --> TF-IDF --> Logistic Regression --> queue/category
  |
  +--> seven structured features --> selected tree/ensemble --> priority
```

`category` is the CLI/artifact name for predicting the dataset's `queue` target.
Priority selection compares Decision Tree, Random Forest, and Gradient Boosting.

## Setup and training

Python 3.12+ and uv are required for the locked workflow. Run from the repository root:

```bash
uv sync --locked
uv run --locked support-ticket-ml train-category --data dataset/dataset-tickets-multi-lang-4-20k.csv
uv run --locked support-ticket-ml train-priority --data dataset/dataset-tickets-multi-lang-4-20k.csv
uv run --locked python -m unittest discover -s tests -v
```

Place the UTF-8 CSV at the path above, or supply another path with `--data`.
Raw datasets are excluded from Git. **Dataset provenance and license remain
unverified**; no verified upstream source is recorded. See
[dataset setup](dataset/README.md) for details.

| Task | Required CSV columns |
| --- | --- |
| Queue/category | `language`, `subject`, `body`, `queue` |
| Priority | `language`, `subject`, `body`, `priority` |

Only exact `language == "en"` rows are used. Queue/category training drops missing
bodies and treats missing subjects as empty. Priority training treats missing
subjects and bodies as empty. Missing/blank targets fail validation. Extra columns
are ignored by the models.

Both commands accept `--artifacts PATH` and `--test-size FRACTION`. Priority also
accepts `--cv-folds N` and `--n-jobs N` (default 1). Defaults live in `config.py`.
Repeated runs overwrite the same task's artifacts; use separate output directories
to retain experiments.

Alternatively, install into an active virtual environment with
`python -m pip install -e .`, then use `support-ticket-ml` or
`python -m ai_support_ticket_intelligence_platform.cli`. The pip installation
resolves declared dependencies rather than the exact `uv.lock` versions.
Runtime dependencies are pandas, scikit-learn, and joblib; tests use unittest.

## Methodology and leakage safeguards

Both tasks use an 80/20 stratified train/test split and seed 42. All supported
estimators and shuffled CV also use seed 42.

**Queue/category:** a single pipeline joins subject and body, fits default
unigram TF-IDF on training rows only, and fits Logistic Regression with
`max_iter=5000`. This is a fixed baseline without hyperparameter search. The
category notebook also records a Multinomial Naive Bayes comparison.

**Priority:** five shuffled stratified CV folds on training data compare 18
configurations: six Decision Trees, eight regularized Random Forests, and four
Gradient Boosting models. The highest full-precision mean CV accuracy selects
the configuration; it is refit on all training rows before test evaluation.
The explicit search in `priority.parameter_grid()` retains regularization
motivated by the notebook's overfitting Random Forest baseline.

Priority features are deterministic, stateless calculations from subject/body:

| Feature | Definition |
| --- | --- |
| `urgency_keyword_count` | Substring occurrences of urgency terms |
| `incident_keyword_count` | Substring occurrences of incident terms |
| `negative_word_count` | Substring occurrences of negative terms/phrases |
| `action_request_count` | Substring occurrences of action requests |
| `digit_count` | Digit characters |
| `uppercase_word_ratio` | Uppercase words of at least two letters / all word tokens |
| `subject_body_similarity` | Jaccard overlap of lowercase subject/body token sets |

The notebook's keyword lists, multilingual terms, and four-decimal ratio rounding
are preserved. Substring matches can occur inside longer words. Missing values
never become literal `nan` text. These features learn no corpus statistics, so
computing them before CV does not fit preprocessing on validation rows.

Only `subject` and `body` reach either model: no `answer`, tags, ticket type,
queue/priority labels, or future/outcome fields are prediction features. TF-IDF
never fits on test data, and priority selection never uses test scores. Both saved
pipelines include preprocessing. Stratification requires sufficient rows in each
class; default priority CV requires at least five training rows per class.

## Verified evaluation

Local runs used the original 20,000-row CSV, seed 42, Python 3.12.0, pandas 3.0.5,
and scikit-learn 1.9.0 from the lockfile. Queue/category used 11,922 eligible rows;
priority used 11,923. Both test sets contain 2,385 rows. Input CSV SHA-256:
`9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635`.

| Metric | Queue/category: Logistic Regression | Priority: selected Decision Tree |
| --- | ---: | ---: |
| Train accuracy | 0.5815 | 0.4681 |
| Training CV accuracy mean | n/a (fixed baseline) | 0.4636 |
| Training CV accuracy std | n/a | 0.0075 |
| Test accuracy | 0.4361 | 0.4512 |
| Majority baseline test accuracy | 0.2864 | 0.4155 |
| Test macro F1 | 0.3139 | 0.3243 |
| Test weighted F1 | 0.4060 | 0.3904 |

Priority CV selected `max_depth=5`, `min_samples_leaf=8`. Logistic Regression
converged in 110 iterations. Priority improves only modestly over the majority
baseline and has zero recall for the low-priority class in this run.

Accuracy can hide weak minority-class performance. Macro F1 weights classes
equally; weighted F1 weights by support. Both commands print/save per-class
precision, recall, F1, support, and confusion matrices alongside the summary
metrics. Matrix rows are actual labels and columns are predictions, ordered by
the JSON `labels` array. Priority reports every candidate's CV mean/std, selected
hyperparameters, and impurity-based feature importances.

**Impurity-based importance is relative model importance, not an effect size.
It does not show causal direction or establish causality.**

## Artifacts and inference

Generated outputs are excluded from Git; `artifacts/.gitkeep` retains the directory:

- `artifacts/category_pipeline.joblib`
- `artifacts/category_metrics.json`
- `artifacts/priority_model.joblib`
- `artifacts/priority_metrics.json`

JSON reports record dependency versions, split settings, and the CSV checksum.
Models remain fitted on training rows only. With the package installed in the
same dependency environment, load trusted artifacts and predict from text:

```python
import joblib
import pandas as pd

tickets = pd.DataFrame([{"subject": "Service unavailable", "body": "Please fix the outage."}])
category = joblib.load("artifacts/category_pipeline.joblib")
priority = joblib.load("artifacts/priority_model.joblib")
print(category.predict(tickets))
print(priority.predict(tickets))
```

Inference requires only `subject` and `body`; callers must supply English tickets
for v1. Synthetic tests need no external dataset and cover validation, missing
values, feature schema/semantics, training-only vocabulary and CV inputs,
selection by CV mean, determinism, inference, artifact reloads, and CLI output.

## Repository layout

```text
src/ai_support_ticket_intelligence_platform/
  config.py       shared configuration
  data.py         loading, validation, English filtering, splits
  features.py     text and structured features
  category.py     queue/category pipeline
  priority.py     training-only CV selection
  evaluate.py     metrics and artifact saving
  cli.py          training commands
notebooks/
  01_data_audit.ipynb
  02_category_baseline.ipynb
  03_priority_tree.ipynb
tests/            synthetic unittest suite
dataset/          setup README and ignored local CSVs
artifacts/        ignored generated pipelines and reports
```

## Limitations and next steps

Language labels are noisy; some English-labeled rows contain other languages.
Technical Support, IT Support, and Product Support overlap semantically. Class
imbalance, noisy queue/priority labels, and limited handcrafted features constrain
quality. Labels are not silently corrected.

The preserved notebooks include exploratory test comparisons. The historical
holdout was viewed during development, so it is **not a pristine blind
benchmark**. The package uses training-only CV for selection, but random row
splits do not establish performance on new customers, future tickets, or
near-duplicate groups.

Next steps are to verify dataset provenance/license, audit labels with domain
experts, clarify queue definitions, and evaluate on a fresh untouched dataset.
Check duplicate groups and time/customer-based splits. Compare richer
creation-time features and imbalance-aware objectives through training CV before
freezing the procedure for a new holdout.
