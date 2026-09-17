# Local dataset

Place `dataset-tickets-multi-lang-4-20k.csv` in this directory to reproduce the
original notebook experiments. This is the CLI default. Alternatively pass any
UTF-8 CSV path with `--data path/to/tickets.csv`. Paths are relative to the current
working directory; no machine-specific path is required.

From the repository root, for example:

```bash
uv run --locked support-ticket-ml train-category --data dataset/dataset-tickets-multi-lang-4-20k.csv
uv run --locked support-ticket-ml train-priority --data path/to/tickets.csv
```

Required columns are `language`, `subject`, `body`, plus `queue` for category
training or `priority` for priority training. Only exact `language == "en"` rows
are used. Empty CSV cells are read as missing values using pandas defaults.
Category training drops missing bodies and treats missing subjects as empty.
Priority training treats missing subjects and bodies as empty. Missing/blank
target labels fail validation. Extra columns are ignored by the models.

The original local CSV has 20,000 rows; 11,923 are labeled English. Its filename
comes from the existing notebooks. No upstream URL or redistribution license is
recorded in this repository. Dataset provenance and license remain unverified;
obtain an authorized copy from your data provider.
The raw data is intentionally excluded from Git. This directory is retained
instead of introducing a second `data/` directory.
