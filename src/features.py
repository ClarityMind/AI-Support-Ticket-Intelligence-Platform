"""Stateless creation-time features recovered from the priority notebook."""

import re
from collections.abc import Mapping

import pandas as pd

from .data import validate_columns

URGENCY_TERMS = ("urgent", "asap", "immediately", "critical", "терміново", "срочно", "dringend")
INCIDENT_TERMS = ("crash", "breach", "loss", "blocked", "down", "failure", "outage", "error", "помилка")
NEGATIVE_TERMS = ("problem", "cannot", "failed", "failure", "unresolved", "not working", "issue")
ACTION_TERMS = ("please fix", "need help", "investigate", "restore", "resolve", "recover")
FEATURE_COLUMNS = (
    "urgency_keyword_count", "incident_keyword_count", "negative_word_count",
    "action_request_count", "digit_count", "uppercase_word_ratio", "subject_body_similarity",
)


def text_or_empty(value: object) -> str:
    """Never stringify pandas/NumPy missing values as 'nan' or '<NA>'."""
    return "" if pd.isna(value) else str(value)


def combine_text(frame: pd.DataFrame) -> pd.Series:
    validate_columns(frame, ("subject", "body"))
    return frame["subject"].map(text_or_empty) + " " + frame["body"].map(text_or_empty)


def extract_priority_features(row: Mapping | pd.Series) -> dict[str, int | float]:
    """Preserve notebook substring counts and rounded token-set similarity."""
    raw_subject, raw_body = (text_or_empty(row[key]) for key in ("subject", "body"))
    subject, body = raw_subject.lower(), raw_body.lower()
    combined = f"{subject} {body}"
    words = re.findall(r"\b\w+\b", combined)
    uppercase = re.findall(r"\b[A-ZА-ЯІЇЄ]{2,}\b", f"{raw_subject} {raw_body}")
    subject_words = set(re.findall(r"\b\w+\b", subject))
    body_words = set(re.findall(r"\b\w+\b", body))
    union = subject_words | body_words
    counts = [sum(combined.count(term) for term in terms)
              for terms in (URGENCY_TERMS, INCIDENT_TERMS, NEGATIVE_TERMS, ACTION_TERMS)]
    values = counts + [
        sum(character.isdigit() for character in combined),
        round(len(uppercase) / len(words), 4) if words else 0.0,
        round(len(subject_words & body_words) / len(union), 4) if union else 0.0,
    ]
    return dict(zip(FEATURE_COLUMNS, values, strict=True))


def priority_features(frame: pd.DataFrame) -> pd.DataFrame:
    validate_columns(frame, ("subject", "body"))
    return pd.DataFrame(
        [extract_priority_features(row) for row in frame[["subject", "body"]].to_dict("records")],
        index=frame.index, columns=FEATURE_COLUMNS,
    ).astype(float)
