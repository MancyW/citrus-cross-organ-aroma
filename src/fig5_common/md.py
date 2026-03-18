from __future__ import annotations

from typing import Iterable, List

import pandas as pd

def df_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:

    if df is None:

        return ""

    if len(df) > max_rows:

        df = df.head(max_rows).copy()

    cols = list(df.columns)

    rows = [[("" if pd.isna(v) else str(v)) for v in df.iloc[i].tolist()] for i in range(len(df))]

    widths = [len(str(c)) for c in cols]

    for r in rows:

        for j, v in enumerate(r):

            widths[j] = max(widths[j], len(v))

    def fmt_row(items):

        return "| " + " | ".join(str(items[j]).ljust(widths[j]) for j in range(len(cols))) + " |"

    header = fmt_row(cols)

    sep = "| " + " | ".join("-"*widths[j] for j in range(len(cols))) + " |"

    body = "\n".join(fmt_row(r) for r in rows)

    return "\n".join([header, sep, body]) if body else "\n".join([header, sep])

def write_md(path, text: str):

    from pathlib import Path

    Path(path).write_text(text, encoding="utf-8")
