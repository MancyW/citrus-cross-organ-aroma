import pandas as pd

import numpy as np

def load_voc_data(pred_long_file: str):

    df = pd.read_parquet(pred_long_file)

    true_w = df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first").fillna(0.0)

    pred_w = df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first").fillna(0.0).reindex(true_w.index)

    return true_w, pred_w

def calculate_variability(true_w: pd.DataFrame):

    var = true_w.var(axis=0)

    cv = true_w.std(axis=0) / (true_w.mean(axis=0) + 1e-12)

    return var, cv

def get_top_vocs_by_variability(var: pd.Series, cv: pd.Series, top_n: int = 20):

    top_vocs = cv.sort_values(ascending=False).head(top_n).index.tolist()

    return top_vocs

pred_long_file = "results/pred_vectors/run_20260130_132441_S1/pred_vectors_long.parquet"

true_w, pred_w = load_voc_data(pred_long_file)

var, cv = calculate_variability(true_w)

top_vocs = get_top_vocs_by_variability(var, cv, top_n=30)

print(top_vocs)
