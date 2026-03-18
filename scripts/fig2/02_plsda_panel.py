import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import StratifiedKFold, GroupKFold

from sklearn.pipeline import Pipeline

from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

from matplotlib.patches import Ellipse, Patch

plt.rcParams["font.family"] = "Arial"

mpl.rcParams["pdf.fonttype"] = 42

mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["svg.fonttype"] = "none"

INPUT_PATH = "data/GCMS_leaf.csv"

OUTPUT_DIR = "output"

FIG_BASENAME = os.path.join(OUTPUT_DIR, "Fig1B_leaf_PLSDA_QC")

STAGE_ORDER = ["S1", "S2", "S3", "S4"]

LEAF_STAGE_COLORS = {

    "S1": "#DCD7EB",

    "S2": "#FCE6CF",

    "S3": "#D5EAD9",

    "S4": "#7DC69B",

}

N_COMPONENTS = 2

CV_N_SPLITS = 7

CV_RANDOM_STATE = 42

ENABLE_QC_FILTER = True

QC_MODE = "BY_CULTIVAR"

QC_GROUP_COL = "Cultivar"

QC_MAD_MULT_ZEROFRAC = 5.0

QC_MAD_MULT_LOGTIC  = 5.0

QC_ABS_TIC_FLOOR = None

USE_LOG1P = True

RUN_GROUP_CV_BY_CULTIVAR = True

FIGSIZE = (3.2, 3.2)

FIG_DPI = 300

SAVE_DPI = 800

def add_confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):

    x = np.asarray(x)

    y = np.asarray(y)

    if x.size <= 2:

        return

    cov = np.cov(x, y)

    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]

    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width = 2 * n_std * np.sqrt(vals[0])

    height = 2 * n_std * np.sqrt(vals[1])

    ellipse = Ellipse(

        xy=(np.mean(x), np.mean(y)),

        width=width,

        height=height,

        angle=theta,

        facecolor=facecolor,

        **kwargs

    )

    ax.add_patch(ellipse)

def _mad(x):

    x = np.asarray(x, dtype=float)

    med = np.median(x)

    return np.median(np.abs(x - med))

def _ensure_numeric_df(X: pd.DataFrame) -> pd.DataFrame:

    X_num = X.apply(pd.to_numeric, errors="coerce")

    if X_num.isna().any().any():

        nan_cols = X_num.columns[X_num.isna().any()].tolist()

        raise ValueError(

            f"Non-numeric/NaN detected after coercion in {len(nan_cols)} columns. "

            f"Example columns: {nan_cols[:10]}"

        )

    return X_num

def compute_qc_flags(df_meta: pd.DataFrame, X_raw: pd.DataFrame) -> pd.DataFrame:

    zero_frac = (X_raw == 0).mean(axis=1).astype(float)

    tic = X_raw.sum(axis=1).astype(float)

    log1p_tic = np.log1p(tic)

    qc = df_meta.copy()

    qc["zero_frac"] = zero_frac.values

    qc["tic"] = tic.values

    qc["log1p_tic"] = log1p_tic.values

    def _flag_one(block):

        zf = block["zero_frac"].values

        lt = block["log1p_tic"].values

        zf_med, zf_mad = np.median(zf), _mad(zf)

        lt_med, lt_mad = np.median(lt), _mad(lt)

        zf_mad = zf_mad if zf_mad > 0 else 1e-12

        lt_mad = lt_mad if lt_mad > 0 else 1e-12

        flag_zf = zf > (zf_med + QC_MAD_MULT_ZEROFRAC * zf_mad)

        flag_tic = lt < (lt_med - QC_MAD_MULT_LOGTIC * lt_mad)

        if QC_ABS_TIC_FLOOR is not None:

            flag_abs = block["tic"].values < float(QC_ABS_TIC_FLOOR)

        else:

            flag_abs = np.zeros_like(flag_zf, dtype=bool)

        out = block.copy()

        out["zf_med"] = zf_med

        out["zf_mad"] = zf_mad

        out["lt_med"] = lt_med

        out["lt_mad"] = lt_mad

        out["flag_zero_frac"] = flag_zf

        out["flag_low_tic"] = flag_tic

        out["flag_abs_tic"] = flag_abs

        out["flag_any"] = flag_zf | flag_tic | flag_abs

        return out

    if QC_MODE.upper() == "BY_CULTIVAR" and QC_GROUP_COL in qc.columns:

        qc_out = []

        for g, block in qc.groupby(QC_GROUP_COL, sort=False):

            qc_out.append(_flag_one(block))

        qc = pd.concat(qc_out, axis=0).sort_index()

    else:

        qc = _flag_one(qc)

    return qc

def cv_predict_plsda(X: pd.DataFrame, Y: np.ndarray, labels: np.ndarray):

    steps = []

    if USE_LOG1P:

        steps.append(("log1p", FunctionTransformer(np.log1p, validate=False)))

    steps += [

        ("scaler", StandardScaler()),

        ("pls", PLSRegression(n_components=N_COMPONENTS)),

    ]

    pipe = Pipeline(steps)

    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

    Y_cv_pred = np.zeros_like(Y, dtype=float)

    for tr, te in skf.split(X, labels):

        pipe.fit(X.iloc[tr], Y[tr])

        Y_cv_pred[te] = pipe.predict(X.iloc[te])

    tss_y = ((Y - Y.mean(axis=0)) ** 2).sum()

    press = ((Y - Y_cv_pred) ** 2).sum()

    q2 = 1 - press / tss_y

    sse_row = ((Y - Y_cv_pred) ** 2).sum(axis=1)

    pred_ranges = []

    for j in range(Y.shape[1]):

        col = Y_cv_pred[:, j]

        pred_ranges.append((float(col.min()), float(col.max()), float(col.mean())))

    return Y_cv_pred, q2, press, tss_y, sse_row, pred_ranges

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🔹 Reading: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"   Rows: {df.shape[0]} | Cols: {df.shape[1]}")

    print(f"   First 10 columns: {df.columns[:10].tolist()}")

    if "Organ" in df.columns:

        df_leaf = df[df["Organ"] == "Leaf"].copy()

    else:

        df_leaf = df.copy()

    print(f"\n🔹 Leaf samples: {df_leaf.shape[0]}")

    meta_cols = ["SampleID", "Cultivar", "Organ", "Stage", "Batch", "Rep"]

    if not all(c in df_leaf.columns for c in meta_cols):

        meta_cols = df_leaf.columns[:6].tolist()

        print(f"   Note: using first 6 columns as metadata: {meta_cols}")

    else:

        print(f"   Metadata columns: {meta_cols}")

    df_leaf = df_leaf[df_leaf["Stage"].isin(STAGE_ORDER)].copy()

    df_leaf["Stage"] = pd.Categorical(df_leaf["Stage"], categories=STAGE_ORDER, ordered=True)

    print(f"   Stage counts: {df_leaf['Stage'].value_counts().sort_index().to_dict()}")

    X = df_leaf.iloc[:, 6:].copy()

    X = _ensure_numeric_df(X)

    n_features_raw = X.shape[1]

    X = X.loc[:, X.std(axis=0) > 0]

    n_features_used = X.shape[1]

    print(f"\n🔹 VOC features: raw={n_features_raw} | used(after zero-variance filter)={n_features_used}")

    df_meta = df_leaf[meta_cols].copy()

    qc_df = compute_qc_flags(df_meta, X)

    qc_path = os.path.join(OUTPUT_DIR, "Leaf_QC_flags.csv")

    qc_df.to_csv(qc_path, index=False)

    print(f"\n🔹 QC table saved: {qc_path}")

    summary_cols = ["zero_frac", "tic"]

    group_cols = []

    if "Cultivar" in qc_df.columns:

        group_cols.append("Cultivar")

    if "Stage" in qc_df.columns:

        group_cols.append("Stage")

    if group_cols:

        qc_sum = qc_df.groupby(group_cols)[summary_cols].agg(["median", "mean"]).reset_index()

        qc_sum_path = os.path.join(OUTPUT_DIR, "Leaf_QC_summary_byCultivarStage.csv")

        qc_sum.to_csv(qc_sum_path, index=False)

        print(f"🔹 QC summary saved: {qc_sum_path}")

    if ENABLE_QC_FILTER:

        flagged = qc_df[qc_df["flag_any"]].copy()

        print(f"   QC-flagged samples: {flagged.shape[0]}")

        if flagged.shape[0] > 0:

            show_cols = [c for c in ["SampleID","Cultivar","Stage","zero_frac","tic","flag_zero_frac","flag_low_tic","flag_abs_tic"] if c in flagged.columns]

            print(flagged[show_cols].to_string(index=False))

        keep_mask = ~qc_df["flag_any"].values

        df_use = df_leaf.loc[keep_mask].copy()

        X_use = X.loc[keep_mask].copy()

        print(f"   Samples kept for PLS-DA/CV: {df_use.shape[0]}")

    else:

        df_use = df_leaf.copy()

        X_use = X.copy()

        print("   QC filtering disabled; using all samples.")

    try:

        enc = OneHotEncoder(sparse_output=False, categories=[STAGE_ORDER])

    except TypeError:

        enc = OneHotEncoder(sparse=False, categories=[STAGE_ORDER])

    Y = enc.fit_transform(df_use[["Stage"]])

    labels = df_use["Stage"].astype(str).values

    X_fit = np.log1p(X_use.values) if USE_LOG1P else X_use.values

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_fit)

    pls = PLSRegression(n_components=N_COMPONENTS)

    T_scores, _ = pls.fit_transform(X_scaled, Y)

    scores_df = pd.DataFrame(T_scores, columns=["LV1", "LV2"])

    scores_df["Stage"] = labels

    centroids = scores_df.groupby("Stage")[["LV1","LV2"]].mean().reindex(STAGE_ORDER)

    Y_pred = pls.predict(X_scaled)

    tss_y_fit = ((Y - Y.mean(axis=0)) ** 2).sum()

    sse_y_fit = ((Y - Y_pred) ** 2).sum()

    R2Y = 1 - sse_y_fit / tss_y_fit

    X_hat = pls.x_scores_ @ pls.x_loadings_.T

    ssx = (X_scaled ** 2).sum()

    resx = ((X_scaled - X_hat) ** 2).sum()

    R2X = 1 - resx / ssx

    print("\n🔹 Fit metrics (for plot):")

    print(f"   R²X(cum) = {R2X:.3f}")

    print(f"   R²Y(cum) = {R2Y:.3f}")

    print("\n🔹 Cross-validation (StratifiedKFold + Pipeline; no leakage)")

    Y_cv_pred, Q2, press, tss_y_cv, sse_row, pred_ranges = cv_predict_plsda(X_use, Y, labels)

    y_pred_cls = np.array(STAGE_ORDER)[np.argmax(Y_cv_pred, axis=1)]

    bal_acc = balanced_accuracy_score(labels, y_pred_cls)

    macro_f1 = f1_score(labels, y_pred_cls, average="macro")

    cm = confusion_matrix(labels, y_pred_cls, labels=STAGE_ORDER)

    print(f"   TSS_y = {tss_y_cv:.3f} | PRESS = {press:.3f} | PRESS/TSS = {press/tss_y_cv:.3f}")

    print(f"   Q²(cum) = {Q2:.3f}  (note: regression-style Q² on one-hot Y can be extreme if predictions overshoot)")

    print(f"   Balanced accuracy = {bal_acc:.3f} | Macro-F1 = {macro_f1:.3f}")

    print("   Confusion matrix (rows=true, cols=pred; S1..S4):")

    print(cm)

    print("   CV prediction ranges per class (min, max, mean):")

    for st, (mn, mx, mu) in zip(STAGE_ORDER, pred_ranges):

        print(f"   - {st}: min={mn:.3f}, max={mx:.3f}, mean={mu:.3f}")

    cv_diag = df_use[meta_cols].copy()

    cv_diag["sse_row"] = sse_row

    cv_diag["pred_stage"] = y_pred_cls

    cv_diag_path = os.path.join(OUTPUT_DIR, "Leaf_PLSDA_CV_diagnostics.csv")

    cv_diag.to_csv(cv_diag_path, index=False)

    print(f"\n🔹 CV diagnostics saved: {cv_diag_path}")

    top = cv_diag.sort_values("sse_row", ascending=False).head(10)

    print("\n🔎 Top-10 SSE samples (most influential in PRESS):")

    print(top.to_string(index=False))

    if RUN_GROUP_CV_BY_CULTIVAR and "Cultivar" in df_use.columns:

        print("\n🔹 Optional Group CV by Cultivar (GroupKFold)")

        groups = df_use["Cultivar"].values

        n_groups = len(pd.unique(groups))

        n_splits = min(5, n_groups) if n_groups >= 2 else 2

        gkf = GroupKFold(n_splits=n_splits)

        steps = []

        if USE_LOG1P:

            steps.append(("log1p", FunctionTransformer(np.log1p, validate=False)))

        steps += [("scaler", StandardScaler()), ("pls", PLSRegression(n_components=N_COMPONENTS))]

        pipe = Pipeline(steps)

        Yg_pred = np.zeros_like(Y, dtype=float)

        for tr, te in gkf.split(X_use, labels, groups=groups):

            pipe.fit(X_use.iloc[tr], Y[tr])

            Yg_pred[te] = pipe.predict(X_use.iloc[te])

        tss_g = ((Y - Y.mean(axis=0)) ** 2).sum()

        press_g = ((Y - Yg_pred) ** 2).sum()

        q2_g = 1 - press_g / tss_g

        y_pred_g = np.array(STAGE_ORDER)[np.argmax(Yg_pred, axis=1)]

        bal_g = balanced_accuracy_score(labels, y_pred_g)

        f1_g = f1_score(labels, y_pred_g, average="macro")

        cm_g = confusion_matrix(labels, y_pred_g, labels=STAGE_ORDER)

        print(f"   Group-CV Q² = {q2_g:.3f}")

        print(f"   Group-CV Balanced accuracy = {bal_g:.3f} | Macro-F1 = {f1_g:.3f}")

        print("   Group-CV confusion matrix (rows=true, cols=pred; S1..S4):")

        print(cm_g)

    print("\n🔹 Plotting Fig1B (Leaf PLS-DA)")

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)

    for st in STAGE_ORDER:

        sub = scores_df[scores_df["Stage"] == st]

        ax.scatter(

            sub["LV1"], sub["LV2"],

            s=15, c=LEAF_STAGE_COLORS[st],

            edgecolor="k", linewidth=0.3, alpha=0.85, label=st

        )

        add_confidence_ellipse(

            sub["LV1"], sub["LV2"], ax,

            n_std=2.0,

            edgecolor=LEAF_STAGE_COLORS[st],

            facecolor=LEAF_STAGE_COLORS[st],

            alpha=0.18, linewidth=0.8

        )

    ax.plot(

        centroids["LV1"], centroids["LV2"],

        "-o", color="k", linewidth=1.0, markersize=4

    )

    ax.axhline(0, color="lightgray", linewidth=0.5, zorder=0)

    ax.axvline(0, color="lightgray", linewidth=0.5, zorder=0)

    ax.set_xlabel("LV1", fontsize=8)

    ax.set_ylabel("LV2", fontsize=8)

    ax.tick_params(axis="both", labelsize=7)

    legend_handles = [Patch(facecolor=LEAF_STAGE_COLORS[s], edgecolor="none", alpha=0.85, label=s) for s in STAGE_ORDER]

    ax.legend(

        handles=legend_handles,

        loc="upper right",

        fontsize=7,

        frameon=True,

        framealpha=1.0,

        borderpad=0.4,

        title="Stage",

        title_fontsize=8

    )

    q2_text = f"{Q2:.2f}" if Q2 >= 0 else "< 0"

    text_str = (

        f"R²X(cum) = {R2X:.2f}\n"

        f"R²Y(cum) = {R2Y:.2f}\n"

        f"Q²(cum)  {q2_text}\n"

        f"BalAcc   = {bal_acc:.2f}"

    )

    ax.text(

        0.03, 0.97, text_str,

        transform=ax.transAxes,

        ha="left", va="top",

        fontsize=6, linespacing=1.35

    )

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig.savefig(FIG_BASENAME + ".png", dpi=SAVE_DPI)

    fig.savefig(FIG_BASENAME + ".tif", dpi=SAVE_DPI)

    fig.savefig(FIG_BASENAME + ".pdf")

    fig.savefig(FIG_BASENAME + ".svg")

    print(f"\n🔹 Saved: {FIG_BASENAME}.(png/tif/pdf/svg)")

    plt.show()

    plt.close()

    print("\n✅ Done.")

if __name__ == "__main__":

    main()
