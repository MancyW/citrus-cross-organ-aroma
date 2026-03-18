import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd

import numpy as np

import re

try:

    from IPython.display import display

    _HAS_IPY = True

except Exception:

    _HAS_IPY = False

import plotly.graph_objects as go

RUN_MAIN = True

USE_SIGNIF_FILTER = True

Q_THRESHOLD = 0.05

ABS_R_MIN = 0.30

USE_IDF_AXIS = True

IDF_SMOOTH = 1.0

EPS = 1e-9

COV_LEAF = 0.90

COV_AXIS = 0.90

MAX_LEAF_AXIS = 4

MAX_AXIS_PEEL = 6

MIN_WEIGHT = 1.0

TOP_PAIRS_PER_LINK = 10

TOP_PAIRS_IN_HOVER = 5

NODE_HOVER_MAX = None

BASE_DIR = Path(".").resolve()

DATA_DIR = BASE_DIR / "data"

META_DIR = DATA_DIR / "meta"

OUT_DIR  = BASE_DIR / "output"

META_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPOUND_ANNOT_FILE = DATA_DIR / "compound_classification.csv"

PEEL_STAGE_COLORS = ["#9BD7F3", "#D8EEFB", "#FBDDDD", "#F2A1A7"]

LEAF_STAGE_COLORS = ["#DCD7EB", "#FCE6CF", "#D5EAD9", "#7DC69B"]

AXIS_NODE_COLORS = [

    "rgba(160,160,160,0.85)",

    "rgba(242,161,167,0.75)",

    "rgba(155,215,243,0.75)",

    "rgba(213,234,217,0.85)",

    "rgba(252,230,207,0.85)",

]

OUT_ANNOT_CLEAN = META_DIR / "Fig1E_CompoundAnnotation_clean.csv"

OUT_LEAF_ANN    = META_DIR / "Fig1E_LeafModules_annotated.csv"

OUT_PEEL_ANN    = META_DIR / "Fig1E_PeelModules_annotated.csv"

OUT_PAIRS_ANN   = META_DIR / "Fig1E_Pairs_annotated_AIprior_v6C.csv"

OUT_AGG         = META_DIR / "Fig1E_Aggregated_ModuleAxisModule_AIprior_v6C.csv"

OUT_EDGES       = META_DIR / "Fig1E_SankeyEdges_AIprior_v6C.csv"

OUT_NODES       = META_DIR / "Fig1E_SankeyNodes_AIprior_v6C.csv"

OUT_PRIOR_LA    = META_DIR / "Fig1E_Prior_LeafAxis_v6C.csv"

OUT_PRIOR_AP    = META_DIR / "Fig1E_Prior_AxisPeel_v6C.csv"

OUT_LABEL_SUG   = META_DIR / "Fig1E_ModuleLabel_suggestions.csv"

MANUAL_LABEL_FILE = META_DIR / "Fig1E_ModuleLabel_manual.csv"

OUT_MOD_TOPVOC     = META_DIR / "Fig1E_ModuleTopVOCs.csv"

OUT_AXIS_TOPPAIRS  = META_DIR / "Fig1E_AxisTopPairs.csv"

OUT_LINK_TOPPAIRS  = META_DIR / "Fig1E_LinkTopPairs_long.csv"

OUT_MOD_VOC_LONG = META_DIR / "Fig1E_ModuleVOCs_long.csv"

OUT_MOD_VOC_WIDE = META_DIR / "Fig1E_ModuleVOCs_wide.csv"

OUT_HTML = OUT_DIR / "Fig1E_Sankey_AIprior_v6C.html"

OUT_PNG  = OUT_DIR / "Fig1E_Sankey_AIprior_v6C.png"

OUT_SVG  = OUT_DIR / "Fig1E_Sankey_AIprior_v6C.svg"

OUT_PDF  = OUT_DIR / "Fig1E_Sankey_AIprior_v6C.pdf"

def preview_df(df: pd.DataFrame, name: str, n=6):

    print(f"\n--- Preview: {name} (top {n}) ---")

    if _HAS_IPY:

        display(df.head(n))

    else:

        print(df.head(n).to_string(index=False))

def find_best_file(patterns, prefer_contains=None):

    cands = []

    for f in META_DIR.glob("*.csv"):

        nm = f.name

        for pat in patterns:

            if re.search(pat, nm, flags=re.IGNORECASE):

                score = 0

                if prefer_contains:

                    for kw in prefer_contains:

                        if kw.lower() in nm.lower():

                            score += 3

                score += max(0, 60 - len(nm)) / 60

                cands.append((score, f))

                break

    if not cands:

        return None

    cands.sort(key=lambda x: x[0], reverse=True)

    return cands[0][1]

def resolve_inputs():

    leaf_map = find_best_file(

        patterns=[r"leaf.*module", r"leaf.*cluster", r"3f.*module", r"3f.*cluster", r"leaf_cluster"],

        prefer_contains=["leaf","3f","module","cluster"]

    )

    peel_map = find_best_file(

        patterns=[r"peel.*module", r"peel.*cluster", r"2f.*module", r"2f.*cluster", r"peel_cluster"],

        prefer_contains=["peel","2f","module","cluster"]

    )

    pair_file = find_best_file(

        patterns=[r"4c", r"pair", r"edge", r"network", r"corr", r"correlation"],

        prefer_contains=["4c","pair","edge","network"]

    )

    print("\n[File discovery]")

    print("  Leaf module mapping:", leaf_map if leaf_map else "NOT FOUND")

    print("  Peel module mapping:", peel_map if peel_map else "NOT FOUND")

    print("  4C pairs file:", pair_file if pair_file else "NOT FOUND")

    return leaf_map, peel_map, pair_file

def load_compound_annotation(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]

    if not {"Compound","Family"}.issubset(df.columns):

        raise ValueError(f"compound_classification 缺少列：{ {'Compound','Family'} - set(df.columns)}")

    if "Pathway" in df.columns:

        df = df.rename(columns={"Pathway":"Pathway_raw"})

    elif "Subclass" in df.columns:

        df = df.rename(columns={"Subclass":"Pathway_raw"})

    else:

        df["Pathway_raw"] = ""

    out = df[["Compound","Family","Pathway_raw"]].copy()

    for c in out.columns:

        out[c] = out[c].astype(str).str.strip()

    return out

def normalize_module_table(df: pd.DataFrame, organ: str) -> pd.DataFrame:

    cols = {c.lower(): c for c in df.columns}

    voc_col = None

    for key in ["voc","compound","metabolite","feature","name"]:

        if key in cols:

            voc_col = cols[key]

            break

    if voc_col is None:

        voc_col = df.columns[0]

    mod_col = None

    for key in ["leaf_module","peel_module","module","cluster","kmeans","group","class","cluster_id"]:

        if key in cols:

            mod_col = cols[key]

            break

    if mod_col is None:

        if df.shape[1] < 2:

            raise ValueError("模块映射表列数不足，无法识别 module/cluster 列")

        mod_col = df.columns[1]

    out = df[[voc_col, mod_col]].copy()

    out.columns = ["VOC","MODULE_RAW"]

    out["VOC"] = out["VOC"].astype(str).str.strip()

    out["MODULE_RAW"] = out["MODULE_RAW"].astype(str).str.strip()

    if organ == "leaf":

        out["Leaf_Module"] = out["MODULE_RAW"]

        return out[["VOC","Leaf_Module"]]

    else:

        out["Peel_Module"] = out["MODULE_RAW"]

        return out[["VOC","Peel_Module"]]

def load_leaf_modules(file_path: Path, annot: pd.DataFrame) -> pd.DataFrame:

    df = pd.read_csv(file_path)

    df.columns = [c.strip() for c in df.columns]

    base = normalize_module_table(df, "leaf")

    out = base.merge(annot, left_on="VOC", right_on="Compound", how="left")

    out["Leaf_Family"] = out["Family"]

    out["Leaf_Pathway_raw"] = out["Pathway_raw"]

    return out[["VOC","Leaf_Module","Leaf_Family","Leaf_Pathway_raw"]].copy()

def load_peel_modules(file_path: Path, annot: pd.DataFrame) -> pd.DataFrame:

    df = pd.read_csv(file_path)

    df.columns = [c.strip() for c in df.columns]

    base = normalize_module_table(df, "peel")

    out = base.merge(annot, left_on="VOC", right_on="Compound", how="left")

    out["Peel_Family"] = out["Family"]

    out["Peel_Pathway_raw"] = out["Pathway_raw"]

    return out[["VOC","Peel_Module","Peel_Family","Peel_Pathway_raw"]].copy()

def detect_pair_columns(df: pd.DataFrame):

    cols = {c.lower(): c for c in df.columns}

    leaf_col = None

    peel_col = None

    r_col = None

    for k in ["leaf_voc","leaf","leafcompound","leaf_compound","voc_leaf"]:

        if k in cols:

            leaf_col = cols[k]; break

    for k in ["peel_voc","peel","peelcompound","peel_compound","voc_peel"]:

        if k in cols:

            peel_col = cols[k]; break

    for k in ["r","rho","spearman_r","spearman","corr","correlation"]:

        if k in cols:

            r_col = cols[k]; break

    if leaf_col is None:

        for c in df.columns:

            if "leaf" in c.lower():

                leaf_col = c; break

    if peel_col is None:

        for c in df.columns:

            if "peel" in c.lower():

                peel_col = c; break

    return leaf_col, peel_col, r_col

def detect_signif_column(df: pd.DataFrame):

    cols = {c.lower(): c for c in df.columns}

    for k in ["fdr","q","qvalue","q_value","padj","p_adj","p.adjust","fdr_bh"]:

        if k in cols:

            return cols[k], "q"

    for k in ["p","pvalue","p_value"]:

        if k in cols:

            return cols[k], "p"

    return None, None

def assign_axis(row) -> str:

    lf = str(row.get("Leaf_Family","")).lower()

    pf = str(row.get("Peel_Family","")).lower()

    lp = str(row.get("Leaf_Pathway_raw","")).lower()

    pp = str(row.get("Peel_Pathway_raw","")).lower()

    if ("aldehyde" in pp and "aliphatic" in pp) or ("fatty acid-derived" in pf and "aldehyde" in pp):

        return "Fatty-acid–derived aldehyde coordination axis"

    if ("monoterpen" in lf) or ("monoterpen" in pf) or ("sesquiterpen" in lf) or ("sesquiterpen" in pf)
       or ("terpen" in lp) or ("terpen" in pp):

        return "Terpenoid coordination axis"

    if ("benzen" in lp) or ("benzen" in pp) or ("phenyl" in lp) or ("phenyl" in pp)
       or ("benzenoid" in lf) or ("benzenoid" in pf) or ("shikimate" in lp) or ("shikimate" in pp):

        return "Aromatic-related coordination axis"

    return "Mixed / low-coherence axis"

def load_pairs(pair_file: Path, leaf_ann: pd.DataFrame, peel_ann: pd.DataFrame) -> pd.DataFrame:

    df = pd.read_csv(pair_file)

    df.columns = [c.strip() for c in df.columns]

    leaf_col, peel_col, r_col = detect_pair_columns(df)

    if leaf_col is None or peel_col is None or r_col is None:

        raise ValueError(f"pairs 文件列名无法识别：{df.columns.tolist()}")

    sig_col, sig_type = detect_signif_column(df)

    if sig_col is not None:

        df = df.rename(columns={sig_col: "q_or_p"})

    df = df.rename(columns={leaf_col:"Leaf_VOC", peel_col:"Peel_VOC", r_col:"r"})

    df["Leaf_VOC"] = df["Leaf_VOC"].astype(str).str.strip()

    df["Peel_VOC"] = df["Peel_VOC"].astype(str).str.strip()

    df["r"] = pd.to_numeric(df["r"], errors="coerce")

    if "q_or_p" in df.columns:

        df["q_or_p"] = pd.to_numeric(df["q_or_p"], errors="coerce")

    m = df.merge(leaf_ann, left_on="Leaf_VOC", right_on="VOC", how="left").drop(columns=["VOC"])

    m = m.merge(peel_ann, left_on="Peel_VOC", right_on="VOC", how="left").drop(columns=["VOC"])

    m["Axis_group"] = m.apply(assign_axis, axis=1)

    m["abs_r"] = m["r"].abs()

    m = m.dropna(subset=["Leaf_Module","Peel_Module","Axis_group","r","abs_r"]).copy()

    if ABS_R_MIN is not None and ABS_R_MIN > 0:

        m = m[m["abs_r"] >= ABS_R_MIN].copy()

    if USE_SIGNIF_FILTER and ("q_or_p" in m.columns) and m["q_or_p"].notna().any():

        m = m[m["q_or_p"].notna() & (m["q_or_p"] <= Q_THRESHOLD)].copy()

        m["sig_type"] = sig_type if sig_type is not None else "unknown"

    else:

        m["sig_type"] = "none"

    return m.reset_index(drop=True)

def export_module_top_vocs(leaf_ann: pd.DataFrame, peel_ann: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:

    def _top(df, mod_col, fam_col, path_col, organ_tag):

        x = df.copy()

        x[fam_col] = x[fam_col].fillna("Unknown")

        x[path_col] = x[path_col].fillna("Unknown")

        x["VOC"] = x["VOC"].astype(str)

        fam_cnt = x.groupby([mod_col, fam_col]).size().reset_index(name="n_family")

        fam_top = fam_cnt.sort_values([mod_col, "n_family"], ascending=[True, False]).groupby(mod_col).head(3)

        fam_top = fam_top.groupby(mod_col).apply(

            lambda d: "; ".join([f"{r[fam_col]}({int(r['n_family'])})" for _, r in d.iterrows()])

        ).reset_index(name="TopFamilies")

        x = x.sort_values([mod_col, fam_col, path_col, "VOC"], ascending=[True, True, True, True])

        rows = []

        for m, g in x.groupby(mod_col):

            vocs = g["VOC"].dropna().unique().tolist()[:top_n]

            topfam = fam_top.loc[fam_top[mod_col] == m, "TopFamilies"].values[0] if (fam_top[mod_col] == m).any() else ""

            rows.append({

                "Organ": organ_tag,

                "Module": f"{organ_tag[0].upper()}_{m}",

                "nVOCs_inModule": int(g["VOC"].nunique()),

                "TopFamilies": topfam,

                "TopVOCs": "; ".join(vocs),

            })

        return pd.DataFrame(rows)

    leaf_tbl = _top(leaf_ann, "Leaf_Module", "Leaf_Family", "Leaf_Pathway_raw", "Leaf")

    peel_tbl = _top(peel_ann, "Peel_Module", "Peel_Family", "Peel_Pathway_raw", "Peel")

    out = pd.concat([leaf_tbl, peel_tbl], ignore_index=True)

    out.to_csv(OUT_MOD_TOPVOC, index=False)

    return out

def export_axis_top_pairs(pairs_ann: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:

    x = pairs_ann.copy()

    x["abs_r"] = x["r"].abs()

    x = x.sort_values(["Axis_group", "abs_r"], ascending=[True, False])

    keep_cols = [

        "Axis_group", "Leaf_VOC", "Peel_VOC", "r", "abs_r",

        "Leaf_Module", "Peel_Module",

        "Leaf_Family", "Peel_Family",

        "Leaf_Pathway_raw", "Peel_Pathway_raw",

        "sig_type"

    ]

    for c in keep_cols:

        if c not in x.columns:

            x[c] = np.nan

    out = x.groupby("Axis_group").head(top_n)[keep_cols].reset_index(drop=True)

    out.to_csv(OUT_AXIS_TOPPAIRS, index=False)

    return out

def export_module_vocs_all(leaf_ann: pd.DataFrame, peel_ann: pd.DataFrame):

    leaf_long = leaf_ann[["Leaf_Module","VOC","Leaf_Family","Leaf_Pathway_raw"]].copy()

    leaf_long["Organ"] = "Leaf"

    leaf_long = leaf_long.rename(columns={

        "Leaf_Module":"Module",

        "Leaf_Family":"Family",

        "Leaf_Pathway_raw":"Pathway_raw"

    })

    peel_long = peel_ann[["Peel_Module","VOC","Peel_Family","Peel_Pathway_raw"]].copy()

    peel_long["Organ"] = "Peel"

    peel_long = peel_long.rename(columns={

        "Peel_Module":"Module",

        "Peel_Family":"Family",

        "Peel_Pathway_raw":"Pathway_raw"

    })

    out_long = pd.concat([leaf_long, peel_long], ignore_index=True)

    out_long["VOC"] = out_long["VOC"].astype(str).str.strip()

    out_long["Family"] = out_long["Family"].fillna("Unknown")

    out_long["Pathway_raw"] = out_long["Pathway_raw"].fillna("Unknown")

    out_long = out_long.drop_duplicates(subset=["Organ","Module","VOC"])

    out_long = out_long.sort_values(["Organ","Module","Family","Pathway_raw","VOC"]).reset_index(drop=True)

    out_long.to_csv(OUT_MOD_VOC_LONG, index=False)

    out_wide = (

        out_long.groupby(["Organ","Module"])

        .agg(

            nVOC=("VOC","nunique"),

            TopFamilies=("Family", lambda x: "; ".join(pd.Series(x).value_counts().head(3).index.tolist())),

            VOCs=("VOC", lambda x: "; ".join(sorted(pd.unique(x))))

        )

        .reset_index()

        .sort_values(["Organ","Module"])

    )

    out_wide.to_csv(OUT_MOD_VOC_WIDE, index=False)

    return out_long, out_wide

def compute_idf_axis(pairs_ann: pd.DataFrame) -> pd.Series:

    df = pairs_ann.groupby(["Axis_group","Leaf_Module"]).size().reset_index(name="n")

    df_axis = df.groupby("Axis_group")["Leaf_Module"].nunique()

    N = pairs_ann["Leaf_Module"].nunique() + EPS

    idf = np.log((N + IDF_SMOOTH) / (df_axis + IDF_SMOOTH)) + 1.0

    return idf

def aggregate_ai_weight(pairs_ann: pd.DataFrame) -> pd.DataFrame:

    x = pairs_ann.copy()

    x["abs_r"] = x["r"].abs()

    x["is_pos"] = (x["r"] >= 0).astype(int)

    agg = (

        x.groupby(["Leaf_Module","Axis_group","Peel_Module"])

         .agg(

            n_pairs=("r","size"),

            sum_abs_r=("abs_r","sum"),

            mean_abs_r=("abs_r","mean"),

            mean_r=("r","mean"),

            n_pos=("is_pos","sum")

         )

         .reset_index()

    )

    agg["pos_frac"] = agg["n_pos"] / (agg["n_pairs"] + EPS)

    if USE_IDF_AXIS:

        idf = compute_idf_axis(pairs_ann)

        agg["idf_axis"] = agg["Axis_group"].map(idf).fillna(1.0)

    else:

        agg["idf_axis"] = 1.0

    agg["weight_ai"] = agg["sum_abs_r"] * agg["idf_axis"]

    return agg

def prune_by_coverage(agg: pd.DataFrame,

                     cov_leaf=COV_LEAF, cov_axis=COV_AXIS,

                     max_leaf_axis=MAX_LEAF_AXIS, max_axis_peel=MAX_AXIS_PEEL) -> pd.DataFrame:

    la = agg.groupby(["Leaf_Module","Axis_group"])["weight_ai"].sum().reset_index()

    la = la.sort_values(["Leaf_Module","weight_ai"], ascending=[True, False])

    keep_la = []

    for lm, g in la.groupby("Leaf_Module"):

        g = g.copy()

        g["cum"] = g["weight_ai"].cumsum() / (g["weight_ai"].sum() + EPS)

        g_keep = g[(g["cum"] <= cov_leaf) | (g.index == g.index.min())].head(max_leaf_axis)

        keep_la.append(g_keep[["Leaf_Module","Axis_group"]])

    keep_la = pd.concat(keep_la, ignore_index=True)

    agg2 = agg.merge(keep_la, on=["Leaf_Module","Axis_group"], how="inner")

    ap = agg2.groupby(["Axis_group","Peel_Module"])["weight_ai"].sum().reset_index()

    ap = ap.sort_values(["Axis_group","weight_ai"], ascending=[True, False])

    keep_ap = []

    for ax, g in ap.groupby("Axis_group"):

        g = g.copy()

        g["cum"] = g["weight_ai"].cumsum() / (g["weight_ai"].sum() + EPS)

        g_keep = g[(g["cum"] <= cov_axis) | (g.index == g.index.min())].head(max_axis_peel)

        keep_ap.append(g_keep[["Axis_group","Peel_Module"]])

    keep_ap = pd.concat(keep_ap, ignore_index=True)

    agg3 = agg2.merge(keep_ap, on=["Axis_group","Peel_Module"], how="inner")

    agg3 = agg3[agg3["weight_ai"] >= MIN_WEIGHT].copy()

    return agg3.sort_values("weight_ai", ascending=False).reset_index(drop=True)

def export_prior_matrices(agg: pd.DataFrame):

    la = agg.groupby(["Leaf_Module","Axis_group"])["weight_ai"].sum().reset_index()

    la["Leaf"] = "L_" + la["Leaf_Module"].astype(str)

    la["Axis"] = la["Axis_group"]

    la["w"] = la["weight_ai"]

    la["w_norm_leaf"] = la["w"] / (la.groupby("Leaf")["w"].transform("sum") + EPS)

    la_out = la[["Leaf","Axis","w","w_norm_leaf"]].sort_values(["Leaf","w"], ascending=[True, False])

    la_out.to_csv(OUT_PRIOR_LA, index=False)

    ap = agg.groupby(["Axis_group","Peel_Module"])["weight_ai"].sum().reset_index()

    ap["Axis"] = ap["Axis_group"]

    ap["Peel"] = "P_" + ap["Peel_Module"].astype(str)

    ap["w"] = ap["weight_ai"]

    ap["w_norm_axis"] = ap["w"] / (ap.groupby("Axis")["w"].transform("sum") + EPS)

    ap_out = ap[["Axis","Peel","w","w_norm_axis"]].sort_values(["Axis","w"], ascending=[True, False])

    ap_out.to_csv(OUT_PRIOR_AP, index=False)

    print("  Saved prior matrices:")

    print("   -", OUT_PRIOR_LA)

    print("   -", OUT_PRIOR_AP)

def suggest_module_labels(leaf_ann: pd.DataFrame, peel_ann: pd.DataFrame) -> pd.DataFrame:

    def top_cat(df, mod_col, cat_col, prefix):

        tmp = df.copy()

        tmp[cat_col] = tmp[cat_col].fillna("Unknown")

        cnt = tmp.groupby([mod_col, cat_col]).size().reset_index(name="n")

        top = cnt.sort_values([mod_col,"n"], ascending=[True, False]).groupby(mod_col).head(1)

        top["Node"] = prefix + "_" + top[mod_col].astype(str)

        top["Suggested_Label"] = prefix.capitalize() + " " + top[cat_col] + f" module ({prefix.upper()}{top[mod_col].astype(str)})"

        return top[["Node","Suggested_Label","n"]]

    leaf_s = top_cat(leaf_ann, "Leaf_Module", "Leaf_Family", "L")

    peel_s = top_cat(peel_ann, "Peel_Module", "Peel_Family", "P")

    out = pd.concat([leaf_s, peel_s], ignore_index=True)

    out.to_csv(OUT_LABEL_SUG, index=False)

    return out

def apply_manual_labels(nodes_df: pd.DataFrame) -> pd.DataFrame:

    if MANUAL_LABEL_FILE.exists():

        lab = pd.read_csv(MANUAL_LABEL_FILE)

        if {"Node","Label"}.issubset(lab.columns):

            nodes_df = nodes_df.merge(lab[["Node","Label"]], on="Node", how="left", suffixes=("","_new"))

            nodes_df["Label"] = nodes_df["Label_new"].fillna(nodes_df["Label"])

            nodes_df = nodes_df.drop(columns=["Label_new"])

            print("  Applied manual labels:", MANUAL_LABEL_FILE)

    return nodes_df

def export_link_top_pairs_long(pairs_ann: pd.DataFrame, agg_pruned: pd.DataFrame, out_path: Path, top_n=TOP_PAIRS_PER_LINK) -> pd.DataFrame:

    routes = agg_pruned[["Leaf_Module","Axis_group","Peel_Module"]].drop_duplicates()

    x = pairs_ann.copy()

    x["abs_r"] = x["r"].abs()

    out_rows = []

    for _, rt in routes.iterrows():

        lm, ax, pm = rt["Leaf_Module"], rt["Axis_group"], rt["Peel_Module"]

        sub = x[(x["Leaf_Module"]==lm) & (x["Axis_group"]==ax) & (x["Peel_Module"]==pm)].copy()

        if sub.empty:

            continue

        sub = sub.sort_values("abs_r", ascending=False).head(top_n)

        for rank, r in enumerate(sub.itertuples(index=False), start=1):

            out_rows.append({

                "Leaf_Module": lm,

                "Axis_group": ax,

                "Peel_Module": pm,

                "rank_in_route": rank,

                "Leaf_VOC": getattr(r, "Leaf_VOC"),

                "Peel_VOC": getattr(r, "Peel_VOC"),

                "r": float(getattr(r, "r")),

                "abs_r": float(getattr(r, "abs_r")),

                "sig_type": getattr(r, "sig_type") if hasattr(r, "sig_type") else "none"

            })

    out = pd.DataFrame(out_rows)

    out.to_csv(out_path, index=False)

    return out

def build_link_hover_evidence(pairs_ann: pd.DataFrame, agg_pruned: pd.DataFrame, top_pairs=TOP_PAIRS_IN_HOVER):

    routes = agg_pruned[["Leaf_Module","Axis_group","Peel_Module"]].drop_duplicates()

    la_map = routes.groupby(["Leaf_Module","Axis_group"])["Peel_Module"].apply(set).to_dict()

    ap_map = routes.groupby(["Axis_group","Peel_Module"])["Leaf_Module"].apply(set).to_dict()

    x = pairs_ann.copy()

    x["abs_r"] = x["r"].abs()

    la_ev = {}

    for (lm, ax), peel_set in la_map.items():

        sub = x[(x["Leaf_Module"]==lm) & (x["Axis_group"]==ax) & (x["Peel_Module"].isin(peel_set))].copy()

        if sub.empty:

            la_ev[(lm, ax)] = "No pairs after pruning."

            continue

        sub = sub.sort_values("abs_r", ascending=False)

        n = sub.shape[0]

        s = sub["abs_r"].sum()

        pos = (sub["r"]>=0).mean()*100

        tops = sub.head(top_pairs).apply(lambda r: f"{r['Leaf_VOC']}–{r['Peel_VOC']} (r={r['r']:.2f})", axis=1).tolist()

        la_ev[(lm, ax)] = f"n_pairs={n}; Σ|r|={s:.2f}; pos={pos:.1f}%<br>Top pairs:<br>" + "<br>".join(tops)

    ap_ev = {}

    for (ax, pm), leaf_set in ap_map.items():

        sub = x[(x["Axis_group"]==ax) & (x["Peel_Module"]==pm) & (x["Leaf_Module"].isin(leaf_set))].copy()

        if sub.empty:

            ap_ev[(ax, pm)] = "No pairs after pruning."

            continue

        sub = sub.sort_values("abs_r", ascending=False)

        n = sub.shape[0]

        s = sub["abs_r"].sum()

        pos = (sub["r"]>=0).mean()*100

        tops = sub.head(top_pairs).apply(lambda r: f"{r['Leaf_VOC']}–{r['Peel_VOC']} (r={r['r']:.2f})", axis=1).tolist()

        ap_ev[(ax, pm)] = f"n_pairs={n}; Σ|r|={s:.2f}; pos={pos:.1f}%<br>Top pairs:<br>" + "<br>".join(tops)

    return la_ev, ap_ev

def build_sankey_tables(agg: pd.DataFrame, la_ev=None, ap_ev=None):

    la = agg.groupby(["Leaf_Module","Axis_group"]).agg(

        weight=("weight_ai","sum"),

        n_pairs=("n_pairs","sum"),

        sum_abs_r=("sum_abs_r","sum"),

        mean_r=("mean_r","mean"),

        pos_frac=("pos_frac","mean")

    ).reset_index()

    la["Source"] = "L_" + la["Leaf_Module"].astype(str)

    la["Target"] = "A_" + la["Axis_group"].astype(str)

    la["evidence"] = la.apply(lambda r: la_ev.get((r["Leaf_Module"], r["Axis_group"]), "") if la_ev else "", axis=1)

    ap = agg.groupby(["Axis_group","Peel_Module"]).agg(

        weight=("weight_ai","sum"),

        n_pairs=("n_pairs","sum"),

        sum_abs_r=("sum_abs_r","sum"),

        mean_r=("mean_r","mean"),

        pos_frac=("pos_frac","mean")

    ).reset_index()

    ap["Source"] = "A_" + ap["Axis_group"].astype(str)

    ap["Target"] = "P_" + ap["Peel_Module"].astype(str)

    ap["evidence"] = ap.apply(lambda r: ap_ev.get((r["Axis_group"], r["Peel_Module"]), "") if ap_ev else "", axis=1)

    edges = pd.concat(

        [la[["Source","Target","weight","mean_r","n_pairs","sum_abs_r","pos_frac","evidence"]],

         ap[["Source","Target","weight","mean_r","n_pairs","sum_abs_r","pos_frac","evidence"]]],

        ignore_index=True

    )

    node_ids = pd.unique(pd.concat([edges["Source"], edges["Target"]], ignore_index=True))

    nodes = []

    for n in node_ids:

        if n.startswith("L_"):

            nodes.append({"Node":n, "Node_type":"leaf_module", "Label":n.replace("L_","Leaf ")})

        elif n.startswith("P_"):

            nodes.append({"Node":n, "Node_type":"peel_module", "Label":n.replace("P_","Peel ")})

        else:

            nodes.append({"Node":n, "Node_type":"axis", "Label":n.replace("A_","")})

    nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["Node"])

    return edges, nodes_df

def make_node_colors(nodes_df: pd.DataFrame) -> dict:

    colors = {}

    leaf = nodes_df[nodes_df["Node_type"]=="leaf_module"]["Node"].tolist()

    peel = nodes_df[nodes_df["Node_type"]=="peel_module"]["Node"].tolist()

    axis = nodes_df[nodes_df["Node_type"]=="axis"]["Node"].tolist()

    for i, n in enumerate(leaf):

        colors[n] = LEAF_STAGE_COLORS[i % len(LEAF_STAGE_COLORS)]

    for i, n in enumerate(peel):

        colors[n] = PEEL_STAGE_COLORS[i % len(PEEL_STAGE_COLORS)]

    for i, n in enumerate(axis):

        colors[n] = AXIS_NODE_COLORS[(i+1) % len(AXIS_NODE_COLORS)]

    return colors

def enrich_node_labels(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,

                       leaf_ann: pd.DataFrame, peel_ann: pd.DataFrame) -> pd.DataFrame:

    leaf_n = leaf_ann.groupby("Leaf_Module")["VOC"].nunique().to_dict()

    peel_n = peel_ann.groupby("Peel_Module")["VOC"].nunique().to_dict()

    node_strength = {}

    all_nodes = pd.unique(pd.concat([edges_df["Source"], edges_df["Target"]], ignore_index=True))

    for n in all_nodes:

        s = edges_df.loc[edges_df["Source"]==n, "sum_abs_r"].sum() + edges_df.loc[edges_df["Target"]==n, "sum_abs_r"].sum()

        node_strength[n] = float(s)

    labels = []

    for _, r in nodes_df.iterrows():

        node = r["Node"]

        lab = r["Label"]

        strength = node_strength.get(node, 0.0)

        if node.startswith("L_"):

            mid = node.replace("L_","")

            nVOC = leaf_n.get(mid, np.nan)

            labels.append(f"{lab}<br>(nVOC={nVOC}; Σ|r|={strength:.1f})")

        elif node.startswith("P_"):

            mid = node.replace("P_","")

            nVOC = peel_n.get(mid, np.nan)

            labels.append(f"{lab}<br>(nVOC={nVOC}; Σ|r|={strength:.1f})")

        else:

            labels.append(f"{lab}<br>(Σ|r|={strength:.1f})")

    nodes_df["Label"] = labels

    return nodes_df

def _to_rgba_with_alpha(c: str, alpha: float = 0.40) -> str:

    if c is None:

        return f"rgba(160,160,160,{alpha})"

    c = str(c).strip()

    if re.fullmatch(r"#([0-9a-fA-F]{6})", c):

        r = int(c[1:3], 16); g = int(c[3:5], 16); b = int(c[5:7], 16)

        return f"rgba({r},{g},{b},{alpha})"

    m = re.fullmatch(r"rgb\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", c)

    if m:

        r, g, b = m.group(1), m.group(2), m.group(3)

        return f"rgba({r},{g},{b},{alpha})"

    m = re.fullmatch(r"rgba\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", c)

    if m:

        r, g, b = m.group(1), m.group(2), m.group(3)

        return f"rgba({r},{g},{b},{alpha})"

    return f"rgba(160,160,160,{alpha})"

def plot_sankey(edges_df: pd.DataFrame, nodes_df: pd.DataFrame):

    node_list = nodes_df["Node"].tolist()

    idx = {n:i for i,n in enumerate(node_list)}

    src = [idx[s] for s in edges_df["Source"]]

    tgt = [idx[t] for t in edges_df["Target"]]

    vals = edges_df["weight"].astype(float).tolist()

    cm = make_node_colors(nodes_df)

    node_colors = [cm[n] for n in node_list]

    link_colors = []

    for s in edges_df["Source"]:

        base = cm.get(s, "#A0A0A0")

        link_colors.append(_to_rgba_with_alpha(base, alpha=0.40))

    custom = edges_df[["n_pairs","sum_abs_r","pos_frac","evidence"]].copy()

    custom["pos_pct"] = (custom["pos_frac"]*100).round(1)

    customdata = custom[["n_pairs","sum_abs_r","pos_pct","evidence"]].values.tolist()

    node_custom = nodes_df.get("HoverVOCs", pd.Series([""]*len(nodes_df))).tolist()

    fig = go.Figure(data=[go.Sankey(

        arrangement="snap",

        node=dict(

            pad=18,

            thickness=18,

            line=dict(color="rgba(70,70,70,0.35)", width=0.6),

            label=nodes_df["Label"].tolist(),

            color=node_colors,

            customdata=node_custom,

            hovertemplate=(

                "%{label}<br><br>"

                "<b>VOCs in module</b><br>"

                "%{customdata}<extra></extra>"

            )

        ),

        link=dict(

            source=src,

            target=tgt,

            value=vals,

            color=link_colors,

            customdata=customdata,

            hovertemplate=(

                "n_pairs=%{customdata[0]}<br>"

                "Σ|r|=%{customdata[1]:.2f}<br>"

                "pos=%{customdata[2]}%<br>"

                "%{customdata[3]}<extra></extra>"

            )

        )

    )])

    fig.update_layout(

        title=dict(text="Modular Sankey (data-driven): Leaf modules → coordination axes → Peel modules", x=0.02),

        font=dict(family="Arial", size=14),

        width=1500, height=820,

        margin=dict(l=20, r=20, t=70, b=20)

    )

    return fig

def export_figure(fig: go.Figure):

    fig.write_html(str(OUT_HTML))

    try:

        fig.write_image(str(OUT_PNG), scale=4)

    except Exception as e:

        print("\n⚠ PNG 导出失败：", e)

    try:

        fig.write_image(str(OUT_SVG))

        fig.write_image(str(OUT_PDF))

    except Exception as e:

        print("\n⚠ SVG/PDF 导出失败：", e)

        print("   建议：pip install -U kaleido 并确保 Chrome 可用")

def main():

    print("▶ Fig1E AI-prior pipeline v6C started.")

    leaf_map, peel_map, pair_file = resolve_inputs()

    if leaf_map is None or peel_map is None or pair_file is None:

        raise FileNotFoundError("无法自动发现 leaf/peel 模块文件或 4C pairs 文件（请确认都在 data/meta/）")

    print("\n[Step 1] Load compound annotations…")

    annot = load_compound_annotation(COMPOUND_ANNOT_FILE)

    annot.to_csv(OUT_ANNOT_CLEAN, index=False)

    preview_df(annot, "Fig1E_CompoundAnnotation_clean")

    print("  Saved:", OUT_ANNOT_CLEAN)

    print("\n[Step 2] Load + annotate leaf modules…")

    leaf_ann = load_leaf_modules(leaf_map, annot)

    leaf_ann.to_csv(OUT_LEAF_ANN, index=False)

    preview_df(leaf_ann, "Fig1E_LeafModules_annotated")

    print("  Saved:", OUT_LEAF_ANN)

    print("\n[Step 3] Load + annotate peel modules…")

    peel_ann = load_peel_modules(peel_map, annot)

    peel_ann.to_csv(OUT_PEEL_ANN, index=False)

    preview_df(peel_ann, "Fig1E_PeelModules_annotated")

    print("  Saved:", OUT_PEEL_ANN)

    print("\n[Step 3.5] Label suggestions (optional)…")

    sug = suggest_module_labels(leaf_ann, peel_ann)

    preview_df(sug, "Fig1E_ModuleLabel_suggestions", n=10)

    print("  Saved:", OUT_LABEL_SUG)

    print(f"  Optional override: {MANUAL_LABEL_FILE} (Node,Label)")

    print("\n[Step 4] Load + annotate 4C pairs…")

    pairs_ann = load_pairs(pair_file, leaf_ann, peel_ann)

    pairs_ann.to_csv(OUT_PAIRS_ANN, index=False)

    preview_df(pairs_ann, "Fig1E_Pairs_annotated_AIprior_v6C", n=8)

    print("  Saved:", OUT_PAIRS_ANN, "| retained:", pairs_ann.shape[0])

    print(f"  Filters: ABS_R_MIN={ABS_R_MIN}, USE_SIGNIF_FILTER={USE_SIGNIF_FILTER}, Q_THRESHOLD={Q_THRESHOLD}")

    print("\n[Step 4.5] Export Supplementary evidence (module Top VOCs + axis Top pairs)…")

    mod_top = export_module_top_vocs(leaf_ann, peel_ann, top_n=10)

    preview_df(mod_top, "Fig1E_ModuleTopVOCs", n=6)

    print("  Saved:", OUT_MOD_TOPVOC)

    axis_top = export_axis_top_pairs(pairs_ann, top_n=15)

    preview_df(axis_top, "Fig1E_AxisTopPairs", n=6)

    print("  Saved:", OUT_AXIS_TOPPAIRS)

    print("\n[Step 4.6] Export ALL VOCs per module (long + wide)…")

    mod_long, mod_wide = export_module_vocs_all(leaf_ann, peel_ann)

    preview_df(mod_wide, "Fig1E_ModuleVOCs_wide", n=6)

    print("  Saved:", OUT_MOD_VOC_LONG)

    print("  Saved:", OUT_MOD_VOC_WIDE)

    node_voc_map = {}

    for m, g in leaf_ann.groupby("Leaf_Module"):

        vocs = sorted(pd.unique(g["VOC"].dropna().astype(str)))

        if NODE_HOVER_MAX is not None:

            vocs = vocs[:int(NODE_HOVER_MAX)]

        node_voc_map[f"L_{m}"] = "<br>".join(vocs)

    for m, g in peel_ann.groupby("Peel_Module"):

        vocs = sorted(pd.unique(g["VOC"].dropna().astype(str)))

        if NODE_HOVER_MAX is not None:

            vocs = vocs[:int(NODE_HOVER_MAX)]

        node_voc_map[f"P_{m}"] = "<br>".join(vocs)

    print("\n[Step 5] Aggregate + prune (data-driven)…")

    agg = aggregate_ai_weight(pairs_ann)

    agg = prune_by_coverage(agg)

    agg.to_csv(OUT_AGG, index=False)

    preview_df(agg, "Fig1E_Aggregated_ModuleAxisModule_AIprior_v6C", n=10)

    print("  Saved:", OUT_AGG, "| routes:", agg.shape[0])

    print("\n[Step 5.2] Export AI prior matrices…")

    export_prior_matrices(agg)

    print("\n[Step 5.3] Export per-route evidence (top pairs per pruned route)…")

    link_pairs = export_link_top_pairs_long(pairs_ann, agg, out_path=OUT_LINK_TOPPAIRS, top_n=TOP_PAIRS_PER_LINK)

    preview_df(link_pairs, "Fig1E_LinkTopPairs_long", n=10)

    print("  Saved:", OUT_LINK_TOPPAIRS)

    print("\n[Step 6] Build Sankey edges/nodes + hover evidence…")

    la_ev, ap_ev = build_link_hover_evidence(pairs_ann, agg, top_pairs=TOP_PAIRS_IN_HOVER)

    edges_df, nodes_df = build_sankey_tables(agg, la_ev=la_ev, ap_ev=ap_ev)

    nodes_df["HoverVOCs"] = nodes_df["Node"].map(node_voc_map).fillna("")

    nodes_df = apply_manual_labels(nodes_df)

    nodes_df = enrich_node_labels(nodes_df, edges_df, leaf_ann, peel_ann)

    edges_df.to_csv(OUT_EDGES, index=False)

    nodes_df.to_csv(OUT_NODES, index=False)

    preview_df(edges_df.sort_values("weight", ascending=False), "Fig1E_SankeyEdges_AIprior_v6C", n=10)

    preview_df(nodes_df.sort_values(["Node_type","Node"]), "Fig1E_SankeyNodes_AIprior_v6C", n=20)

    print("  Saved:", OUT_EDGES)

    print("  Saved:", OUT_NODES)

    print("\n[Step 7] Plot + export…")

    fig = plot_sankey(edges_df, nodes_df)

    try:

        fig.show()

        print("  Preview shown in notebook.")

    except Exception:

        print("  Preview skipped (non-notebook environment).")

    export_figure(fig)

    print("  Saved:", OUT_HTML)

    print("  Saved:", OUT_PNG)

    print("  Saved:", OUT_SVG)

    print("  Saved:", OUT_PDF)

    print("\n▶ Fig1E AI-prior pipeline v6C finished successfully.")

    print("\nKey outputs:")

    print(" -", OUT_HTML)

    print(" -", OUT_AGG)

    print(" -", OUT_MOD_VOC_LONG)

    print(" -", OUT_MOD_VOC_WIDE)

    print(" -", OUT_LINK_TOPPAIRS)

    print(" -", OUT_AXIS_TOPPAIRS)

    print(" -", OUT_MOD_TOPVOC)

if RUN_MAIN:

    main()
