from __future__ import annotations
from libs import *
from config_evaluate import *
from model import *
import umap
import hdbscan
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

from sklearn.neighbors import NearestNeighbors

from trainning import forward_model_from_batch, build_loader
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from donwstream_model import fit_logistic_regression, LogisticRegression
from sklearn.model_selection import train_test_split
from loss import collate_fn_refalt_ontology

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from mixed_norme_ontology import zero_shot_low_penetrance_from_cfg

# =============================================================================
# Ontology mapping
# =============================================================================

ONTOLOGY_CATEGORY_PATTERNS = {
    "immune response": [
        r"\bimmune\b", r"\bimmun", r"\binflamm", r"\bcytokine\b", r"\bchemokine\b",
        r"\bleukocyte\b", r"\blymphocyte\b", r"\bantigen\b", r"\binterferon\b",
        r"\binnate immune\b", r"\badaptive immune\b", r"\bhost defense\b",
    ],
    "signal transduction": [
        r"\bsignal transduction\b", r"\bsignaling\b", r"\bsignalling\b",
        r"\bkinase cascade\b", r"\bmapk\b", r"\bjak\b", r"\bstat\b", r"\bpi3k\b",
        r"\bakt\b", r"\bmtor\b", r"\bwnt\b", r"\bnotch\b", r"\btgf", r"\bsmad\b",
        r"\bnf[- ]?kappa", r"\bgpcr\b", r"\breceptor signaling\b", r"\breceptor signalling\b",
    ],
    "DNA cycle/cell cycle": [
        r"\bdna repair\b", r"\bdna replication\b", r"\bdna recombination\b",
        r"\bdna damage\b", r"\bcell cycle\b", r"\bmitotic\b", r"\bmitosis\b",
        r"\bmeiosis\b", r"\bcheckpoint\b", r"\bchromosome segregation\b",
        r"\bchromatin segregation\b", r"\bnuclear division\b", r"\bcell division\b",
    ],
    "Metabolic process": [
        r"\bmetabolic process\b", r"\bmetabolism\b", r"\bbiosynthetic\b",
        r"\bcatabolic\b", r"\bglycolysis\b", r"\boxidative phosphorylation\b",
        r"\brespiratory chain\b", r"\bmitochond", r"\blipid metabolic\b",
        r"\bamino acid metabolic\b", r"\bnucleotide metabolic\b", r"\bglucose metabolic\b",
    ],
    "development process": [
        r"\bdevelopment\b", r"\bdevelopmental\b", r"\bdifferentiation\b",
        r"\bmorphogenesis\b", r"\bembryo", r"\borgan development\b",
        r"\btissue development\b", r"\bcell fate\b", r"\bpattern specification\b",
    ],
    "cell adhesion and migration": [
        r"\badhesion\b", r"\bcell adhesion\b", r"\bcell[- ]cell adhesion\b",
        r"\bextracellular matrix\b", r"\becm\b", r"\bmigration\b",
        r"\bcell migration\b", r"\bmotility\b", r"\bchemotaxis\b",
        r"\binvasion\b", r"\bcell junction\b", r"\bfocal adhesion\b",
    ],
}

NA_LIKE = {
    ".", "NA", "N/A", "nan", "NaN",
    "None", "NULL", "", "null"
}

def is_valid_value(v):
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    return str(v).strip() not in NA_LIKE



def ontology_text_from_row(row: pd.Series) -> str:
    parts = []
    for c in ["GO_BP", "GO_MF", "GO_CC", "KEGG", "HPO"]:
        v = str(row.get(c, ""))
        if v not in NA_LIKE:
            parts.append(v)
    return " | ".join(parts).lower()


def assign_categories(text: str) -> List[str]:
    out = []
    for cat, patterns in ONTOLOGY_CATEGORY_PATTERNS.items():
        if any(re.search(p, text, flags=re.IGNORECASE) for p in patterns):
            out.append(cat)
    return out


def primary_category(cats: List[str]) -> str:
    return cats[0] if cats else "other / unassigned"


def umap_hdbscan_plot(H: np.ndarray, out_html: str = "../downstream_data/umap_hdbscan_refalt.html"):


    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    U = reducer.fit_transform(H)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
    labels = clusterer.fit_predict(U)

    fig = go.Figure()
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

    for i, cl in enumerate(np.unique(labels)):
        mask = labels == cl
        fig.add_trace(go.Scatter(
            x=U[mask,0], y=U[mask,1],
            mode="markers",
            name=f"Cluster {cl}",
            marker=dict(size=6, color=palette[i % len(palette)], opacity=0.85)
        ))

    fig.update_layout(
        title="UMAP + HDBSCAN on Fused REF-ALT Embeddings",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        template="plotly_white", width=900, height=700
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f" Saved interactive plot: {out_html}")
    return labels, U

def shorten_label(s, max_len=30):
    if s is None:
        return "NA"
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def umap_plot_ontology(
    H: np.ndarray,
    df: pd.DataFrame,
    out_html: str = "../downstream_data/umap_fused_ontology.html",
    ontology_col: str = "ontology_categories"
):
    if H is None or len(H) == 0:
        raise ValueError("H is empty.")

    if ontology_col not in df.columns:
        raise ValueError(f"Column '{ontology_col}' not found in df.")

    if len(H) != len(df):
        raise ValueError(f"Shape mismatch: H has {len(H)} rows but df has {len(df)} rows.")

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )
    U = reducer.fit_transform(H)

    fig = go.Figure()
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    categories = df[ontology_col].fillna("NA").astype(str).values
    unique_categories = np.unique(categories)

    for i, cat in enumerate(unique_categories):
        mask = categories == cat
        fig.add_trace(go.Scatter(
            x=U[mask, 0],
            y=U[mask, 1],
            mode="markers",
            # name=str(cat),
            name=shorten_label(cat),
            marker=dict(
                size=6,
                color=palette[i % len(palette)],
                opacity=0.85
            ),
            text=df.loc[mask, ontology_col].astype(str),
            hovertemplate=(
                "Ontology: %{text}<br>"
                "UMAP-1: %{x:.3f}<br>"
                "UMAP-2: %{y:.3f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="UMAP on Fused REF-ALT Embeddings Colored by Ontology",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        template="plotly_white",
        width=900,
        height=700
    )

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved interactive plot: {out_html}")

    return U

def add_ontology_columns(df: pd.DataFrame) -> pd.DataFrame:
    texts = []
    cats_json = []
    primary = []

    for _, row in df.iterrows():
        txt = ontology_text_from_row(row)
        cats = assign_categories(txt)
        texts.append(txt)
        cats_json.append(json.dumps(cats, ensure_ascii=False))
        primary.append(primary_category(cats))

    df = df.copy()
    df["ontology_text"] = texts
    df["ontology_categories"] = cats_json
    df["primary_ontology_category"] = primary

    ordered = [
        "immune response",
        "signal transduction",
        "DNA cycle/cell cycle",
        "Metabolic process",
        "development process",
        "cell adhesion and migration",
    ]
    for cat in ordered:
        df[f"is_{cat}"] = [int(cat in json.loads(x)) for x in df["ontology_categories"]]

    return df






@torch.no_grad()
def extract_embeddings_refalt(model: TCL_TDA_Model_RefAlt, ds: Dataset, cfg: Config):
    print("I am in extract")
    device = cfg.device
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_fn_refalt_ontology)
    H_list, Z_list = [], []
    print("I am in ")
    for step, batch in enumerate(loader):
        (
            z_seq,
            z_bio,
            z_text,
            h_tda,
            h_mut,
            h_bio,
            h_text,
            h_ref,
            h_alt,
        ), bios = forward_model_from_batch(cfg, model, batch)
        H_list.append(h_tda.cpu())
        Z_list.append(h_bio.cpu())

    H = torch.cat(H_list, 0).numpy()
    Z = torch.cat(Z_list, 0).numpy()
    np.save("../downstream_data/embeddings_fused_refalt.npy", H)
    np.save("../downstream_data/embeddings_proj_refalt.npy", Z)
    print("Saved: embeddings_fused_refalt.npy and embeddings_proj_refalt.npy")
    return H, Z



def plot_loss(link : str):
    df = pd.read_csv("checkpoints/training_log.csv", sep=",", low_memory=False)
    print("*"*50)
    print(df)
    df_mean = df.groupby("epoch")["loss"].mean()
    # Plot epoch vs batch
    plt.figure(figsize=(10, 6))
    plt.plot(df_mean.index, df_mean.values, marker="o", linestyle="-")
    # plt.title("Batch en fonction de l'Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Sauvegarde du plot
    plt.savefig(link, dpi=300, bbox_inches="tight")



def compute_prototype_from_label(X: np.ndarray, y: np.ndarray, target_label: int):
    mask = (y == target_label)

    if not np.any(mask):
        raise ValueError(f"No samples found with label {target_label}")

    prototype = X[mask].mean(axis=0)
    return prototype

def compute_weighted_prototype(X, y, weights, target_label):
    mask = (y == target_label)
    X_sel = X[mask]
    w_sel = weights[mask]

    w_sel = w_sel / w_sel.sum()
    return np.sum(X_sel * w_sel[:, None], axis=0)

def mixed_norm_score(X, ref, alpha=0.7, beta=0.3):
    l2 = np.linalg.norm(X, axis=1)
    dist = np.linalg.norm(X - ref[None, :], axis=1)
    return alpha * l2 + beta * dist


def sample_k_per_class(X, y, k, seed=42):
    rng = np.random.default_rng(seed)

    X_sub = []
    y_sub = []

    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < k:
            raise ValueError(f"Class {c} has only {len(idx)} samples, cannot sample {k}.")
        chosen = rng.choice(idx, size=k, replace=False)
        X_sub.append(X[chosen])
        y_sub.append(y[chosen])

    X_sub = np.concatenate(X_sub, axis=0)
    y_sub = np.concatenate(y_sub, axis=0)

    perm = rng.permutation(len(y_sub))
    return X_sub[perm], y_sub[perm]

cont_cols = [
    "QUAL",
    "DP_GERM", "VAF_GERM", "DP_SOM", "VAF_SOM", "DP_UNI", "VAF_UNI",
    "IG_AF", "EVS_EA_MAF", "1000G_Global_AF", "gnomAD_Global_AF", "Kaviar_AF", "MAX_AF", "MAX_POP_AF", "GnomAD_MNV_AF",
    "1000G_AFR_AF", "1000G_AMR_AF", "1000G_EAS_AF", "1000G_EUR_AF", "1000G_SAS_AF", "1000G_AA_AF", "1000G_EA_AF",
    "gnomAD_AFR_AF", "gnomAD_AMR_AF", "gnomAD_ASJ_AF", "gnomAD_EAS_AF", "gnomAD_FIN_AF", "gnomAD_NFE_AF", "gnomAD_OTH_AF", "gnomAD_SAS_AF",
    "PhyloP", "PhastCons", "Distance_Grantham",
    "MaxEntScan_alt", "MaxEntScan_diff", "MaxEntScan_ref",
    "DANN_Score", "FATHMM_Non_Coding_Score", "FATHMM_Coding_Score", "Gene_damage_index",
    "N_PATIENTS_WITH_VARIANT", "N_CANCER_WITH_VARIANT", "N_NONCANCER_WITH_VARIANT", "SOMATIC_REPEAT_COUNT",
]

ordinal_cols = [
    "SIFT_score", "PolyPhen_ord", "PolyPhen_score",
    "Consequence_num", "Impact_num", "VariantClass_num", "Biotype_num", "Canonical_num",
    "Origin_num", "CNV_LOH_status_num",
]

binary_cols = [
    "SIFT_bin",
    "HAS_GERM", "HAS_SOM", "N_GERM_ONLY", "N_GERM_AND_SOM",
    "PATHWAY_HIT", "PATH_DNA_REPAIR",
    "LOH_PARTIAL_PATHO", "LOH_PARTIAL_PATHO_SOM",
    "CNV_HIT", "CNV_pathogenic",
    "IN_ROH", "HET_HIGH_HOM", "HAS_SOMATIC_PATIENT", "GERM_PASS",
    "VARIANT_CLASS_SNV", "VARIANT_CLASS_deletion", "VARIANT_CLASS_insertion", "VARIANT_CLASS_substitution",
    "IMPACT_HIGH", "IMPACT_LOW", "IMPACT_MODERATE", "IMPACT_UNKNOWN",
    "BIOTYPE_CTCF_binding_site", "BIOTYPE_NA", "BIOTYPE_TF_binding_site", "BIOTYPE_enhancer",
    "BIOTYPE_lncRNA", "BIOTYPE_miRNA", "BIOTYPE_misc_RNA", "BIOTYPE_open_chromatin_region",
    "BIOTYPE_promoter", "BIOTYPE_promoter_flanking_region", "BIOTYPE_protein_coding",
    "BIOTYPE_snoRNA", "BIOTYPE_transcribed_pseudogene",
    "ORIGIN_germline", "ORIGIN_somatic",
]

unit_cols = [
    "ada_score",
    "rf_score",
]

ordinal_max_map = {
    "SIFT_score": 1,
    "PolyPhen_ord": 3,
    "PolyPhen_score": 1,
    "Consequence_num": 5,
    "Impact_num": 4,
    "VariantClass_num": 4,
    "Biotype_num": 5,
    "Canonical_num": 1,
    "Origin_num": 3,
    "CNV_LOH_status_num": 2,
}

# def evaluate_over_dataset_size(
#     X_train,
#     y_train,
#     X_test,
#     y_test,
#     sizes=(5, 10, 15),
#     seed=42,
#     max_iter=200,
# ):
#     results = []
#     device = cfg.device
#     feature_dim = X_train.shape[1]
#     num_classes = len(np.unique(y_train))
#     # sizes = sizes + (X_train.shape[0],)
#     for k in sizes:
#         X_sub, y_sub = sample_k_per_class(X_train, y_train, k=k, seed=seed)
#         print(np.unique(y_sub).size)
        
#         lr = 1e-3
#         weight_decay = 1e-4

#         result = fit_logistic_regression(
#             X_sub,
#             y_sub,
#             X_test,
#             y_test
#         )
#         acc = result["best_val_acc"]
#         print(acc)
#         results.append((k, acc))
#         print(result)

#     # result = fit_logistic_regression(
#     #         X_train,
#     #         y_train,
#     #         X_test,
#     #         y_test
#     #     )

#     print("*"*100)
#     print(result)
#     print("*"*100)
#     # acc = result["best_val_acc"]
#     # results.append((100,acc))
#     return results

# from sklearn.model_selection import train_test_split
# import numpy as np


# def evaluate_over_dataset_size(
#     X_train,
#     y_train,
#     X_test,
#     y_test,
#     sizes=(5, 10, 15),
#     seed=42,
# ):
#     results = []

#     for k in sizes:
#         # few-shot subset from training set
#         X_sub, y_sub = sample_k_per_class(X_train, y_train, k=k, seed=seed)

#         print(f"\n{'='*80}")
#         print(f"k = {k}")
#         print(f"Subset shape: {X_sub.shape}")
#         print(f"Num classes in subset: {np.unique(y_sub).size}")

#         # split the subset into train/val
#         X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(
#             X_sub,
#             y_sub,
#             test_size=0.2,
#             random_state=seed,
#             stratify=y_sub
#         )

#         result = fit_logistic_regression(
#             X_train=X_sub_train,
#             y_train=y_sub_train,
#             X_val=X_sub_val,
#             y_val=y_sub_val,
#             X_test=X_test,
#             y_test=y_test,
#             seed=seed,
#         )

#         best_val_metrics = result["best_val_metrics"]
#         test_metrics = result.get("test_metrics", {})

#         row = {
#             "k": k,
#             "best_epoch": result["best_epoch"],
#             "best_val_score": result["best_val_score"],
#             "val_acc": best_val_metrics["acc"],
#             "val_f1_macro": best_val_metrics["f1_macro"],
#             "val_f1_weighted": best_val_metrics["f1_weighted"],
#             "test_acc": test_metrics.get("acc", None),
#             "test_f1_macro": test_metrics.get("f1_macro", None),
#             "test_f1_weighted": test_metrics.get("f1_weighted", None),
#         }
#         results.append(row)

#         print("Best val metrics:", best_val_metrics)
#         print("Test metrics:", test_metrics)

#     return results



def evaluate_over_dataset_size(
    X_train,
    y_train,
    X_test,
    y_test,
    sizes=(5, 10, 15),
    seed=42,
):
    results = []

    for k in sizes:
        X_sub, y_sub = sample_k_per_class(X_train, y_train, k=k, seed=seed)

        result = fit_logistic_regression(
            X_train=X_sub,
            y_train=y_sub,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
        )

        test_metrics = result["test_metrics"]

        results.append({
            "k": k,
            "test_acc": test_metrics["acc"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_f1_weighted": test_metrics["f1_weighted"],
        })

    return results


# def plot_dataset_size_curve(
#     results,
#     title="Classification over dataset size",
#     out_png=None
# ):
#     sizes = [r[0] for r in results]
#     accs = [r[1] for r in results]

#     plt.figure(figsize=(7, 4.8))
#     plt.plot(
#         sizes,
#         accs,
#         linestyle="--",
#         marker="*",
#         markersize=16,
#         linewidth=1.5,
#         color="black"
#     )

#     plt.title(title, fontsize=15)
#     plt.xlabel("Number of samples per class", fontsize=12)
#     plt.ylabel("Test accuracy", fontsize=12)
#     plt.xscale("log")
#     plt.xticks(sizes, [str(s) for s in sizes])
#     plt.grid(True, alpha=0.3)

#     if out_png is not None:
#         os.makedirs(os.path.dirname(out_png), exist_ok=True)
#         plt.savefig(out_png, dpi=200, bbox_inches="tight")
#         print(f"Saved plot to {out_png}")

#     plt.show()


def plot_dataset_size_curve(
    results,
    metric="test_acc",
    title="Few-shot performance vs dataset size",
    out_png=None
):
    sizes = [r["k"] for r in results]
    values = [r[metric] for r in results]

    sizes = np.array(sizes)
    values = np.array(values)

    plt.figure(figsize=(7, 5))

    plt.plot(
        sizes,
        values,
        linestyle="--",
        marker="o",
        linewidth=2,
    )

    plt.title(title, fontsize=15)
    plt.xlabel("Number of samples per class", fontsize=12)
    plt.ylabel(metric.replace("_", " "), fontsize=12)

    plt.xscale("log")
    plt.xticks(sizes, [str(s) for s in sizes])

    plt.grid(True, alpha=0.3)

    if out_png is not None:
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_png}")

    plt.show()

# def plot_multiple_dataset_size_curves(curves, title="Classification over dataset size", out_png=None):
#     plt.figure(figsize=(7, 4.8))

#     for label, results in curves.items():
#         sizes = [r[0] for r in results]
#         accs = [r[1] for r in results]
#         plt.plot(sizes, accs, linestyle="--", marker="o", linewidth=1.5, label=label)

#     plt.title(title, fontsize=15)
#     plt.xlabel("Number of samples per class", fontsize=12)
#     plt.ylabel("Test accuracy", fontsize=12)
#     plt.xscale("log")
#     plt.xticks(sizes, [str(s) for s in sizes])
#     plt.grid(True, alpha=0.3)
#     plt.legend()

#     if out_png is not None:
#         os.makedirs(os.path.dirname(out_png), exist_ok=True)
#         plt.savefig(out_png, dpi=200, bbox_inches="tight")
#         print(f"Saved plot to {out_png}")

#     plt.show()

def plot_multiple_dataset_size_curves(
    curves,
    metric="test_acc",
    title="Classification over dataset size",
    out_png=None
):
    plt.figure(figsize=(7, 5))

    all_sizes = None

    for label, results in curves.items():

        # detect format
        if isinstance(results[0], dict):
            sizes = [r["k"] for r in results]
            values = [r[metric] for r in results]
        else:
            # old format (k, acc)
            sizes = [r[0] for r in results]
            values = [r[1] for r in results]

        sizes = np.array(sizes)
        values = np.array(values)

        plt.plot(
            sizes,
            values,
            linestyle="--",
            marker="o",
            linewidth=2,
            label=label
        )

        # store sizes for xticks (assume same across curves)
        if all_sizes is None:
            all_sizes = sizes

    plt.title(title, fontsize=15)
    plt.xlabel("Number of samples per class", fontsize=12)
    plt.ylabel(metric.replace("_", " "), fontsize=12)

    plt.xscale("log")

    if all_sizes is not None:
        plt.xticks(all_sizes, [str(s) for s in all_sizes])

    plt.grid(True, alpha=0.3)
    plt.legend()

    if out_png is not None:
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_png}")

    plt.show()

if __name__ == "__main__":
    try:
        cfg = Config()
        h = "../downstream_data/embeddings_fused_refalt.npy"
        z = "../downstream_data/embeddings_proj_refalt.npy"
        # plot_loss("../result/epoch_batch_plot2.png")
        if os.path.exists(h) and os.path.exists(z):
            H = np.load(h)
            Z = np.load(z)
            print("End of model loading")
        else:
            # model_student = TCL_TDA_Model_RefAlt(cfg).to(cfg.device)
            # ema_teacher = TCL_TDA_Model_RefAlt(cfg)
            # --- Load checkpoint ---
            


            print("Let' start ", cfg)
            train_ds, n_train = get_data_all(
                seq_tsv=cfg.train_seq,
                bio_tsv=cfg.train_bio,   
                onto_tsv=cfg.train_onto,
                text_tsv=getattr(cfg, "train_text", None),
                textfields_tsv=getattr(cfg, "train_textfields", None),
                labels_path=cfg.train_labels if os.path.exists(cfg.train_labels) else None,
                label_col="labels",
                include_ontology=True,
                include_oncology=False,
                cache_dir="dataset",
                use_cache=True,
                rebuild_cache=True,
                max_bio_line=getattr(cfg, "max_bio_line", None),
                use_text=False,
                seq_offline_embeddings=getattr(cfg, "seq_offline_embeddings", False),
                text_offline_embeddings=False,
                dna_model_name=getattr(cfg, "dna_model_name", None),
                text_model_name=getattr(cfg, "text_model_name", None),
                seq_pool=getattr(cfg, "pool", "mean"),
                text_pool=getattr(cfg, "text_pool", "mean"),
                seq_max_len=getattr(cfg, "max_len", 512),
                text_max_len=getattr(cfg, "text_max_len", 256),
                precompute_batch_size=getattr(cfg, "precompute_batch_size", 32),
                embedding_device=str(cfg.device),
                precomputed_ref_path=getattr(cfg, "train_ref_emb_path", None),
                precomputed_alt_path=getattr(cfg, "train_alt_emb_path", None),
                precomputed_text_path=getattr(cfg, "train_text_emb_path", None),
                cont_cols=cont_cols,
                ordinal_cols=ordinal_cols,
                binary_cols=binary_cols,
                unit_cols=unit_cols,
                ordinal_max_map=ordinal_max_map,
                corr_threshold=None,
            )  

            print("Okay")
            cfg.bio_dim = train_ds.bio_in_dim
            print("shape bio", cfg.bio_dim)
            print("offline embedding",cfg.seq_offline_embeddings)
            cfg.seq_offline_embeddings = False
            ema_teacher = TCL_TDA_Model_RefAlt(cfg)
            ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")
            # model_student.load_state_dict(ckpt["model_state"])
            ema_teacher.load_state_dict(ckpt["teacher_state_dict"], strict=False)
            # ema_teacher.load_state_dict(ckpt["model_state_dict"])
            print("I have loaded the model")
            # model_student.eval()
            ema_teacher.eval()
            print("I am ready to start")
            H, Z = extract_embeddings_refalt(ema_teacher, train_ds, cfg)
        # umap_hdbscan_plot(H, "../downstream_data/umap_hdbscan_refalt.html")
        # umap_hdbscan_plot(Z, "../downstream_data/umap_hdbscan_bio.html")

        meta_df = pd.read_csv(cfg.train_onto, sep="\t")
        df_emb = add_ontology_columns(meta_df)
        # umap_plot_ontology(
        #     H,
        #     df_emb,
        #     out_html ="../downstream_data/umap_fused_ontology.html",
        #     ontology_col="primary_ontology_category"
        # )


        label_df = pd.read_csv(cfg.train_bio, sep="\t")
        print(label_df.nunique())

        label = label_df["label_clinvar"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            H,
            label,
            test_size=0.2,
            random_state=42,
            stratify=label 
        )
        print("*"*100)
        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_test, return_counts=True))
        results_fused = evaluate_over_dataset_size(
            X_train=X_train,
            y_train=(y_train >= 3).astype(int),
            X_test=X_test,
            y_test=(y_test >= 3).astype(int),
            sizes=(10, 100, 200, 400, 500),
            seed=42
        )
        # clf = LogisticRegression(max_iter=2000)
        # clf.fit(X_train, y_train)

        # y_pred = clf.predict(X_test)

        # print("Accuracy:", accuracy_score(y_test, y_pred))
        # print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
        # print(classification_report(y_test, y_pred))

        plot_dataset_size_curve(
            results_fused,
            title="Fused embedding classification over dataset size",
            out_png="../downstream_data/fused_dataset_size_curve.png"
        ) 

        X_train, X_test, y_train, y_test = train_test_split(
            Z,
            label,
            test_size=0.2,
            random_state=42,
            stratify=label  
        )
        print("*"*100)
        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_test, return_counts=True))
        results_bio = evaluate_over_dataset_size(
            X_train=X_train,
            y_train=(y_train >= 3).astype(int),
            X_test=X_test,
            y_test=(y_test >= 3).astype(int),
            sizes=(10, 50, 100, 200, 400, 500),
            seed=42
        )
        # clf = LogisticRegression(max_iter=2000)
        # clf.fit(X_train, y_train)

        # y_pred = clf.predict(X_test)

        # print("Accuracy:", accuracy_score(y_test, y_pred))
        # print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
        # print(classification_report(y_test, y_pred))
        plot_dataset_size_curve(
            results_bio,
            title="Biological embedding classification over dataset size",
            out_png="../downstream_data/bio_dataset_size_curve.png"
        ) 
        plot_multiple_dataset_size_curves(
            {
                "z_bio": results_bio,
                "z_fused": results_fused,
            },
            title="Downstream classification over dataset size",
            out_png="../downstream_data/compare_dataset_size_curves.png"
        )

        y = label_df["labels"].to_numpy()
        P_1 = compute_prototype_from_label(H, y, target_label=1)

        print(P_1.shape)  # (D,)
        print("Nb samples label=1:", (y == 1).sum())
        scores = mixed_norm_score(H, P_1, alpha=0.7, beta=0.3)
        idx_sorted = np.argsort(scores)
        top_k = 50

        info = pd.read_csv(r"/nfs/homes/vomodonfack/Dataset/Final Data Construction/Exome_Data/ALL_FAMILIES__FAMILY_VARIANTS.tsv", sep="\t")
        selected_idx = idx_sorted[:top_k]
        df_selected = info.iloc[selected_idx][["Gene_Name", "CHROM","POS", "REF", "ALT","Prediction_ACMG_tapes"]]
        print(df_selected.head())

        

        # results = zero_shot_low_penetrance_from_cfg(
        #     H=H,
        #     y=label,
        #     evaluate_bio_tsv=r"/nfs/homes/vomodonfack/Dataset/Final Data Construction/Exome_Data/ALL_FAMILIES__FAMILY_VARIANTS.tsv",
        #     out_dir="../downstream_data/results_zero_shot_low_pen",
        #     ontology_cols=("GO_BP", "GO_MF", "GO_CC"),
        #     alpha=0.7,
        #     beta=0.3,
        # )

        # df_ranked = results["df_ranked"]
        # df_selected = results["df_selected"]
        



    except Exception as e:
        print("\n ERROR in main:", e)
        print("Cleaning up RAM...")
        raise

    finally:
        # del df, h, z, carriers
        del df, h, z
        gc.collect()
         
