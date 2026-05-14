"""
PCA assignment: GSE5325 gene expression (README tasks 1–2, plus primer-style extras).
- Figure 1a: GATA3 vs XBP1 scatter, colored by ER status.
- Figure 1b: PC1/PC2 axes from the 2-D covariance (raw expression).
- Figure 1c: PCA on the 105×2 matrix; projection onto PC1 (All / ER- / ER+).
- Scree + biplot: full-matrix PCA (genes standardized), XBP1/CCNB2 loadings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

GATA3_ID = "4359"
XBP1_ID = "4404"
CCNB2_ID = "4459"


def load_class_labels(path: Path) -> np.ndarray:
    lines = path.read_text().strip().splitlines()
    return np.array([int(x) for x in lines], dtype=np.int8)


def load_expression(path: Path) -> pd.DataFrame:
    # Whitespace-separated; first row is probe/gene IDs; .gz handled by pandas
    return pd.read_csv(path, sep=r"\s+", engine="python", compression="infer")


def main() -> None:
    classes = load_class_labels(DATA / "class.tsv")
    expr = load_expression(DATA / "filtered.tsv.gz")

    if len(classes) != len(expr):
        raise ValueError(f"class rows {len(classes)} != expression rows {len(expr)}")

    gata3 = expr[GATA3_ID].to_numpy(dtype=float)
    xbp1 = expr[XBP1_ID].to_numpy(dtype=float)
    X2 = np.column_stack([gata3, xbp1])

    er_pos = classes == 1
    er_neg = classes == 0

    # --- Figure 1a ---
    fig_a, ax_a = plt.subplots(figsize=(6, 5))
    ax_a.scatter(
        gata3[er_neg],
        xbp1[er_neg],
        c="black",
        marker="s",
        s=28,
        edgecolors="k",
        linewidths=0.4,
        label="ER-",
        zorder=2,
    )
    ax_a.scatter(
        gata3[er_pos],
        xbp1[er_pos],
        c="red",
        marker="s",
        s=28,
        edgecolors="darkred",
        linewidths=0.4,
        label="ER+",
        zorder=2,
    )
    ax_a.set_xlabel("GATA3 expression")
    ax_a.set_ylabel("XBP1 expression")
    ax_a.set_title("GATA3 vs XBP1 (colored by ER status)")
    ax_a.legend(frameon=True, loc="best")
    ax_a.grid(True, alpha=0.25)
    fig_a.tight_layout()
    out_a = HERE / "figure_1a_gata3_xbp1.png"
    fig_a.savefig(out_a, dpi=200)
    plt.close(fig_a)

    # --- Figure 1b: PC axes on raw GATA3–XBP1 scatter (eigenvectors of cov) ---
    mean2 = X2.mean(axis=0)
    cov2 = np.cov(X2, rowvar=False)
    evals2, evecs2 = np.linalg.eigh(cov2)
    order2 = np.argsort(evals2)[::-1]
    evals2 = evals2[order2]
    evecs2 = evecs2[:, order2]
    span = 2.5 * np.sqrt(np.clip(evals2[:2], 1e-12, None))
    v1_raw = evecs2[:, 0] * span[0]
    v2_raw = evecs2[:, 1] * span[1]

    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    ax_b.scatter(gata3[er_neg], xbp1[er_neg], c="black", marker="s", s=28, label="ER-", zorder=2)
    ax_b.scatter(gata3[er_pos], xbp1[er_pos], c="red", marker="s", s=28, label="ER+", zorder=2)
    ax_b.axline(mean2, mean2 + v1_raw, color="tab:blue", linewidth=2, label="PC1 (raw)")
    ax_b.axline(mean2, mean2 + v2_raw, color="tab:orange", linewidth=2, label="PC2 (raw)")
    ax_b.set_xlabel("GATA3 expression")
    ax_b.set_ylabel("XBP1 expression")
    ax_b.set_title("PC1 & PC2 directions (covariance of GATA3, XBP1)")
    ax_b.legend(loc="best", fontsize=8)
    ax_b.grid(True, alpha=0.25)
    fig_b.tight_layout()
    out_b = HERE / "figure_1b_pc_axes_raw.png"
    fig_b.savefig(out_b, dpi=200)
    plt.close(fig_b)

    # --- PCA on 2-D gene space, PC1 scores ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X2)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(Xs)
    pc1 = scores[:, 0]

    explained = pca.explained_variance_ratio_

    # --- Figure 1c: strip-style projection onto PC1 ---
    rng = np.random.default_rng(0)
    jitter = 0.12

    fig_c, axes_c = plt.subplots(3, 1, figsize=(7, 4.5), sharex=True)
    panels = [
        ("All", np.arange(len(pc1)), pc1),
        ("ER-", np.where(er_neg)[0], pc1[er_neg]),
        ("ER+", np.where(er_pos)[0], pc1[er_pos]),
    ]
    for ax, (title, idx, vals) in zip(axes_c, panels):
        y = rng.normal(0, jitter, size=len(vals))
        colors = np.where(classes[idx] == 1, "red", "black")
        ax.scatter(vals, y, c=colors, marker="s", s=22, edgecolors="none", alpha=0.9)
        ax.set_ylabel(title, rotation=0, labelpad=36, va="center")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.25)
    axes_c[-1].set_xlabel("Projection onto PC1")
    fig_c.suptitle(
        "PCA on GATA3 & XBP1 (standardized): scores on PC1\n"
        f"PC1 explains {100 * explained[0]:.1f}% variance; PC2 {100 * explained[1]:.1f}%",
        fontsize=10,
        y=1.02,
    )
    fig_c.tight_layout()
    out_c = HERE / "figure_1c_pc1_projection.png"
    fig_c.savefig(out_c, dpi=200, bbox_inches="tight")
    plt.close(fig_c)

    # --- Full expression matrix: PCA (samples × genes), standardized per gene ---
    X_all = expr.to_numpy(dtype=float)
    scaler_all = StandardScaler()
    X_all_s = scaler_all.fit_transform(X_all)
    n_pc_full = min(X_all_s.shape[0], X_all_s.shape[1])
    pca_full = PCA(n_components=n_pc_full, svd_solver="full", random_state=0)
    scores_full = pca_full.fit_transform(X_all_s)
    ratio_full = pca_full.explained_variance_ratio_

    # Scree (first 40 PCs for readability; full stats in summary)
    n_scree = min(40, len(ratio_full))
    fig_d, ax_d = plt.subplots(figsize=(8, 4))
    ax_d.bar(np.arange(n_scree), 100 * ratio_full[:n_scree], color="steelblue", edgecolor="none")
    ax_d.set_xlabel("Principal component (0-based index)")
    ax_d.set_ylabel("Proportion of variance (%)")
    ax_d.set_title("Scree plot — PCA on all probes (genes standardized across samples)")
    ax_d.set_xticks(np.arange(0, n_scree, 5))
    fig_d.tight_layout()
    out_d = HERE / "figure_scree_full_matrix.png"
    fig_d.savefig(out_d, dpi=200)
    plt.close(fig_d)

    # Biplot PC1 vs PC2 with XBP1 and CCNB2 loading arrows (scaled for visibility)
    pcx, pcy = scores_full[:, 0], scores_full[:, 1]
    cols = list(expr.columns)
    j_xbp1 = cols.index(XBP1_ID)
    j_ccnb2 = cols.index(CCNB2_ID)
    load = pca_full.components_.T[:, :2]
    arrow_scale = 25 * max(np.ptp(pcx), np.ptp(pcy), 1.0) / max(
        np.linalg.norm(load[j_xbp1]), np.linalg.norm(load[j_ccnb2]), 1e-9
    )
    lx_xbp1 = load[j_xbp1] * arrow_scale
    lx_ccnb2 = load[j_ccnb2] * arrow_scale

    fig_e, ax_e = plt.subplots(figsize=(6.5, 5.5))
    ax_e.scatter(pcx[er_neg], pcy[er_neg], c="black", marker="s", s=26, label="ER-", zorder=2)
    ax_e.scatter(pcx[er_pos], pcy[er_pos], c="red", marker="s", s=26, label="ER+", zorder=2)
    ax_e.quiver(
        0,
        0,
        lx_xbp1[0],
        lx_xbp1[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        width=0.006,
        zorder=3,
    )
    ax_e.scatter([lx_xbp1[0]], [lx_xbp1[1]], c="limegreen", s=40, zorder=4)
    ax_e.text(lx_xbp1[0], lx_xbp1[1], "  XBP1", fontsize=10, va="center")
    ax_e.quiver(
        0,
        0,
        lx_ccnb2[0],
        lx_ccnb2[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        width=0.006,
        zorder=3,
    )
    ax_e.scatter([lx_ccnb2[0]], [lx_ccnb2[1]], c="limegreen", s=40, zorder=4)
    ax_e.text(lx_ccnb2[0], lx_ccnb2[1], "  CCNB2", fontsize=10, va="center")
    ax_e.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
    ax_e.axvline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
    ax_e.set_xlabel("Projection onto PC1")
    ax_e.set_ylabel("Projection onto PC2")
    ax_e.set_title("Biplot (full PCA): samples + XBP1 / CCNB2 loadings (scaled)")
    ax_e.legend(loc="best")
    ax_e.grid(True, alpha=0.25)
    fig_e.tight_layout()
    out_e = HERE / "figure_biplot_pc1_pc2.png"
    fig_e.savefig(out_e, dpi=200)
    plt.close(fig_e)

    # Text summary for report folder
    summary_path = HERE / "OUTPUT_SUMMARY.txt"
    cum10 = 100 * ratio_full[:10].sum()
    lines = [
        "PCA on gene expression (GSE5325) — assignment output summary",
        "=" * 60,
        f"Samples: {len(classes)}  |  ER+: {int(er_pos.sum())}  |  ER-: {int(er_neg.sum())}",
        f"Genes used (2-gene plots): GATA3 (probe {GATA3_ID}), XBP1 (probe {XBP1_ID})",
        f"Biplot arrows: XBP1 ({XBP1_ID}), CCNB2 ({CCNB2_ID})",
        "",
        "Figure 1a (saved: figure_1a_gata3_xbp1.png)",
        "  Scatter of GATA3 (x) vs XBP1 (y); black = ER-, red = ER+.",
        "  Expect strong positive correlation; ER+ tends toward higher expression.",
        "",
        "Figure 1b (saved: figure_1b_pc_axes_raw.png)",
        "  Same scatter with PC1/PC2 as principal axes of the 2×2 covariance (raw units).",
        "",
        "PCA (2 columns only, after StandardScaler per gene)",
        f"  PC1 variance explained: {100 * explained[0]:.2f}%",
        f"  PC2 variance explained: {100 * explained[1]:.2f}%",
        "  PC loading directions (scaled feature space, sklearn convention):",
        f"    PC1 weights (GATA3, XBP1): {pca.components_[0]}",
        f"    PC2 weights (GATA3, XBP1): {pca.components_[1]}",
        "",
        "Figure 1c (saved: figure_1c_pc1_projection.png)",
        "  Three panels: all samples, ER- only, ER+ only — x = PC1 score.",
        "  Separation along PC1 mirrors separation visible along the long axis in 1a.",
        "",
        "Full matrix PCA (all probes; StandardScaler per gene across the 105 samples)",
        f"  Number of PCs computed: {n_pc_full}",
        f"  PC1 variance explained: {100 * ratio_full[0]:.2f}%",
        f"  PC2 variance explained: {100 * ratio_full[1]:.2f}%",
        f"  Cumulative variance (PC1–PC10): {cum10:.2f}%",
        "",
        "Figure scree (saved: figure_scree_full_matrix.png)",
        "  Bar chart of variance explained for the first PCs (see file for count).",
        "",
        "Figure biplot (saved: figure_biplot_pc1_pc2.png)",
        "  Samples in PC1–PC2 space with scaled loading arrows for XBP1 and CCNB2.",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(
        f"\nWrote: {out_a.name}, {out_b.name}, {out_c.name}, "
        f"{out_d.name}, {out_e.name}, {summary_path.name}"
    )


if __name__ == "__main__":
    main()
