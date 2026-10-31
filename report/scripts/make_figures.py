# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""Report figures that results/figures/ lacks: block size and matrix size comparisons and
the phase breakdown with error bars, plus Portuguese vector re-plots of the strong/weak
scaling and node sweep figures of results/figures/ (same data, same std bars). Reads
results/metrics.csv and results/node_sweep.csv and reuses the style of
scripts/analyze_results.py from the data repo.

Usage: python3 make_figures.py [DATA_REPO]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_REPO = ("/Users/oliveira_mats/Documents/phd_documents/1_semester/"
                "High_perfomance_computer/HPC_Matrix_Multiplication")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
VERSIONS = ["mpi", "mpi_openblas"]
PHASES = {"time_distribution_s": "distribuição", "time_compute_s": "cálculo",
          "time_gather_s": "coleta (gather)"}


def grid(ar, df, col_key, col_vals, col_fmt, line_key, line_vals, line_fmt, ylabel, title,
         name):
    """Rows = version, columns = col_key, one line per line_key value; median vs p, std bars."""
    fig, axes = plt.subplots(2, len(col_vals), figsize=(4.2 * len(col_vals), 6.8),
                             sharex=True, squeeze=False)
    for r, alg in enumerate(VERSIONS):
        for c, cv in enumerate(col_vals):
            ax = axes[r][c]
            for i, lv in enumerate(line_vals):
                s = df[(df.algorithm == alg) & (df[col_key] == cv) & (df[line_key] == lv)]
                s = s.sort_values("processes")
                # ponytail: lower bar clipped so log y stays defined when std > median
                low = np.minimum(s["std"], 0.9 * s["value"])
                ax.errorbar(s.processes, s.value, yerr=[low, s["std"]], color=ar.COLORS[i],
                            marker=ar.MARKERS[i], linestyle=ar.LINESTYLES[i], markersize=5,
                            capsize=3, label=line_fmt(lv))
            ar.log2_axis(ax)
            ax.set_yscale("log")
            ax.set_title("{}, {}".format(ar.LABELS[alg], col_fmt.format(cv)))
        axes[r][0].set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel("processos MPI p")
    axes[0][0].legend(frameon=False)
    fig.suptitle(title + " (mediana de 5 execuções; barras: desvio padrão)")
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def versions_row(ar, df, panel_key, panel_vals, panel_fmt, x_key, xlabel, ylabel, title,
                 name, ref=None, ylog2=False, xlog2=True):
    """One panel per panel_key value, MPI and MPI + OpenBLAS lines, median with std bars
    (as in analyze_results.make_plots, Portuguese labels, vector output)."""
    fig, axes = plt.subplots(1, len(panel_vals), figsize=(4.2 * len(panel_vals), 3.6),
                             squeeze=False)
    for ax, pv in zip(axes[0], panel_vals):
        for i, alg in enumerate(VERSIONS):
            s = df[(df.algorithm == alg) & (df[panel_key] == pv)].sort_values(x_key)
            ax.errorbar(s[x_key], s.value, yerr=s["std"], color=ar.COLORS[i + 1],
                        marker=ar.MARKERS[i + 1], linestyle=ar.LINESTYLES[i], markersize=6,
                        capsize=3, label=ar.LABELS[alg])
        if ref:
            ref(ax, sorted(df[x_key].unique()))
        if xlog2:
            ar.log2_axis(ax)
        if ylog2:  # the ideal line reaches 1024 and would flatten the data
            ar.log2_axis(ax, "y")
        ax.set_title(panel_fmt.format(pv))
        ax.set_xlabel(xlabel)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend(frameon=False)
    fig.suptitle(title + " (mediana de 5 execuções; barras: desvio padrão)")
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default=DEFAULT_REPO, help="data repo root")
    args = parser.parse_args()

    sys.dont_write_bytecode = True  # the data repo is read-only: no __pycache__ there
    sys.path.insert(0, os.path.join(args.repo, "scripts"))
    import analyze_results as ar

    allm = pd.read_csv(os.path.join(args.repo, "results", "metrics.csv"))
    # Strong scaling, packed layout only (drops the node sweep and the weak runs).
    packed = allm.nodes <= np.maximum(1, -(-allm.processes // ar.CORES_PER_NODE))
    m = allm[(allm.scaling == "strong") & packed & allm.algorithm.isin(VERSIONS)]
    sizes, blocks = sorted(m.matrix_size.unique()), sorted(m.block_size.unique())

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3dd", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8980", "lines.linewidth": 2})
    os.makedirs(OUT, exist_ok=True)

    t = m[m.metric == "time_total_s"]
    grid(ar, t, "matrix_size", sizes, "N = {}", "block_size", blocks, "bloco {}".format,
         "tempo total T [s]", "Tempo total vs p por tamanho de bloco", "tempo_blocos.pdf")
    g = m[m.metric == "gflops"]
    grid(ar, g, "block_size", blocks, "bloco {}", "matrix_size", sizes, "N = {}".format,
         "desempenho [GFLOP/s]", "Desempenho vs p por tamanho de matriz",
         "gflops_tamanhos.pdf")
    ph = m[m.metric.isin(list(PHASES))]
    for b in blocks:
        grid(ar, ph[ph.block_size == b], "matrix_size", sizes, "N = {}", "metric",
             list(PHASES), PHASES.get, "tempo da fase [s] (máx. entre ranks)",
             "Tempo por fase, bloco {}".format(b), "fases_b{}.pdf".format(b))

    # Portuguese re-plots of results/figures/{speedup_parallel,efficiency_strong}_b*.png,
    # efficiency_weak.png and node_sweep_b*.png.
    ideal = lambda ax, ps: ax.plot(ps, ps, color="#8a8980", linewidth=1, linestyle=":",
                                   label="ideal (S_p = p)")
    half = lambda ax, ps: ax.axhline(ar.SATURATION_EFFICIENCY, color="#8a8980", linewidth=1,
                                     linestyle=":", label="E_p = 0,5 (saturação)")
    for b in blocks:
        for metric, ylabel, ref, ylog2 in (
                ("speedup_parallel", "speedup paralelo S_p = T_1 / T_p", ideal, True),
                ("efficiency_strong", "eficiência E_p = S_p / p", half, False)):
            versions_row(ar, m[(m.metric == metric) & (m.block_size == b)], "matrix_size",
                         sizes, "N = {}", "processes", "processos MPI p", ylabel,
                         "Escalabilidade forte, bloco {}".format(b),
                         "{}_b{}.pdf".format(metric, b), ref, ylog2)
    w = allm[(allm.scaling == "weak") & (allm.metric == "efficiency_weak")]
    one = lambda ax, ps: ax.axhline(1.0, color="#8a8980", linewidth=1, linestyle=":",
                                    label="E_w = 1")
    versions_row(ar, w, "block_size", blocks, "bloco {}", "processes",
                 "processos MPI p (N = 2000, 4000, 6000)", "eficiência fraca E_w = T_1 / T_p",
                 "Escalabilidade fraca", "efficiency_weak.pdf", one, xlog2=False)
    sw = pd.read_csv(os.path.join(args.repo, "results", "node_sweep.csv")).rename(
        columns={"median_total_s": "value", "std_total_s": "std"})
    for b in blocks:
        versions_row(ar, sw[sw.block_size == b], "matrix_size", sizes, "N = {}", "nodes",
                     "nós (p = 128 fixo)", "tempo total T [s]",
                     "Varredura de nós com p = 128 fixo, bloco {}".format(b),
                     "node_sweep_b{}.pdf".format(b))


if __name__ == "__main__":
    main()
