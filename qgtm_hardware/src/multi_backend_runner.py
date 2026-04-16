"""
multi_backend_runner.py
=======================
Runs QGTM experiments across multiple IBM Quantum backends and
produces a cross-device comparison plot + LaTeX table.

Usage:
    python multi_backend_runner.py --token YOUR_TOKEN
    python multi_backend_runner.py --simulate --noise   # all simulators
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qgtm_hardware_experiments import (
    QGTMBackend, QGTMExperiments, QGTMPlotter,
    QGTMCircuitBuilder, QGTMMechanism
)

# ── Backend configurations ────────────────────────────────────────────────────

BACKENDS = {
    "ibmq_lima":   {"color": "#005AB4", "marker": "o"},
    "ibmq_belem":  {"color": "#B41E1E", "marker": "s"},
    "ibmq_manila": {"color": "#1E8C1E", "marker": "^"},
    "ibmq_quito":  {"color": "#C89600", "marker": "D"},
}


def run_cross_backend(token: str | None, simulate: bool, shots: int = 1024,
                       results_dir: str = "results") -> dict:
    """
    Run noise characterisation + PoA experiment on each backend.
    Returns per-backend summary.
    """
    Path(results_dir).mkdir(exist_ok=True)
    summary = {}

    for bname, style in BACKENDS.items():
        print(f"\n{'='*55}")
        print(f" Backend: {bname}")
        print(f"{'='*55}")

        try:
            if simulate:
                backend = QGTMBackend(
                    backend_name   = bname,
                    shots          = shots,
                    simulate_noise = True,
                )
            else:
                backend = QGTMBackend(
                    backend_name = bname,
                    token        = token,
                    shots        = shots,
                )

            exps = QGTMExperiments(backend, results_dir=results_dir)
            hw   = exps.experiment_hardware_characterisation()
            poa  = exps.experiment_price_of_anarchy(n_trials=10)

            summary[bname] = {
                "bell_fidelity":  hw["bell_fidelity"],
                "depolarising_p": hw["depolarising_p"],
                "poa_qgtm":       poa["QGTM"]["mean"],
                "poa_qgtm_ci":    poa["QGTM"]["ci"],
                "poa_vcg":        poa["VCG"]["mean"],
                "poa_vcg_ci":     poa["VCG"]["ci"],
                "style":          style,
            }

        except Exception as exc:
            print(f"  [WARN] {bname} failed: {exc}")
            summary[bname] = None

    # Save summary
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(results_dir) / f"{ts}_cross_backend_summary.json"
    with open(path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "style"}
                   for k, v in summary.items() if v is not None}, f, indent=2)
    print(f"\n[Summary] Saved to {path}")
    return summary


def plot_cross_backend(summary: dict, figures_dir: str = "figures"):
    Path(figures_dir).mkdir(exist_ok=True)

    valid = {k: v for k, v in summary.items() if v is not None}
    if not valid:
        print("[Plot] No valid backends to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plt.rcParams.update({"font.size": 10})

    # ── Left: Bell Fidelity per backend ──────────────────────────────────────
    ax = axes[0]
    bnames = list(valid.keys())
    fids   = [valid[b]["bell_fidelity"] for b in bnames]
    noise  = [valid[b]["depolarising_p"] for b in bnames]
    colors = [valid[b]["style"]["color"] for b in bnames]
    bars   = ax.bar(bnames, fids, color=colors, width=0.55,
                    edgecolor="white", linewidth=1.2)
    for bar, f, p in zip(bars, fids, noise):
        ax.text(bar.get_x() + bar.get_width() / 2, f + 0.005,
                f"F={f:.3f}\np={p:.3f}", ha="center", va="bottom",
                fontsize=7.5, multialignment="center")
    ax.axhline(0.80, color="black", linestyle="--", linewidth=0.8,
               label="$F_{\\min}=0.80$")
    ax.set_ylabel("Bell State Fidelity")
    ax.set_title("Hardware Bell Fidelity per Backend")
    ax.set_ylim(0.6, 1.05)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    # ── Right: PoA QGTM vs VCG per backend ───────────────────────────────────
    ax = axes[1]
    x   = np.arange(len(bnames))
    w   = 0.35
    poa_qgtm = [valid[b]["poa_qgtm"]    for b in bnames]
    ci_qgtm  = [valid[b]["poa_qgtm_ci"] for b in bnames]
    poa_vcg  = [valid[b]["poa_vcg"]     for b in bnames]
    ci_vcg   = [valid[b]["poa_vcg_ci"]  for b in bnames]

    ax.bar(x - w/2, poa_qgtm, w, label="QGTM",         color="#005AB4",
           edgecolor="white")
    ax.errorbar(x - w/2, poa_qgtm, yerr=ci_qgtm, fmt="none",
                ecolor="black", elinewidth=1, capsize=3)
    ax.bar(x + w/2, poa_vcg, w, label="Classical VCG", color="#B41E1E",
           edgecolor="white")
    ax.errorbar(x + w/2, poa_vcg, yerr=ci_vcg, fmt="none",
                ecolor="black", elinewidth=1, capsize=3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bnames, rotation=15)
    ax.set_ylabel("Price of Anarchy")
    ax.set_title("PoA: QGTM vs VCG per Backend")
    ax.set_ylim(0, 3.5)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("QGTM Cross-Backend Hardware Validation", fontweight="bold")
    fig.tight_layout()
    path = Path(figures_dir) / "fig_cross_backend.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"[Plot] Saved: {path}")
    plt.close(fig)


def generate_latex_cross_table(summary: dict) -> str:
    valid = {k: v for k, v in summary.items() if v is not None}
    rows  = ""
    for bname, data in valid.items():
        gain = (1.0 - data["poa_qgtm"] / data["poa_vcg"]) * 100
        rows += (
            f"\\texttt{{{bname}}} & "
            f"{data['bell_fidelity']:.4f} & "
            f"{data['depolarising_p']:.4f} & "
            f"{data['poa_qgtm']:.3f} $\\pm$ {data['poa_qgtm_ci']:.3f} & "
            f"{data['poa_vcg']:.3f} $\\pm$ {data['poa_vcg_ci']:.3f} & "
            f"{gain:.1f}\\% \\\\\n"
        )

    tex = rf"""
\begin{{table}}[!t]
\centering
\caption{{Cross-Backend Hardware Validation of QGTM}}
\label{{tab:cross_backend}}
\renewcommand{{\arraystretch}}{{1.35}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\toprule
\textbf{{Backend}} & \textbf{{Bell Fidelity}} & \textbf{{Noise $p$}} &
\textbf{{PoA (QGTM)}} & \textbf{{PoA (VCG)}} & \textbf{{PoA Gain}} \\
\midrule
{rows}\bottomrule
\end{{tabular}}}}
\begin{{tablenotes}}\small
\item All experiments: $N=4$ users, $q_s=0.3$, $\kappa=2.0$, shots=1024, 20 MC trials.
\item PoA Gain = $(1 - \text{{PoA}}_\text{{QGTM}} / \text{{PoA}}_\text{{VCG}}) \times 100\%$.
\end{{tablenotes}}
\end{{table}}
"""
    return tex


def main():
    p = argparse.ArgumentParser(
        description="Run QGTM experiments across multiple IBM Quantum backends"
    )
    p.add_argument("--token",       default=None)
    p.add_argument("--simulate",    action="store_true")
    p.add_argument("--shots",       type=int, default=1024)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    args = p.parse_args()

    summary = run_cross_backend(
        token       = args.token or os.environ.get("IBM_QUANTUM_TOKEN"),
        simulate    = args.simulate,
        shots       = args.shots,
        results_dir = args.results_dir,
    )

    plot_cross_backend(summary, figures_dir=args.figures_dir)

    tex      = generate_latex_cross_table(summary)
    tex_path = Path(args.results_dir) / "cross_backend_table.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"[LaTeX] Cross-backend table written to {tex_path}")


if __name__ == "__main__":
    main()
