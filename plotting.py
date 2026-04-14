"""
All plotting functions — fit curves, convergence, summary, deconv bars.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def setup_matplotlib(cfg):
    """Set matplotlib font parameters from config."""
    plt.rc('font', family='Liberation Sans', size=cfg.base_fontsize)


def safe_savefig(fig, path, max_image_dim, **kwargs):
    """Save figure ensuring no dimension exceeds max_image_dim pixels."""
    w_in, h_in = fig.get_size_inches()
    max_in = max(w_in, h_in)
    dpi = min(150, max_image_dim / max_in)
    fig.savefig(path, dpi=dpi, **kwargs)


def _diverging_colors(n, cmap_name='PRGn'):
    """
    Sample n colours from a diverging cmap, skipping both the washed-out
    white centre and the near-black extremes.
    Maps evenly into [0.15, 0.4] ∪ [0.6, 0.9].
    """
    cmap = plt.get_cmap(cmap_name)
    if n == 1:
        return [cmap(0.15)]
    t_vals = np.linspace(0, 0.55, n)   # total usable range = 0.25 + 0.30 = 0.55
    colors = []
    for t in t_vals:
        if t <= 0.25:
            colors.append(cmap(0.15 + t))   # [0.15, 0.4]
        else:
            colors.append(cmap(0.35 + t))   # [0.6, 0.9]
    return colors


def _deconv_colors(n):
    """
    Return n perceptually distinct colors for deconvolution plots.
    j=0 (all non-specific) is purple; j=n-1 (all specific) is green.
    """
    return _diverging_colors(n)


def plot_deconv_stacked(
    L_vals_out,
    contrib_stack,
    S,
    N,
    title_prefix,
    cfg,
    outline_totals=None,
    outline_err=None,
    outline_label="F_exp total"
):
    """
    Stacked bar plots for each apparent bound count i across ligand concentrations.
    Uses a shared legend and shared axis labels for cleaner layout.
    """
    base_fontsize = cfg.base_fontsize
    num_species = S + N + 1
    x = np.arange(len(L_vals_out))
    colors = _deconv_colors(S + 1)

    start_i = 1
    n_panels = num_species - start_i
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 2.6 * nrows),
        squeeze=False,
        sharex=True
    )

    for idx, i in enumerate(range(start_i, num_species)):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        max_j = min(i, S)
        bottom = np.zeros(len(L_vals_out))
        panel_handles = []
        for j in range(max_j + 1):
            bars = ax.bar(
                x,
                contrib_stack[:, i, j],
                bottom=bottom,
                width=0.8,
                color=colors[j],
                edgecolor='none',
            )
            panel_handles.append(
                Patch(facecolor=colors[j], edgecolor='none',
                      label=f"{j}S+{i - j}N"))
            bottom += contrib_stack[:, i, j]

        if outline_totals is not None:
            outline = ax.bar(
                x,
                outline_totals[:, i],
                width=0.8,
                facecolor='none',
                edgecolor='black',
                linewidth=0.8,
            )
            panel_handles.append(
                Patch(facecolor='none', edgecolor='black', label=outline_label))
            if outline_err is not None and outline_err.shape[1] > i:
                ax.errorbar(
                    x,
                    outline_totals[:, i],
                    yerr=outline_err[:, i],
                    fmt='none',
                    ecolor='black',
                    elinewidth=0.8,
                    capsize=2,
                    alpha=0.9
                )
        ax.legend(
            handles=panel_handles,
            fontsize=base_fontsize * 0.5,
            ncol=2,
            loc=cfg.deconv_legend_loc,
            frameon=False
        )
        ax.set_title(f"{title_prefix} {i} ligand binding", fontsize=base_fontsize * 0.7)
        ax.set_xticks(x)
        if r == nrows - 1:
            ax.set_xticklabels([f"{v:.3g}" for v in L_vals_out], rotation=45, ha='right', fontsize=base_fontsize * 0.6)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=base_fontsize * 0.6)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    for j in range(n_panels, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis('off')

    fig.supxlabel(f"Total Ligand Concentration ({cfg.output_unit})", fontsize=base_fontsize * 0.85)
    fig.supylabel("Fraction", fontsize=base_fontsize * 0.85)
    fig.tight_layout()
    return fig


def plot_fit_curves(L_totals_M, F_exps, L_grid_M, F_grid, num_species, title, output_svg, cfg, n_specific=None):
    """Plot model curves vs experimental scatter — used by both models."""
    colors = _diverging_colors(num_species)

    fig = plt.figure(figsize=(8, 6))
    for j in range(num_species):
        ls = '--' if (n_specific is not None and j > n_specific) else '-'
        plt.plot(L_grid_M * cfg.scale_m_to_out, F_grid[:, j], lw=2.5, color=colors[j],
                 linestyle=ls, label=f'P·L$_{{{j}}}$')
        if j < F_exps.shape[1]:
            plt.scatter(L_totals_M * cfg.scale_m_to_out, F_exps[:, j], s=60,
                        edgecolor='k', facecolor=colors[j], alpha=0.8)

    plt.xlabel(f'Total Ligand Concentration ({cfg.output_unit})', fontsize=cfg.base_fontsize * 1.0)
    plt.ylabel('Mole Fraction', fontsize=cfg.base_fontsize * 1.0)
    plt.title(title, fontsize=cfg.base_fontsize * 1.1)
    plt.xticks(fontsize=cfg.base_fontsize * 0.9)
    plt.yticks(fontsize=cfg.base_fontsize * 0.9)
    plt.legend(ncol=2, fontsize=cfg.base_fontsize * 0.85, title='Species')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if cfg.save_plots:
        safe_savefig(fig, output_svg, cfg.max_image_dim, bbox_inches='tight')
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_convergence(ssr_history, output_svg, cfg):
    """Plot SSR convergence trace — used by both models."""
    fig = plt.figure(figsize=(6, 4))
    plt.plot(ssr_history, marker='.', linestyle='-', markersize=8)
    plt.xlabel('Optimizer Function Call', fontsize=cfg.base_fontsize * 1.0)
    plt.ylabel('Sum of Squared Residuals (SSR)', fontsize=cfg.base_fontsize * 1.0)
    plt.yscale('log')
    plt.title('Convergence Trace', fontsize=cfg.base_fontsize * 1.1)
    plt.xticks(fontsize=cfg.base_fontsize * 0.9)
    plt.yticks(fontsize=cfg.base_fontsize * 0.9)
    plt.grid(True)
    plt.tight_layout()
    if cfg.save_plots:
        safe_savefig(fig, output_svg, cfg.max_image_dim)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_summary_fit(ref_L, F_exp_mean, F_exp_std, L_grid_M, F_calc_mean, F_calc_std, output_svg, num_species, cfg, n_specific=None):
    """Plot mean experimental vs mean model curves."""
    colors = _diverging_colors(num_species)

    fig = plt.figure(figsize=(8, 6))
    for j in range(num_species):
        ls = '--' if (n_specific is not None and j > n_specific) else '-'
        plt.plot(L_grid_M * cfg.scale_m_to_out, F_calc_mean[:, j], lw=2.5, color=colors[j],
                 linestyle=ls, label=f'P·L$_{{{j}}}$')
        if cfg.summary_show_calc_shade and F_calc_std is not None:
            lower = F_calc_mean[:, j] - F_calc_std[:, j]
            upper = F_calc_mean[:, j] + F_calc_std[:, j]
            plt.fill_between(
                L_grid_M * cfg.scale_m_to_out,
                lower,
                upper,
                color=colors[j],
                alpha=0.15,
                linewidth=0
            )
        if j < F_exp_mean.shape[1]:
            mask = ~np.isnan(F_exp_mean[:, j])
            if F_exp_std is not None and F_exp_std.shape[1] > j:
                plt.errorbar(
                    ref_L[mask] * cfg.scale_m_to_out,
                    F_exp_mean[mask, j],
                    yerr=F_exp_std[mask, j],
                    fmt='o',
                    color=colors[j],
                    ecolor=colors[j],
                    capsize=3,
                    alpha=0.8
                )
            else:
                plt.scatter(
                    ref_L[mask] * cfg.scale_m_to_out,
                    F_exp_mean[mask, j],
                    s=50,
                    edgecolor='k',
                    facecolor=colors[j],
                    alpha=0.8
                )

    plt.xlabel(f'Total Ligand Concentration ({cfg.output_unit})', fontsize=cfg.base_fontsize * 1.0)
    plt.ylabel('Mole Fraction', fontsize=cfg.base_fontsize * 1.0)
    plt.title('Summary Fit: Mean Experimental vs Mean Model', fontsize=cfg.base_fontsize * 1.1)
    plt.xticks(fontsize=cfg.base_fontsize * 0.9)
    plt.yticks(fontsize=cfg.base_fontsize * 0.9)
    plt.legend(ncol=2, fontsize=cfg.base_fontsize * 0.85, title='Species')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    safe_savefig(fig, output_svg, cfg.max_image_dim, bbox_inches='tight')
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)
