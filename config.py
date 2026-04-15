"""
RunConfig dataclass + YAML loader for multi-system, multi-temperature runs.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os

UNIT_MAP = {'M': 1.0, 'mM': 1e-3, 'uM': 1e-6, 'nM': 1e-9, 'pM': 1e-12}


@dataclass
class RunConfig:
    # --- Paths ---
    base_dir: str = ""
    out_dir: str = ""
    csv_name_wildcard: str = ""
    data_path: Optional[str] = None

    # --- Units ---
    input_unit: str = "M"
    output_unit: str = "uM"
    p_total_val: float = 1.0
    p_total_unit: str = "uM"

    # --- Model parameters ---
    s: int = 4
    s_mode: str = "auto"
    n_override: Optional[int] = None
    auto_adjust_s: bool = True
    min_species_frac: float = 0.01

    # --- Models to run (names from models.REGISTRY) ---
    models: List[str] = field(default_factory=lambda: ["specific_binding", "geometric_nonspecific"])

    # --- Deconvolution ---
    deconv_enable: bool = True
    deconv_source: str = "calc"
    deconv_use_grid: bool = False
    deconv_grid_points: int = 40
    deconv_csv_path: Optional[str] = None
    report_ligand_conc: List[float] = field(default_factory=lambda: [30])

    # --- Plot/Debug ---
    save_plots: bool = True
    show_plots: bool = False
    debug_validate: bool = True
    debug_index: int = 0
    debug_ligand_conc: Optional[float] = 30
    debug_i_index: int = 4
    deconv_legend_loc: str = "best"

    # --- Summary ---
    summary_enable: bool = True
    summary_show_calc_shade: bool = False

    # --- Rendering ---
    max_image_dim: int = 1800
    base_fontsize: int = 16

    # --- Derived (computed in __post_init__) ---
    scale_l_in_to_m: float = field(init=False, default=0.0)
    scale_p_in_to_m: float = field(init=False, default=0.0)
    scale_m_to_out: float = field(init=False, default=0.0)
    p_total_m: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.scale_l_in_to_m = UNIT_MAP[self.input_unit]
        self.scale_p_in_to_m = UNIT_MAP[self.p_total_unit]
        self.scale_m_to_out = 1.0 / UNIT_MAP[self.output_unit]
        self.p_total_m = self.p_total_val * self.scale_p_in_to_m


def load_configs(yaml_path: str) -> List[RunConfig]:
    """
    Parse a YAML config file with 'defaults' and 'systems' sections.
    Returns one RunConfig per (system, temperature) pair.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    systems = raw.get("systems", [])
    configs = []

    for sys_cfg in systems:
        temps = sys_cfg.pop("temperatures", [25])
        name = sys_cfg.pop("name", "unknown")
        base_dir = sys_cfg.pop("base_dir", "")
        wildcard_fmt = sys_cfg.pop("wildcard_fmt", "*.csv")
        out_fmt = sys_cfg.pop("out_fmt", "output_qS_{t}C")

        # Merge: defaults < system-level overrides
        merged = {**defaults, **sys_cfg}

        for t in temps:
            wildcard = wildcard_fmt.format(t=t)
            out_dir = os.path.join(base_dir, out_fmt.format(t=t))

            cfg_dict = {
                **merged,
                "base_dir": base_dir,
                "out_dir": out_dir,
                "csv_name_wildcard": wildcard,
            }

            # Only pass keys that RunConfig accepts
            valid_keys = {f.name for f in RunConfig.__dataclass_fields__.values() if f.init}
            filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
            configs.append(RunConfig(**filtered))

    return configs
