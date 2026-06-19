"""Create curated pipeline notebooks from the organized project modules."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "pipelines"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


COMMON_SETUP = """
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / "src").exists():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("Project root:", PROJECT_ROOT)
"""


def main() -> None:
    write_notebook(
        "00_project_overview_and_paths.ipynb",
        [
            md(
                """
# 00 Project Overview And Paths

这个 notebook 用来确认项目结构、数据路径和输出路径。建议每次开始分析前先运行本 notebook。

核心目录：

- `src/ia_analysis`: 正式 Python 包。
- `notebooks/raw_20260618`: 原始 notebook 归档。
- `src/ia_analysis/notebook_pipelines/exports`: 原始 notebook 代码导出的 `.py` 脚本。
- `notebooks/pipelines`: 新的分主题 pipeline notebook。
- `configs/example_paths.json`: 路径配置示例。
"""
            ),
            code(COMMON_SETUP),
            code(
                """
import json

CONFIG_PATH = PROJECT_ROOT / "configs" / "example_paths.json"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

config
"""
            ),
            md(
                """
## Pipeline Order

推荐顺序：

1. 运行 global catalog pipeline，生成 subhalo 级 HDF5。
2. 运行 alignment figure pipeline，检查形状和潮汐对齐。
3. 运行 HOD 和 population pipeline，构造 LRG/ELG 与固定 number density 样本。
4. 运行 power spectrum pipeline，测量 $P(k)$。
5. 运行 correlation pipeline，测量 real-space IA 相关函数。
6. 运行 TNG dynamics pipeline，检查 shell-wise 和 layered shape-tide。
"""
            ),
        ],
    )

    write_notebook(
        "01_global_catalog_generation.ipynb",
        [
            md(
                """
# 01 Global Catalog Generation

本 notebook 负责 ClusterSims 和 TNG 的 subhalo 级总表生成。产物是 columnar HDF5，后续 alignment、HOD、$P(k)$ 和 correlation pipeline 都以它为输入。
"""
            ),
            code(COMMON_SETUP),
            code(
                """
from ia_analysis.pipelines import run_cs, run_tng

# 这里仅展示命令模板。实际大规模运行建议在 HPC job script 中执行。
cs_command = (
    "python run_cs.py "
    "--basepath /path/to/ClusterSims/L302_N1136_GR "
    "--snap 21 --nworker 16 --out outputs/global_cs_GR_s021.hdf5"
)

tng_command = (
    "python run_tng.py "
    "--nworker 16 --out outputs/global_tng_s099.hdf5 "
    "--api-key $TNG_API_KEY"
)

print(cs_command)
print(tng_command)
"""
            ),
            md(
                """
## Checks

生成后建议检查：

- HDF5 根目录是否有 `SubhaloID`, `GroupID`, `DM`, `Star`, `Tidal_grp`, `Tidal_tot`。
- `DM/I` 和 `Star/I` 是否为 $N \\times 3 \\times 3$。
- `Star/cos_err` 是否有限且满足后续筛选阈值。
"""
            ),
        ],
    )

    write_notebook(
        "02_alignment_figure_suite.ipynb",
        [
            md(
                """
# 02 Alignment Figure Suite

本 notebook 负责读取 MA 或 MArenew 对齐 catalog，调用 `arts_IA.py` 中的画图 API，生成形状、潮汐、速度和径向方向的对齐图。
"""
            ),
            code(COMMON_SETUP),
            code(
                """
from ia_analysis.visualization import arts_IA

arts_IA.set_paper_style()

# 修改为你的实际 MArenew.pkl 或 HDF5 catalog root。
MA_PATH = "MArenew.pkl"

# 如果文件存在，可以取消注释：
# MAset, flags, snap_list = arts_IA.load_marenew_pickle(MA_PATH)
# arts_IA.set_alignment_context(MAset, flags, arts_IA.ZMAP_ALL, snap_list=snap_list)
# arts_IA.list_alignment_chapters()
"""
            ),
            code(
                """
# 示例：画某一组 alignment。
# arts_IA.plot_alignment_pair("star_shape_dm_shape_major", save=True, show=True)

# 示例：按 chapter 批量画图。
# arts_IA.plot_alignment_chapter("shape_shape", save=True, show=False)
"""
            ),
            md(
                """
## Source Mapping

原始 notebook 中的画图和分析代码已经导出到 `src/ia_analysis/notebook_pipelines/exports`。优先使用 `ia_analysis.visualization.arts_IA` 作为正式 API，导出脚本作为追溯和补充。
"""
            ),
        ],
    )

    write_notebook(
        "03_hod_population_pipeline.ipynb",
        [
            md(
                """
# 03 HOD And Population Pipeline

本 notebook 负责 HOD、LRG/ELG、satellite radial distribution 和 merger population 相关分析。
"""
            ),
            code(COMMON_SETUP),
            code(
                """
# 相关原始 notebook 代码导出：
exports = PROJECT_ROOT / "src" / "ia_analysis" / "notebook_pipelines" / "exports"
for name in [
    "hod_lrg_elg_nb.py",
    "hod_data_nb.py",
    "hod_measure_lrg_elg_nb.py",
    "maset_satellite_radial_distribution_nb.py",
    "maset_satellite_radial_distribution_compare_nb.py",
    "merger_align_nb.py",
    "merger_stripping_nb.py",
]:
    print(exports / name)
"""
            ),
            md(
                """
## Suggested Pipeline

1. 从 global HDF5 或原始 FoF/Subhalo catalog 读取样本。
2. 按 stellar mass、SFR、central/satellite、host mass 分组。
3. 生成 HOD 表、satellite radial profiles 和 merger diagnostics。
4. 输出表格到 `outputs/tables`，输出图片到 `outputs/figures/hod_population`。
"""
            ),
        ],
    )

    write_notebook(
        "04_power_spectrum_pipeline.ipynb",
        [
            md(
                """
# 04 Power Spectrum Pipeline

本 notebook 负责 folded mesh、IA fields、matter density fields 和 $P(k)$ 测量。
"""
            ),
            code(COMMON_SETUP),
            code(
                """
from ia_analysis.spectra.ia_pk_cs import parse_pk_types, spec_keys_from_pk_types

pk_types = parse_pk_types("core")
spec_keys = spec_keys_from_pk_types(pk_types)

pk_types, spec_keys
"""
            ),
            code(
                """
# HPC 命令模板：
command = (
    "python ia_pk_cs.py --flag GR --snap 21 "
    "--threads 8 --nmesh 512 --folds 1,2,4,8,16,32 "
    "--pk-types full --outdir outputs/pks"
)
print(command)
"""
            ),
            md(
                """
## Outputs

主要输出是 `pks_FLAG_SNAP.hdf5`。每个 sample 下包含 folded spectra、stitched native spectra、stitched target-$k$ spectra 和 noise-corrected spectra。
"""
            ),
        ],
    )

    write_notebook(
        "05_correlation_pipeline.ipynb",
        [
            md(
                """
# 05 Correlation Pipeline

本 notebook 负责 real-space IA correlation，包括 jackknife covariance 和 mass-bin sample。
"""
            ),
            code(COMMON_SETUP),
            code(
                """
# 运行模板：
command = (
    "python ia_corr.py --flag GR --snap 21 "
    "--boxsize 205.0 --nsub 3 --nthreads 8 "
    "--out outputs/cfs/cfs_GR_s021.hdf5"
)
print(command)
"""
            ),
            md(
                """
## Notes

该 pipeline 依赖 `halotools`。如果只需要查看流程，可先阅读 `src/ia_analysis/notebook_pipelines/exports/ia_corr_nb.py` 和 `ia_corr_abundance_nb.py`。
"""
            ),
        ],
    )

    write_notebook(
        "06_tng_dynamics_layered_pipeline.ipynb",
        [
            md(
                """
# 06 TNG Dynamics And Layered Shape-Tide Pipeline

本 notebook 负责 TNG halo dynamics、cross-redshift tracking、layered ellipsoidal shells 和 tidal comparison。
"""
            ),
            code(COMMON_SETUP),
            code(
                """
from ia_analysis.dynamics import halo_dynamics

# 轻量检查：核心线性代数函数可以直接使用。
import numpy as np
M = np.eye(3)
halo_dynamics.eigh_sorted_desc(M)
"""
            ),
            code(
                """
# 大规模 layered TNG 命令模板：
command = (
    "python tng_layered_shape_tide.py "
    "--snap 99 --base /path/to/tng_data "
    "--out outputs/tng_layered_s099.hdf5"
)
print(command)
"""
            ),
            md(
                """
## Related Raw Notebooks

原始流程包括 `hd_tng_crossZ.ipynb`、`hd_tng_plot.ipynb`、`crossz.ipynb` 和 `TNGCatLoader.ipynb`。对应代码已导出到 notebook exports 目录。
"""
            ),
        ],
    )

    write_notebook(
        "07_orbits_and_shell_visualization.ipynb",
        [
            md(
                """
# 07 Orbits And Shell Visualization

本 notebook 负责 NFW orbit mock、radial shell plots、binding shell plots 和 3D visual diagnostics。
"""
            ),
            code(COMMON_SETUP),
            code(
                """
from ia_analysis.orbits.halo_maker import gen_nfw, transform_points_to_ellipsoid

pts = gen_nfw(size=128)
unrotated, rotated = transform_points_to_ellipsoid(
    pts,
    a=2.0,
    b=1.0,
    c=0.5,
    principal_axis=[1.0, 1.0, 0.2],
)

pts.shape, rotated.shape
"""
            ),
            md(
                """
## Visualization Modules

正式绘图工具位于 `ia_analysis.visualization`。`arts.py` 包含 shell visualization，`orbit_viz.py` 和 `orbit_viz2.py` 负责 orbit movie 和 preview frame。
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()

