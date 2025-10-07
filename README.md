# Clinical-ComBAT

Reference package for ComBAT harmonization of clinical MRI data. It ships the
ComBAT implementations for adapting clinical sites to a reference site
along with ready-to-run scripts to prepare datasets, fit a model, apply the
harmonization and analyze the outputs. While Clinical-ComBAT was designed and tested for the harmonization of diffusion MRI metrics (like fractional anisotropy, mean diffusivity, apparent fiber density) it can also be used on other type of data like volumetric data.

## References

- Girard, G., Edde, M., Dumais, F., et al. (2025). *Clinical-ComBAT: a diffusion MRI harmonization method for clinical normative modeling applications*.  (to be submitted).
- Jodoin, P.-M., Edde, M., Girard, G., et al. (2025). ComBAT harmonization for diffusion MRI: Challenges and best practices. (submitted). https://arxiv.org/abs/2505.14722
- Fortin, J.-P., Parker, D., Tun¸c, B., et al. (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170. https://doi.org/10.1016/j.neuroimage.2017.08.047

## Quick installation

```bash
# 1) create a Python >= 3.9 environment
python -m venv .venv
source .venv/bin/activate

# 2) install the core dependencies
pip install -r requirements.txt

# 3) install the library in editable mode
pip install --no-build-isolation --user -e .
```

The toolbox mainly depends on `numpy`, `pandas`, `matplotlib`, and `seaborn`.
All scripts accept compressed or uncompressed CSV files.

## Project layout

| Folder / file | Description |
| --- | --- |
| `clinical_combat/` | Python package (harmonization, utilities, visualization). |
| `scripts/` | Production-ready scripts to fit, apply, and visualize ComBAT. |
| `scripts/tests/` | Automated checks (for example `pytest scripts/tests/test_combat_quick.py`). |
| `scripts_dev/` | Research helpers and additional plotting utilities (optional for end-users). |
| `docs/data/` | Example datasets and sample figures. |
| `requirements.txt` | Minimal Python dependencies. |
| `setup.py` | Package metadata and script entry points. |

## Expected data format

Scripts expect CSV files containing at least the columns below:

```
sid,site,bundle,metric,mean,age,sex,handedness,disease
```

- `sid`: subject identifier
- `site`: site name (string)
- `bundle`: bundle or region name
- `metric`: diffusion metric (for example `md`, `fa`)
- `mean`: numeric value per bundle (mean, median, etc.)
- `age`, `sex`, `handedness`: covariates
  - use integer values (1 or 2) for `sex` and `handedness`; when a covariate is unknown, add the column filled with `1`and the scripts will disable that effect automatically
- `disease` acts as a flag; any row whose value is not `HC` is dropped before fitting the model

`docs/data/` contains fully fledged examples (`CamCAN.md.raw.csv.gz` and
`ShamCamCAN.md.raw.csv.gz`) illustrating the column layout
distribution.

## Choosing a ComBAT variant
The code supports two harmonization modes, namely clinic and pairwise. In both cases, the procedure harmonizes data from a moving site onto a reference site.
| Method | Description |
| --- | --- |
| `clinic` (default) | Harmonizes a moving site to a normative reference following the Clinical-ComBAT method (Girard et al., 2025). It fits site-specific polynomial covariate models, anchors variance with Bayesian priors suited to small cohorts, and auto-tunes the hyperparameters to keep the harmonized metrics consistent with the reference population. |
| `pairwise` | Adaptation of the original ComBAT (Fortin et al., 2017) that still fits both sites together but explicitly anchors the harmonization to a chosen reference site. For more details, see Jodoin et al. (2025), *ComBAT Harmonization for Diffusion MRI: Challenges and Best Practices* (arXiv:2505.14722). |

Common options for both methods:
- age filtering (`--limit_age_range`)
- covariate selection (`--ignore_sex`, `--ignore_handedness`)
- age effect polynomial order (`--degree`)

## Easy start

Run the bundled example once to check your setup:

```bash
# From the project root
python scripts/combat_quick.py \
    docs/data/CamCAN.md.raw.csv.gz \
    docs/data/ShamCamCAN.md.raw.csv.gz \
    --method clinic \
    --out_dir quickstart_demo/
```

This produces:
- a fitted model (`quickstart_demo/ShamCamCAN-CamCAN.md.clinic.model.csv`)
- harmonized data (`quickstart_demo/ShamCamCAN.md.clinic.csv.gz`)
- QC metrics and figures inside `quickstart_demo/`

## Main scripts

### Combined workflow

`combat_quick.py` runs the full pipeline in sequence (fit → apply → QC → figures) and logs
each spawned command.

- `ref_data` *(required)*: reference-site CSV (`*.raw.csv[.gz]`).
- `mov_data` *(required)*: moving-site CSV to harmonize.
- `--method {clinic,pairwise}` (default `clinic`): harmonization strategy.
- `--degree` (default 2 for clinic, 1 for pairwise when omitted): polynomial degree for age.
- `--limit_age_range` (default disabled): drop reference subjects outside the moving age range.
- `--ignore_sex` (default disabled): remove sex from the covariate model.
- `--ignore_handedness` (default disabled): remove handedness from the model.
- `--no_empirical_bayes` (default disabled): skip empirical Bayes estimation.
- `--robust` (default disabled, not implemented): placeholder for robust mode.
- `--regul_ref` *(clinic only, default 0)*: ridge penalty applied to reference regression.
- `--regul_mov` *(clinic only, default -1; pairwise falls back to 0)*: moving-site penalty or auto-tuning.
- `--nu` *(clinic only, default 5)*: variance hyperparameter for the moving site.
- `--tau` *(clinic only, default 2)*: covariate hyperparameter for the moving site.
- `--bundles` (default `mni_IIT_mask_skeletonFA` in plots): bundle subset for figures (`all` for every bundle).
- `--degree_qc` (default 0): QC model degree override (0 reuses the harmonization degree).
- `--out_dir` (default `./`): root directory for models, results, and figures.
- `--output_model_filename` (default auto-generated): custom name for the saved model.
- `--output_results_filename` (default auto-generated): custom name for the harmonized CSV.
- `--verbose/-v` (default `WARNING`): logging verbosity (`INFO` with `-v`, `DEBUG` with `-v DEBUG`).
- `--overwrite/-f` (default disabled): allow overwriting existing files.

Example:

```bash
python scripts/combat_quick.py docs/data/CamCAN.md.raw.csv.gz \
    docs/data/ShamCamCAN.md.raw.csv.gz \
    --method clinic \
    --out_dir results/clinic_pipeline/
```

### Model fitting

`combat_quick_fit.py` estimates harmonization parameters and writes a `*.model.csv`.

- `ref_data` *(required)*: reference-site CSV.
- `mov_data` *(required)*: moving-site CSV.
- `--method {clinic,pairwise}` (default `clinic`): harmonization variant.
- `--degree` (default 2 for clinic, 1 for pairwise when omitted): polynomial age order.
- `--limit_age_range` (default disabled): match reference ages to the moving-site range.
- `--ignore_sex` (default disabled): drop sex from the design matrix.
- `--ignore_handedness` (default disabled): drop handedness from the design matrix.
- `--no_empirical_bayes` (default disabled): rely on classical estimates for alpha/sigma.
- `--ignore_bundles` (default `left_ventricle right_ventricle`): bundles removed prior to fitting.
- `--regul_ref` *(clinic only, default 0)*: ridge penalty on the reference regression.
- `--regul_mov` *(clinic only, default -1; pairwise falls back to 0)*: moving-site penalty or auto-tuning.
- `--nu` *(clinic only, default 5)*: variance hyperparameter for the moving site.
- `--tau` *(clinic only, default 2)*: covariate hyperparameter for the moving site.
- `--out_dir` (default `./`): directory for the generated model.
- `--output_model_filename` (default auto-generated): custom model filename.
- `--verbose/-v` (default `WARNING`): logging verbosity.
- `--overwrite/-f` (default disabled): authorize overwriting existing files.

Example:

```bash
python scripts/combat_quick_fit.py docs/data/CamCAN.md.raw.csv.gz \
    docs/data/ShamCamCAN.md.raw.csv.gz \
    --method pairwise \
    --out_dir models/pairwise/
```

### Model application

`combat_quick_apply.py` consumes a moving-site CSV and a saved `*.model.csv`, then produces
harmonized measurements (`site.metric.method.csv.gz` by default).

- `mov_data` *(required)*: moving-site CSV to transform.
- `model` *(required)*: harmonization model generated by `combat_quick_fit.py` or `combat_quick.py`.
- `--out_dir` (default `./`): directory for the harmonized output.
- `--output_results_filename` (default auto-generated): custom output filename.
- `--verbose/-v` (default `WARNING`): logging level.
- `--overwrite/-f` (default disabled): allow overwriting.

Example:

```bash
python scripts/combat_quick_apply.py docs/data/ShamCamCAN.md.raw.csv.gz \
    models/pairwise/ShamCamCAN-CamCAN.md.pairwise.model.csv \
    --out_dir harmonized/pairwise/
```

### Evaluation and quality control (QC) to assess the alignment of the harmonized population.

- `combat_quick_QC.py`: reports Bhattacharyya distances between reference and moving datasets.
  - `ref_data` *(required)*: reference-site CSV (HC subjects only are used).
  - `mov_data` *(required)*: moving-site CSV.
  - `model` *(required)*: harmonization model (`*.model.csv`).
  - `--degree_qc` (default 0): QC polynomial degree (0 reuses the model degree).
  - `--ignore_bundles` (default `left_ventricle right_ventricle`): bundles to drop.
  - `--print_only` (default disabled): skip writing the distance file.
  - `--out_dir` (default `./`): directory for QC outputs.
  - `--output_results_filename` (default auto-generated): custom QC filename.
  - `--verbose/-v` (default `WARNING`), `--overwrite/-f` (default disabled).
  - Example:
    ```bash
    python scripts/combat_quick_QC.py docs/data/CamCAN.md.raw.csv.gz \
        docs/data/ShamCamCAN.md.raw.csv.gz \
        models/pairwise/ShamCamCAN-CamCAN.md.pairwise.model.csv \
        --out_dir qc_reports/
    ```

### Visualization

Common helper flags: each script accepts `-v/--verbose` (default `WARNING`) and `-f/--overwrite`
(default disabled).

- `combat_visualize_data.py`: scatterplots for raw or harmonized datasets.
  - `in_files` *(required, one or more)*: CSV files to display (reference first for legend clarity).
  - `--bundles` (default `mni_IIT_mask_skeletonFA`; use `all` for everything): bundles drawn.
  - `--display_marginal_hist` (default disabled): add marginal histograms.
  - `--hide_disease` (default disabled): remove non-HC subjects.
  - `--out_dir` (default `./`), `--outname` (default none), `--add_suffix` (default none): figure export controls.
  - `--fixed_ylim` (default auto): clamp Y axis to provided `[min max]`.
  - `--xlim` (default `20 90`): X-axis age range.
  - `--no_background` (default disabled): export without background styling.
  - Example:
    ```bash
    python scripts/combat_visualize_data.py docs/data/CamCAN.md.raw.csv.gz \
        docs/data/ShamCamCAN.md.raw.csv.gz \
        harmonized/pairwise/ShamCamCAN.md.pairwise.csv.gz \
        --bundles mni_AF_L mni_AF_R \
        --out_dir figures/data/
    ```

- `combat_visualize_model.py`: overlays regression models with data.
  - `in_reference` *(required)*: reference raw CSV.
  - `in_moving` *(required)*: moving raw CSV.
  - `in_model` *(required)*: harmonization model CSV.
  - `--bundles` (default `mni_IIT_mask_skeletonFA`; `all` for everything).
  - `--hide_disease` (default disabled): remove non-HC rows.
  - `--display_marginal_hist` (default disabled): add marginal histograms.
  - `--out_dir` (default `./`), `--outname` (default none), `--add_suffix` (default none).
  - `--fixed_color` (default palette-driven): manually set reference/moving colors.
  - `--lightness` (default 1.0): scale palette brightness.
  - `--only_models` (default disabled): hide scatter data and show regression lines only.
  - `--line_width` (default 2.5): width of regression lines.
  - `--fixed_ylim` (default auto) and `--xlim` (default `20 90`): axis limits.
  - `--no_background` (default disabled): export without background styling.
  - Example:
    ```bash
    python scripts/combat_visualize_model.py docs/data/CamCAN.md.raw.csv.gz \
        docs/data/ShamCamCAN.md.raw.csv.gz \
        models/pairwise/ShamCamCAN-CamCAN.md.pairwise.model.csv \
        --out_dir figures/model/ \
        --only_models
    ```

- `combat_visualize_harmonization.py`: age curves before/after harmonization.
  - `in_reference` *(required)*: reference raw CSV.
  - `in_movings` *(required, two or more)*: moving raw CSV plus harmonized CSV (order matters).
  - `--out_dir` (default `./`), `--outname` (default none), `--add_suffix` (default none).
  - `--bundles` (default `mni_IIT_mask_skeletonFA`; `all` allowed), `--ages` (default `20 90`).
  - `--sexes`, `--handednesses`, `--diseases` (default all values present): cohort filters.
  - `--hide_disease` (default disabled): remove disease rows entirely.
  - `--display_point` (default disabled): scatter representation for moving site.
  - `--display_marginal_hist` (default disabled): add marginal histograms.
  - `--hide_percentiles` (default disabled): swap percentile bands for SD bands.
  - `--window_size` (default 20), `--window_count` (default 10), `--no_dynamic_window` (default disabled): sliding window controls.
  - `--min_subject_per_site` (default 10): minimum subjects per site retained.
  - `--randomize_line` (default disabled) or `--line_style` (default dashed): adjust moving-line style.
  - `--increase_ylim` (default 5): percentage padding on the Y axis when not fixed.
  - `--fixed_ylim` (default auto): clamp Y axis to specified bounds.
  - `--y_axis_percentile` (default `1 99`): percentile range used for automatic Y limits.
  - `--percentiles` (default `5 25 50 75 95`): percentile bands drawn.
  - `--line_widths` (default `0.25 1 2 1 0.25`): line widths for percentile envelopes.
  - `--display_errors` (default disabled) & `--error_metric {uncertainty,bounds}` (default `uncertainty`): plot error bars for single-subject harmonization outputs.
  - Example:
    ```bash
    python scripts/combat_visualize_harmonization.py docs/data/CamCAN.md.raw.csv.gz \
        docs/data/ShamCamCAN.md.raw.csv.gz \
        harmonized/pairwise/ShamCamCAN.md.pairwise.csv.gz \
        --bundles all \
        --out_dir figures/harmonization/
    ```

### Dataset inspection

- `combat_info.py`: prints population statistics for a single CSV.
  - `in_file` *(required)*: dataset summarised. No optional switches.
  - Example:
    ```bash
    python scripts/combat_info.py docs/data/CamCAN.md.raw.csv.gz
    ```

## Typical pipeline

1. **Inspect the datasets**
   ```bash
   python scripts/combat_info.py docs/data/CamCAN.md.raw.csv.gz
   ```
2. **Fit a harmonization model**
   ```bash
   python scripts/combat_quick_fit.py \
       docs/data/CamCAN.md.raw.csv.gz \
       docs/data/ShamCamCAN.md.raw.csv.gz \
       --method clinic \
       --out_dir out/models/
   ```
3. **Apply the harmonization**
   ```bash
   python scripts/combat_quick_apply.py \
       docs/data/ShamCamCAN.md.raw.csv.gz \
       out/models/ShamCamCAN-CamCAN.md.clinic.model.csv \
       --out_dir out/harmonized/
   ```
4. **Quality control**
   ```bash
   python scripts/combat_quick_QC.py \
       docs/data/CamCAN.md.raw.csv.gz \
       docs/data/ShamCamCAN.md.raw.csv.gz \
       out/models/ShamCamCAN-CamCAN.md.clinic.model.csv
   ```
5. **Visualize the results**
   ```bash
   python scripts/combat_visualize_harmonization.py \
       docs/data/CamCAN.md.raw.csv.gz \
       docs/data/ShamCamCAN.md.raw.csv.gz \
       out/harmonized/ShamCamCAN.md.clinic.csv.gz \
       --out_dir out/figures/
   ```

`combat_quick.py` can execute steps 2 through 5 in sequence and logs each
invoked command.
