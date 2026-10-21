# MechStat-Guided-DDPM

Code for the paper: Deng Z, Dong L, Chen Z, Ning Z, Tang X. Probabilistic Battery Capacity Prediction for Real-World Vehicles via a Mechanism- and Statistical Feature-Driven Diffusion Model. IEEE Transactions on Transportation Electrification. DOI: [10.1109/TTE.2026.3746250].

If you use this code or the proposed method in your research, please cite this paper.

The implementation includes statistical and mechanistic feature extraction,
conditional DDPM training, prediction, and evaluation using 24 weeks of
historical vehicle data.

## Installation

Install the required packages in your Python environment:

```console
python -m pip install -r requirements.txt
```

PyTorch uses CUDA when available; otherwise the scripts use the CPU.

## Project structure

```text
MechStat-Guided-DDPM/
    DDPM.py
    diffusion_utilities_TS_deep.py
    statistical_features_extraction.py
    mechanistic_features_extraction.py
    README.md
    requirements.txt
    .gitignore
    data/
        raw/
            <20 vehicle CSV files>
    results/
        weights/
```

`diffusion_utilities_TS_deep.py` defines the model imported by `DDPM.py`.
Keep all four Python files in the same folder.

## Data preparation

Download the source data from
[Battery charging data of on-road electric vehicles](https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles).
Place the 20 vehicle CSV files directly in `data/raw/`, or set `RAW_DATA_DIR`
in `statistical_features_extraction.py` to the directory containing those CSVs.
Files are read in natural filename order, so vehicle 2 precedes vehicle 10.

When using the dataset, cite its original publication:

Deng Z, Xu L, Liu H, Hu X, Duan Z, Xu Y. Prognostics of battery capacity based
on charging data and data-driven methods for on-road vehicles. Applied Energy.
2023;339:120954. [DOI](https://doi.org/10.1016/j.apenergy.2023.120954).

## Configuration

Paths default to folders next to the scripts, so the project can be placed on
any drive. For cell-by-cell execution without `__file__`, set the working
directory to the project folder.

- Use the same `DATA_DIR` in both feature extraction scripts and `DDPM.py`.
- Set `RAW_DATA_DIR` in the statistical script to change the source CSV folder.
- Set `OUTPUT_DIR` in `DDPM.py` to change the prediction and checkpoint folder.
- Set `PLOT_FEATURES = True` in the extraction scripts to display diagnostic
  plots. These plots are disabled by default.

## Run order

```console
python statistical_features_extraction.py
python mechanistic_features_extraction.py
python DDPM.py
```

The first two scripts generate the processed inputs. The final command runs
cross-validation training, prediction, evaluation, and metric box plots.

## Generated files

| File | Purpose |
| --- | --- |
| `data/All_20vehs_data.pkl` | Charging segments and capacity estimates |
| `data/complete_point_features_base.pkl` | Full weekly capacity and statistical feature trajectories |
| `data/point_features_base.pkl` | Selected statistical features for the first 24 weeks |
| `data/point_features_rescombined_24weeks.pkl` | Fused point and resistance features used by DDPM |
| `data/ic_tensor_24weeks.pt` | IC input tensor used by DDPM |
| `results/weights/model_fold_<n>_best.pth` | Saved checkpoint for each fold |
| `results/24weeks_raw_predictions.pkl` | Predictions from all folds |
| `results/CNN_DDPM_results.pkl` | Predictions and evaluation metrics |

For the 20-vehicle dataset, each fused feature table has 24 rows and 17 columns:
one time column and 16 numerical inputs. The IC tensor has shape `(480, 3, 100)`:
20 vehicles, 24 weeks, three channels (`avg`, `max`, `min`), and 100 voltage points.
Full weekly capacity trajectories are retained as prediction targets.

The statistical script reuses `All_20vehs_data.pkl` when it exists. Move that
cache out of `DATA_DIR` before rerunning if you change the source CSV data.
Downloaded data and generated files are excluded from Git by `.gitignore`.
