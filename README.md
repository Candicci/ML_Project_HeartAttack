# CS-433 Project 1: Heart Disease Prediction (group CIM)

This project predicts coronary heart disease (MICHD) from the BRFSS health survey (328,135 labelled respondents, 321 features) with six linear models implemented from scratch in NumPy. All six are trained and tuned with the same preprocessing and evaluation protocol. Our submission uses logistic regression with a decision cutoff tuned on a validation set.

**AIcrowd result:** F1 = 0.416, accuracy = 0.860.

The report is [latex-example-paper/groupCIM-submission.pdf](latex-example-paper/groupCIM-submission.pdf).

## Repository structure

| Path | Content |
|---|---|
| `implementations.py` | The 6 required methods and their loss/gradient helpers (NumPy only) |
| `helpers.py` | `load_csv_data` and `create_csv_submission`, provided by the course; the loader was changed to use less memory (see step 1) |
| `run.ipynb` | Full pipeline: loading, preprocessing, training and sweeps, comparison, submission |
| `submission.csv` | The predictions we submitted to AIcrowd |
| `grading_tests/` | The course's public tests and their conda environment (`environment.yml`) |
| `latex-example-paper/` | The report (`groupCIM-submission.tex` / `.pdf`) and its figure |
| `data/` | Dataset; the CSV files are not tracked in git (see Setup) |

## Setup

**Environment.** Only the Python standard library, NumPy, matplotlib and seaborn are used, as the course rules require. The course's environment works:

```bash
conda env create --file=grading_tests/environment.yml --name=project1-grading
conda activate project1-grading
pip install ipykernel   # to run the notebook
```

Any environment with `numpy`, `matplotlib`, `seaborn` and `jupyter` works too. We also ran everything with Python 3.11 and NumPy 2.4.

**Data.** Put `x_train.csv`, `y_train.csv` and `x_test.csv` directly in `data/`. They come from the [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1) or from the zip in this repo:

```bash
unzip data/dataset_to_release.zip -d data && mv data/dataset/* data/ && rmdir data/dataset
```

## Reproducing the submission

Open `run.ipynb` and run all cells. It writes `submission.csv`, which is identical to our AIcrowd submission.

- Everything random (data split, SGD samples) is seeded with `SEED = 42`, so reruns give the same results.
- The first code cell loads the full data (`sub_sample=False`). Training on the full data takes tens of minutes on a laptop CPU, mostly the gradient-descent sweeps, and needs about 3 GB of RAM.
- For a quick check, set `sub_sample=True`. This uses every 50th row and the whole notebook runs in under a minute, but the numbers differ from the report.

## Pipeline

### 1. Loading
`helpers.load_csv_data` reads the CSV files with `np.loadtxt` and turns empty fields into `NaN`. The original version used `np.genfromtxt`, which needed more than 6 GB of RAM on `x_train.csv` and crashed the kernel; `np.loadtxt` needs about 1 GB. Labels are returned in {−1, 1}.

### 2. Train / validation / test split
The labelled data is shuffled (seed 42) and split 80/10/10:

| Split | Rows | Used for |
|---|---|---|
| train | 262,508 | fitting the weights and the preprocessing statistics |
| validation | 32,813 | choosing γ, λ and the decision cutoff, and selecting the final method |
| test | 32,814 | reporting results only, never used for any choice |

`x_test.csv` has no labels and is only used for the submission. We use this single hold-out split rather than k-fold cross-validation. Each split has about 33,000 rows, and validation and test scores agree closely (see Results).

### 3. Preprocessing
All statistics are computed on the train split only and then applied unchanged to every other split.
1. Drop columns that are constant or entirely missing in the train split. That's 6 columns on the full data.
2. Standardize each column: `(x − mean) / std`. This is required because some raw columns reach about 10⁹, which makes the sigmoid overflow.
3. Replace missing values (about 45% of all entries) with 0, which equals the train mean after standardizing.
4. Add a bias column of ones. The result has 316 columns.
5. Labels: {−1, 1} for the MSE methods; `(y + 1) / 2` gives {0, 1} for the logistic methods.

### 4. Models (`implementations.py`)

| Function | Method | Labels |
|---|---|---|
| `mean_squared_error_gd` | Linear regression, gradient descent on the MSE loss (with the ½ factor) | {−1, 1} |
| `mean_squared_error_sgd` | Linear regression, SGD with one random sample per step | {−1, 1} |
| `least_squares` | Normal equations XᵀX w = Xᵀy (`np.linalg.solve`) | {−1, 1} |
| `ridge_regression` | (XᵀX + 2Nλ I) w = Xᵀy | {−1, 1} |
| `logistic_regression` | Gradient descent on the negative log-likelihood | {0, 1} |
| `reg_logistic_regression` | Same, plus the penalty λ‖w‖² in the gradient | {0, 1} |

Every function returns `(w, loss)` for the last weight vector. The loss never includes the penalty term. The log-likelihood is computed with `np.logaddexp` so it can't overflow.

### 5. Training and hyperparameter sweeps
- All iterative methods start from `w0 = 0`.
- **Loss curves:** each method runs in chunks of iterations, and every chunk restarts from the previous `w`. This gives exactly the same weights as one long run and records the training loss after every epoch.
  - GD and logistic regression: 1 epoch = 1 full-gradient iteration.
  - SGD: 1 epoch = N single-sample steps.
- **Selection:** each hyperparameter is chosen by the lowest **validation** loss: MSE for the MSE methods, negative log-likelihood for the logistic ones.

| Method | Grid | Iterations | Chosen |
|---|---|---|---|
| MSE GD | γ ∈ {0.001, 0.003, 0.01, 0.03, 0.1} | 200 | γ = 0.03 (0.1 diverges) |
| MSE SGD | γ ∈ {1e-4, 3e-4, 1e-3, 3e-3} | 10 epochs | γ = 1e-4 (≥ 3e-4 diverges) |
| Least squares | none | closed form | none |
| Ridge | λ: 17 log-spaced values in [1e-8, 1] | closed form | λ = 3e-6 |
| Logistic | γ ∈ {0.01, 0.03, 0.1, 0.3, 1} | 200 | γ = 1 |
| Reg. logistic | λ ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}, with the best logistic γ | 200 | λ = 1e-5 |

### 6. Decision cutoff
Every model predicts +1 if `xᵀw ≥ t`. The default `t = 0` (a 50% risk for logistic regression) almost never predicts +1, because only 8.8% of the labels are positive. For each method, `tune_threshold` tries 200 quantiles of the validation scores `xᵀw` and keeps the cutoff with the highest validation F1.

The chosen cutoffs:
- **MSE models:** about −0.57.
- **Logistic regression:** −1.478, which means flagging anyone whose predicted risk is above about 19%.

### 7. Model selection and submission
The method with the highest validation F1 is chosen: logistic regression, with validation F1 0.417. Its weights and tuned cutoff produce the predictions on `x_test.csv`, which are mapped to {−1, 1} and written with `create_csv_submission`.

## Results (test split, 32,814 rows)

| Method | Accuracy | F1, cutoff 0 | F1, tuned cutoff |
|---|---|---|---|
| MSE GD | 0.860 | 0.004 | 0.400 |
| MSE SGD | 0.865 | 0.007 | 0.386 |
| Least squares | 0.864 | 0.017 | 0.411 |
| Ridge | 0.862 | 0.016 | 0.412 |
| **Logistic** | 0.858 | 0.183 | **0.413** |
| Reg. logistic | 0.851 | 0.183 | 0.412 |

Accuracy is measured at the tuned cutoff. Tuning the cutoff is by far the biggest improvement. After tuning, all six methods are within 0.03 F1 of each other, and train and test scores are nearly identical, so the models don't overfit. The report discusses these results.

## Tests and formatting

The course's public tests check the function signatures and outputs. Run them from `grading_tests/`:

```bash
cd grading_tests
pytest --github_link .. .
```

With a local path, `test_github_link_format` fails by design; all other tests for the six functions pass. Code is formatted with `black` (`black .`).

## Building the report

```bash
cd latex-example-paper
pdflatex groupCIM-submission
pdflatex groupCIM-submission
```

There are no references, so the `bibtex` step from the course instructions isn't needed.
