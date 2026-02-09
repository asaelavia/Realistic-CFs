# Realistic Counterfactual Explanations via Denial Constraints

This repository contains the implementation for generating realistic counterfactual explanations that adhere to Denial Constraints (DCs), as described in our paper. The core technique is a **perturb-and-project** framework that combines existing CF generation methods with SMT-solver-based projection onto the constraint-satisfying space.

## Overview

### Counterfactual Explanations with Denial Constraints

Given a classifier and an input instance, **counterfactual explanations (CFs)** identify minimal changes that flip the model's prediction, revealing influential features. However, existing CF methods often produce unrealistic explanations that violate domain constraints. Our approach ensures that generated CFs satisfy **Denial Constraints** — expressive logical constraints capturing data integrity rules.

**Key aspects:**
- Generates diverse, proximate, and realistic counterfactual explanations
- Ensures all CFs satisfy denial constraints with respect to the database
- Supports both unary and binary denial constraints
- Handles immutable attributes that cannot be modified

### Two Approaches for CF Generation

- **Perturb-and-Project (P&P)**: Generate CFs with existing methods (e.g., DiCE), then project them onto the constraint-satisfying space. Works with any black-box model (neural networks).
- **Linear Integrated**: Directly encode both classifier and constraints into the solver. Superior quality for linear classifiers (SVM).

### Core Subroutine: Tuple Projection

The projection step finds the nearest valid tuple that satisfies all denial constraints. This is the key subroutine enabling realism in CF generation.

**Projection methods:**
- **Single Solver (Preprocessing)**: Fastest per-instance runtime after preprocessing
- **Suspect Set (No Preprocessing)**: Zero upfront cost, filters constraint space on-the-fly
- **Exhaustive Search**: Baseline for small datasets
- **Best-in-Dataset**: Selects closest valid tuple from dataset

## Key Features

- **Realistic CFs**: Zero constraint violations across all generated counterfactuals
- **Comparable Quality**: Proximity and diversity within ~10% and ~3% of unconstrained baselines on most datasets
- **Solver Optimizations**: Up to 63× speedup over vanilla solver usage via preprocessing and suspect-set filtering
- **Diversity Optimization**: Explicit diversity constraints to avoid redundant projections
- **Comprehensive Metrics**: Proximity (MAD, L0, L1), diversity (DPP, pairwise, minimum), constraint violations

## Installation

### Using Conda (Recommended - Tested Configuration)
```bash
# 1. Create conda environment with Python 3.8.5
conda create -n projection python=3.8.5

# 2. Activate environment
conda activate projection

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

### Using Python Virtual Environment (Alternative - Not Fully Tested)

If you prefer to use venv instead of conda:
```bash
# 1. Ensure Python 3.8+ is installed
python --version

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt
```

**Note:** This repository has been tested with conda and Python 3.8.5 using the dependency versions listed in `requirements.txt`. While Python 3.11 with updated dependencies (numpy==1.23.3, scikit_learn==1.2.0) is expected to work, full compatibility is not guaranteed.

### Requirements

```
ipython==8.12.3
jsonschema==3.2.0
numpy==1.22.0
pandas==1.3.5
raiutils==0.4.1
scikit_learn==0.23.2
tensorflow==2.13.0
torch==2.0.1
tqdm==4.50.2
z3_solver==4.13.0.0
```

## Repository Structure

```
.
├── perturb_test.py          # Main script for CF generation with projection
├── projection_test.py       # Projection-only experiments
├── evaluate.py              # Evaluation metrics computation
├── perturb.py              # Core projection algorithms
├── eval.py                 # Evaluation utility functions
├── class_models.py         # Neural network models
├── scripts/                # Example scripts per dataset
│   ├── adult_*.sh
│   ├── ny_*.sh
│   ├── census_*.sh
│   └── tax_*.sh
├── data/
│   ├── datasets/           # CSV datasets
│   └── constraints/        # Denial constraint files
└── README.md
```

## Quick Start

### 1. Counterfactual Generation (Main Use Case)

Generate realistic counterfactuals for neural network models:

```bash
python perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 --k_upper 6 \
    --num_samples 10 \
    --exp_name adult_neural_cfs \
    --solver_timeout 10000
```

### 2. Linear Model Integrated Approach

For linear models with integrated constraint handling:

```bash
python perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 --k_upper 6 \
    --linear_model \
    --exp_name adult_linear_integrated \
    --solver_timeout 50000
```

### 3. Projection Only

Test projection algorithms in isolation (without CF generation):

```bash
python projection_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 1 --k_upper 2 \
    --num_samples 10 \
    --exp_name adult_projection_test \
    --projection_mode solver
```

## Parameter Reference

### Core Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--cont_feat` | list | Continuous feature names | `age education_num hours_per_week` |
| `--fixed_feat` | list | Immutable features (e.g., age, race, sex) | `age race sex` |
| `--dataset_path` | str | Path to CSV dataset | `data/datasets/adult.csv` |
| `--constraints_path` | str | Path to denial constraints file | `data/constraints/adult_dcs.txt` |
| `--k_lower` | int | Minimum number of CFs to generate (included) | `5` |
| `--k_upper` | int | Maximum number of CFs to generate (not included) | `6` |
| `--num_samples` | int | Number of test instances | `10` |
| `--exp_name` | str | Experiment name for output directory | `adult_experiment` |

### Model Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--epochs` | int | Training epochs for neural network | `10` |
| `--load_model` | flag | Load pre-trained neural model | `False` |
| `--linear_model` | flag | Use linear SVM classifier | `False` |
| `--linear_pandp` | flag | Use perturb-and-project with linear model | `False` |
| `--load_linear_model` | flag | Load pre-trained linear model | `False` |

### Solver Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--solver_timeout` | int | Solver timeout in milliseconds | `10000` |
| `--timeout` | int | Projection timeout in seconds | `1000` |
| `--distance` | str | Distance function for projection optimization | `MAD` |
| `--gamma` | float | Diversity constraint parameter (min mutable attributes differing by >1 MAD from previous projections) | `2` |
| `--delta` | float | Diversity weight parameter | `0.5` |
| `--fixed_flag` | flag | Reset solver cache between projections | `False` |

**Distance function options (`--distance`):**
- `MAD`: MAD-normalized distance (default, as used in the paper)
- `L0`: Number of changed attributes
- `Custom`: User-defined distance function

### Projection Mode

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `--projection_mode` | str | Projection algorithm | `solver`, `exhaustive`, `best_in_dataset` |

**Options:**
- `solver`: SMT solver-based projection (recommended)
- `exhaustive`: Exhaustive search (only for small datasets)
- `best_in_dataset`: Select closest tuple from dataset

## Dataset Format

### CSV Dataset Structure

```csv
age,education_num,hours_per_week,workclass,education,...,label
39,13,40,State-gov,Bachelors,...,0
50,13,13,Self-emp-not-inc,Bachelors,...,0
```

- Last column must be named `label` (0 or 1 for binary classification)
- Categorical features will be auto-detected
- Continuous features specified via `--cont_feat`

### Denial Constraints Format

Constraints use the format: `¬{condition1 ∧ condition2 ∧ ...}`

Example (`adult_dcs.txt`):

```
¬{ t0.relationship == "Wife" ∧ t0.sex != "Female"}
¬{ t0.education == t1.education ∧ t0.education_num != t1.education_num }
```

- `t0.` refers to the counterfactual instance
- `t1.` refers to database tuples (for binary constraints)
- Operators: `==`, `!=`, `<`, `<=`, `>`, `>=`

## Example Scripts

See the `scripts/` directory for complete example scripts for each dataset:

### Adult-Income Dataset

```bash
# Neural network Perturb and Project
bash scripts/adult_neural_solver_CFs.sh

# Neural network Perturb and Project solver per sample
bash scripts/adult_neural_solver_per_sample_CFs.sh

# Projection only with solver
bash scripts/adult_projection_solver.sh
```

### NY-Housing Dataset (Small Dataset)

```bash
# Linear model counterfactuals with model integrated
bash scripts/ny_linear_model_CFs.sh

# Domain exhaustive search projection
bash scripts/ny_projection_exhaustive.sh
```

### Census-Income Dataset (Large Dataset)

```bash
# Linear model counterfactuals with model integrated
bash scripts/census_linear_CFs.sh

# Linear model counterfactuals with model integrated, solver per sample
bash scripts/census_linear_CFs_solver_per_sample.sh
```

### Tax Dataset

```bash
# Linear model counterfactuals with model integrated
bash scripts/tax_linear_CFs.sh

# Linear model counterfactuals perturb and project
bash scripts/tax_linear_perturb_and_project_CFs.sh
```

## Output

Results are saved in `data/{exp_name}/`:

- `dice_cfs_sample{i}_k{k}.csv` - Original DiCE counterfactuals
- `solver_pandp_cfs_sample{i}_k{k}.csv` - Projected counterfactuals
- `solver_linear_cfs_sample{i}_k{k}.csv` - Linear integrated CFs
- `commandline_args.txt` - Command-line arguments used
- `ml_model_state_dict.pth` - Trained neural model weights
- `linear_model.pkl` - Trained SVM model

### Metrics Computed

**Proximity Metrics:**
- MAD-normalized distance
- L0 distance (sparsity)
- L1 distance

**Diversity Metrics:**
- DPP (Determinantal Point Process)
- Pairwise diversity
- Minimum distance diversity

**Constraint Violations:**
- Total violations per CF
- Unary constraint violations
- Tuple conflicts (binary constraints)

## Algorithm Variants

### CF Generation

| Variant | Use When | Advantages | How to Set |
|---------|----------|------------|------------|
| Perturb-and-Project | Neural networks | Works with any black-box model | Default (omit `--linear_model`) |
| Linear Integrated | Linear classifiers (SVM) | Superior quality, joint optimization | Add `--linear_model` flag |

### Projection Optimizations

| Variant | Use When | Advantages | How to Set |
|---------|----------|------------|------------|
| Single Solver (Preprocessing) | Multiple projections, can amortize cost | Fastest per-instance after preprocessing | Default mode |
| Suspect Set (No Preprocessing) | Few projections, zero upfront cost | Filters constraint space automatically | Filtering based on `--fixed_feat` |

## Troubleshooting

### Out of Memory Errors

For large datasets (Census), reduce batch size or use Suspect Set variant:

```bash
# Already uses Suspect Set automatically
python perturb_test.py --fixed_flag --dataset_path data/datasets/census.csv ...
```

### Solver Timeouts

Increase solver timeout for complex constraint spaces:

```bash
python perturb_test.py --solver_timeout 50000 ...  # 50 seconds
```

## Citation (TODO)

<!-- If you use this code, please cite our paper:

```bibtex
@inproceedings{yourpaper2026,
  title={Realistic Counterfactual Explanations via Denial Constraints},
  author={Your Name and Coauthors},
  booktitle={Conference Name},
  year={2026}
}
``` -->

## License (TODO)

<!-- [Your License Here] -->

## Contact (TODO)