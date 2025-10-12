#!/bin/bash
# Adult Dataset - Neural Network CFs with Solver Projection
# This script generates k=5 counterfactuals using a neural network classifier
# and projects them using the SMT solver approach to satisfy denial constraints.
echo "Adult projection neural network solver experiment starts!"
python -u perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --exp_name adult_neural_solver_CFs \
    --mode hard \
    --solver_timeout 10000 \
    --num_samples 10 \
    --epochs 10 \
    > logs/adult_neural_solver_CFs.out


# Parameters explained:
# --cont_feat: Continuous features (age, education_num, hours_per_week)
# --fixed_feat: Immutable features that cannot change (age, race, sex)
# --dataset_path: Path to the Adult dataset CSV file
# --constraints_path: Path to denial constraints for Adult dataset
# --k_lower, --k_upper: Generate 5 counterfactuals per instance
# --model_state_dict_path: Path to save/load neural network model
# --exp_name: Name for experiment output directory (data/adult_neural_solver/)
# --mode: "hard" means strict constraint satisfaction (no violations allowed)
# --solver_timeout: SMT solver timeout in milliseconds (10 seconds)
# --delta: Diversity weight parameter (higher = more diverse CFs)
# --num_samples: Number of test instances to generate CFs for
# --epochs: Training epochs for neural network (omit if --load_model)

echo "Adult projection neural network solver experiment complete!"
echo "Results saved to: data/adult_neural_solver_CFs/"
echo "Check logs/adult_neural_solver_CFs.out for detailed output"