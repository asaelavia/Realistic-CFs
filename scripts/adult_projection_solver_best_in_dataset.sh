#!/bin/bash
# Adult Dataset - Neural Network with Best-in-Dataset Projection
# This script generate 1 counterfactual using DiCE and projects it 
# using the closest tuple from the dataset.
echo "Adult projection best in dataset experiment starts!"
python -u projection_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 1 \
    --k_upper 2 \
    --exp_name adult_project_best_in_dataset \
    --timeout 50000 \
    --projection_mode best_in_dataset \
    > logs/adult_project_best_in_dataset.out
# Parameters explained:
# --cont_feat: Continuous features (age, education_num, hours_per_week)
# --fixed_feat: Immutable features that cannot change (age, race, sex)
# --dataset_path: Path to the Adult dataset CSV file
# --constraints_path: Path to denial constraints for Adult dataset
# --k_lower, --k_upper: Generate 1 counterfactuals per instance
# --exp_name: Name for experiment output directory (data/adult_neural_solver/)
# --solver_timeout: SMT solver timeout in milliseconds (10 seconds)
# --num_samples: Number of test instances to generate CFs for
# --epochs: Training epochs for neural network (omit if --load_model)

echo "Adult projection best in dataset experiment complete!"
echo "Results saved to: data/adult_project_best_in_dataset/"
echo "Check logs/adult_project_best_in_dataset.out for detailed output"