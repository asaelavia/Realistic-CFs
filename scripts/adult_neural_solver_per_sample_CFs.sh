#!/bin/bash
# Adult Dataset - Neural Network CFs with Solver Projection - Solver Per Sample
echo "Adult neural network solver experiment starts!"
python -u perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --exp_name adult_neural_solver_per_sample_CFS \
    --mode hard \
    --solver_timeout 10000 \
    --num_samples 10 \
    --epochs 10 \
    --fixed_flag \
    > logs/adult_neuraladult_neural_solver_per_sample_CFS_solver.out



echo "Adult neural network solver experiment complete!"
echo "Results saved to: data/adult_neural_solver_per_sample_CFS/"
echo "Check logs/adult_neural_solver_per_sample_CFS.out for detailed output"