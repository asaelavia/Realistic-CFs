#!/bin/bash
# Census Dataset - Linear Model CFS With Integrated Model Solver Per Sample
echo "Census linear model CFs solver per sample experiment starts!"
echo "Check logs/census_linear_solver_per_sample_CFS.out for detailed output"
python -u perturb_test.py \
    --cont_feat age wage_per_hour weeks_worked_in_year capital_gains capital_losses num_person_Worked_employer dividend_from_Stocks \
    --fixed_feat age hispanic_origin sex race marital_status country_self country_mother citizenship \
    --dataset_path data/datasets/census.csv \
    --constraints_path data/constraints/census_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --num_samples 11 \
    --epochs 2 \
    --exp_name census_linear_solver_per_sample_CFS \
    --solver_timeout 300000 \
    --timeout 200000 \
    --fixed_flag \
    --linear_model \
    > logs/census_linear_solver_per_sample_CFS.out

echo "Census linear model CFs solver per sample experiment complete!"
echo "Results saved to: data/census_linear_solver_per_sample_CFS/"
echo "Check logs/census_linear_solver_per_sample_CFS.out for detailed output"