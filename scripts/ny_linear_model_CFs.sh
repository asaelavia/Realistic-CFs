#!/bin/bash
# NY Dataset - Linear Model CFS with Integrated Model
# This script generate 5 CFs using SMT Solver
echo "NY linear model CFs experiment starts!"
echo "Check logs/ny_linear_cfs.out for detailed output"
python -u perturb_test.py \
    --cont_feat beds bath propertysqft \
    --fixed_feat type locality sublocality \
    --dataset_path data/datasets/nyhouse.csv \
    --constraints_path data/constraints/ny_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --epochs 100 \
    --linear_model \
    --exp_name ny_linear_cfs \
    > logs/ny_linear_cfs.out

echo "NY linear model CFs experiment complete!"
echo "Results saved to: data/ny_linear_cfs/"
echo "Check logs/ny_linear_cfs.out for detailed output"