#!/bin/bash
# NY Dataset - Neural Network with Exhaustive Projection
# This script generate 1 counterfactual using DiCE and projects it 
# using on exhaustive search on whole domain of the data.
echo "NY projection exhaustive experiment starts!"
echo "Check logs/adult_project_best_in_dataset.out for detailed output"
python -u projection_test.py \
    --cont_feat beds bath propertysqft \
    --fixed_feat type locality sublocality \
    --dataset_path data/datasets/nyhouse.csv \
    --constraints_path data/constraints/ny_dcs.txt \
    --k_lower 1 \
    --k_upper 2 \
    --epochs 100 \
    --exp_name ny_exhaust_search \
    --projection_mode exhaustive \
    > logs/ny_exhaust_search.out

echo "NY projection exhaustive experiment complete!"
echo "Results saved to: data/adult_project_best_in_dataset/"
echo "Check logs/adult_project_best_in_dataset.out for detailed output"