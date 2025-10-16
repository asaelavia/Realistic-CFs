#!/bin/bash
# Tax Dataset - Linear Model CFS With Integrated Model
echo "Tax pertrub linear model CFs experiment starts!"
echo "Check logs/tax_linear_CFs.out for detailed output"
python -u perturb_test.py \
    --fixed_feat Genderstr \
    --cont_feat Salaryint SingleExempint MarriedExempint ChildExempint \
    --dataset_path data/datasets/tax.csv \
    --constraints_path data/constraints/tax_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --mode hard \
    --exp_name tax_linear_CFs \
    --solver_timeout 10000 \
    --linear_model \
    > logs/tax_linear_CFs.out

echo "Tax pertrub linear model CFs experiment complete!"
echo "Results saved to: data/tax_linear_CFs/"
echo "Check logs/tax_linear_CFs.out for detailed output"