#!/bin/bash
# Tax Dataset - Linear Model CFS Perturb-and-Project
echo "Tax linear model perturb and project experiment starts!"
echo "Check logs/tax_solver_linear_pandp.out for detailed output"
python -u perturb_test.py \
    --fixed_feat Genderstr \
    --cont_feat Salaryint SingleExempint MarriedExempint ChildExempint \
    --dataset_path data/datasets/tax.csv \
    --constraints_path data/constraints/tax_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --mode hard \
    --exp_name tax_solver_linear_pandp \
    --solver_timeout 10000 \
    --linear_pandp \
    > logs/tax_solver_linear_pandp.out

echo "Tax linear model perturb and project experiment complete!"
echo "Results saved to: data/tax_solver_linear_pandp/"
echo "Check logs/tax_solver_linear_pandp.out for detailed output"