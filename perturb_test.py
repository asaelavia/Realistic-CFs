import dice_ml
from perturb import *
from eval import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from dice_ml.utils.helpers import DataTransfomer
from torch.utils.data import TensorDataset, DataLoader
from class_models import Mlp, pretrain

if __name__ == '__main__':

    args = parse_arguments()

    if not os.path.exists(f'data/{args.exp_name}'):
        os.makedirs(f'data/{args.exp_name}')
    with open(f'data/{args.exp_name}/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    constraints, dic_cols, cons_function, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons = load_constraints(
        args.constraints_path)

    dice_runtimes = []
    df = pd.read_csv(args.dataset_path)
    for col in df.columns:
        if col in args.cont_feat + ['label']:
            continue
        df[col] = df[col].astype('object')
    y = df['label']
    train_dataset, test_dataset, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0, stratify=y)
    x_train = train_dataset.drop('label', axis=1, errors='ignore')
    x_test = test_dataset.drop('label', axis=1, errors='ignore')
    categorical_column_names = [col for col in x_train.columns if col not in args.cont_feat]
    norm_params, normalized_medians = create_normalization_params(x_train, args.cont_feat, categorical_column_names)
    # Initialize coefs_dic and intercept
    coefs_dic = None
    intercept = None
    category_mappings = {}

    if args.load_test:
        with open(f'data/{args.exp_name}/rows_to_run_{args.exp_name}.pkl', 'rb') as f:
            x_test = pickle.load(f)
    else:
        with open(f'data/{args.exp_name}/rows_to_run_{args.exp_name}.pkl', 'wb') as f:
            pickle.dump(x_test[y_test == 0][:10], f)
    # exit(0)
    dataset = x_train.copy()
    dataset_full = df.copy()
    x_test_int = x_test.copy()

    # Step 1: Convert dataset_full to category type FIRST (it has all categories)
    dataset_full[categorical_column_names] = dataset_full[categorical_column_names].astype('category')

    # Step 2: Extract the category ordering from dataset_full

    for col in categorical_column_names:
        # Get the categories from dataset_full (the complete set)
        categories = dataset_full[col].cat.categories
        category_mappings[col] = {v: k for k, v in enumerate(categories)}

        # Apply the SAME categories to all DataFrames
        dataset[col] = pd.Categorical(dataset[col], categories=categories)
        x_test_int[col] = pd.Categorical(x_test_int[col], categories=categories)

    # Step 3: Now convert to integer codes - all will use the same mapping
    dataset[categorical_column_names] = dataset[categorical_column_names].apply(lambda x: x.cat.codes)
    dataset_full[categorical_column_names] = dataset_full[categorical_column_names].apply(lambda x: x.cat.codes)
    x_test_int[categorical_column_names] = x_test_int[categorical_column_names].apply(lambda x: x.cat.codes)
    model_lin = None
    if args.linear_model or args.linear_pandp:
        model_lin = train_model(dataset, dataset, y_train, y_train, f'data/{args.exp_name}/linear_model.pkl',
                                preload=args.load_linear_model)
        coefs, intercept = model_lin.coef_[0], model_lin.intercept_[0]
        coefs_dic = {col: val for col, val in zip(x_train.columns, coefs)}

    d = dice_ml.Data(dataframe=df, continuous_features=args.cont_feat, outcome_name='label')
    with open(f'data/{args.exp_name}/transformer_{args.exp_name}.pkl', 'wb') as f:
        pickle.dump(d, f)
    if args.load_transformer:
        with open(f'data/{args.exp_name}/transformer_{args.exp_name}.pkl', 'rb') as f:
            d = pickle.load(f)
    transformer = DataTransfomer('ohe-min-max')
    transformer.feed_data_params(d)
    transformer.initialize_transform_func()

    df_train = df.drop('label', axis=1, errors='ignore')
    X_train = transformer.transform(x_train)
    DF_train = transformer.transform(df_train)
    model = Mlp(DF_train.shape[1], [100, 1])
    model.train()

    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train.values.astype(float)), torch.Tensor(y_train.values.astype(float))),
                              64, shuffle=True)
    print('Train started')
    if args.load_model:
        model.load_state_dict(torch.load(f'data/{args.exp_name}/ml_model_state_dict.pth'))
    else:
        model = pretrain(model, 'cpu', train_loader, lr=1e-4, epochs=args.epochs)
        torch.save(model.state_dict(), f'data/{args.exp_name}/ml_model_state_dict.pth')

    # Initialize explainer as None
    explainer = None

    if args.linear_pandp:
        df_for_dice = dataset_full.copy()  # dataset is already integerized from earlier
        df_for_dice['label'] = y.values

        # Initialize DiCE Data object
        # Even though cats are ints, we should still tell DiCE they're categorical
        d_linear = dice_ml.Data(
            dataframe=df_for_dice,
            continuous_features=args.cont_feat,  # Only truly continuous features
            outcome_name='label'
        )

        m_linear = dice_ml.Model(
            model=model_lin,
            backend='sklearn',
            model_type='classifier'
        )
        exp_random_lin = dice_ml.Dice(d_linear, m_linear, method="random")

    m = dice_ml.Model(model=model, backend='PYT', func="ohe-min-max")
    exp_random = dice_ml.Dice(d, m, method="gradient", constraints=False)
    model_labels = model(torch.Tensor(transformer.transform(x_test).values.astype(float))).round().detach().numpy().flatten()
    x_test = x_test[model_labels == 0]
    x_test_int = x_test_int[model_labels == 0]
    features_to_vary = list(df.columns)
    for feat in args.fixed_feat + ['label']:
        features_to_vary.remove(feat)
    print('Features to vary:', features_to_vary)
    if args.linear_pandp:
        # First run in DiCE to ensure everything is set up correctly
        exp_random_lin.generate_counterfactuals(
            x_test_int[0:1],
            total_CFs=2,
            desired_class=1,
            verbose=False,
            features_to_vary=features_to_vary,
        )

    exp_random.generate_counterfactuals(x_test[0:1], total_CFs=2, desired_class=1,
                                        verbose=False,
                                        features_to_vary=features_to_vary, min_iter=500, max_iter=600,
                                        learning_rate=6e-1, proximity_weight=args.delta / 100)

    bfs_runtimes = {'solver': {}}
    solver_kwargs = {
        'constraints': constraints,
        'dic_cols': dic_cols,
        'cons_function': cons_function,
        'cons_feat': cons_feat,
        'categorical_column_names': categorical_column_names,
        'coefs_dic': coefs_dic,
        'intercept': intercept,
        'unary_cons_lst': unary_cons_lst,
        'unary_cons_lst_single': unary_cons_lst_single,
        'bin_cons': bin_cons,
    }
    project_runtimes = {solver: [] for solver in
                        ['exhaustive', 'solver', 'solver_linear', 'best_in_dataset', 'solver_pandp']}
    projection_metrics = {solver: [] for solver in
                          ['exhaustive', 'solver', 'solver_linear', 'dice_linear', 'dice', 'best_in_dataset',
                           'solver_pandp']}
    perturb_metrics = {solver: [] for solver in
                       ['exhaustive', 'solver', 'solver_linear', 'dice_linear', 'dice', 'best_in_dataset',
                        'solver_pandp']}
    perturb_runtimes = {solver: {} for solver in
                        ['exhaustive', 'solver', 'solver_linear', 'dice_linear', 'dice', 'best_in_dataset',
                         'solver_pandp']}
    projection_config = ProjectionConfig(
        args=args,
        df=df,
        d=d,
        exp_random=exp_random,
        exp_random_lin=exp_random_lin if args.linear_pandp else None,
        transformer=transformer,
        category_mappings=category_mappings,
        mode='solver', projection_func=project_solver,
        project_runtimes=project_runtimes,
        projection_metrics=projection_metrics,
        norm_params=norm_params,
        normalized_medians=normalized_medians,
        model=model,
        model_lin=model_lin,
        **solver_kwargs
    )

    # Access results
    # for i, cfs in enumerate(results['counterfactuals']):
    #     if cfs is not None:
    #         print(f"Instance {i}: Generated {len(cfs)} counterfactuals")
    #         print(f"  Approximation quality: {results['approximation_quality'][i]['accuracy']:.2%}")
    prev = {}
    for k in range(args.k_lower, args.k_upper):

        print(f'K = {k}')
        dice_runtimes = {}
        dice_linear_runtimes = {}
        for i in range(args.num_samples):
            print(f'Sample: {i}')
            # dice_exp_random = None
            dice_cfs = None
            # query_instances = x_test[y_test == 0][i:i + 1]

            st = time.time()
            query_instances = x_test[i:i + 1]
            if args.linear_pandp:
                query_instances_int = x_test_int[i:i + 1]
                dice_exp_genetic = exp_random_lin.generate_counterfactuals(
                    query_instances_int,
                    total_CFs=k,
                    desired_class=1,
                    verbose=True,
                    features_to_vary=features_to_vary,
                    # proximity_weight=args.delta,
                    # diversity_weight=1.0  # Add this - genetic benefits from diversity tuning
                )
                dice_cfs_orig = convert_codes_to_categories_df(dice_exp_genetic.cf_examples_list[0].final_cfs_df_sparse,
                                                               category_mappings)
            else:
                dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=k, desired_class=1,
                                                                      verbose=True,
                                                                      features_to_vary=features_to_vary, max_iter=500,
                                                                      learning_rate=6e-1,
                                                                      proximity_weight=args.delta / 100)
                dice_cfs_orig = dice_exp_random.cf_examples_list[0].final_cfs_df_sparse
            dice_cfs_orig.to_csv(f'data/{args.exp_name}/dice_cfs_sample{i}_k{k}.csv', index=False)
            et = time.time()
            dice_cfs = [dice_cfs_orig.copy()]
            dice_dpp, dice_distance, l0, l1, pwise_div, min_dist_div = n_cfs_score(
                dice_cfs_orig.drop('label', axis=1, errors='ignore'), query_instances.iloc[0], projection_config)
            uncons, rows, cons_all = 0, 0, 0
            for _, cf in dice_cfs_orig.drop('label', axis=1, errors='ignore').iterrows():
                cons_res = cons_score(cf, projection_config)
                uncons += cons_res[0]
                rows += cons_res[1]
                cons_all += cons_res[2]
            perturb_metrics['dice'].append(
                (dice_distance, dice_dpp, uncons / len(dice_cfs_orig), rows / len(dice_cfs_orig),
                 cons_all / len(dice_cfs_orig), l0, l1, pwise_div, min_dist_div))

            elapsed_time = et - st
            perturb_runtimes['dice'][k] = perturb_runtimes['dice'].get(k, []) + [elapsed_time]

            mode = 'solver'
            projection_config.mode = mode

            if args.linear_model:
                projection_config.mode = 'solver_linear'
                # Prepare the query instance - need to integerize it like the training data
                query_instance_int = query_instances.copy()

                # Apply the same categorical encoding as training
                query_instance_int[categorical_column_names] = query_instance_int[categorical_column_names].astype(
                    'category')
                for col in categorical_column_names:
                    if col in query_instance_int.columns:
                        query_instance_int[col] = query_instance_int[col].map(category_mappings[col])

                # Create a combined dataframe with integerized features for DiCE
                df_for_dice = dataset.copy()  # dataset is already integerized from earlier
                df_for_dice['label'] = y_train.values

                # Initialize DiCE Data object
                # Even though cats are ints, we should still tell DiCE they're categorical
                d_linear = dice_ml.Data(
                    dataframe=df_for_dice,
                    continuous_features=args.cont_feat,  # Only truly continuous features
                    outcome_name='label'
                )

                # Wrap the linear SVC model for DiCE
                m_linear = dice_ml.Model(
                    model=model_lin,
                    backend='sklearn',
                    model_type='classifier'
                )

                # Create DiCE explainer for linear model
                exp_linear = dice_ml.Dice(
                    d_linear,
                    m_linear,
                    method="random"  # or "genetic" for linear models
                )

                # Generate counterfactuals using DiCE
                st_linear = time.time()
                dice_exp_linear = exp_linear.generate_counterfactuals(
                    query_instances=query_instance_int[0:1],
                    total_CFs=k,
                    desired_class=1,
                    features_to_vary=features_to_vary,
                    verbose=False
                )
                elapsed_time = time.time() - st_linear
                perturb_runtimes['dice_linear'][k] = perturb_runtimes['dice_linear'].get(k, []) + [elapsed_time]
                # perturb_runtimes['dice_linear'][k] = perturb_runtimes['dice_linear'].get(k, []) + [elapsed_time]
                # print(f'DiCE runtime for linear model: {perturb_runtimes['dice_linear'][k][-1]} seconds')
                # # Extract the counterfactuals
                linear_dice_cfs = dice_exp_linear.cf_examples_list[0].final_cfs_df_sparse
                linear_dice_cfs_orig = convert_codes_to_categories_df(linear_dice_cfs, category_mappings)
                # Add runtime tracking
                dice_lin_dpp, dice_lin_distance, l0, l1, pwise_div, min_dist_div = n_cfs_score(
                    linear_dice_cfs_orig.drop('label', axis=1, errors='ignore'), query_instances.iloc[0],
                    projection_config)
                uncons, rows, cons_all = 0, 0, 0
                for i, cf in linear_dice_cfs_orig.drop('label', axis=1, errors='ignore').iterrows():
                    cons_res = cons_score(cf, projection_config)
                    uncons += cons_res[0]
                    rows += cons_res[1]
                    cons_all += cons_res[2]
                perturb_metrics['dice_linear'].append(
                    (dice_lin_distance, dice_lin_dpp, uncons / len(linear_dice_cfs_orig),
                     rows / len(linear_dice_cfs_orig), cons_all / len(linear_dice_cfs_orig), l0, l1, pwise_div,
                     min_dist_div))
                linear_dice_cfs_orig.to_csv(f'data/{args.exp_name}/dice_linear_cfs_sample{i}_k{k}.csv', index=False)
                # Now project the DiCE counterfactuals with all required parameters
                st_linear = time.time()
                algorithm_cfs, lin_cat_cfs = lin_model_counterfactuals(query_instances.iloc[0],
                                                                       query_instance_int.iloc[0], k, projection_config)
                project_runtimes['solver_linear'].append(time.time() - st_linear)
                print(algorithm_cfs)
                lin_dpp, lin_distance, l0, l1, pwise_div, min_dist_div = n_cfs_score(
                    algorithm_cfs.drop('label', axis=1, errors='ignore'), query_instances.iloc[0], projection_config)
                print(f'Linear model DPP: {lin_dpp}, Distance: {lin_distance}')

                uncons, rows, cons_all = 0, 0, 0
                for i, cf in algorithm_cfs.drop('label', axis=1, errors='ignore').iterrows():
                    cons_res = cons_score(cf, projection_config)
                    uncons += cons_res[0]
                    rows += cons_res[1]
                    cons_all += cons_res[2]
                perturb_metrics['solver_linear'].append(
                    (lin_distance, lin_dpp, uncons / len(algorithm_cfs), rows / len(algorithm_cfs),
                     cons_all / len(algorithm_cfs), l0, l1, pwise_div, min_dist_div))
                algorithm_cfs.to_csv(f'data/{args.exp_name}/solver_linear_cfs_sample{i}_k{k}.csv', index=False)
                print(
                    f'Uncons: {uncons / len(algorithm_cfs)}, Rows: {rows / len(algorithm_cfs)}, Cons: {cons_all / len(algorithm_cfs)}')

            else:
                projection_config.projection_func = project_solver
                projection_config.mode = 'solver'
                st = time.time()
                algorithm_cfs_pandp = bfs_counterfactuals(dice_cfs, projection_config, k, **solver_kwargs)
                prev[i] = x_train[1:0]
                for _ in range(k):
                    row = dice_cfs[0].iloc[0].drop('label', errors='ignore')
                    best_dataset = best_from_dataset(algorithm_cfs_pandp, row, prev[i], projection_config, timeout=None)
                    prev[i] = best_dataset
                algorithm_cfs_pandp = prev[i]
                # perturb_runtimes['solver_pandp'][k].append(time.time() - st)
                perturb_runtimes['solver_pandp'][k] = perturb_runtimes['solver_pandp'].get(k, []) + [
                    time.time() - st + elapsed_time]
                print(f'Runtime for solver_pandp: {perturb_runtimes["solver_pandp"][k][-1]} seconds')
                dpp, distance, l0, l1, pwise_div, min_dist_div = n_cfs_score(
                    algorithm_cfs_pandp.drop('label', axis=1, errors='ignore'), query_instances.iloc[0],
                    projection_config)
                uncons, rows, cons_all = 0, 0, 0
                for _, cf in algorithm_cfs_pandp.drop('label', axis=1, errors='ignore').iterrows():
                    cons_res = cons_score(cf, projection_config)
                    uncons += cons_res[0]
                    rows += cons_res[1]
                    cons_all += cons_res[2]
                perturb_metrics['solver_pandp'].append(
                    (distance, dpp, uncons / len(dice_cfs_orig), rows / len(dice_cfs_orig),
                     cons_all / len(dice_cfs_orig), l0, l1, pwise_div, min_dist_div))
                print((distance, dpp, uncons / len(dice_cfs_orig), rows / len(dice_cfs_orig),
                       cons_all / len(dice_cfs_orig), l0, l1, pwise_div, min_dist_div))
                print(f'Final projected CFs from solver_pandp:')
                print(algorithm_cfs_pandp)
                algorithm_cfs_pandp.to_csv(f'data/{args.exp_name}/solver_pandp_cfs_sample{i}_k{k}.csv', index=False)
                if args.fixed_flag:
                    reset_solver_cache()
                else:
                    if args.gamma > 0:
                        reset_gamma_constraints()
                pass

        for mode in ['solver']:
            if project_runtimes[mode]:
                if mode == 'solver':
                    max_index = project_runtimes[mode].index(max(project_runtimes[mode]))
                    project_runtimes[mode].pop(max_index)

                print(f'Number of results for {mode}: {len(projection_metrics[mode])}')

                print(f'{mode} Mean runtimes: {np.mean(project_runtimes[mode])}')
                print(f'{mode} STD runtimes: {np.std(project_runtimes[mode])}')
                print(f'{mode} Max runtimes: {np.max(project_runtimes[mode])}')
                print(f'{mode} Min runtimes: {np.min(project_runtimes[mode])}')

                mean_distance = np.mean([metric[0] for metric in projection_metrics[mode]])
                std_distance = np.std([metric[0] for metric in projection_metrics[mode]])
                max_distance = np.max([metric[0] for metric in projection_metrics[mode]])
                min_distance = np.min([metric[0] for metric in projection_metrics[mode]])
                print(f'{mode} Mean projection distance: {mean_distance}')
                print(f'{mode} Mean STD distance: {std_distance}')
                print(f'{mode} Max projection distance: {max_distance}')
                print(f'{mode} Min projection distance: {min_distance}')

                mean_l0_distance = np.mean([metric[1] for metric in projection_metrics[mode]])
                std_l0_distance = np.std([metric[1] for metric in projection_metrics[mode]])
                max_l0_distance = np.max([metric[1] for metric in projection_metrics[mode]])
                min_l0_distance = np.min([metric[1] for metric in projection_metrics[mode]])
                print(f'{mode} Mean L0 projection distance: {mean_l0_distance}')
                print(f'{mode} Mean L0 STD distance: {std_l0_distance}')
                print(f'{mode} Max L0 projection distance: {max_l0_distance}')
                print(f'{mode} Min L0 projection distance: {min_l0_distance}')

                mean_l1_distance = np.mean([metric[2] for metric in projection_metrics[mode]])
                std_l1_distance = np.std([metric[2] for metric in projection_metrics[mode]])
                max_l1_distance = np.max([metric[2] for metric in projection_metrics[mode]])
                min_l1_distance = np.min([metric[2] for metric in projection_metrics[mode]])
                print(f'{mode} Mean L1 projection distance: {mean_l1_distance}')
                print(f'{mode} Mean L1 STD distance: {std_l1_distance}')
                print(f'{mode} Max L1 projection distance: {max_l1_distance}')
                print(f'{mode} Min L1 projection distance: {min_l1_distance}')

        if args.linear_model:
            print(f'solver_linear Mean projection runtimes: {np.mean(project_runtimes["solver_linear"])}')
            print(f'solver_linear STD projection runtimes: {np.std(project_runtimes["solver_linear"])}')
            print(f'solver_linear Max projection runtimes: {np.max(project_runtimes["solver_linear"])}')
            print(f'solver_linear Min projection runtimes: {np.min(project_runtimes["solver_linear"])}')
            print(f'Mean DICE linear runtimes: {np.mean(perturb_runtimes["dice_linear"][k])}')
            print(f'STD DICE linear runtimes: {np.std(perturb_runtimes["dice_linear"][k])}')
            print(f'Max DICE linear runtimes: {np.max(perturb_runtimes["dice_linear"][k])}')
            print(f'Min DICE linear runtimes: {np.min(perturb_runtimes["dice_linear"][k])}')

    for mode in ['dice', 'dice_linear', 'solver_linear', 'solver_pandp']:
        if len(perturb_runtimes[mode]) != 0:
            for k in perturb_runtimes[mode]:
                print(f'Mean K = {k} Mode {mode} runtimes: {np.mean(perturb_runtimes[mode][k][1:])}')
                print(f'STD K = {k} Mode {mode} runtimes: {np.std(perturb_runtimes[mode][k][1:])}')
                print(f'Max K = {k} Mode {mode} runtimes: {np.max(perturb_runtimes[mode][k][1:])}')
                print(f'Min K = {k} Mode {mode} runtimes: {np.min(perturb_runtimes[mode][k][1:])}')
        if perturb_metrics[mode]:
            print(f'{mode} Mean DPP: {np.mean([metric[1] for metric in perturb_metrics[mode]])}')
            print(f'{mode} STD DPP: {np.std([metric[1] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Max DPP: {np.max([metric[1] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Min DPP: {np.min([metric[1] for metric in perturb_metrics[mode]])}')

            print(f'{mode} Mean Pairwise Div: {np.mean([metric[7] for metric in perturb_metrics[mode]])}')
            print(f'{mode} STD Pairwise Div: {np.std([metric[7] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Max Pairwise Div: {np.max([metric[7] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Min Pairwise Div: {np.min([metric[7] for metric in perturb_metrics[mode]])}')

            print(f'{mode} Mean Minimal Dist Div: {np.mean([metric[8] for metric in perturb_metrics[mode]])}')
            print(f'{mode} STD Minimal Dist Div: {np.std([metric[8] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Max Minimal Dist Div: {np.max([metric[8] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Min Minimal Dist Div: {np.min([metric[8] for metric in perturb_metrics[mode]])}')

            print(f'{mode} Mean MAD Distance: {np.mean([metric[0] for metric in perturb_metrics[mode]])}')
            print(f'{mode} STD MAD Distance: {np.std([metric[0] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Max MAD Distance: {np.max([metric[0] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Min MAD Distance: {np.min([metric[0] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Mean L0 Distance: {np.mean([metric[5] for metric in perturb_metrics[mode]])}')
            print(f'{mode} STD L0 Distance: {np.std([metric[5] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Max L0 Distance: {np.max([metric[5] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Min L0 Distance: {np.min([metric[5] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Mean l1 Distance: {np.mean([metric[6] for metric in perturb_metrics[mode]])}')
            print(f'{mode} STD l1 Distance: {np.std([metric[6] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Max l1 Distance: {np.max([metric[6] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Min l1 Distance: {np.min([metric[6] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Mean Uncons: {np.mean([metric[2] for metric in perturb_metrics[mode]])}')
            print(f'{mode} STD Uncons: {np.std([metric[2] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Mean Rows: {np.mean([metric[3] for metric in perturb_metrics[mode]])}')
            print(f'{mode} STD Rows: {np.std([metric[3] for metric in perturb_metrics[mode]])}')
            print(f'{mode} Mean Cons: {np.mean([metric[4] for metric in perturb_metrics[mode]])}')
            print(f'{mode} STD Cons: {np.std([metric[4] for metric in perturb_metrics[mode]])}')

    metric_names = ['MAD Distance', 'DPP', 'Uncons', 'Rows', 'Cons', 'L0', 'L1', 'Pairwise Div', 'Min Dist Div']
    pairs = [('solver_pandp', 'dice'), ('solver_pandp', 'dice_linear'), ('solver_linear', 'dice_linear')]
    print('\n--- Wilcoxon Signed-Rank Tests ---')
    for m1, m2 in pairs:
        if not perturb_metrics[m1] or not perturb_metrics[m2]:
            continue
        n = min(len(perturb_metrics[m1]), len(perturb_metrics[m2]))
        print(f'\n{m1} vs {m2} (n={n}):')
        for idx, name in enumerate(metric_names):
            a = [x[idx] for x in perturb_metrics[m1][:n]]
            b = [x[idx] for x in perturb_metrics[m2][:n]]
            if len(set(np.array(a) - np.array(b))) <= 1:
                print(f'  {name}: skipped (all differences identical)')
                continue
            stat, p = wilcoxon(a, b)
            print(f'  {name}: stat={stat:.4f}, p={p:.4f}')
