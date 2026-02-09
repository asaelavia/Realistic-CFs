import argparse
import operator
import bisect
import json
import numbers
import pickle
import random
import signal
import time
import os
import copy
import re
import pandas as pd
import numpy as np
import torch
import itertools
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from z3 import *
from eval import *
from solver_distance import build_distance_for_solver, calculate_custom_distance

ops = {'==': operator.eq, '!=': operator.ne, '<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge}
op_bound = {'==': 'point', '!=': 'not_point', '<': 'right_exclusive', '<=': 'right_inclusive', '>': 'left_exclusive',
            '>=': 'left_inclusive'}
op_rev_bound = {'==': 'point', '!=': 'not_point', '<': 'left_exclusive', '<=': 'left_inclusive', '>': 'right_exclusive',
                '>=': 'right_inclusive'}
event_type = {'left_exclusive': ('increase', 1), 'left_inclusive': ('increase', 0), 'right_exclusive': ('decrease', 1),
              'right_inclusive': ('decrease', 0), 'not_point': ('not_point', 0), 'point': ('point', 0)}
_solver_cache = {}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Counterfactual generation with configurable parameters')
    parser.add_argument('--fixed_feat', nargs='+', default=[], help='Fixed features')
    parser.add_argument('--cont_feat', nargs='+', default=['age', 'education_num', 'hours_per_week'],
                        help='Continuous features')
    parser.add_argument('--dataset_path', type=str, default='data/adult_clean.csv', help='Path to dataset file')
    parser.add_argument('--constraints_path', type=str, default='data/adult_good_adcs_test.txt',
                        help='Path to constraints file')
    parser.add_argument('--num_samples', type=int, default=11, help='Number of samples to generate counterfactuals for')
    parser.add_argument('--k_lower', type=int, default=3, help='Lower range of number of counterfactuals to generate')
    parser.add_argument('--k_upper', type=int, default=4, help='Upper range of number of counterfactuals to generate')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs to train model')
    parser.add_argument('--exp_name', type=str, default='adult_test', help='Name of the dataset')
    parser.add_argument('--mode', type=str, default='hard', help='Soft or Hard projection')
    parser.add_argument('--gamma', type=float, default=0, help='Diversity constraint parameter')
    parser.add_argument('--delta', type=float, default=50, help='Proximity weight parameter')
    parser.add_argument('--timeout', type=int, default=1000, help='Timeout for projection function in seconds')
    parser.add_argument('--solver_timeout', type=int, default=10000, help='Timeout for projection function in seconds')
    parser.add_argument('--load_model', action='store_true', default=False, help='Whether to load a pre-trained model')
    parser.add_argument('--load_test', action='store_true', default=False, help='Whether to load test samples')
    parser.add_argument('--linear_model', action='store_true', default=False,
                        help='Whether to use a binary linear model')
    parser.add_argument('--linear_pandp', action='store_true', default=False,
                        help='Whether to use perturb and project with binary linear model')
    parser.add_argument('--load_transformer', action='store_true', help='Whether to load a pre-trained transformer')
    parser.add_argument('--load_linear_model', action='store_true', help='Whether to load a pre-trained linear model')
    parser.add_argument('--distance', type=str, default='MAD', help='Projection Distance Function')
    parser.add_argument('--fixed_flag', action='store_true', help='Whether to load a pre-trained linear model')
    parser.add_argument('--projection_mode', type=str, default='solver',
                        choices=['solver', 'exhaustive', 'best_in_dataset'],
                        help='Projection mode to use: solver, exhaustive, or best_in_dataset')
    return parser.parse_args()


def reset_solver_cache():
    """Reset the solver cache to empty."""
    _solver_cache.clear()


def reset_gamma_constraints():
    for cache_key in list(_solver_cache.keys()):
        s, vars, column_types = _solver_cache[cache_key]
        s.pop()  # Pop Level 1 (removes gamma constraints)
        s.push()  # Push new Level 1 (fresh gamma level)
    print(f'Reset gamma constraints for {len(_solver_cache)} cached solvers')


def create_normalization_params(x_train, non_cat_cols, cat_cols):
    """
    Create normalization parameters and normalized medians for continuous features.

    Args:
        x_train: Training dataset (without label column)
        non_cat_cols: List of continuous column names
        cat_cols: List of categorical column names

    Returns:
        norm_params: Dictionary with min and range for each continuous column
        normalized_medians: Dictionary with MAD for each normalized continuous column
    """

    # Initialize dictionaries
    norm_params = {}
    normalized_medians = {}

    # Process continuous columns
    for col in non_cat_cols:
        # Calculate min-max normalization parameters
        col_min = x_train[col].min()
        col_max = x_train[col].max()
        col_range = col_max - col_min

        # Handle constant columns (zero range)
        if col_range <= 1e-8:
            col_range = 1

        norm_params[col] = {
            'min': col_min,
            'range': col_range
        }

        # Normalize the training data for this column
        normalized_col = (x_train[col] - col_min) / col_range
        median = normalized_col.median()
        mad = np.median(np.abs(normalized_col - median))

        # Store the MAD (median absolute deviation)
        normalized_medians[col] = mad

    return norm_params, normalized_medians


def convert_codes_to_categories(row, category_mappings):
    """
    Convert category codes back to their original string values.

    Args:
        row: pandas Series or dict containing the data with category codes
        category_mappings: dictionary mapping category names to their code mappings

    Returns:
        dict: Data with category codes converted back to original string values
    """
    result = row.copy()
    for col, mapping in category_mappings.items():
        if col in row:
            # Reverse the mapping (code -> category)
            reverse_mapping = {v: k for k, v in mapping.items()}
            # Convert the code back to the category string
            if isinstance(row, pd.Series):
                result[col] = reverse_mapping[int(row[col])]
            else:
                result[col] = reverse_mapping[int(row[col])]
    return result


def convert_codes_to_categories_df(df, category_mappings):
    """
    Convert category codes back to their original string values.

    Args:
        row: pandas Series or dict containing the data with category codes
        category_mappings: dictionary mapping category names to their code mappings

    Returns:
        dict: Data with category codes converted back to original string values
    """
    df_copy = df.copy()
    df_copy = df_copy.apply(lambda row: convert_codes_to_categories(row, category_mappings), axis=1)
    return df_copy


def extract_names_and_conditions(line):
    constraints = []
    pattern = r'([\w.-]+)\s*([<>=!]=?)\s*([\w."]+)'
    matches = re.findall(pattern, line.rstrip())
    for match in matches:
        lhs, op, rhs = match
        constraints.append((lhs, op, rhs))
    return constraints


def get_column_type(dataset, col):
    """
    Determine if a column is integer, float, or categorical
    Returns: 'integer', 'float', or 'categorical'
    """
    try:
        values = dataset[col].dropna()
        if len(values) == 0:
            return 'categorical'  # or handle empty columns as needed

        # Try to convert to numeric
        numeric_values = pd.to_numeric(values, errors='coerce')

        # If any values couldn't be converted to numeric, it's categorical
        if numeric_values.isna().any():
            return 'categorical'

        # Check if all numeric values are close to their integer representation
        if np.allclose(numeric_values, numeric_values.astype(int)):
            return 'integer'
        else:
            return 'float'

    except:
        return 'categorical'


def classify_columns(dataset):
    """
    Classify all columns in dataset
    Returns: dict with 'integer', 'float', 'categorical' keys containing lists of column names
    """
    col_types = {}
    for col in dataset.columns:
        col_type = get_column_type(dataset, col)
        col_types[col] = col_type

    return col_types


def is_integer_column(dataset, col):
    """Check if a column contains only integer values"""
    try:
        values = dataset[col].dropna()
        if len(values) == 0:
            return False
        # Check if all values are close to their integer representation
        return np.allclose(values, values.astype(int))
    except:
        return False


def smart_project_intervals(row_val, val_col, dataset, constraints, dic_cols, cons_function, cont_feat):
    """Exact interval-based projection for continuous variables only"""

    # Build suspect sets for all constraints
    val_cons_dfs = {}
    for cons in range(len(constraints)):
        val_cons_set = dataset[cons_function(dataset, row_val, cons, [val_col])]
        if len(val_cons_set) != 0 and (cons not in dic_cols.get(val_col, [])):
            return None  # Unavoidable violation
        val_cons_dfs[cons] = val_cons_set

    # Check if it's integer-valued
    is_integer = is_integer_column(dataset, val_col)

    # Collect all forbidden regions
    forbidden_intervals = []
    forbidden_points = []
    required_values = []

    # Process constraints
    for cons in dic_cols.get(val_col, []):
        if len(val_cons_dfs[cons]) == 0:
            continue

        for pred in constraints[cons]:
            if 'cf_row' not in pred:
                continue

            parts = pred.split(' ')
            first_clause, op, second_clause = parts[0], parts[1], parts[2]
            cf_pred = first_clause if 'cf_row' in first_clause else second_clause
            cf_col = cf_pred.split('.')[1]

            if val_col != cf_col:
                continue

            df_pred = first_clause if cf_pred == second_clause else second_clause

            # Get values to process
            if 'df.' in df_pred:
                values = val_cons_dfs[cons][cf_col].unique()
                cf_is_second = (cf_pred == second_clause)
            else:
                # Constant value
                if '"' in df_pred:
                    const_value = float(df_pred[1:-1])
                else:
                    const_value = float(df_pred)
                values = np.array([const_value])
                cf_is_second = False

            # Extract forbidden regions based on operator
            if op == '==':
                # Must avoid these exact values
                forbidden_points.extend(values)
            elif op == '!=':
                # Can only be these values
                required_values.extend(values)
            else:
                # Extract forbidden intervals
                intervals = extract_forbidden_intervals(values, op, cf_is_second, is_integer)
                if intervals:
                    forbidden_intervals.extend(intervals)

            break  # Process only first relevant predicate

    # Get domain bounds
    domain_min, domain_max = dataset[val_col].min(), dataset[val_col].max()

    # Handle required values (from != constraints)
    if required_values:
        unique_required = np.unique(required_values)
        if len(unique_required) > 1:
            return None  # Contradiction
        required_value = unique_required[0]

        # Check validity
        if required_value in forbidden_points or not (domain_min <= required_value <= domain_max):
            return None

        # Check if in any forbidden interval
        for start, end in forbidden_intervals:
            if start <= required_value <= end:
                return None

        return int(required_value) if is_integer else float(required_value)

    # Add forbidden points as intervals
    for point in set(forbidden_points):
        forbidden_intervals.append((point, point))

    # Compute valid intervals
    valid_intervals = compute_valid_intervals(forbidden_intervals, domain_min, domain_max, is_integer)

    if not valid_intervals:
        return None

    # Select closest valid value
    original_value = row_val[val_col]
    best_value = None
    min_distance = float('inf')

    for start, end in valid_intervals:
        # Find closest point in this interval
        if start <= original_value <= end:
            # Original value is valid
            return int(original_value) if is_integer else original_value
        elif original_value < start:
            candidate = start
        else:
            candidate = end

        distance = abs(candidate - original_value)
        if distance < min_distance:
            min_distance = distance
            best_value = candidate

    if best_value is not None and is_integer:
        best_value = int(best_value)

    return best_value


def extract_forbidden_intervals(values, op, cf_is_second, is_integer):
    """Extract forbidden intervals based on constraint operator

    Returns list of (start, end) tuples representing closed intervals [start, end]
    """
    values = np.asarray(values)
    intervals = []

    if op == '>':
        if cf_is_second:
            # Original: keeps values >= max_val
            # So forbid values < max_val, i.e., (-∞, max_val-1] for ints, (-∞, max_val) for floats
            max_val = values.max()
            if is_integer:
                intervals.append((float('-inf'), max_val - 1))
            else:
                # For floats, we'll handle exclusive boundary by subtracting small epsilon
                intervals.append((float('-inf'), max_val - 1e-10))
        else:
            # Original: keeps values <= min_val
            # So forbid values > min_val, i.e., [min_val+1, ∞) for ints, (min_val, ∞) for floats
            min_val = values.min()
            if is_integer:
                intervals.append((min_val + 1, float('inf')))
            else:
                intervals.append((min_val + 1e-10, float('inf')))

    elif op == '>=':
        if cf_is_second:
            # Original: keeps values > max_val
            # So forbid values <= max_val, i.e., (-∞, max_val]
            max_val = values.max()
            intervals.append((float('-inf'), max_val))
        else:
            # Original: keeps values < min_val
            # So forbid values >= min_val, i.e., [min_val, ∞)
            min_val = values.min()
            intervals.append((min_val, float('inf')))

    elif op == '<':
        if cf_is_second:
            # Original: keeps values <= min_val
            # So forbid values > min_val
            min_val = values.min()
            if is_integer:
                intervals.append((min_val + 1, float('inf')))
            else:
                intervals.append((min_val + 1e-10, float('inf')))
        else:
            # Original: keeps values >= max_val
            # So forbid values < max_val
            max_val = values.max()
            if is_integer:
                intervals.append((float('-inf'), max_val - 1))
            else:
                intervals.append((float('-inf'), max_val - 1e-10))

    elif op == '<=':
        if cf_is_second:
            # Original: keeps values < min_val
            # So forbid values >= min_val
            min_val = values.min()
            intervals.append((min_val, float('inf')))
        else:
            # Original: keeps values > max_val
            # So forbid values <= max_val
            max_val = values.max()
            intervals.append((float('-inf'), max_val))

    return intervals


def compute_valid_intervals(forbidden_intervals, domain_min, domain_max, is_integer):
    """Compute valid intervals as complement of forbidden regions"""

    if not forbidden_intervals:
        return [(domain_min, domain_max)]

    # Convert to numpy array and clip to domain
    intervals = []
    for start, end in forbidden_intervals:
        start = max(start, domain_min)
        end = min(end, domain_max)
        if start <= end:
            intervals.append([start, end])

    if not intervals:
        return [(domain_min, domain_max)]

    # Sort by start position
    intervals.sort(key=lambda x: x[0])

    # Merge overlapping intervals
    merged = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if is_integer:
            # For integers, adjacent intervals should be merged
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        else:
            # For floats, check with small epsilon
            if start <= current_end + 1e-10:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end

    merged.append((current_start, current_end))

    # Generate valid intervals as gaps between forbidden regions
    valid_intervals = []

    # Before first forbidden region
    if merged[0][0] > domain_min:
        if is_integer:
            # For integers, valid up to one less than forbidden start
            valid_intervals.append((domain_min, merged[0][0] - 1))
        else:
            # For floats, valid up to just before forbidden start
            valid_intervals.append((domain_min, merged[0][0] - 1e-10))

    # Between forbidden regions
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]

        if is_integer:
            # For integers, gap is [end+1, next_start-1]
            if gap_end - gap_start >= 2:
                valid_intervals.append((gap_start + 1, gap_end - 1))
        else:
            # For floats, gap is (end, next_start)
            if gap_end - gap_start > 2e-10:
                valid_intervals.append((gap_start + 1e-10, gap_end - 1e-10))

    # After last forbidden region
    if merged[-1][1] < domain_max:
        if is_integer:
            valid_intervals.append((merged[-1][1] + 1, domain_max))
        else:
            valid_intervals.append((merged[-1][1] + 1e-10, domain_max))

    return valid_intervals


def smart_project_intervals_approximate(row_val, val_col, dataset, gamma, constraints, dic_cols, cons_function,
                                        bin_cons):
    val_cons_dfs = {}
    starting_viol = set()
    epsilon = 1e-10
    is_integer = all(isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer())
                     for x in dataset[val_col].head(100))

    epsilon = 1 if is_integer else 1e-10
    # Process initial constraints
    for cons in range(len(constraints)):
        indices = cons_function(dataset, row_val, cons, [val_col])
        val_cons_set = dataset[indices]
        if len(val_cons_set) != 0 and (cons not in dic_cols[val_col]):
            starting_viol.update(np.where(indices)[0])
            if len(starting_viol) > gamma:
                return None
        val_cons_dfs[cons] = val_cons_set

    starting_viol_len = len(starting_viol)  # Cache this expensive operation

    bounds = []

    # Process constraints that affect val_col
    for count, cons in enumerate(dic_cols[val_col]):
        if cons in bin_cons:
            if len(val_cons_dfs[cons]) == 0:
                continue

            for pred in constraints[cons]:
                if pred.find('cf_row') == -1:
                    continue

                first_clause, op, second_clause = pred.split(' ')
                cf_pred = first_clause if 'cf_row' in first_clause else second_clause
                cf_col = cf_pred.split('.')[1]

                if val_col != cf_col:
                    continue

                df_pred = first_clause if cf_pred == second_clause else second_clause

                if 'df.' in df_pred:
                    # Create mask based on operation
                    if cf_pred == second_clause and op in ['>', '>=', '<', '<=']:
                        bounds += [(val, op_rev_bound[op], 1) for val in val_cons_dfs[cons][cf_col]]
                    else:
                        bounds += [(val, op_bound[op], 1) for val in val_cons_dfs[cons][cf_col]]
                    # Update violation dictionary


                else:
                    const_value = float(df_pred) if '"' not in df_pred else df_pred[1:-1]
                    bounds += [(const_value, op_bound[op], len(val_cons_dfs[cons]))]
                break
        else:
            if len(val_cons_dfs[cons]) == 0:
                continue

            for pred in constraints[cons]:
                if pred.find('cf_row') == -1:
                    continue

                first_clause, op, second_clause = pred.split(' ')
                cf_pred = first_clause if 'cf_row' in first_clause else second_clause
                cf_col = cf_pred.split('.')[1]

                if val_col != cf_col:
                    continue

                df_pred = first_clause if cf_pred == second_clause else second_clause

                # Get values to process
                # Create mask based on operation
                const_value = float(df_pred) if '"' not in df_pred else df_pred[1:-1]
                bounds += [(const_value, op_bound[op], 2 * gamma)]

                break
    min_val, max_val = dataset[val_col].min(), dataset[val_col].max()
    events = []
    events.append((min_val, 'start', 0))
    events.append((max_val, 'end', 0))
    base_violations = starting_viol_len
    for bound_val, bound_type, violation_count in bounds:

        if bound_type == 'left_inclusive':
            # Violations start at this point (includes the point)
            events.append((bound_val, 'increase', violation_count))
        elif bound_type == 'left_exclusive':
            # Violations start after this point (excludes the point)
            events.append((bound_val + epsilon, 'increase', violation_count))
        elif bound_type == 'right_inclusive':
            # Violations end after this point (includes the point)
            events.append((bound_val + epsilon, 'decrease', violation_count))
            base_violations += violation_count
        elif bound_type == 'right_exclusive':
            # Violations end at this point (excludes the point)
            events.append((bound_val, 'decrease', violation_count))
            base_violations += violation_count
        elif bound_type == 'point':
            # Violation only at this exact point
            events.append((bound_val, 'increase', violation_count))
            events.append((bound_val + epsilon, 'decrease', violation_count))
        elif bound_type == 'not_point':
            # Subtract violations only at the specific point
            events.append((bound_val, 'decrease', violation_count))
            events.append((bound_val + epsilon, 'increase', violation_count))
            base_violations += violation_count

    # Sort events by position, with special ordering for same position
    event_order = {'start': 0, 'decrease': 1, 'increase': 2, 'end': 3}
    events.sort(key=lambda x: (x[0], event_order[x[1]]))

    # Sweep line to build intervals
    intervals = []
    current_violations = base_violations  # Start with base violations
    last_pos = min_val

    for pos, event_type, violation_count in events:
        # Close previous interval if we moved to a new position
        if pos > last_pos:
            intervals.append((last_pos, pos - epsilon, current_violations))

        # Process event to update violation count
        if event_type == 'increase':
            current_violations += violation_count
        elif event_type == 'decrease':
            current_violations -= violation_count
        last_pos = pos

    # Remove empty intervals and check gamma constraint
    valid_intervals = []
    for left, right, violations in intervals:
        if violations < gamma:
            valid_intervals.append((left, right, violations))

    if not valid_intervals:
        return None

    # Find best interval - minimum violations, closest to target
    target_val = row_val[val_col]

    # min_violations = min(violations for _, _, violations in valid_intervals)
    # best_intervals = [(left, right, violations) for left, right, violations in valid_intervals
    #                  if violations == min_violations]

    # Among best intervals, find closest to target
    def interval_distance(interval_tuple):
        left, right, _ = interval_tuple
        if left <= target_val <= right:
            return 0  # Target is inside interval
        elif target_val < left:
            return left - target_val
        else:
            return target_val - right

    best_interval = min(valid_intervals, key=interval_distance)
    left, right, _ = best_interval

    # Return target if it's in the best interval, otherwise return closest boundary
    if left <= target_val <= right:
        return target_val
    elif target_val < left:
        return left
    else:
        return right

    pass


def smart_project(row_val, val_col, dataset, constraints, dic_cols, cons_function, cont_feat):
    val_cons_dfs = {}
    for cons in range(len(constraints)):
        # non_follow_cons = ~cons_df.swifter.apply(cons_function(row_val, cons), axis=1)
        # val_cons_set = cons_df[~cons_df.apply(cons_function(row_val, cons, [val_col]), axis=1)]
        # print(cons_function(dataset, row_val, cons, [val_col]))
        val_cons_set = dataset[cons_function(dataset, row_val, cons, [val_col])]
        if len(val_cons_set) != 0 and (cons not in dic_cols[val_col]):
            return None
        val_cons_dfs[cons] = val_cons_set

    if val_col in cont_feat:
        possible_values = list(range(dataset[val_col].min(), dataset[val_col].max() + 1))
    else:
        possible_values = list(dataset[val_col].unique())

    for count, cons in enumerate(dic_cols[val_col]):
        prev_values = possible_values
        # if len(val_cons_dfs[cons] == 0):
        #     continue
        if len(val_cons_dfs[cons]) == 0:
            continue
        for pred in constraints[cons]:
            if pred.find('cf_row') == -1:
                continue
            first_clause, op, second_clause = pred.split(' ')
            cf_pred = first_clause if 'cf_row' in first_clause else second_clause
            cf_col = cf_pred.split('.')[1]

            if val_col != cf_col:
                continue
            # if len(val_cons_dfs[cons][cf_col]) == 0:
            #     break
            df_pred = first_clause if cf_pred == second_clause else second_clause

            if 'df.' in df_pred:
                if op == '==':
                    possible_values = [elem for elem in possible_values if
                                       elem not in val_cons_dfs[cons][cf_col].unique()]
                elif op == '!=':
                    possible_values = [elem for elem in possible_values if
                                       elem in val_cons_dfs[cons][cf_col].unique()]
                elif op == '>':
                    if cf_pred == second_clause:
                        max_val = val_cons_dfs[cons][cf_col].unique().max()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np >= max_val])
                    else:
                        min_val = val_cons_dfs[cons][cf_col].unique().min()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np <= min_val])
                elif op == '>=':
                    if cf_pred == second_clause:
                        max_val = val_cons_dfs[cons][cf_col].unique().max()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np > max_val])
                    else:
                        min_val = val_cons_dfs[cons][cf_col].unique().min()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np < min_val])
                elif op == '<':
                    if cf_pred == second_clause:
                        min_val = val_cons_dfs[cons][cf_col].unique().min()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np <= min_val])
                    else:
                        max_val = val_cons_dfs[cons][cf_col].unique().max()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np >= max_val])
                elif op == '<=':
                    if cf_pred == second_clause:
                        min_val = val_cons_dfs[cons][cf_col].unique().min()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np < min_val])
                    else:
                        max_val = val_cons_dfs[cons][cf_col].unique().max()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np > max_val])

            else:
                const_value = float(df_pred) if '"' not in df_pred else df_pred[1:-1]

                if op == '==':
                    possible_values = [elem for elem in possible_values if
                                       elem not in [const_value]]
                elif op == '!=':
                    possible_values = [elem for elem in possible_values if
                                       elem in [const_value]]
                elif op == '>':
                    min_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np <= min_val])
                elif op == '>=':
                    min_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np < min_val])
                elif op == '<':
                    max_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np >= max_val])
                elif op == '<=':
                    max_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np > max_val])
            if not possible_values:
                # if count < args.cons_lim:
                return None
                # if isinstance(prev_values[0], (int, float, complex)) and not isinstance(prev_values[0], bool):
                #     return min(prev_values, key=lambda x: abs(x - row_val[val_col]))
                # return prev_values[0]
            break
    if isinstance(possible_values[0], numbers.Number) and not isinstance(possible_values[0], bool):
        return min(possible_values, key=lambda x: abs(x - row_val[val_col]))
    return possible_values[0]


def smart_project_soft(row_val, val_col, dataset, gamma, constraints, dic_cols, cons_function, cont_feat):
    val_cons_dfs = {}
    starting_viol = set()
    for cons in range(len(constraints)):
        # non_follow_cons = ~cons_df.swifter.apply(cons_function(row_val, cons), axis=1)
        # val_cons_set = cons_df[~cons_df.apply(cons_function(row_val, cons, [val_col]), axis=1)]
        indices = cons_function(dataset, row_val, cons, [val_col])
        val_cons_set = dataset[indices]
        if len(val_cons_set) != 0 and (cons not in dic_cols[val_col]):
            starting_viol.update(np.where(indices)[0])
            if len(starting_viol) > gamma:
                return None
        val_cons_dfs[cons] = val_cons_set

    # if val_col in args.cont_feat:
    #     possible_values = list(range(dataset[val_col].min(), dataset[val_col].max() + 1))
    # else:
    possible_values = list(dataset[val_col].unique())

    violation_dic = {key: set() for key in possible_values}

    for count, cons in enumerate(dic_cols[val_col]):
        if len(val_cons_dfs[cons]) == 0:
            continue
        for pred in constraints[cons]:
            if pred.find('cf_row') == -1:
                continue
            first_clause, op, second_clause = pred.split(' ')
            cf_pred = first_clause if 'cf_row' in first_clause else second_clause
            cf_col = cf_pred.split('.')[1]

            if val_col != cf_col:
                continue
            # if len(val_cons_dfs[cons][cf_col]) == 0:
            #     break
            df_pred = first_clause if cf_pred == second_clause else second_clause
            keys = np.array(list(violation_dic.keys()))

            # Get the column values as a numpy array
            column_values = val_cons_dfs[cons][cf_col].values
            column_array = np.array(column_values)
            category_array = np.array(keys)

            if 'df.' in df_pred:
                if op == '==':
                    mask = category_array[:, np.newaxis] == column_array
                elif op == '!=':
                    mask = category_array[:, np.newaxis] != column_array
                elif op == '>':
                    if cf_pred == second_clause:
                        mask = category_array[:, np.newaxis] < column_array
                    else:
                        mask = category_array[:, np.newaxis] > column_array
                elif op == '>=':
                    if cf_pred == second_clause:
                        mask = category_array[:, np.newaxis] <= column_array
                    else:
                        mask = category_array[:, np.newaxis] >= column_array
                elif op == '<':
                    if cf_pred == second_clause:
                        mask = category_array[:, np.newaxis] > column_array
                    else:
                        mask = category_array[:, np.newaxis] < column_array
                elif op == '<=':
                    if cf_pred == second_clause:
                        mask = category_array[:, np.newaxis] >= column_array
                    else:
                        mask = category_array[:, np.newaxis] <= column_array
                        # Update the dictionary
                for i, key in enumerate(keys):
                    violated_indices = np.where(mask[i])[0]
                    violation_dic[key].update(violated_indices)
                if (min(len(violations) for violations in violation_dic.values()) + len(starting_viol)) > gamma:
                    return None
            else:
                const_value = float(df_pred) if '"' not in df_pred else df_pred[1:-1]

                if op == '==':
                    possible_values = [elem for elem in possible_values if
                                       elem not in [const_value]]
                    violation_dic = {k: violation_dic[k] for k in possible_values}
                elif op == '!=':
                    possible_values = [elem for elem in possible_values if
                                       elem in [const_value]]
                    violation_dic = {k: violation_dic[k] for k in possible_values}
                elif op == '>':
                    min_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np <= min_val])
                    violation_dic = {k: violation_dic[k] for k in possible_values}
                elif op == '>=':
                    min_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np < min_val])
                    violation_dic = {k: violation_dic[k] for k in possible_values}
                elif op == '<':
                    max_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np >= max_val])
                    violation_dic = {k: violation_dic[k] for k in possible_values}
                elif op == '<=':
                    max_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np > max_val])
                    violation_dic = {k: violation_dic[k] for k in possible_values}
            if not possible_values:
                # if count < args.cons_lim:
                return None
    # possible_return_vals = [key for key, value in violation_dic.items() if value < gamma]
    # if isinstance(possible_return_vals[0], (int, float, complex)) and not isinstance(possible_return_vals[0], bool):
    #     return min(possible_return_vals, key=lambda x: abs(x - row_val[val_col]))
    # return min(violation_dic,key=lambda x:violation_dic.get(x))
    possible_return_vals = [key for key, violations in violation_dic.items() if
                            (len(violations) + len(starting_viol)) < gamma]

    if isinstance(possible_values[0], numbers.Number) and not isinstance(possible_values[0], bool):
        return min(possible_return_vals, key=lambda x: abs(x - row_val[val_col]))
    return min(violation_dic, key=lambda x: len(violation_dic[x]))


def single_pred_cons(val_iter, col, dic_cols, constraints, unary_cons_lst_single):
    if col not in dic_cols:
        return val_iter
    common_elements = set(unary_cons_lst_single) & set(dic_cols[col])
    for cons in common_elements:
        _, op, constant = constraints[cons][0].split(' ')
        if op == '>':
            constant = float(constant)
            ind = bisect.bisect_right(val_iter, constant)
            val_iter = val_iter[:ind]
        elif op == '>=':
            constant = float(constant)
            ind = bisect.bisect_left(val_iter, constant)
            val_iter = val_iter[:ind]
        elif op == '<':
            constant = float(constant)
            ind = bisect.bisect_left(val_iter, constant)
            val_iter = val_iter[ind:]
        elif op == '<=':
            constant = float(constant)
            ind = bisect.bisect_right(val_iter, constant)
            val_iter = val_iter[ind]
        elif op == '==':
            if constant in val_iter:
                val_iter.remove(constant)
        else:
            if constant in val_iter:
                val_iter = [constant]
        pass
    return val_iter


def bounds_builder(row_val, comb, dataset, constraints, cons_feat, dic_cols, cons_function):
    val_cons_dfs = {}
    comb_set = set(comb)
    for cons in range(len(constraints)):
        val_cons_set = dataset[cons_function(dataset, row_val, cons, comb)]
        if len(val_cons_set) != 0 and len(comb_set | set(cons_feat[cons])) == 0:
            return None  # Unavoidable violation
        val_cons_dfs[cons] = val_cons_set

    # Check if it's integer-valued
    column_types = classify_columns(dataset)

    # Collect all forbidden regions
    full_bounds = []  # List of DataFrames, one per constraint
    attribute_to_dataframes = {}  # Dictionary mapping attribute names to dataframe indices
    # Process constraints
    for cons in range(len(constraints)):
        if cons not in set([val for col in comb for val in dic_cols[col]]):
            full_bounds.append(pd.DataFrame())  # No bounds for this constraint
            continue
        if len(val_cons_dfs[cons]) == 0:
            full_bounds.append(pd.DataFrame())  # No bounds for this constraint
            continue
        count_preds = 0
        comb_cons = []

        for pred in constraints[cons]:
            if 'cf_row' not in pred:
                continue

            parts = pred.split(' ')
            first_clause, op, second_clause = parts[0], parts[1], parts[2]
            cf_pred = first_clause if 'cf_row' in first_clause else second_clause
            cf_col = cf_pred.split('.')[1]

            if cf_col not in comb_set:
                continue

            df_pred = first_clause if cf_pred == second_clause else second_clause
            unary = 1
            col_type = column_types.get(cf_col, 'unknown')
            # Get values to process
            if 'df.' in df_pred:
                values = val_cons_dfs[cons][cf_col]
                cf_is_second = (cf_pred == second_clause)
            else:
                # Constant value
                if '"' in df_pred:
                    if col_type == 'float':
                        const_value = float(df_pred[1:-1])
                    elif col_type == 'integer':
                        const_value = int(df_pred[1:-1])
                    else:
                        const_value = df_pred[1:-1]
                else:
                    if col_type == 'float':
                        const_value = float(df_pred)
                    elif col_type == 'integer':
                        const_value = int(df_pred)
                    else:
                        const_value = df_pred
                val = const_value
                cf_is_second = False
                unary = len(val_cons_dfs[cons])
            if unary == 1:
                comb_cons.append([(cf_col, val, op_rev_bound[op] if cf_is_second else op_bound[op]) for val in values])
            else:
                comb_cons.append([(cf_col, val, op_rev_bound[op] if cf_is_second else op_bound[op])] * unary)
            count_preds += 1
            if count_preds >= len(comb):
                break

        # Create bounds for this constraint
        constraint_bounds = []
        for i in range(len(comb_cons[0])):
            full_bound = []
            for j in range(len(comb_cons)):
                col, val, op = comb_cons[j][i]
                full_bound += [col, val, op]
            constraint_bounds.append(full_bound)

        # Convert constraint bounds to DataFrame
        if constraint_bounds:
            # Fast approach: use the first bound to determine structure
            first_bound = constraint_bounds[0]
            num_cols = len(first_bound)

            # Create column headers from first bound only
            column_headers = []
            constraint_attributes = set()
            for i in range(0, num_cols, 3):
                col_name = first_bound[i]
                constraint_attributes.add(col_name)
                column_headers.extend([col_name, f"{col_name}_value", f"{col_name}_op"])

            # Create DataFrame directly from constraint_bounds (no padding needed if all same length)
            # If lengths vary, use numpy for fast padding
            if len(set(len(bound) for bound in constraint_bounds)) == 1:
                # All bounds same length - direct DataFrame creation
                constraint_df = pd.DataFrame(constraint_bounds, columns=column_headers).drop_duplicates()
            else:
                # Different lengths - use numpy for fast padding
                max_cols = len(first_bound)  # Assume first bound has max length
                padded_array = np.full((len(constraint_bounds), max_cols), None, dtype=object)
                for i, bound in enumerate(constraint_bounds):
                    padded_array[i, :len(bound)] = bound
                constraint_df = pd.DataFrame(padded_array, columns=column_headers).drop_duplicates()

            full_bounds.append(constraint_df)

            # Update attribute_to_dataframes mapping
            current_df_index = len(full_bounds) - 1
            for attr in constraint_attributes:
                if attr not in attribute_to_dataframes:
                    attribute_to_dataframes[attr] = []
                attribute_to_dataframes[attr].append(current_df_index)
    return column_types, full_bounds, attribute_to_dataframes


def project_constraints_exact_fast(row, projection_config):
    # Extract all needed parameters from kwargs
    args = projection_config.args
    d = projection_config.d
    df = projection_config.df
    constraints = projection_config.constraints
    cons_function = projection_config.cons_function
    cons_feat = projection_config.cons_feat
    categorical_column_names = projection_config.categorical_column_names
    mode = projection_config.mode
    dic_cols = projection_config.dic_cols
    dataset = projection_config.df
    transformer = projection_config.transformer
    unary_cons_lst_single = projection_config.unary_cons_lst_single
    exp_random = projection_config.exp_random
    bin_cons = projection_config.bin_cons

    try:
        # dataset = d.data_df
        # dataset = cf_example.data_interface.data_df
        dataset = pd.read_csv(args.dataset_path)
        viable_cols = [col for col in dataset.columns if col not in args.fixed_feat]
        viable_cols = [col for col in viable_cols if col in dic_cols]

        must_cons = set()
        need_proj = False
        violation_set = []
        cons_set = [[]] * len(constraints)
        starting_viol = set()

        for cons in range(len(constraints)):
            non_follow_cons = cons_function(dataset, row, cons)
            violation_set.append(dataset.loc[non_follow_cons])
            if len(non_follow_cons) == 0:
                continue
            count = non_follow_cons.sum()
            if count > 0:
                if cons in bin_cons:
                    indices = np.where(non_follow_cons)[0]
                    starting_viol.update(indices)
                    cons_set[cons] = indices
                need_proj = True
                must_cons.add(cons)
        if not need_proj:
            return row.copy()
        # Order based on number of violations
        # Order Based on previous permutations
        #
        for i in range(1, len(viable_cols) + 1):
            random.shuffle(viable_cols)
            combs = list(itertools.combinations(viable_cols, i))
            for comb in combs:
                if not all(any(feat in comb for feat in cons_feat[cons]) for cons in must_cons):
                    continue
                row_per = row.copy()
                comb_iters = []
                for col in comb[:-1]:
                    if col in args.cont_feat:
                        # TODO FIX
                        # epsilon = args.epsilon.get(col, None)
                        epsilon = None
                        if epsilon is None:
                            if (dataset[col].max() - dataset[col].min()) > 100:
                                val_iter = np.linspace(dataset[col].min(), dataset[col].max(), 100, dtype=int)
                            else:
                                val_iter = np.arange(dataset[col].min(), dataset[col].max() + 1, dtype=int)
                        else:
                            val_iter = np.arange(dataset[col].min(), dataset[col].max() + 1, epsilon, dtype=int)
                        val_iter = list(val_iter)
                        val_iter = single_pred_cons(val_iter, col, dic_cols, constraints, unary_cons_lst_single)
                        val_iter.sort(key=lambda x: abs(x - row_per[col]))
                        if row_per[col] in val_iter:
                            val_iter.remove(row_per[col])
                    else:
                        val_iter = list(dataset[col].unique())
                        val_iter = single_pred_cons(val_iter, col, dic_cols, constraints, unary_cons_lst_single)
                        random.shuffle(val_iter)
                        if row_per[col] in val_iter:
                            val_iter.remove(row_per[col])
                    comb_iters.append([(val, col) for val in val_iter])
                all_combinations = itertools.product(*comb_iters)
                for vals in all_combinations:
                    row_val = row_per.copy()
                    for val, val_col in vals:
                        row_val[val_col] = val
                    if args.mode == 'soft':
                        st = time.time()
                        res = smart_project_soft(row_val, comb[-1], dataset, args.gamma * len(dataset), constraints,
                                                 dic_cols, cons_function, cont_feat)
                        print(f'Time to project soft: {time.time() - st:.6f} seconds')
                        if comb[-1] in args.cont_feat:
                            st = time.time()
                            res2 = smart_project_intervals_approximate(row_val, comb[-1], dataset,
                                                                       args.gamma * len(dataset), constraints, dic_cols,
                                                                       cons_function, bin_cons)
                            print(f'Time to project intervals soft: {time.time() - st:.6f} seconds')
                            if res2 != res:
                                res2 = smart_project_intervals_approximate(row_val, comb[-1], dataset,
                                                                           args.gamma * len(dataset), constraints,
                                                                           dic_cols, cons_function, bin_cons)
                                print(f'Warning: Interval projection {res2} does not match soft projection {res}')
                            pass
                    else:
                        if comb[-1] in args.cont_feat:
                            res = smart_project_intervals(row_val, comb[-1], dataset, constraints, dic_cols,
                                                          cons_function, args.cont_feat)
                        else:
                            res = smart_project(row_val, comb[-1], dataset, constraints, dic_cols, cons_function,
                                                args.cont_feat)
                        # st = time.time()
                        # res = smart_project(row_val, comb[-1], dataset)
                        # print(f'Time to project: {time.time() - st:.6f} seconds')
                        # if comb[-1] in args.cont_feat:
                        #     st = time.time()
                        #     res2 = smart_project_intervals(row_val, comb[-1], dataset)
                        #     print(f'Time to project intervals: {time.time() - st:.6f} seconds')
                        #     pass
                        #     if res2 != res:
                        #         print(f'Warning: Interval projection {res2} does not match soft projection {res}')

                    if res is not None:
                        row_val[comb[-1]] = res
                        print('Success')
                        print(row_val)
                        return row_val
        else:
            return None
    except TimeoutError:
        print("Projection function timed out")
        return None
    finally:
        signal.alarm(0)


def diversity_slack_variable(s, vars, found_points, norm_params, normalized_medians, non_cat_cols, cat_cols):
    """
    Solution 1: Use a slack variable to represent minimum distance.
    Instead of computing min with nested Ifs, we add constraints that enforce
    the slack variable to be <= all distances.

    This is MUCH more solver-friendly than nested If statements.
    """
    if len(found_points) == 0:
        return 0

    # Create a slack variable for minimum distance
    min_dist = Real('min_diversity_distance')

    # Add constraints: min_dist must be <= distance to each point
    for i in range(len(found_points)):
        distance_to_point = 0

        for col in non_cat_cols:
            if col in vars:
                med_scaled = normalized_medians[col] * norm_params[col]['range']
                if med_scaled <= 1e-10:
                    med_scaled = 1.0
                distance_to_point += Abs(vars[col] - found_points.iloc[i][col]) / med_scaled

        for col in cat_cols:
            if col in vars:
                distance_to_point += If(vars[col] == found_points.iloc[i][col], 0, 1)

        # Key constraint: min_dist must be less than or equal to this distance
        s.add(min_dist <= distance_to_point)

    # Add bounds to help solver
    s.add(min_dist >= 0)
    # Optional: add upper bound based on problem knowledge
    # s.add(min_dist <= len(non_cat_cols) + len(cat_cols))

    return min_dist


def project_solver(row, projection_config):
    """
    Project a single row using the constraint solver method.

    Args:
        row: pandas Series representing the instance to project
        projection_config: ProjectionConfig object containing all necessary parameters

    Returns:
        pandas Series representing the projected instance, or None if projection failed
    """
    st_project = time.time()

    # Extract frequently used variables
    args = projection_config.args
    d = projection_config.d
    df = projection_config.df
    constraints = projection_config.constraints
    cons_function = projection_config.cons_function
    cons_feat = projection_config.cons_feat
    categorical_column_names = projection_config.categorical_column_names
    mode = projection_config.mode

    dataset = projection_config.df
    orig_dataset = dataset.copy()
    orig_row = row.copy()

    print(f'Projecting row: {row}')

    # Setup viable columns (excluding fixed features)
    fixed_feat = args.fixed_feat
    fixed_feat1 = [] if not args.fixed_flag else fixed_feat  # Currently empty in your code
    viable_cols = [col for col in dataset.columns
                   if col not in fixed_feat1 and col in projection_config.dic_cols]

    # Prepare combined dataset
    combined = pd.concat([dataset, row.to_frame().T], ignore_index=True)
    combined[categorical_column_names] = combined[categorical_column_names].astype('category')
    cat_cols = combined.select_dtypes(include=['category']).columns
    category_mappings = {col: {v: k for k, v in enumerate(combined[col].cat.categories)}
                         for col in cat_cols}

    # Convert categorical columns to integer codes
    combined[cat_cols] = combined[cat_cols].apply(lambda x: x.cat.codes)

    # Split back into dataset and row
    dataset = combined.iloc[:-1]
    row = combined.iloc[-1]
    dataset[categorical_column_names] = dataset[categorical_column_names].astype('category')

    # Handle constraint checking for 'none' mode
    if mode != 'solver_linear':
        must_cons = set()
        violation_set = []
        for cons in range(len(constraints)):
            non_follow_cons = cons_function(orig_dataset, orig_row, cons)
            violation_set.append(orig_dataset.loc[non_follow_cons])
            if len(non_follow_cons) == 0:
                continue
            count = non_follow_cons.sum()
            if count > 0:
                must_cons.add(cons)

        if len(must_cons) == 0:
            print('No constraints to project')
            return orig_row.copy()
    else:
        must_cons = set(range(len(constraints)))
    # Shuffle viable columns for randomization
    random.shuffle(viable_cols)
    solver_add_time = []
    constraints_list_add_time = []

    # Try projection with all viable columns
    for i in range(len(viable_cols), len(viable_cols) + 1):
        combs = list(itertools.combinations(viable_cols, i))
        for comb in combs:
            result = _try_projection_combination(
                comb, row, orig_row, dataset, orig_dataset,
                projection_config, must_cons, category_mappings,
                solver_add_time, constraints_list_add_time, st_project, fixed_feat1
            )
            # result = _try_projection_combination_scored(
            #     comb, row, orig_row, dataset, orig_dataset,
            #     projection_config, must_cons, category_mappings,
            #     solver_add_time, constraints_list_add_time, st_project, fixed_feat1
            # )
            # result = _try_projection_combination_violated(
            #     comb, row, orig_row, dataset, orig_dataset,
            #     projection_config, must_cons, category_mappings,
            #     solver_add_time, constraints_list_add_time, st_project, fixed_feat1
            # )
            if result is not None:
                return result

    return None


def _try_projection_combination(comb, row, orig_row, dataset, orig_dataset,
                                projection_config, must_cons, category_mappings,
                                solver_add_time, constraints_list_add_time, st_project, fixed_feat1):
    """
    Helper function to try a specific combination of features for projection.
    """
    st_preprocess = time.time()
    args = projection_config.args
    constraints = projection_config.constraints
    cons_feat = projection_config.cons_feat
    dic_cols = projection_config.dic_cols
    cons_function = projection_config.cons_function
    mode = projection_config.mode
    categorical_column_names = projection_config.categorical_column_names
    found_points = getattr(projection_config, 'found_points', None)
    coefs_dic = getattr(projection_config, 'coefs_dic', None)
    intercept = getattr(projection_config, 'intercept', None)

    # Create cache key
    cache_key = (tuple(sorted(comb)), tuple(orig_row[fixed_feat1]))

    if cache_key not in _solver_cache:
        # Check constraint coverage for 'none' mode
        if mode != 'solver_linear':
            if not all(any(feat in comb for feat in cons_feat[cons]) for cons in must_cons):
                return None

        # Build solver
        column_types = classify_columns(orig_dataset)
        s = Optimize()
        vars = {}

        st_constraints = time.time()

        # Create variables
        for col in dataset.columns:
            if col == 'label':
                continue
            if col in args.cont_feat:
                if column_types[col] == 'float':
                    var = Real(col)
                else:
                    var = Int(col)
                vars[col] = var
                s.add(var >= dataset[col].min(), var <= dataset[col].max())
            else:
                var = Int(col)
                vars[col] = var
                s.add(var >= 0, var <= len(dataset[col].cat.categories) - 1)

        # Build bounds and add constraints
        st = time.time()
        column_types, full_bounds, attribute_to_dataframes = bounds_builder(
            orig_row, comb, orig_dataset, constraints, cons_feat, dic_cols, cons_function
        )
        print(f'Time taken to build bounds: {time.time() - st:.4f} seconds')

        # Map categorical values to codes
        for attr in attribute_to_dataframes:
            if attr not in categorical_column_names:
                continue
            for bound in attribute_to_dataframes[attr]:
                full_bounds[bound][f'{attr}_value'] = full_bounds[bound][f'{attr}_value'].map(
                    category_mappings[attr]
                )

        # Add constraint bounds
        for bound in full_bounds:
            if len(bound) == 0:
                continue

            for idx, row_bound in bound.iterrows():
                row_constraints = []

                for i in range(0, len(row_bound), 3):
                    col = row_bound[i + 0]
                    val = row_bound[i + 1]
                    op = row_bound[i + 2]

                    if col not in vars:
                        continue

                    # Create constraint based on operator
                    if op == 'point':
                        constraint = vars[col] == val
                    elif op == 'not_point':
                        constraint = vars[col] != val
                    elif op == 'left_exclusive':
                        constraint = vars[col] > val
                    elif op == 'left_inclusive':
                        constraint = vars[col] >= val
                    elif op == 'right_exclusive':
                        constraint = vars[col] < val
                    elif op == 'right_inclusive':
                        constraint = vars[col] <= val
                    else:
                        continue

                    st = time.time()
                    row_constraints.append(constraint)
                    constraints_list_add_time.append(time.time() - st)

                # Add negation of conjunction
                if row_constraints:
                    st = time.time()
                    s.add(Not(And(row_constraints)))
                    solver_add_time.append(time.time() - st)

        print(f'Time to create constraints: {time.time() - st_constraints:.2f} seconds')
        print(f'##################### Amount of solver constraints: {len(s.assertions())}###############')

        # Push to create Level 1 (gamma constraints level)
        s.push()

        _solver_cache[cache_key] = (s, copy.deepcopy(vars), column_types)

        print(f'Preprocessing time for combination {comb}: {time.time() - st_preprocess:.2f} seconds')
    else:
        # Use cached solver (already has preprocessing + Level 1 push)
        st = time.time()
        s, vars, column_types = _solver_cache[cache_key]
        vars = copy.deepcopy(vars)
        print(f'Time to copy cached solver: {time.time() - st:.2f} seconds')

    # Push for Level 2 (per-projection constraints)
    s.push()

    # Add linear model constraint if needed
    if mode == 'solver_linear':
        added_intercept = 0
        for coefs_key in coefs_dic:
            if coefs_key not in vars:
                added_intercept += coefs_dic[coefs_key] * row[coefs_key]
        s.add(Sum([coefs_dic[var] * vars[var] for var in vars]) + intercept + added_intercept >= 0.1)

    # Add fixed feature constraints
    for col in args.fixed_feat:
        s.add(vars[col] == row[col])

    # Add found_points exclusion constraints
    if found_points is not None and len(found_points) > 0:
        for i in range(len(found_points)):
            s.add(Or([vars[col] != found_points.iloc[i][col]
                      for col in vars if col != 'label']))

    # Build objective function
    st = time.time()
    distance = 0
    if args.distance == 'L0':
        distance = distance_L0(row, projection_config, args, vars)
    elif args.distance == 'MAD':
        distance = distance_MAD(row, projection_config, args, vars)
    else:
        distance = build_distance_for_solver(vars, row, projection_config, category_mappings)

    if found_points is not None:
        total_dist = -args.delta * diversity_slack_variable(s, vars,
                                                            found_points.drop('label', axis=1, errors='ignore'),
                                                            projection_config.norm_params,
                                                            projection_config.normalized_medians, args.cont_feat,
                                                            categorical_column_names) + distance
    else:
        total_dist = distance

    opt = s.minimize(total_dist)
    print(f'Time to create objective: {time.time() - st:.2f} seconds')

    # Solve
    st_solve = time.time()
    s.set(timeout=args.solver_timeout)
    res = s.check().r
    s.lower(opt)
    print(f'Time to solve: {time.time() - st_solve:.2f} seconds')
    print(res)

    if res != -1:
        m = s.model()
        row_val = row.copy()

        # Extract solution
        for col in comb:
            model_val = m[vars[col]]
            if model_val is None:
                print(f'Warning: Model value for {col} is None, skipping')
                continue
            print(f'Column: {col}, Model Value: {model_val}')
            if col in args.cont_feat:
                if column_types[col] == 'float':
                    value = model_val.as_decimal(10).replace('?', '')
                    row_val[col] = float(value)
                else:
                    value = model_val.as_long()
                    row_val[col] = value
            else:
                row_val[col] = dataset[col].cat.categories[model_val.as_long()]
        print('row_val:', row_val)
        row_val_orig = convert_codes_to_categories(row_val, category_mappings)

        if row_val_orig.equals(row):
            print('Row is equal to original row, returning None')

        print('Success')
        print(row_val_orig)
        print(f'Solver add time: {np.sum(solver_add_time):.6f} seconds')
        print(f'Constraints list add time: {np.sum(constraints_list_add_time):.6f} seconds')
        print(f'Time to project: {time.time() - st_project:.6f} seconds')

        # Pop Level 2 (per-projection constraints)
        s.pop()

        if projection_config.gamma > 0:
            print(f'Adding gamma constraint: at least {int(projection_config.gamma)} features must differ')

            # Build coded values dict from model - ONLY for comb
            row_val_coded = {}
            for var in comb:
                model_val = m[vars[var]]
                if model_val is None:
                    continue

                if var in args.cont_feat:
                    if column_types[var] == 'float':
                        row_val_coded[var] = float(model_val.as_decimal(10).replace('?', ''))
                    else:
                        row_val_coded[var] = model_val.as_long()
                else:
                    row_val_coded[var] = model_val.as_long()  # Store integer code

            feature_differences = []

            for var in comb:
                if var not in row_val_coded:
                    continue

                if var in args.cont_feat:
                    # For continuous: consider "different" if change exceeds a threshold
                    med_scaled = projection_config.normalized_medians.get(var, 1.0) * \
                                 projection_config.norm_params.get(var, {'range': 1.0})['range']
                    if med_scaled <= 1e-10:
                        med_scaled = 1.0

                    # Is the change significant? (at least 1 normalized unit)
                    diff = Abs(vars[var] - row_val_coded[var])
                    is_different = diff > med_scaled
                    feature_differences.append(If(is_different, 1, 0))
                else:
                    # For categorical: simple equality check
                    is_different = vars[var] != row_val_coded[var]
                    feature_differences.append(If(is_different, 1, 0))

            # Count total number of different features
            num_different_features = Sum(feature_differences)

            # Require at least gamma features to be different
            gamma_constraint = num_different_features >= int(projection_config.gamma)

            # Add gamma constraint at Level 1 (persists within sample, cleared between samples)
            s.add(gamma_constraint)

            # No need to save back to cache - s is mutable and already referenced there
            print(f'Gamma constraint added: {int(projection_config.gamma)} features must differ')

        if mode != 'solver_linear':
            return row_val_orig
        else:
            return row_val_orig, row_val
    else:
        s.pop()
        return None


def distance_MAD(row, projection_config, args, vars):
    distance = 0
    for var in vars:
        if var in args.cont_feat:
            med_scaled = projection_config.normalized_medians[var] * projection_config.norm_params[var]['range']
            if med_scaled <= 1e-10:
                med_scaled = 1.0

            distance += Abs(vars[var] - row[var]) / med_scaled
        else:
            distance += If(vars[var] == row[var], 0, 1)
    return distance


def distance_L0(row, projection_config, args, vars):
    distance = 0
    for var in vars:
        distance += If(vars[var] == row[var], 0, 1)
    return distance


class ProjectionConfig:
    """Configuration object to hold all projection parameters."""

    def __init__(self, args, df, d, exp_random, exp_random_lin, transformer,
                 constraints, dic_cols, cons_function, cons_feat,
                 categorical_column_names, mode, category_mappings,
                 project_runtimes, projection_metrics, model,
                 unary_cons_lst, unary_cons_lst_single, bin_cons, model_lin=None,
                 norm_params=None, normalized_medians=None,
                 projection_func=None,
                 coefs_dic=None, intercept=None, found_points=None, **kwargs):
        # Core objects
        self.args = args
        self.df = df
        self.d = d
        self.model = model
        self.model_lin = model_lin
        self.exp_random = exp_random
        self.exp_random_lin = exp_random_lin
        self.transformer = transformer
        self.category_mappings = category_mappings

        # Constraint-related parameters
        self.constraints = constraints
        self.dic_cols = dic_cols
        self.cons_function = cons_function
        self.cons_feat = cons_feat
        self.categorical_column_names = categorical_column_names
        self.unary_cons_lst = unary_cons_lst
        self.unary_cons_lst_single = unary_cons_lst_single
        self.bin_cons = bin_cons

        # Projection mode and metrics
        self.mode = mode
        self.project_runtimes = project_runtimes
        self.projection_metrics = projection_metrics
        self.projection_func = projection_func

        # Optional linear model parameters
        self.coefs_dic = coefs_dic
        self.intercept = intercept

        # Optional diversity parameters
        self.found_points = found_points

        # Normalization parameters
        self.normalized_medians = normalized_medians
        self.norm_params = norm_params
        # Gamma for solver diversity constraint
        self.gamma = args.gamma if model_lin is None else 0


def project_constraints_exact_fast_long(row, projection_config):
    args = projection_config.args
    d = projection_config.d
    df = projection_config.df
    constraints = projection_config.constraints
    cons_function = projection_config.cons_function
    cons_feat = projection_config.cons_feat
    categorical_column_names = projection_config.categorical_column_names
    mode = projection_config.mode
    dic_cols = projection_config.dic_cols
    dataset = projection_config.df
    transformer = projection_config.transformer
    unary_cons_lst_single = projection_config.unary_cons_lst_single
    exp_random = projection_config.exp_random
    viable_cols = [col for col in dataset.columns if col not in args.fixed_feat]
    viable_cols = [col for col in viable_cols if col in dic_cols]

    must_cons = set()
    need_proj = False
    violation_set = []
    for cons in range(len(constraints)):
        non_follow_cons = cons_function(dataset, row, cons)
        violation_set.append(dataset.loc[non_follow_cons])
        if len(non_follow_cons) == 0:
            continue
        count = non_follow_cons.sum()
        if count > 0:
            need_proj = True
            must_cons.add(cons)
    if not need_proj:
        return row.copy()

    # print(len(viable_cols))
    best_row = None
    min_distance = float('inf')
    comb_times = []
    viable_combs = 0
    j = 0
    for i in range(1, len(viable_cols) + 1):
        if i - j > 3:
            print(f"Skipping combinations of size {i} as it exceeds the limit of 4")
            break
        # Comb size l0 test
        # if best_row is not None:
        #     break
        random.shuffle(viable_cols)
        combs = list(itertools.combinations(viable_cols, i))
        for comb in combs:
            # best in comb test
            # if best_row is not None:
            #     break
            start_time = time.time()
            if args.mode != 'soft':
                if not all(any(feat in comb for feat in cons_feat[cons]) for cons in must_cons):
                    continue
            viable_combs += 1
            row_per = row.copy()
            comb_iters = []
            for col in comb[:-1]:
                if col in args.cont_feat:
                    if (dataset[col].max() - dataset[col].min()) > 100:
                        val_iter = range(dataset[col].min(), dataset[col].max() + 1,
                                         (dataset[col].max() - dataset[col].min()) // 100)
                    else:
                        val_iter = range(dataset[col].min(), dataset[col].max() + 1)
                    val_iter = list(val_iter)
                    val_iter = single_pred_cons(val_iter, col, dic_cols, constraints, unary_cons_lst_single)
                    val_iter.sort(key=lambda x: abs(x - row_per[col]))
                    if row_per[col] in val_iter:
                        val_iter.remove(row_per[col])
                else:
                    val_iter = list(dataset[col].unique())
                    val_iter = single_pred_cons(val_iter, col, dic_cols, constraints, unary_cons_lst_single)
                    random.shuffle(val_iter)
                    if row_per[col] in val_iter:
                        val_iter.remove(row_per[col])
                comb_iters.append([(val, col) for val in val_iter])
            all_combinations = itertools.product(*comb_iters)
            for vals in all_combinations:
                row_val = row_per.copy()
                for val, val_col in vals:
                    row_val[val_col] = val
                if args.mode == 'soft':
                    res = smart_project_soft(row_val, comb[-1], dataset, args.gamma * len(dataset))
                else:
                    # res = smart_project(row_val, comb[-1], dataset, constraints, dic_cols, cons_function)
                    if comb[-1] in args.cont_feat:
                        res = smart_project_intervals(row_val, comb[-1], dataset, constraints, dic_cols, cons_function,
                                                      args.cont_feat)
                    else:
                        res = smart_project(row_val, comb[-1], dataset, constraints, dic_cols, cons_function,
                                            args.cont_feat)
                    # res = smart_project_optimized_safe(row_val, comb[-1], dataset)
                if res is not None:
                    row_val[comb[-1]] = res
                    distance = compute_dist(
                        torch.tensor(transformer.transform(row_val.drop('label').to_frame().T).values).flatten(),
                        torch.tensor(transformer.transform(row.drop('label').to_frame().T).values).flatten(),
                        exp_random)
                    if distance < min_distance:
                        min_distance = distance
                        best_row = row_val
                        j = i
            end_time = time.time()
            comb_times.append(end_time - start_time)
    # print(f"Combination times: {comb_times}")
    # print(f"Viable combinations: {viable_combs}")
    # print(f"Mean combination time: {np.mean(comb_times)}")
    # print(f"STD combination time: {np.std(comb_times)}")
    print(best_row)
    return best_row


def project_instances(cf_example, projection_config):
    """
    Project individual counterfactual instances within a single DataFrame.

    Args:
        cf_example: DataFrame containing counterfactual examples
        projection_config: ProjectionConfig object containing all necessary parameters

    Returns:
        DataFrame of projected counterfactual instances
    """
    # Prepare the counterfactual DataFrame
    project_cfs_df = cf_example.copy()

    # Ensure categorical columns have proper categories
    # for col in projection_config.df.columns:
    #     if col in projection_config.args.cont_feat + ['label']:
    #         continue
    #     project_cfs_df[col] = pd.Categorical(
    #         project_cfs_df[col],
    #         categories=projection_config.d.data_df[col].cat.categories
    #     )
    for col in projection_config.df.columns:
        if col in projection_config.args.cont_feat + ['label']:
            continue

        categories = projection_config.category_mappings.get(col)
        if categories is not None:
            project_cfs_df[col] = pd.Categorical(project_cfs_df[col], categories=categories)

    # Initialize empty DataFrame for results
    projected_cfs_df = project_cfs_df.copy()
    projected_cfs_df = projected_cfs_df[1:0]  # Keep structure but remove all rows

    # Project each row
    for index, row in project_cfs_df.iterrows():
        print(f'Project instance {index}')

        st = time.time()
        proj_row = projection_config.projection_func(row, projection_config)
        et = time.time()

        elapsed_time = et - st
        projection_config.project_runtimes[projection_config.mode].append(elapsed_time)

        if proj_row is not None:
            # Compute distance metrics
            if not proj_row.equals(row):
                dist = compute_dist(
                    torch.tensor(projection_config.transformer.transform(
                        proj_row.drop('label').to_frame().T).values).flatten(),
                    torch.tensor(projection_config.transformer.transform(
                        row.drop('label').to_frame().T).values).flatten(),
                    projection_config.exp_random
                )

                l0_distance = (proj_row != row).sum()
                l1_distance = l1_distance_with_cont_feat(proj_row.drop('label'), row.drop('label'),
                                                         cont_feat=projection_config.args.cont_feat)
                projection_config.projection_metrics[projection_config.mode].append((dist, l0_distance, l1_distance))
            else:
                dist = 0.0
                l0_distance = 0
                l1_distance = 0
                print('Row is equal to original row')

            # Add to results
            projected_cfs_df.loc[len(projected_cfs_df)] = proj_row

    return projected_cfs_df


def project_counterfactuals(cf_examples_list, projection_config):
    """
    Project a list of counterfactual examples using the specified projection method.

    Args:
        cf_examples_list: List of counterfactual DataFrames to project
        projection_config: ProjectionConfig object containing all necessary parameters

    Returns:
        List of projected counterfactual DataFrames
    """
    all_instances_cfs = []
    for cf_example in cf_examples_list:
        projected_instances = project_instances(cf_example, projection_config)
        all_instances_cfs.append(projected_instances)
    return all_instances_cfs


def lin_model_counterfactuals(row, row_int, thresh, projection_config):
    """Generate counterfactuals using a linear model approach.

    Args:
        row: The original instance to generate counterfactuals for
        thresh: Threshold for the number of counterfactuals to generate
        projection_config: ProjectionConfig object containing all necessary parameters

    Returns:
        DataFrame of generated counterfactuals
    """
    print(f'Generating {thresh} counterfactuals for row: {row}')
    current_cfs = None
    found_points = None

    # Set mode to 'linear' for the linear model solver
    original_mode = projection_config.mode
    projection_config.mode = 'solver_linear'

    for i in range(thresh):
        st = time.time()
        projection_config.found_points = found_points
        cfs = project_solver(row, projection_config)
        elapsed_time = time.time() - st
        projection_config.project_runtimes['solver'].append(elapsed_time)

        if cfs is None:
            print(f'returned {len(current_cfs) if current_cfs is not None else 0} counterfactuals')

            # If we have some counterfactuals but not enough, pad with copies of the original row
            if current_cfs is not None and len(current_cfs) < thresh:
                remaining = thresh - len(current_cfs)

                # Create copies of the original row for padding
                row_copies = pd.concat([row.to_frame().T] * remaining, ignore_index=True)
                row_int_copies = pd.concat([row_int.to_frame().T] * remaining, ignore_index=True)

                # Add the copies to reach thresh length
                current_cfs = pd.concat([current_cfs, row_copies], ignore_index=True)
                found_points = pd.concat([found_points, row_int_copies], ignore_index=True)

            # If we have no counterfactuals at all, return thresh copies of the original row
            elif current_cfs is None:
                current_cfs = pd.concat([row.to_frame().T] * thresh, ignore_index=True)
                found_points = pd.concat([row_int.to_frame().T] * thresh, ignore_index=True)

            projection_config.mode = original_mode
            return current_cfs, found_points

        if isinstance(cfs, tuple):
            cf, cf_coded = cfs[0], cfs[1]
        else:
            cf = cfs
            cf_coded = cfs

        dist = compute_dist(
            torch.tensor(projection_config.transformer.transform(
                cf.drop('label', errors='ignore').to_frame().T).values).flatten(),
            torch.tensor(projection_config.transformer.transform(
                row.drop('label', errors='ignore').to_frame().T).values).flatten(),
            projection_config.exp_random
        )
        l1_distance = l1_distance_with_cont_feat(cf.drop('label', errors='ignore'), row.drop('label', errors='ignore'),
                                                 cont_feat=projection_config.args.cont_feat)
        projection_config.projection_metrics['solver'].append(
            (dist, (cf.drop('label', errors='ignore') != row.drop('label', errors='ignore')).sum(), l1_distance)
        )

        if current_cfs is None:
            current_cfs = cf.to_frame().T
            found_points = cf_coded.to_frame().T
        else:
            current_cfs = pd.concat([current_cfs, cf.to_frame().T], ignore_index=True)
            found_points = pd.concat([found_points, cf_coded.to_frame().T], ignore_index=True)

    projection_config.mode = original_mode
    return current_cfs, found_points


def bfs_counterfactuals(dice_cfs, projection_config, threshold, **proj_kwargs):
    transformer = projection_config.transformer
    exp_random = projection_config.exp_random
    exp_random_lin = projection_config.exp_random_lin
    model = projection_config.model
    df = projection_config.df
    args = projection_config.args
    category_mappings = projection_config.category_mappings
    categorical_column_names = projection_config.categorical_column_names
    projected_cfs = project_counterfactuals(dice_cfs, projection_config)
    all_accepted_instances = []
    for i, cfs in enumerate(projected_cfs):
        accepted_final_cfs = []
        cfs = cfs.drop('label', axis=1)
        if args.linear_pandp:
            cfs_int = cfs.copy()
            for col in categorical_column_names:
                if col in cfs_int.columns:
                    cfs_int[col] = cfs_int[col].map(category_mappings[col])
            probs = projection_config.model_lin.predict_proba(cfs_int)[:, 1]
        else:
            probs = model(torch.tensor(transformer.transform(cfs).values).float()).detach().numpy()
        labels = np.round(probs)
        accepted = cfs[labels == 1]
        print('###########ACCEPTED#############')
        print(accepted)
        accepted_final_cfs.append(accepted)

        not_accepted = cfs[labels == 0]
        print('###########NOT ACCEPTED#############')
        print(not_accepted)
        if len(not_accepted) == 0:
            return accepted
        features_to_vary = list(df.columns)
        for feat in args.fixed_feat + ['label']:
            features_to_vary.remove(feat)

        not_accepted_final_cfs = []
        while len(accepted) < threshold:
            print(f'Accepted so far: {len(accepted)}')
            print(f'Not accepted so far: {len(not_accepted)}')
            try:
                if args.linear_pandp:
                    for col in category_mappings:
                        not_accepted[col] = not_accepted[col].map(category_mappings[col])
                    dice_exp_random = exp_random_lin.generate_counterfactuals(
                        not_accepted,
                        total_CFs=threshold,
                        desired_class=1,
                        verbose=True,
                        features_to_vary=features_to_vary,
                    )
                    dice_cfs_orig = [
                        convert_codes_to_categories_df(dice_exp_random.cf_examples_list[i].final_cfs_df_sparse,
                                                       category_mappings) for i in
                        range(len(dice_exp_random.cf_examples_list))]
                else:
                    dice_exp_random = exp_random.generate_counterfactuals(not_accepted, total_CFs=threshold,
                                                                          desired_class=1,
                                                                          verbose=True,
                                                                          features_to_vary=features_to_vary,
                                                                          max_iter=500,
                                                                          learning_rate=6e-1,
                                                                          proximity_weight=args.delta / 100)
                    dice_cfs_orig = [dice_exp_random.cf_examples_list[i].final_cfs_df_sparse for i in
                                     range(len(dice_exp_random.cf_examples_list))]

            except:
                break

            projected_cfs_not_accepted = projected_cfs = project_counterfactuals(dice_cfs_orig, projection_config)
            # projected_cfs_not_accepted = project_counterfactuals(dice_exp_random, projection_func)
            # projected_cfs_not_accepted = project_counterfactuals([dice_exp_random.cf_examples_list[i].final_cfs_df_sparse for i in range(len(dice_exp_random.cf_examples_list))],df, args, d, exp_random, projection_func)
            not_accepted = None
            for cfs in projected_cfs_not_accepted:
                cfs = cfs.drop('label', axis=1)
                probs = model(
                    torch.tensor(transformer.transform(cfs).values).float()).detach().numpy()
                labels = np.round(probs)
                accepted_2 = cfs[labels == 1]
                not_accepted_final_cfs.append(accepted_2)
                if not_accepted is None:
                    not_accepted = cfs[labels == 0]
                else:
                    not_accepted = pd.concat([not_accepted, cfs[labels == 0]])
                # Filter for diversity instead of just dropping duplicates
                # not_accepted = filter_diverse_samples(not_accepted)
                if len(accepted_2) != 0:
                    accepted = pd.concat([accepted, accepted_2], ignore_index=True)
                    accepted_final_cfs.append(accepted_2)

        print('###########ACCEPTED NEW#############')
        print(accepted)
        return accepted
    pass


def load_constraints(path):
    f = open(path, 'r')
    constraints_txt = []
    for line in f:
        constraint = extract_names_and_conditions(line.rstrip())
        if 't1.' in line:
            if any(op in line for op in ['>=', '<=', '< ', '> ']):
                rev_constraint = []
                for pred in constraint:
                    if 't0.' in pred[0]:
                        rev_constraint.append((pred[0].replace('t0.', 't1.'), pred[1], (pred[2].replace('t1.', 't0.'))))
                    else:
                        rev_constraint.append((pred[0].replace('t1.', 't0.'), pred[1], (pred[2].replace('t1.', 't0.'))))
                # constraints_txt.append(rev_constraint)
                rev_constraint_fixed = []
                for pred in rev_constraint:
                    rev_constraint_fixed.append(
                        f'{pred[0].replace("t0.", "cf_row.").replace("t1.", "df.")} {pred[1]} {pred[2].replace("t0.", "cf_row.").replace("t1.", "df.")}')
                constraints_txt.append(rev_constraint_fixed)
        constraint_fixed = []
        for pred in constraint:
            constraint_fixed.append(
                f'{pred[0].replace("t0.", "cf_row.").replace("t1.", "df.")} {pred[1]} {pred[2].replace("t0.", "cf_row.").replace("t1.", "df.")}')
        constraints_txt.append(constraint_fixed)

    def cons_func(df, cf_row, cons_id, exclude=None):
        if exclude is None:
            exclude = []
        df_mask = '('
        for pred in constraints_txt[cons_id]:
            if (pred.split('.')[1].split(' ')[0] in exclude) and ('cf_row' in pred):
                continue
            df_mask += f'({pred}) & '
        # print(df_mask[:-1]+')')
        if len(df_mask[:-1]) == 0:
            if len(exclude) == 0:
                return pd.Series([False] * len(df), index=df.index)
            else:
                return pd.Series([True] * len(df), index=df.index)
        res = eval(df_mask[:-3] + ')')
        if type(res) not in [bool, np.bool_]:
            return res
        if not res:
            return pd.Series([False] * len(df), index=df.index)
        else:
            return pd.Series([True] * len(df), index=df.index)

    dic_cols = {}
    unary_cons_lst = []
    unary_cons_lst_single = []
    bin_cons = []
    cons_feat = [[] for _ in range(len(constraints_txt))]
    for index, cons in enumerate(constraints_txt):
        unary = True
        for pred in cons:
            col = pred.split('.')[1].split(' ')[0]
            if index not in dic_cols.get(col, []):
                dic_cols[col] = dic_cols.get(col, []) + [index]
            cons_feat[index].append(col)
            if unary:
                if 'df.' in pred:
                    unary = False
        if unary:
            if len(cons) == 1:
                unary_cons_lst_single.append(index)
            else:
                unary_cons_lst.append(index)
        else:
            bin_cons.append(index)
    f.close()
    return constraints_txt, dic_cols, cons_func, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons


def train_model(x_train, x_test, y_train, y_test, model_name='model.pkl', preload=False):
    file_name = model_name
    if preload:
        if os.path.exists(file_name):
            print('Loading linear model from file')
            with open(file_name, 'rb') as f:
                return pickle.load(f)
    model = SVC(kernel='linear', probability=True)
    # model = LogisticRegression(max_iter=1000,class_weight={0: 1, 1: 3})
    # model = LogisticRegression()
    print('Training linear model')
    model.fit(x_train.iloc[:1000], y_train.iloc[:1000])
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)
    print("Model accuracy score: " + str(accuracy_score(y_test.iloc[:20000], model.predict(x_test.iloc[:20000]))))
    return model

