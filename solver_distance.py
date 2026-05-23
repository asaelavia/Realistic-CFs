"""
Custom distance function for Z3 solver incorporating semantic categorical distances.
"""

from z3.z3 import *
import numpy as np

# =============================================================================
# CATEGORICAL DISTANCE MATRICES (from categorical_distance.py)
# =============================================================================

# WORKCLASS
WORKCLASS_GROUPS = {
    'State_gov': 'gov', 'Federal_gov': 'gov', 'Local_gov': 'gov',
    'Private': 'private',
    'Self_emp_not_inc': 'self_emp', 'Self_emp_inc': 'self_emp',
    'Without_pay': 'other', 'Never_worked': 'other'
}

WORKCLASS_GROUP_DIST = {
    ('gov', 'gov'): 0.5, ('private', 'private'): 0, ('self_emp', 'self_emp'): 0.5, ('other', 'other'): 0.5,
    ('gov', 'private'): 1, ('gov', 'self_emp'): 2, ('gov', 'other'): 3,
    ('private', 'self_emp'): 1.5, ('private', 'other'): 3, ('self_emp', 'other'): 3,
}

# EDUCATION (Ordinal)
EDUCATION_ORDINAL = {
    'Preschool': 0, '1st_4th': 2, '5th_6th': 4, '7th_8th': 6, '9th': 7, '10th': 8,
    '11th': 9, '12th': 10, 'HS_grad': 11, 'Some_college': 12, 'Assoc_voc': 13,
    'Assoc_acdm': 13, 'Bachelors': 15, 'Masters': 17, 'Prof_school': 19, 'Doctorate': 20
}
EDUCATION_SCALE = 0.2

# MARITAL STATUS
MARITAL_GROUPS = {
    'Never_married': 'never', 'Married_civ_spouse': 'married', 'Married_spouse_absent': 'married',
    'Married_AF_spouse': 'married', 'Separated': 'separated', 'Divorced': 'divorced', 'Widowed': 'widowed'
}

MARITAL_DIST = {
    ('never', 'never'): 0, ('never', 'married'): 1.5, ('never', 'separated'): 3,
    ('never', 'divorced'): 3, ('never', 'widowed'): float('inf'),
    ('married', 'never'): float('inf'), ('married', 'married'): 0.5, ('married', 'separated'): 1,
    ('married', 'divorced'): 1.5, ('married', 'widowed'): float('inf'),
    ('separated', 'never'): float('inf'), ('separated', 'married'): 1, ('separated', 'separated'): 0,
    ('separated', 'divorced'): 1, ('separated', 'widowed'): float('inf'),
    ('divorced', 'never'): float('inf'), ('divorced', 'married'): 1.5, ('divorced', 'separated'): 3,
    ('divorced', 'divorced'): 0, ('divorced', 'widowed'): float('inf'),
    ('widowed', 'never'): float('inf'), ('widowed', 'married'): 1.5, ('widowed', 'separated'): 3,
    ('widowed', 'divorced'): 3, ('widowed', 'widowed'): 0,
}

# OCCUPATION
OCCUPATION_TIERS = {
    'Prof_specialty': 'P', 'Exec_managerial': 'P',
    'Tech_support': 'S', 'Sales': 'S', 'Adm_clerical': 'S', 'Craft_repair': 'S', 'Protective_serv': 'S',
    'Other_service': 'V', 'Priv_house_serv': 'V',
    'Handlers_cleaners': 'L', 'Machine_op_inspct': 'L', 'Transport_moving': 'L', 'Farming_fishing': 'L',
    'Armed_Forces': 'X'
}

OCCUPATION_TIER_DIST = {
    ('P', 'P'): 0.5, ('S', 'S'): 0.5, ('V', 'V'): 0.5, ('L', 'L'): 0.5, ('X', 'X'): 0,
    ('P', 'S'): 2, ('P', 'V'): 3, ('P', 'L'): 4, ('P', 'X'): 3,
    ('S', 'V'): 1, ('S', 'L'): 1.5, ('S', 'X'): 3,
    ('V', 'L'): 1, ('V', 'X'): 3, ('L', 'X'): 3,
}

# RELATIONSHIP
RELATIONSHIP_DIST = {
    ('Husband', 'Husband'): 0, ('Husband', 'Wife'): float('inf'), ('Husband', 'Not_in_family'): 1.5,
    ('Husband', 'Own_child'): float('inf'), ('Husband', 'Unmarried'): 1.5, ('Husband', 'Other_relative'): 2,
    ('Wife', 'Husband'): float('inf'), ('Wife', 'Wife'): 0, ('Wife', 'Not_in_family'): 1.5,
    ('Wife', 'Own_child'): float('inf'), ('Wife', 'Unmarried'): 1.5, ('Wife', 'Other_relative'): 2,
    ('Not_in_family', 'Husband'): 1.5, ('Not_in_family', 'Wife'): 1.5, ('Not_in_family', 'Not_in_family'): 0,
    ('Not_in_family', 'Own_child'): 1.5, ('Not_in_family', 'Unmarried'): 0.5, ('Not_in_family', 'Other_relative'): 1.5,
    ('Own_child', 'Husband'): 3, ('Own_child', 'Wife'): 3, ('Own_child', 'Not_in_family'): 1.5,
    ('Own_child', 'Own_child'): 0, ('Own_child', 'Unmarried'): 1.5, ('Own_child', 'Other_relative'): 1.5,
    ('Unmarried', 'Husband'): 1.5, ('Unmarried', 'Wife'): 1.5, ('Unmarried', 'Not_in_family'): 0.5,
    ('Unmarried', 'Own_child'): 1.5, ('Unmarried', 'Unmarried'): 0, ('Unmarried', 'Other_relative'): 1.5,
    ('Other_relative', 'Husband'): 2, ('Other_relative', 'Wife'): 2, ('Other_relative', 'Not_in_family'): 1.5,
    ('Other_relative', 'Own_child'): float('inf'), ('Other_relative', 'Unmarried'): 1.5, ('Other_relative', 'Other_relative'): 0,
}


# =============================================================================
# HELPER FUNCTIONS TO BUILD Z3 DISTANCE EXPRESSIONS
# =============================================================================

def get_categorical_distance_value(col, val1, val2):
    """
    Get the distance between two categorical values for a given column.
    Returns a float distance value.
    """
    if val1 == val2:
        return 0.0
    
    if col == 'workclass':
        g1 = WORKCLASS_GROUPS.get(val1)
        g2 = WORKCLASS_GROUPS.get(val2)
        if g1 is None or g2 is None:
            return 1.0  # fallback
        key = (g1, g2) if (g1, g2) in WORKCLASS_GROUP_DIST else (g2, g1)
        return WORKCLASS_GROUP_DIST.get(key, 1.0)
    
    elif col == 'education':
        o1 = EDUCATION_ORDINAL.get(val1)
        o2 = EDUCATION_ORDINAL.get(val2)
        if o1 is None or o2 is None:
            return 1.0
        return abs(o1 - o2) * EDUCATION_SCALE
    
    elif col == 'marital_status':
        g1 = MARITAL_GROUPS.get(val1)
        g2 = MARITAL_GROUPS.get(val2)
        if g1 is None or g2 is None:
            return 1.0
        dist = MARITAL_DIST.get((g1, g2), 1.0)
        return dist if dist != float('inf') else 1000.0  # Large value for impossible transitions
    
    elif col == 'occupation':
        t1 = OCCUPATION_TIERS.get(val1)
        t2 = OCCUPATION_TIERS.get(val2)
        if t1 is None or t2 is None:
            return 1.0
        key = (t1, t2) if (t1, t2) in OCCUPATION_TIER_DIST else (t2, t1)
        return OCCUPATION_TIER_DIST.get(key, 1.0)
    
    elif col == 'relationship':
        dist = RELATIONSHIP_DIST.get((val1, val2), 1.0)
        return dist if dist != float('inf') else 1000.0
    
    else:
        # Default: simple 0/1 distance
        return 1.0


def build_categorical_distance_z3(var, current_value, col, category_codes, category_names):
    """
    Build a Z3 expression for categorical distance.
    
    Args:
        var: Z3 Int variable representing the categorical value (as code)
        current_value: The current category name (string)
        col: Column name
        category_codes: Dict mapping category name -> integer code
        category_names: List of category names in order of codes
    
    Returns:
        Z3 expression for the distance
    """
    # Start with default distance (for unknown categories)
    expr = RealVal(1.0)
    
    # Build nested If expressions for each possible target category
    for target_name in category_names:
        target_code = category_codes.get(target_name)
        if target_code is None:
            continue
        
        dist = get_categorical_distance_value(col, current_value, target_name)
        expr = If(var == target_code, RealVal(dist), expr)
    
    return expr


# =============================================================================
# NON-SOLVER DISTANCE CALCULATION (between two samples)
# =============================================================================

def calculate_custom_distance(row1, row2, cont_feat, normalized_medians=None, norm_params=None):
    """
    Calculate custom distance between two samples without Z3 solver.
    Uses semantic categorical distances and MAD-normalized continuous distances.
    
    Args:
        row1: First sample (pandas Series) - original values (not coded)
        row2: Second sample (pandas Series) - original values (not coded)
        cont_feat: List of continuous feature names
        normalized_medians: Dict of MAD values per feature (optional)
        norm_params: Dict with 'min' and 'range' per feature (optional)
    
    Returns:
        float: Total distance between the two samples
    """
    distance = 0.0
    
    for col in row1.index:
        if col == 'label':
            continue
        
        val1 = row1[col]
        val2 = row2[col]
        
        if col in cont_feat:
            # Continuous: MAD-normalized distance
            if normalized_medians and norm_params and col in normalized_medians:
                med_scaled = normalized_medians[col] * norm_params[col]['range']
                if med_scaled <= 1e-10:
                    med_scaled = 1.0
                distance += abs(val1 - val2) / med_scaled
            else:
                distance += abs(val1 - val2)
        else:
            # Categorical: custom semantic distance
            distance += get_categorical_distance_value(col, val1, val2)
    
    return distance


def calculate_custom_distance_breakdown(row1, row2, cont_feat, normalized_medians=None, norm_params=None):
    """
    Calculate custom distance with per-feature breakdown.
    
    Returns:
        dict: Distance per feature and total
    """
    breakdown = {}
    total = 0.0
    
    for col in row1.index:
        if col == 'label':
            continue
        
        val1 = row1[col]
        val2 = row2[col]
        
        if col in cont_feat:
            if normalized_medians and norm_params and col in normalized_medians:
                med_scaled = normalized_medians[col] * norm_params[col]['range']
                if med_scaled <= 1e-10:
                    med_scaled = 1.0
                dist = abs(val1 - val2) / med_scaled
            else:
                dist = abs(val1 - val2)
        else:
            dist = get_categorical_distance_value(col, val1, val2)
        
        breakdown[col] = dist
        total += dist
    
    breakdown['total'] = total
    return breakdown


# =============================================================================
# INTEGRATION WITH _try_projection_combination
# =============================================================================

def build_distance_for_solver(vars, row, projection_config, category_mappings):
    """
    Drop-in replacement for the distance calculation in _try_projection_combination.
    
    USAGE in _try_projection_combination:
    
    Replace this:
        distance = 0
        for var in vars:
            if var in args.cont_feat:
                med_scaled = projection_config.normalized_medians[var] * projection_config.norm_params[var]['range']
                if med_scaled <= 1e-10:
                    med_scaled = 1.0
                distance += Abs(vars[var] - row[var]) / med_scaled
            else:
                distance += If(vars[var] == row[var], 0, 1)
    
    With this:
        from solver_distance import build_distance_for_solver
        distance = build_distance_for_solver(vars, row, projection_config, category_mappings)
    
    Args:
        vars: Dict of Z3 variables {column_name: z3_var}
        row: Current row values (pandas Series with integer codes for categoricals)
        projection_config: ProjectionConfig object with args, normalized_medians, norm_params
        category_mappings: Dict mapping column -> {category_name: code}
    
    Returns:
        Z3 expression for total distance
    """
    args = projection_config.args
    
    distance = RealVal(0)
    
    for var_name, var in vars.items():
        if var_name == 'label':
            continue
            
        if var_name in args.cont_feat:
            # Continuous: MAD-normalized distance
            med_scaled = projection_config.normalized_medians[var_name] * projection_config.norm_params[var_name]['range']
            if med_scaled <= 1e-10:
                med_scaled = 1.0
            distance = distance + Abs(var - row[var_name]) / med_scaled
            
        elif var_name in category_mappings:
            # Categorical: custom semantic distance
            reverse_mapping = {v: k for k, v in category_mappings[var_name].items()}
            current_name = reverse_mapping[int(row[var_name])]
            category_names = list(category_mappings[var_name].keys())
            
            cat_dist = build_categorical_distance_z3(
                var, current_name, var_name,
                category_mappings[var_name], category_names
            )
            distance = distance + cat_dist
        else:
            # Fallback: simple 0/1 distance (shouldn't reach here normally)
            distance = distance + If(var == row[var_name], 0, 1)
    
    return distance


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Create Z3 variables and compute distance
    from z3 import Int, Real, Optimize, Abs, If
    
    # Simulate category mappings
    category_mappings = {
        'workclass': {'Private': 0, 'State_gov': 1, 'Self_emp_not_inc': 2},
        'education': {'HS_grad': 0, 'Bachelors': 1, 'Masters': 2, 'Doctorate': 3},
        'marital_status': {'Never_married': 0, 'Married_civ_spouse': 1, 'Divorced': 2},
        'occupation': {'Tech_support': 0, 'Prof_specialty': 1, 'Exec_managerial': 2},
        'relationship': {'Not_in_family': 0, 'Husband': 1, 'Wife': 2}
    }
    
    # Create Z3 variables
    vars = {
        'age': Int('age'),
        'education_num': Int('education_num'),
        'hours_per_week': Int('hours_per_week'),
        'workclass': Int('workclass'),
        'education': Int('education'),
        'marital_status': Int('marital_status'),
        'occupation': Int('occupation'),
        'relationship': Int('relationship')
    }
    
    # Simulate current row (with integer codes)
    import pandas as pd
    row = pd.Series({
        'age': 35,
        'education_num': 13,
        'hours_per_week': 40,
        'workclass': 0,  # Private
        'education': 1,  # Bachelors
        'marital_status': 0,  # Never_married
        'occupation': 0,  # Tech_support
        'relationship': 0  # Not_in_family
    })
    
    # Test categorical distance
    print("Testing categorical distance values:")
    print(f"workclass: Private -> State_gov = {get_categorical_distance_value('workclass', 'Private', 'State_gov')}")
    print(f"education: Bachelors -> Masters = {get_categorical_distance_value('education', 'Bachelors', 'Masters')}")
    print(f"marital_status: Never_married -> Divorced = {get_categorical_distance_value('marital_status', 'Never_married', 'Divorced')}")
    print(f"occupation: Tech_support -> Prof_specialty = {get_categorical_distance_value('occupation', 'Tech_support', 'Prof_specialty')}")
    print(f"relationship: Not_in_family -> Husband = {get_categorical_distance_value('relationship', 'Not_in_family', 'Husband')}")
    
    # Build Z3 distance expression for one categorical variable
    print("\nBuilding Z3 expression for workclass distance:")
    workclass_dist = build_categorical_distance_z3(
        vars['workclass'], 
        'Private',  # current value
        'workclass',
        category_mappings['workclass'],
        list(category_mappings['workclass'].keys())
    )
    print(f"Z3 expression: {workclass_dist}")
    
    # Test with solver
    print("\nTesting with Z3 solver:")
    s = Optimize()
    
    # Add bounds
    s.add(vars['workclass'] >= 0, vars['workclass'] <= 2)
    
    # Minimize distance
    s.minimize(workclass_dist)
    
    if s.check() == sat:
        m = s.model()
        print(f"Optimal workclass code: {m[vars['workclass']]}")
        reverse_mapping = {v: k for k, v in category_mappings['workclass'].items()}
        print(f"Optimal workclass: {reverse_mapping[m[vars['workclass']].as_long()]}")
    
    # ==========================================================================
    # EXAMPLE: How to modify _try_projection_combination
    # ==========================================================================
    print("\n" + "="*70)
    print("INTEGRATION EXAMPLE FOR _try_projection_combination")
    print("="*70)
    
    example_code = '''
    # In _try_projection_combination, replace:
    
    # OLD CODE:
    st = time.time()
    distance = 0
    
    for var in vars:
        if var in args.cont_feat:
            med_scaled = projection_config.normalized_medians[var] * projection_config.norm_params[var]['range']
            if med_scaled <= 1e-10:
                med_scaled = 1.0
            distance += Abs(vars[var] - row[var]) / med_scaled
        else:
            distance += If(vars[var] == row[var], 0, 1)
    
    # NEW CODE:
    from solver_distance import build_distance_for_solver
    
    st = time.time()
    distance = build_distance_for_solver(vars, row, projection_config, category_mappings)
    '''
    print(example_code)
    
    # ==========================================================================
    # Test non-solver distance calculation
    # ==========================================================================
    print("\n" + "="*70)
    print("TESTING NON-SOLVER DISTANCE CALCULATION")
    print("="*70)
    
    # Create two sample rows (with original string values, not codes)
    row1 = pd.Series({
        'age': 35,
        'education_num': 13,
        'hours_per_week': 40,
        'workclass': 'Private',
        'education': 'Bachelors',
        'marital_status': 'Never_married',
        'occupation': 'Tech_support',
        'relationship': 'Not_in_family'
    })
    
    row2 = pd.Series({
        'age': 40,
        'education_num': 16,
        'hours_per_week': 45,
        'workclass': 'State_gov',
        'education': 'Masters',
        'marital_status': 'Married_civ_spouse',
        'occupation': 'Prof_specialty',
        'relationship': 'Husband'
    })
    
    cont_feat = ['age', 'education_num', 'hours_per_week']
    
    # Simple distance (no MAD normalization)
    simple_dist = calculate_custom_distance(row1, row2, cont_feat)
    print(f"\nSimple distance (no MAD): {simple_dist}")
    
    # With breakdown
    breakdown = calculate_custom_distance_breakdown(row1, row2, cont_feat)
    print("\nDistance breakdown:")
    for col, dist in breakdown.items():
        if col != 'total':
            print(f"  {col}: {dist}")
    print(f"  TOTAL: {breakdown['total']}")
    
    # ==========================================================================
    # Integration example for project_instances
    # ==========================================================================
    print("\n" + "="*70)
    print("INTEGRATION EXAMPLE FOR project_instances")
    print("="*70)
    
    project_instances_code = '''
    # In project_instances, add after computing proj_row:
    
    from solver_distance import calculate_custom_distance
    
    # After: proj_row = projection_config.projection_func(row, projection_config)
    
    if proj_row is not None and not proj_row.equals(row):
        # Existing distance calculations...
        dist = compute_dist(...)
        l0_distance = (proj_row != row).sum()
        l1_distance = l1_distance_with_cont_feat(...)
        
        # NEW: Custom semantic distance
        custom_dist = calculate_custom_distance(
            row.drop('label'),
            proj_row.drop('label'),
            cont_feat=projection_config.args.cont_feat,
            normalized_medians=projection_config.normalized_medians,
            norm_params=projection_config.norm_params
        )
        
        # Add to metrics
        projection_config.projection_metrics[projection_config.mode].append(
            (dist, l0_distance, l1_distance, custom_dist)  # Added custom_dist
        )
    '''
    print(project_instances_code)