from model import Model
import numpy as np
from math import floor, ceil, inf
from datetime import datetime, timedelta
from collections import defaultdict
import copy
import sys
import csv


# Zeilenaktivitäten im Ausgang
def get_row_activity(matrix, redund, lower, upper):
    row_act = defaultdict(list)

    for constraint in range(len(matrix)):
        if constraint in redund:
            continue

        row_index = np.where(matrix[constraint])[0]
        row_min, row_max = 0, 0

        for var in row_index:
            coeff = matrix[constraint][var]

            if coeff < 0:
                row_min += coeff * upper[var]
                row_max += coeff * lower[var]

            elif coeff > 0:
                row_min += coeff * lower[var]
                row_max += coeff * upper[var]

        row_act[constraint] = [row_min, row_max]
    return row_act


# best Shift - Algorithmus
def best_shift(variable, lower, upper, b, matrix, redund):
    column_index = np.setdiff1d(np.where(matrix[:, variable])[0], list(redund), assume_unique=True)
    q = list()

    for constraint in column_index:
        coeff = matrix[constraint][variable]

        if b[constraint] < 0 and coeff < 0:

            t_best = ceil(b[constraint] / coeff)

            if lower[variable] <= t_best <= upper[variable]:
                q.append((t_best, -1))

        elif b[constraint] >= 0 and coeff > 0:
            t_best = floor(b[constraint] / coeff) + 1

            if lower[variable] <= t_best <= upper[variable]:
                q.append((t_best, 1))

    if len(q) == 0:
        return 0

    sigma, t_star, t_before, phi_star = 0, 0, 0, 0
    q_sort = sorted(q, key=lambda x: x[0])

    for element in q_sort:
        if element[0] > t_before and sigma < phi_star:
            phi_star, t_star = sigma, t_before
        t_before = element[0]
        sigma += element[1]

    if sigma < phi_star:
        t_star = t_before

    return t_star


# Herausnahme aller stetigen Variablen
def eliminate(b, matrix, con):
    if len(con) > 0:
        for var in con:
            column_index = np.where(matrix[:, var])[0]

            for constraint in column_index:

                coeff = matrix[constraint][var]

                if coeff < 0:
                    b[constraint] -= coeff * m.var_name[var].ub
                elif coeff > 0:
                    b[constraint] -= coeff * m.var_name[var].lb
                matrix[constraint][var] = 0


# Reihenfolge der Variablen anhand ursprünglicher Verletzung
def first_violation(b, matrix, integers):
    count = np.full(len(matrix[0]), 0)
    for var in integers:
        column_index = np.where(matrix[:, var])[0]

        if all(b[k] < 0 for k in column_index):
            count[var] = len(column_index)

        else:
            for constraint in column_index:
                if b[constraint] < 0:
                    count[var] += 1

    return np.argsort(count, kind='stable')[::-1]


# Reihenfolge der Variablen anhand der Bedeutung im Problem
def importance(matrix, integers):
    count = np.full(len(matrix[0]), 0)

    for var in integers:
        column_index = np.where(matrix[:, var])[0]
        sums = 0

        for constraint in column_index:
            sums += abs(matrix[constraint][var])

        count[var] += sums + len(column_index)

    return np.argsort(count, kind='stable')[::-1]


# Menge aller redundanten Nebenbedingungen
def redundant(b):
    redund = set()

    for k, value in enumerate(b):
        if value == inf:
            redund.add(k)

    return redund


# propagation des Shifts von j über t
def propagate(t_val, variable, b, matrix, redund, row_act, lower, upper, fix, propagated):
    column_index = np.setdiff1d(np.where(matrix[:, variable])[0], list(redund), assume_unique=True)
    b_orig, fix_orig, row_act_orig, lower_orig, upper_orig = \
        b.copy(), fix.copy(), row_act.copy(), lower.copy(), upper.copy()

    temp_lower, temp_upper = lower[variable], upper[variable]
    lower[variable], upper[variable] = t_val, t_val
    fix.add(variable)

    for constraint in column_index:

        update_act_right(variable, constraint, b, matrix, row_act, lower, temp_lower, temp_upper, redund, propagated)

        if row_act[constraint][0] > b[constraint]:
            return False, b_orig, fix_orig, lower_orig, upper_orig, row_act_orig

        if constraint in propagated:
            continue

        row_vars = np.setdiff1d(np.where(matrix[constraint])[0], list(fix), assume_unique=True)

        for var in row_vars:

            if var == variable:
                continue

            if not linear_prop(var, constraint, b, matrix, row_act, lower, upper, fix, redund, propagated):
                return False, b_orig, fix_orig, lower_orig, upper_orig, row_act_orig
    return True, b, fix, lower, upper, row_act


# propagation aller anderen involvierten Variablen
def linear_prop(variable, constraint, b, matrix, row_act, lower, upper, fix, redund, propagated):
    coefficient = matrix[constraint][variable]
    new_lb, new_ub = lower[variable], upper[variable]

    if coefficient < 0:
        new_lb = ceil((b[constraint] - (row_act[constraint][0] - coefficient * upper[variable])) / coefficient)

    elif coefficient > 0:
        new_ub = floor((b[constraint] - (row_act[constraint][0] - coefficient * lower[variable])) / coefficient)

    if new_ub < lower[variable] or new_lb > upper[variable]:
        return False

    temp_lb, temp_ub = lower[variable], upper[variable]

    if lower[variable] < new_lb <= upper[variable]:
        lower[variable] = new_lb

    if lower[variable] <= new_ub < upper[variable]:
        upper[variable] = new_ub

    if lower[variable] == upper[variable]:
        column = np.setdiff1d(np.where(matrix[:, variable])[0], list(redund), assume_unique=True)
        fix.add(variable)

        for const in column:
            update_act_right(variable, const, b, matrix, row_act, lower, temp_lb, temp_ub, redund, propagated)

            if row_act[const][0] > b[const]:
                return False

    propagated.add(constraint)
    return True


# Aktualisierung der Zeilenaktivitäten
def update_act_right(variable, constraint, b, matrix, row_act, lower, old_lb, old_ub, redund, propagated):
    b_bef = b[constraint]
    coefficient = matrix[constraint][variable]
    b[constraint] -= coefficient * lower[variable]

    if coefficient < 0:
        row_act[constraint][0] -= coefficient * old_ub
        row_act[constraint][1] -= coefficient * old_lb

    elif coefficient > 0:
        row_act[constraint][0] -= coefficient * old_lb
        row_act[constraint][1] -= coefficient * old_ub

    if row_act[constraint][1] <= b[constraint]:
        redund.add(constraint)

    if b_bef != b[constraint] and constraint in propagated:
        propagated.remove(constraint)


if __name__ == '__main__':
    m = Model(sys.argv[1])
    m.normalize()

    coefficient_matrix = copy.deepcopy(m.matrix)
    right = copy.deepcopy(m.right_hand_side())
    lower_bound = {key: int(val.lb) for key, val in enumerate(m.variables) if m.get_type(val) != 'C'}
    upper_bound = {key: int(val.ub) for key, val in enumerate(m.variables) if m.get_type(val) != 'C'}

    continuous = np.array([k for k in range(m.model.NumVars) if m.get_type(m.var_name[k]) == 'C'])
    integer = np.array([k for k in range(m.model.NumVars) if m.get_type(m.var_name[k]) != 'C'])

    eliminate(right, coefficient_matrix, continuous)
    redundant_list = redundant(right)
    row_activity = get_row_activity(coefficient_matrix, redundant_list, lower_bound, upper_bound)

    # zufällige Reihenfolge
    # variables = integer.copy()
    # np.random.shuffle(variables)

    # first violation Reihenfolge
    variables = np.setdiff1d(first_violation(right, coefficient_matrix, integer), continuous, assume_unique=True)

    # importance Reihenfolge
    # variables = np.setdiff1d(importance(coefficient_matrix, integer), continuous, assume_unique=True)

    cutoffs = set()
    fixed = set()
    backtrack_limit = 15
    prop = set()
    timelimit = 2

    start = datetime.now()
    # Start der Heuristik
    while not all(right >= 0) and len(variables) > 0:
        next_var = variables[0]

        t = best_shift(next_var, lower_bound, upper_bound, right, coefficient_matrix, redundant_list)

        decision, right, fixed, lower_bound, upper_bound, row_activity = propagate(t, next_var, right,
                                                                                   coefficient_matrix,
                                                                                   redundant_list, row_activity,
                                                                                   lower_bound, upper_bound, fixed,
                                                                                   prop)

        if decision:
            variables = np.array([x for x in variables if x not in fixed])

        else:
            cutoffs.add(next_var)
            variables = np.array([x for x in variables if x not in cutoffs])

            if len(cutoffs) > backtrack_limit:
                break

        if datetime.now() - start > timedelta(minutes=timelimit):
            break

    current = np.zeros(m.model.NumVars)

    if all(right >= 0):
        for i in integer:
            if i in fixed:
                m.var_name[i].lb, m.var_name[i].ub = lower_bound[i], upper_bound[i]
                current[i] = lower_bound[i]
            else:
                m.var_name[i].lb, m.var_name[i].ub = 0, 0

    end = datetime.now()

    print()
    print('Time: ', end - start)

    if all(right >= 0):
        if len(continuous) > 0:
            m.model.optimize()
            if m.model.getAttr('Status') != 2:
                print('no solution')
            else:
                print('Solution: ', m.optimum(m.current_solution()))
        else:
            print('Solution: ', m.optimum(current))
    else:
        print('no solution')
        print()
