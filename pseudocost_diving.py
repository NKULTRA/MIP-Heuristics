from model import Model
import numpy as np
from math import isclose, ceil, floor, sqrt
from datetime import datetime, timedelta
import sys
import csv


# Menge der gebrochenen Variablen
def fraction_index(sol):
    return np.unique(
        [k for k in range(len(sol)) if ((sol[k] != round(sol[k]) and m.get_type(m.var_name[k]) != "C")
                                        and not isclose(sol[k], round(sol[k]), abs_tol=0.000001))]
    )


# Menge der binären Variablen
def get_binary(variables):
    return np.array(
        [k for k in variables if (m.var_name[k].lb == 0 and m.var_name[k].ub == 1)]
    )


# Kennzeichnung aller trivial rundbarer Variablen und welcher Art von Rundung - auf oder ab
def trivial(variable):
    column = m.get_col(variable)
    col_index = m.get_col_index(variable)
    if all([column[i] >= 0 for i in col_index]):
        return 1
    if all([column[i] <= 0 for i in col_index]):
        return 2
    return 0


# Feststellung der Rundungsrichtung
def calculate(lp_sol, variables, trivial_vars, curr_solution, ps_down, ps_up, roundup, curr_obj, obj):
    ratio = dict()
    for index in variables:
        ps_down, ps_up = update_pseudocosts(ps_down, ps_up, index, curr_solution, curr_obj, obj)

        if trivial_vars[index] == 0:
            if abs(curr_solution[index] - lp_sol[index]) >= 0.4:
                if curr_solution[index] < lp_sol[index]:
                    roundup[index] = False
                else:
                    roundup[index] = True

            elif abs(curr_solution[index] - round(curr_solution[index])) < 0.3:
                # wegen Rundungsfehler
                if isclose((curr_solution[index] - abs(curr_solution[index] - round(curr_solution[index]))),
                           round(curr_solution[index]), abs_tol=0.000001):
                    roundup[index] = False
                else:
                    roundup[index] = True

            else:
                if ps_down[index][0] < ps_up[index][0]:
                    roundup[index] = False
                else:
                    roundup[index] = True
        else:
            if trivial_vars[index] == 1:
                roundup[index] = True
            else:
                roundup[index] = False

        if roundup[index]:
            ratio[index] = sqrt(abs(curr_solution[index] - round(curr_solution[index]))) * \
                           ((1 + ps_down[index][0]) / (1 + ps_up[index][0]))
        else:
            ratio[index] = sqrt(1 - abs(curr_solution[index] - round(curr_solution[index]))) * \
                           ((1 + ps_up[index][0]) / (1 + ps_down[index][0]))
    return roundup, ratio, ps_down, ps_up


# Aktualisierung der Pseudokosten
def update_pseudocosts(ps_down, ps_up, var, curr_solution, curr_obj, obj):
    curr_ps_down = pseudocosts(curr_solution, var, "down", curr_obj, obj)
    curr_ps_up = pseudocosts(curr_solution, var, "up", curr_obj, obj)

    if var in ps_down:
        count = ps_down[var][1]
        old_ps_down = ps_down[var][0]
        old_ps_up = ps_up[var][0]
    else:
        count = 1
        old_ps_down = 0
        old_ps_up = 0

    ps_down[var] = [(old_ps_down * count + curr_ps_down) / (count + 1), (count + 1)]
    ps_up[var] = [(old_ps_up * count + curr_ps_up) / (count + 1), (count + 1)]

    return ps_down, ps_up


# Berechnung der Pseudokosten in einem Knoten
def pseudocosts(curr_solution, variable, direction, curr_obj, obj):
    coeff = obj[variable]
    if direction == "down":
        new_obj = curr_obj - coeff * (curr_solution[variable] - floor(curr_solution[variable]))
        return abs(curr_obj - new_obj) / (curr_solution[variable] - floor(curr_solution[variable]))

    new_obj = curr_obj + coeff * (ceil(curr_solution[variable]) - curr_solution[variable])
    return abs(curr_obj - new_obj) / (ceil(curr_solution[variable]) - curr_solution[variable])


# Simple Round Algorithmus
def simple_rounding(curr):
    ind = fraction_index(curr)
    solved = True

    for var in ind:

        col = m.get_col(var)
        col_index = m.get_col_index(var)

        if all([col[i] >= 0 for i in col_index]):
            curr[var] = floor(curr[var])
        elif all([col[i] <= 0 for i in col_index]):
            curr[var] = ceil(curr[var])
        else:
            solved = False
            break
    return solved, curr


if __name__ == '__main__':
    m = Model(sys.argv[1])
    m.relaxation()

    lp_solution = m.current_solution()
    current = m.current_solution()
    objective = m.objective()
    current_objective = np.sum(current * objective)
    pseudo_down, pseudo_up = dict(), dict()
    new_value = dict()

    indices = fraction_index(current)
    binary = get_binary(indices)

    # wenn binäre Variablen vorhanden, dann Vorzug
    if len(binary) > 0:
        indices = binary

    trivial_round = dict(zip(indices, map(trivial, indices)))

    # Ausschluss aller trivial rundbaren Variablen
    if len(trivial_round) > 0:
        real_indices = np.array(
            [k for k in indices if trivial_round[k] == 0]
        )
        if len(real_indices) > 0:
            indices = real_indices

    cnt = 0
    limit = 10000000
    timelimit = 2
    solution = False
    simple_solution = False
    if len(indices) == 0:
        solution = True

    start = datetime.now()
    # Start Heuristik
    while cnt < limit and len(indices) > 0:

        new_value, quotient, pseudo_down, pseudo_up = calculate(lp_solution, indices, trivial_round, current,
                                                                pseudo_down, pseudo_up, new_value,
                                                                sum(current * objective), objective)
        # max Ratio
        change = max(quotient, key=quotient.get)

        if new_value[change]:
            m.var_name[change].lb = ceil(current[change])
            m.var_name[change].ub = ceil(current[change])
        elif not new_value[change]:
            m.var_name[change].lb = floor(current[change])
            m.var_name[change].ub = floor(current[change])

        m.model.optimize()

        # Anzahl innerhalb der Optimierung zählt dazu
        cnt += 1 + m.model.getAttr("IterCount")

        # wenn Model unlösbar, dann switch auf andere Lösung -> wegen binären Variablen
        if m.model.getAttr("Status") == 3:
            if new_value[change]:
                m.var_name[change].lb = floor(current[change])
                m.var_name[change].ub = floor(current[change])
            elif not new_value[change]:
                m.var_name[change].lb = ceil(current[change])
                m.var_name[change].ub = ceil(current[change])

            m.model.optimize()

            cnt += m.model.getAttr("IterCount")

            # wenn weiterhin unlösbar, dann Ende
            if m.model.getAttr('Status') == 3:
                break

        current = m.current_solution()
        indices = fraction_index(current)
        binary = get_binary(indices)

        if len(binary) > 0:
            indices = binary

        trivial_round = dict(zip(indices, map(trivial, indices)))

        if len(trivial_round) > 0:
            real_indices = np.array(
                [k for k in indices if trivial_round[k] == 0]
            )
            if len(real_indices) > 0:
                indices = real_indices

        if len(indices) == 0:
            solution = True
            break

        it_is, simple = simple_rounding(current.copy())

        if it_is:
            current = simple
            solution = True
            simple_solution = True
            break

        if datetime.now() - start > timedelta(minutes=timelimit):
            break

    end = datetime.now()

    if solution:
        print()
        print("Iterations: ", cnt)
        print("Time: ", end - start)
        print()
        print("MIP Startwert: " + str(m.optimum(current)))
    else:
        print()
        print('no solution!')
        print('Iterations: ', cnt)
        print("Time: ", end - start)
        print()
