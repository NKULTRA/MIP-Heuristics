from model import Model
import numpy as np
from datetime import datetime
from math import isclose
import sys
import csv


# Menge der gebrochenen Variablen
def fraction_index(sol):
    return np.unique(
        [k for k in range(len(sol)) if (sol[k] != round(sol[k]) and m.get_type(m.var_name[k]) != "C")]
    )


# Menge aller positiven slacks
def get_pos_slacks(cur_slacks):
    return set(
        ind for ind in range(len(cur_slacks))if cur_slacks[ind] > 0
    )


# Berechnung von lb und ub
def get_lb_ub(lower, upper, coeff, slack):
    if coeff > 0:
        if upper > slack / coeff:
            upper = slack / coeff
    elif coeff < 0:
        if lower > -slack / coeff:
            lower = -slack / coeff
    return lower, upper


# Aktualisierung von ZI
def update_fraction(curr_fraction, bound, ind):
    if curr_fraction[ind] - bound < 0:
        curr_fraction[ind] = 1 - (curr_fraction[ind] + bound)
    else:
        curr_fraction[ind] -= bound
    # wegen Rundungsfehler
    if isclose(curr_fraction[ind], round(curr_fraction[ind]), abs_tol=0.000001):
        curr_fraction[ind] = round(curr_fraction[ind])


# Aktualisierung der aktuellen Lösung
def update_curr(curr, ind, bound, direction):
    if direction == "up":
        curr[ind] += bound
    else:
        curr[ind] -= bound
    # wegen Rundungsfehler
    if isclose(curr[ind], round(curr[ind]), abs_tol=0.000001):
        curr[ind] = round(curr[ind])


# Aktualisierung der betroffenen slacks
def update_slacks(slack, variable, bound, direction):
    column = m.get_col(variable)
    column_index = m.get_col_index(variable)

    if direction == "down":
        for ind in column_index:
            if column[ind] > 0:
                slack[ind] += bound * column[ind]
            else:
                slack[ind] -= bound * -column[ind]
            # wegen Rundungsfehler
            if isclose(slack[ind], round(slack[ind]), abs_tol=0.000001):
                slack[ind] = round(slack[ind])

    elif direction == "up":
        for ind in column_index:
            if column[ind] > 0:
                slack[ind] -= bound * column[ind]
            else:
                slack[ind] += bound * - column[ind]
            # wegen Rundungsfehler
            if isclose(slack[ind], round(slack[ind]), abs_tol=0.000001):
                slack[ind] = round(slack[ind])


# Auswahl von LB oder UB
def choose(curr, curr_fraction, slack, zi, lower, upper, ind, obj):
    fraction_lb, fraction_ub = curr_fraction.copy(), curr_fraction.copy()
    update_fraction(fraction_lb, lower, ind)
    update_fraction(fraction_ub, upper, ind)

    zi_lb, zi_ub = np.sum(fraction_lb), np.sum(fraction_ub)

    if zi_ub == zi_lb and zi_ub < zi:
        curr_lb, curr_ub = curr.copy(), curr.copy()
        curr_lb[ind] -= lower
        curr_ub[ind] += upper
        optimum_lb = np.sum(curr_lb * obj)
        optimum_ub = np.sum(curr_ub * obj)

        if optimum_ub < optimum_lb:
            update_curr(curr, ind, upper, "up")
            update_slacks(slack, ind, upper, "up")
            return zi_ub, fraction_ub
        else:
            update_curr(curr, ind, lower, "down")
            update_slacks(slack, ind, lower, "down")
            return zi_lb, fraction_lb

    elif zi_ub < zi_lb and zi_ub < zi:
        update_curr(curr, ind, upper, "up")
        update_slacks(slack, ind, upper, "up")
        return zi_ub, fraction_ub

    elif zi_lb < zi_ub and zi_lb < zi:
        update_curr(curr, ind, lower, "down")
        update_slacks(slack, ind, lower, "down")
        return zi_lb, fraction_lb
    return zi, current_fraction


if __name__ == '__main__':
    m = Model(sys.argv[1])
    m.relaxation()

    current = m.current_solution()
    slacks = m.slacks()

    current_fraction = np.array(
        [abs(current[k] - round(current[k])) if m.get_type(m.var_name[k]) != "C" else 0 for k in range(len(current))]
    )

    curr_slacks = get_pos_slacks(slacks)
    objective = m.objective()

    ZI = np.sum(current_fraction)

    epsilon = 0.000001
    change = True
    solution = True

    start = datetime.now()
    # Start der Heuristik
    while ZI > 0 and change:
        curr_vars = fraction_index(current)
        zi_before = ZI

        if len(curr_vars) == 0 or len(curr_slacks) == 0:
            break

        for index in curr_vars:
            col = m.get_col(index)
            col_index = m.get_col_index(index)
            # Standardwert für lb und ub
            lb, ub = m.var_name[index].ub, m.var_name[index].ub

            for element in col_index:
                # wenn der aktuelle slack nicht positiv ist
                if element not in curr_slacks:
                    if col[element] < 0:
                        lb = 0
                    else:
                        ub = 0
                    continue

                lb, ub = get_lb_ub(lb, ub, col[element], slacks[element])
                if lb == 0 and ub == 0:
                    break

            LB = min(lb, current[index] - m.var_name[index].lb)
            UB = min(ub, m.var_name[index].ub - current[index])

            if UB < epsilon and LB < epsilon:
                continue

            ZI, current_fraction = choose(current, current_fraction,
                                          slacks, ZI, LB, UB, index,
                                          objective)
            # wegen Rundungsfehler
            if isclose(current[index], round(current[index]), abs_tol=0.000001):
                current[index] = round(current[index])

            curr_slacks = get_pos_slacks(slacks)

        # keine Änderung innerhalb der letzten Iteration
        if ZI == zi_before and ZI > 0:
            change = False
            solution = False

    end = datetime.now()

    with open('C:/Users/Nik/Desktop/Testfiles/zi_round_2.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if solution:
            print()
            print("Time: ", end - start)
            print()
            print("MIP Startwert: " + str(m.optimum(current)))
            writer.writerow([sys.argv[1].replace('C:/Users/Nik/Desktop/Testfiles/Benchmark\\', '').replace('.mps', ''),
                             str(end - start), str(m.optimum(current)).replace('.', ',')])
        else:
            print()
            print('no solution!')
            print("Time: ", end - start)
            writer.writerow([sys.argv[1].replace('C:/Users/Nik/Desktop/Testfiles/Benchmark\\', '').replace('.mps', ''),
                             str(end - start), 'unsolvable'])
        print()
