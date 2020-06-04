from rounding import Rounding
from datetime import datetime
import numpy as np
from math import isclose
import os


__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


if __name__ == "__main__":

    m = Rounding(os.path.join(__location__, 'fast0507.mps'))

    current = m.current_solution()
    slacks = m.slacks()

    current_fraction = np.array([abs(a - round(a)) for a in current])

    curr_slacks = m.get_pos_slacks(slacks)
    objective = m.objective()

    ZI = np.sum(current_fraction)

    cnt = 0
    epsilon = 0.000001

    start = datetime.now()
    while ZI > 0 and cnt < 3:
        curr_vars = m.fraction_index(current)
        # current_not_null = find_null(current)

        if len(curr_vars) == 0 or len(curr_slacks) == 0:
            break

        for index in curr_vars:
            col = m.get_col(index)
            col_index = m.get_col_index(index)
            lb, ub = 1, 1
            new = False
            for element in col_index:
                if element not in curr_slacks:
                    if col[element] < 0:
                        lb = 0
                    else:
                        ub = 0
                    continue
                lb, ub = m.get_lb_ub(lb, ub, col[element], slacks[element])
                # bei gleichheitsproblemen haben wir jeweils zwei nebenbedingungen mit pos und neg Vorzeichen
                # das zwingt auf 0
                if lb == 0 and ub == 0:
                    '''ZI, new = is_there_slack(current, current_fraction, index,
                                             constraint_index, curr_slacks,
                                             slacks, current_not_null)
                    if not new:
                        break'''
                    break
            # if new:
            # continue
            LB = min(lb, current[index] - m.var_name[index].lb)
            UB = min(ub, m.var_name[index].ub - current[index])
            if UB < epsilon and LB < epsilon:
                continue
            ZI, current_fraction = m.choose(current, current_fraction,
                                          slacks, ZI, LB, UB, index,
                                          objective)

            if isclose(current[index], round(current[index]), abs_tol=0.000001):
                current[index] = round(current[index])

            curr_slacks = m.get_pos_slacks(slacks)
        cnt += 1

    end = datetime.now()
    print()
    print("Iterations: ", cnt)
    print("Time: ", end - start)
    print()
    print("MIP Startwert: " + str(m.optimum(current)))
