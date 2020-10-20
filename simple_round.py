from model import Model
import numpy as np
from math import floor, ceil
from datetime import datetime
import sys
import csv


# Menge der gebrochenen Variablen
def fraction_index(solution):
    return np.array(
        [i for i in range(len(solution)) if (solution[i] != round(solution[i]) and m.get_type(m.var_name[i]) != "C")]
    )


if __name__ == '__main__':
    m = Model(sys.argv[1])
    m.relaxation()

    current = m.current_solution()
    indices = fraction_index(current)

    solved = True

    start = datetime.now()
    # Start der Heuristik
    for var in indices:

        col = m.get_col(var)
        col_index = m.get_col_index(var)

        if all([col[i] >= 0 for i in col_index]):
            current[var] = floor(current[var])

        elif all([col[i] <= 0 for i in col_index]):
            current[var] = ceil(current[var])
        else:
            solved = False
            break
    end = datetime.now()

    print()
    print("Time: ", end - start)
    if solved:
        print("MIP Startwert: " + str(m.optimum(current)))
    else:
        print("unsolvable")
        print()
