from rounding import Rounding
from datetime import datetime
from math import ceil, floor
import os


__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

if __name__ == "__main__":

    m = Rounding(os.path.join(__location__, 'fast0507.mps'))

    current = m.current_solution()
    indices = m.fraction_index(current)

    #equals = m.get_equal()
    solved = True

    start = datetime.now()
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
    print()
    if solved:
        print("MIP Startwert: " + str(m.optimum(current)))
    else:
        print("unsolvable")
