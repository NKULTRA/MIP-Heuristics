from pumping import Pumping
from datetime import datetime
from gurobipy import *
import numpy as np
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

if __name__ == "__main__":

    m = Pumping(os.path.join(__location__, 'fast0507.mps'))

    current = m.current_solution()
    rounded_solution = np.around(current)
    objective = m.objective()

    #obj = m.model.getObjective()
    #first_value = obj.getValue()

    cnt = 0
    limit = 10000

    start = datetime.now()
    if not m.feasible(rounded_solution):
        already = list()

        while not m.feasible(rounded_solution) and cnt < limit:
            it_is = False
            expr = LinExpr()

            for i, val in m.var_name.items():
                if m.get_type(val) == "C":
                    continue
                if rounded_solution[i] == 0:
                    expr += val
                elif rounded_solution[i] == 1:
                    expr += (1 - val)

            m.model.setObjective(expr, GRB.MINIMIZE)
            m.model.optimize()

            current = m.current_solution()

            rounded_solution = np.array(
                [round(current[i]) if m.get_type(m.var_name[i]) != "C" else current[i] for i in range(len(current))]
            )

            new_objective = m.objective()

            for element in already:
                if np.array_equal(rounded_solution, element):
                    it_is = True
                    break

            if it_is:
                m.perturb(current, rounded_solution)
            else:
                already.append(rounded_solution)

            # + m.model.getAttr("IterCount")
            cnt += 1

            if m.feasible(rounded_solution):
                break
    end = datetime.now()

    print()
    print("Iterations: ", cnt)
    print("Time: ", end - start)
    print()
    if cnt == limit:
        print("no solution found")
    else:
        print("MIP-Startwert: ", np.sum(objective * rounded_solution))
        print(m.feasible(rounded_solution))