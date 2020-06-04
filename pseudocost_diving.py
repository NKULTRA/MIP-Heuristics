from diving import Diving
import numpy as np
import os
from datetime import datetime

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

if __name__ == "__main__":

    m = Diving(os.path.join(__location__, 'fast0507.mps'))

    lp_solution = m.current_solution()
    current = m.current_solution()
    objective = m.objective()

    indices = m.fraction_index(current)
    trivial_round = dict(zip(indices, map(m.trivial, indices)))

    if len(trivial_round) > 0:
        real_indices = np.array(
            [indices[k] for k in range(len(indices)) if trivial_round[indices[k]] == 0]
        )
        if len(real_indices) > 0:
            indices = real_indices

    cnt = 0
    limit = 1000

    start = datetime.now()
    while cnt < limit:

        current_fraction = np.array([abs(a - round(a)) for a in lp_solution])
        current_objective = np.sum(current * objective)

        new_value, alternative = m.calculate(lp_solution, indices, trivial_round, current,
                                             current_fraction, current_objective, objective)

        change = m.choose(new_value, alternative, current_fraction,
                          current, current_objective, objective)

        m.var_name[change].lb = new_value[change]
        m.var_name[change].ub = new_value[change]

        m.model.optimize()
        # + m.model.getAttr("IterCount")
        cnt += 1

        if m.model.getAttr("Status") == 3:
            # diese Bedingung ist noch fraglich!!
            # ein Backtrack solls eigentlich sein
            m.var_name[change].lb = alternative[change]
            m.var_name[change].ub = alternative[change]
        else:
            current = m.current_solution()
            indices = m.fraction_index(current)
            trivial_round = dict(zip(indices, map(m.trivial, indices)))

            if len(trivial_round) > 0:
                real_indices = np.array(
                    [indices[k] for k in range(len(indices)) if trivial_round[indices[k]] == 0]
                )
                if len(real_indices) > 0:
                    indices = real_indices

        if len(indices) == 0:
            break

        it_is, simple = m.simple_rounding(current.copy())

        if it_is:
            current = simple
            break
    end = datetime.now()

    print()
    print("Iterations: ", cnt)
    print("Time: ", end - start)
    print()
    print("MIP Startwert: " + str(m.optimum(current)))
