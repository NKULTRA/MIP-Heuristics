from model import Model
import numpy as np
import random as rd
from datetime import datetime, timedelta
import sys
import csv


# Störfunktion flip
def flip(current_sol, rounded_sol):
    # randint ist das intervall der Anzahl an veränderten Variablen
    flip_index = error(current_sol, rounded_sol, rd.randint(10, 30))
    for i in flip_index:
        if m.get_type(m.var_name[i]) == 'C':
            continue
        if rounded_sol[i] == 0:
            rounded_sol[i] = 1
        elif rounded_sol[i] == 1:
            rounded_sol[i] = 0


# Störfunktion perturbation
def perturb(current_sol, rounded_sol):
    for i in range(len(rounded_sol)):
        if m.get_type(m.var_name[i]) == 'C':
            continue
        if abs(current_sol[i] - rounded_sol[i]) + max(rd.random() - 0.3, 0) > 0.5:
            if rounded_sol[i] == 0:
                rounded_sol[i] = 1
            elif rounded_sol[i] == 1:
                rounded_sol[i] = 0


# Differenz zwischen gerundeter und ungerundeter Lösung absteigend sortiert, Anzahl von T höchsten Werten
def error(current_sol, rounded_sol, number):
    return np.argsort(abs(current_sol - rounded_sol))[-number:]


if __name__ == '__main__':
    m = Model(sys.argv[1])
    m.relaxation()

    current = m.current_solution()

    integer = np.array(
        [k for k in range(len(current)) if m.get_type(m.var_name[k]) != "C"]
    )

    rounded = np.array(
        [round(current[k]) if m.get_type(m.var_name[k]) != "C" else current[k] for k in range(len(current))]
    )

    rounded_orig = rounded.copy()

    objective = m.objective()
    coefficients = [0 for _ in range(len(rounded))]
    const = 0

    timelimit = 2
    cnt = 0

    start = datetime.now()
    # Start Heuristik
    if not m.feasible(rounded):
        expr = [0 for _ in range(len(rounded))]
        already = dict()
        while not m.feasible(rounded) and cnt < 100:
            # Erstellung der Distanzfunktion
            for index in integer:
                # ab dem zweiten Durchgang, sollte sich ein geänderter Index bei einer 1 befinden, dann
                # muss die Konstante verkleinert werden
                if coefficients[index] == -1:
                    const -= 1
                if rounded[index] == 0:
                    coefficients[index] = 1
                else:
                    const += 1
                    coefficients[index] = -1

            expr = m.get_expr(const, coefficients)
            m.set_objective(expr)
            m.model.optimize()

            last_rounded = np.copy(rounded)
            current = m.current_solution()

            rounded = np.array(
                [round(current[k]) if m.get_type(m.var_name[k]) != "C" else current[k] for k in range(len(current))]
            )

            if m.feasible(rounded):
                break

            if np.all(last_rounded[integer] == rounded[integer]):
                flip(current, rounded)

            if id(rounded) in already and (len(already) >= 3 or cnt >= 100):
                perturb(current, rounded)
                already = dict()
                already[id(rounded)] = None
            else:
                already[id(rounded)] = None

            if len(already) > 3:
                already.pop(list(already.keys())[0])

            if m.feasible(rounded):
                break

            if datetime.now() - start > timedelta(minutes=timelimit):
                break

            cnt += 1

    end = datetime.now()
    if m.feasible(rounded):
        print()
        print("Iterations: ", cnt)
        print("Time: ", end - start)
        print()
        print("MIP Startwert: " + str(sum(objective * rounded)))
    else:
        print()
        print('no solution!')
        print("Iterations: ", cnt)
        print("Time: ", end - start)
        print()
