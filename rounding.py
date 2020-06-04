from model import Model
import numpy as np
from math import isclose


class Rounding(Model):

    def __init__(self, model):
        super().__init__(model)
        self.relaxation()

    def fraction_index(self, solution):
        return np.array(
            [i for i in range(len(solution)) if (solution[i] != round(solution[i])
                                                 and self.get_type(self.var_name[i]) != "C")]
        )

    def get_equal(self):
        equal = set()
        for i, constraint in self.constraint_name.items():
            if constraint.constrname.endswith("_1"):
                equal.add((i, i + 1))
        return equal

    @staticmethod
    def find_null(solution):
        return np.unique(
            [k for k in range(len(solution)) if (solution[k] > 0)]
        )

    @staticmethod
    def get_pos_slacks(current_slacks):
        return set(
            ind for ind in range(len(current_slacks)) if current_slacks[ind] > 0
        )

    @staticmethod
    def get_lb_ub(lower, upper, coeff, slack):
        if coeff > 0:
            if upper > slack / coeff:
                upper = slack / coeff
        elif coeff < 0:
            if lower > -slack / coeff:
                lower = -slack / coeff
        return lower, upper

    @staticmethod
    def update_fraction(curr_fraction, bound, index):
        if curr_fraction[index] - bound < 0:
            curr_fraction[index] = 1 - (curr_fraction[index] + bound)
        else:
            curr_fraction[index] -= bound
        if isclose(curr_fraction[index], round(curr_fraction[index]), abs_tol=0.000001):
            curr_fraction[index] = round(curr_fraction[index])

    @staticmethod
    def update_curr(current, index, bound, direction):
        if direction == "up":
            current[index] += bound
        else:
            current[index] -= bound
        if isclose(current[index], round(current[index]), abs_tol=0.000001):
            current[index] = round(current[index])

    def update_slacks(self, slack, variable, bound, direction):
        column = self.get_col(variable)
        column_index = self.get_col_index(variable)

        if direction == "down":
            for ind in column_index:
                if column[ind] > 0:
                    slack[ind] += bound * column[ind]
                else:
                    slack[ind] -= bound * -column[ind]
                if isclose(slack[ind], round(slack[ind]), abs_tol=0.000001):
                    slack[ind] = round(slack[ind])

        elif direction == "up":
            for ind in column_index:
                if column[ind] > 0:
                    slack[ind] -= bound * column[ind]
                else:
                    slack[ind] += bound * - column[ind]
                if isclose(slack[ind], round(slack[ind]), abs_tol=0.000001):
                    slack[ind] = round(slack[ind])

    def choose(self, current, current_fraction, slack, zi, lower, upper, index, objective):
        fraction_lb, fraction_ub = current_fraction.copy(), current_fraction.copy()
        self.update_fraction(fraction_lb, lower, index)
        self.update_fraction(fraction_ub, upper, index)

        zi_lb, zi_ub = np.sum(fraction_lb), np.sum(fraction_ub)

        if zi_ub == zi_lb and zi_ub < zi:
            curr_lb, curr_ub = current.copy(), current.copy()
            curr_lb[index] -= lower
            curr_ub[index] += upper
            optimum_lb = np.sum(curr_lb * objective)
            optimum_ub = np.sum(curr_ub * objective)

            if optimum_ub < optimum_lb:
                self.update_curr(current, index, upper, "up")
                self.update_slacks(slack, index, upper, "up")
                return zi_ub, fraction_ub
            else:
                self.update_curr(current, index, lower, "down")
                self.update_slacks(slack, index, lower, "down")
                return zi_lb, fraction_lb

        elif zi_ub < zi_lb and zi_ub < zi:
            self.update_curr(current, index, upper, "up")
            self.update_slacks(slack, index, upper, "up")
            return zi_ub, fraction_ub

        elif zi_lb < zi_ub and zi_lb < zi:
            self.update_curr(current, index, lower, "down")
            self.update_slacks(slack, index, lower, "down")
            return zi_lb, fraction_lb

    def get_unbounded(self, const_index, ind, curr_not_null):
        row = self.get_row(const_index)
        pos_coeff = np.array(
            [k for k in range(len(row)) if row[k] != 0 and k != ind]
        )
        return np.intersect1d(pos_coeff, curr_not_null)

    def is_there_slack(self, curr, curr_fraction, ind, const_index, cur_slacks, slack, curr_not_null):
        variables = self.get_unbounded(const_index, ind, curr_not_null)
        answer = False
        if len(variables) > 0:
            for var in variables:
                column = self.get_col(var)
                # hier muss ich noch im Model nur die Paare von equal NBs sammeln und prüfen
                if len([column[k] for k in range(len(column)) if column[k] != 0]) == 1:
                    coeff = self.get_coeff(const_index, var)
                    if coeff > 0:
                        slack[const_index] += coeff * (curr[var] - self.var_name[var].lb)
                        if curr[var] % 1 != 0 and self.get_type(self.var_name[var]) != "C":
                            # fraglich ob das immer zutrifft
                            curr_fraction[var] = 0
                        curr[var] -= (curr[var] - self.var_name[var].lb)
                        # bei anderen problemen mag dies zutreffen
                        if curr[var] % 1 != 0:
                            print(curr[var])
                    elif coeff < 0:
                        # zudem muss ich die Equations mit == sammeln.. am Ende funktioniert diese funktion überhaupt nicht
                        slack[const_index] += - coeff * (self.var_name[var].ub - curr[var])
                        if curr[var] % 1 != 0 and self.get_type(self.var_name[var]) != "C":
                            curr_fraction[var] = 0
                        curr[var] += (self.var_name[var].ub - curr[var])
                    if curr[var] == 0:
                        np.delete(curr_not_null, var)
                    cur_slacks.add(const_index)
                    answer = True
        return np.sum(curr_fraction), answer