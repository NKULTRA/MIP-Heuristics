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
