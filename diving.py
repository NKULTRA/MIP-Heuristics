from model import Model
import numpy as np
from math import isclose, ceil, floor


class Diving(Model):

    def __init__(self, model):
        super().__init__(model)
        self.relaxation()

    def fraction_index(self, solution):
        return np.unique(
            [k for k in range(len(solution)) if (solution[k] != round(solution[k])
                                                 and self.get_type(self.var_name[k]) != "C")]
        )

    def trivial(self, variable):
        column = self.get_col(variable)
        col_index = self.get_col_index(variable)
        if all([column[i] >= 0 for i in col_index]):
            return 1
        if all([column[i] <= 0 for i in col_index]):
            return 2
        return 0

    def calculate(self, lp_solution, variables, trivial_vars, current, current_fraction, curr_obj, obj):
        new, alt = dict(), dict()
        for index in variables:
            if trivial_vars[index] == 0:
                if abs(current[index] - lp_solution[index]) >= 0.4:
                    if current[index] < lp_solution[index]:
                        new[index], alt[index] = floor(current[index]), ceil(current[index])
                    else:
                        new[index], alt[index] = ceil(current[index]), floor(current[index])

                elif current_fraction[index] < 0.3:
                    if isclose((current[index] - current_fraction[index]),
                               round(current[index]), abs_tol=0.000001):
                        new[index], alt[index] = floor(current[index]), ceil(current[index])
                    else:
                        new[index], alt[index] = ceil(current[index]), floor(current[index])

                else:
                    pseudo_down = self.pseudocosts(current, index, "down", curr_obj, obj)
                    pseudo_up = self.pseudocosts(current, index, "up", curr_obj, obj)

                    if pseudo_down < pseudo_up:
                        new[index], alt[index] = floor(current[index]), ceil(current[index])
                    else:
                        new[index], alt[index] = ceil(current[index]), floor(current[index])
            else:
                if trivial_vars[index] == 1:
                    new[index], alt[index] = ceil(current[index]), floor(current[index])
                else:
                    new[index], alt[index] = floor(current[index]), ceil(current[index])
        return new, alt

    @staticmethod
    def pseudocosts(current, index, direction, curr_obj, obj):
        coeff = obj[index]
        if direction == "down":
            new_obj = curr_obj - coeff * (current[index] - floor(current[index]))
            return abs(curr_obj - new_obj) / (current[index] - floor(current[index]))

        new_obj = curr_obj + coeff * (ceil(current[index]) - current[index])
        return abs(curr_obj - new_obj) / (ceil(current[index]) - current[index])

    def choose(self, new_val, alt, curr_fraction, curr_solution, curr_obj, obj):
        best = 0
        index = None

        for i in new_val.keys():
            if new_val[i] < alt[i]:
                pseudo_new = self.pseudocosts(curr_solution, i, "down", curr_obj, obj)
                pseudo_alt = self.pseudocosts(curr_solution, i, "up", curr_obj, obj)
            else:
                pseudo_new = self.pseudocosts(curr_solution, i, "up", curr_obj, obj)
                pseudo_alt = self.pseudocosts(curr_solution, i, "down", curr_obj, obj)

            ratio = curr_fraction[i] * (pseudo_alt / pseudo_new)

            if ratio >= best:
                best = ratio
                index = i
        return index

    def simple_rounding(self, curr):
        ind = self.fraction_index(curr)
        solved = True

        for var in ind:

            col = self.get_col(var)
            col_index = self.get_col_index(var)

            if all([col[i] >= 0 for i in col_index]):
                curr[var] = floor(curr[var])
            elif all([col[i] <= 0 for i in col_index]):
                curr[var] = ceil(curr[var])
            else:
                solved = False
                break
        return solved, curr
