from model import Model
import numpy as np
import random as rd


class Pumping(Model):

    def __init__(self, model):
        super().__init__(model)
        self.relaxation()

    def perturb(self, current_sol, rounded_sol):
        flip_index = self.error(current_sol, rounded_sol, rd.randint(10, 30))
        for i in flip_index:
            if self.get_type(i) == "C":
                continue
            if abs(current_sol[i] - rounded_sol[i]) + max(rd.random() - 0.3, 0) > 0.5:
                if rounded_sol[i] == 0:
                    rounded_sol[i] = 1
                else:
                    rounded_sol[i] = 0

    @staticmethod
    def error(current_sol, rounded_sol, number):
        return np.argsort(current_sol - rounded_sol)[-number:]

