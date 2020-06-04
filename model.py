from gurobipy import *
import numpy as np


class Model:

    def __init__(self, model):
        self.original = read(model)
        self.model = None
        self.variables = None
        self.constraints = None
        self.var_index = None
        self.var_name = None
        self.constraint_index = None
        self.constraint_name = None
        self.matrix = None

    def __canon(self):
        for constraint in self.model.getConstrs():
            if constraint.sense == ">":
                self.model.addConstr(self.model.getRow(constraint) * (-1),
                                     sense="<",
                                     rhs=constraint.RHS * (-1),
                                     name=constraint.constrname)
                self.model.remove(constraint)
            elif constraint.sense == "=":
                self.model.addConstr(self.model.getRow(constraint),
                                     sense="<",
                                     rhs=constraint.RHS,
                                     name=constraint.constrname + "_1")
                self.model.addConstr(self.model.getRow(constraint) * (-1),
                                     sense="<",
                                     rhs=constraint.RHS * (-1),
                                     name=constraint.constrname + "_2")
                self.model.remove(constraint)
        return self

    def relaxation(self):
        self.model = self.original.relax()
        self.__canon()
        self.model.setParam("OutputFlag", 0)
        # bei manchen Problemen führt die duale Lösung zu einem besseren Ergebnis
        self.model.setParam("Method", 1)
        self.model.optimize()
        self.variables = self.model.getVars()
        self.constraints = self.model.getConstrs()
        # hier evtl. auf einen Standard einigen
        self.var_index = {key: n for n, key in enumerate(self.variables)}
        self.var_name = {key: n for key, n in enumerate(self.variables)}
        self.constraint_index = {key: n for n, key in enumerate(self.constraints)}
        self.constraint_name = {key: n for key, n in enumerate(self.constraints)}
        self.matrix = np.array(self.model.getA().A)

    def normalize(self):
        self.model = self.original
        self.__canon()
        self.model.setParam("OutputFlag", 0)
        self.variables = self.model.getVars()
        self.constraints = self.model.getConstrs()
        # hier evtl. auf einen Standard einigen
        self.var_index = {key: n for n, key in enumerate(self.variables)}
        self.var_name = {key: n for key, n in enumerate(self.variables)}
        self.constraint_index = {key: n for n, key in enumerate(self.constraints)}
        self.constraint_name = {key: n for key, n in enumerate(self.constraints)}
        self.matrix = np.array(self.model.getA().A)

    def objective(self):
        return np.array(
            self.model.getAttr("Obj", self.variables)
        )

    def slacks(self):
        return np.array(
            self.model.getAttr("Slack", self.constraints)
        )

    def right_hand_side(self):
        return np.array(
            self.model.getAttr("RHS", self.constraints)
        )

    def current_solution(self):
        return np.array(
            self.model.getAttr("x", self.variables)
        )

    def optimum(self, solution):
        return np.sum(
            solution * self.objective()
        )

    def feasible(self, solution):
        return all(
            np.dot(self.matrix, solution) <= self.right_hand_side()
        )

    def get_type(self, variable):
        return self.original.getVarByName(variable.varname).VType

    def get_col(self, variable):
        return self.matrix[:, variable]

    def get_col_index(self, variable):
        return np.where(self.matrix[:, variable])[0]

    def get_row(self, constraint):
        return self.matrix[constraint]

    def get_row_index(self, constraint):
        return np.where(self.matrix[constraint])[0]

    def get_coeff(self, constraint, variable):
        return self.matrix[constraint][variable]

