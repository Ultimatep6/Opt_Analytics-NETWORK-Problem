from pyomo.environ import SolverFactory, ConcreteModel, RangeSet, Set, Var
from pyomo.environ import NonNegativeReals
import os

from read_data import read_data
from utils import generate_neighbor_pairings_row_major

class Solver:
    def __init__(self, solver_name='glpk', problem_dir='default.in'):
        self.solver_name = solver_name
        self.solver = SolverFactory(solver_name)

        self.problem_dir = rf'./in_files/{problem_dir}'
        self.settings = read_data(self.problem_dir)

        self.models = {}


    def solve(self):
        results = self.solver.solve(self.model)
        return results
    
    def get_components(self):
        self.rows = self.settings['rows']
        self.cols = self.settings['cols']
        self.time_steps = self.settings['sim_time']
        self.total_materials = self.settings['n_resources'] + self.settings['n_products'] 

        self.connections = generate_neighbor_pairings_row_major(self.rows,self.cols)
        # self.connections_bad = [(i, j, k, l) 
        #         for i in range(self.rows) 
        #         for j in range(self.cols) 
        #         for k in range(self.rows) 
        #         for l in range(self.cols) 
        #         if (i, j) != ((k, l))]
        
        # print(f"Optimized connections is {len(self.connections_bad)/len(self.connections)} times smaller than necessary.")

        # Print the size of the problem
        print(f"Connections: {len(self.connections)}")
        print(f"Time Steps: {self.time_steps}")
        print(f"Total Materials: {self.total_materials}")
        print("*"*40)
        print(f"Problem Size: {len(self.connections)*self.time_steps*self.total_materials}")
    
    def model_variables(self, model):
        # A single slither of the time-expanded network
        model.connections = Set(initialize=self.connections, dimen=2)

        model.desc_var = Var(self.connections, RangeSet(0, self.total_materials - 1), domain=NonNegativeReals)

    def model_params(self, model):
        self.
    
    def model_constraints(self, model):
        raise NotImplementedError
    
    def build_model_t(self, time_step):
        model = ConcreteModel()

        # Get all the components
        self.get_components()

        # Define the Variables
        self.model_variables(model)

        # Define the Parameters
        self.model_params(model)


        self.models[time_step] = model


    
    def display_results(self, model):
        raise NotImplementedError
    
if __name__ == '__main__':
    solver = Solver(solver_name='glpk', problem_dir='default.in')
    solver.get_components()
    