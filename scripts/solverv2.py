import json
from xml.parsers.expat import model
from pyomo.environ import SolverFactory, ConcreteModel, RangeSet, Set, Var, Param, Constraint, Objective, minimize, maximize, ConstraintList
from pyomo.environ import NonNegativeReals, Any, Reals, Binary
import os
import random
import numpy as np
import cloudpickle as cpickle

import plotly.graph_objs as go
import pandas as pd

from read_data import read_data
from utils import generate_neighbor_pairings_row_major,generate_neighbor_pairings_2d, neighbors, all_neighbors, get_desc_variables, node_to_row_index, row_index_to_node
from utils import compute_net_flow, compute_required_flow, compute_net_flow_total, compute_total_throughput, tuple_key_to_str


class SolverV2:
    def __init__(self, solver_name='glpk', problem_dir='default.in'):
        self.solver_name = solver_name
        self.solver = SolverFactory(solver_name)

        self.problem_dir = rf'./in_files/{problem_dir}'
        self.settings = read_data(self.problem_dir)

        self.models = {}
        self.solutions = np.zeros((self.settings['rows'], self.settings['cols'], int(self.settings['sim_time'])))

    def setup_modelSets(self):
        # Grid Sets
        self.R, self.C = self.settings['rows'], self.settings['cols']
        self.model = ConcreteModel()
        self.model.rows = RangeSet(0, self.R - 1)
        self.model.cols = RangeSet(0, self.C - 1)
        self.model.nodes = Set(initialize=[(r, c) for r in range(self.R) for c in range(self.C)])
        # Bidirectional connections between neighboring nodes
        # ROW-MAJOR order
        self.model.connections = Set(initialize=generate_neighbor_pairings_2d(self.R, self.C))
        
        # For simple solution
        self.quarter_div_col = self.C // 4
        self.quarter_div_row = self.R // 3

        # Define Sources, Factories, Sinkholes
        self.sources = [ f'source_{i}' for i in range(1,self.settings['n_sources']+1) ]
        self.model.sources = Set(initialize=self.sources)
        self.factories = [ f'fact_{i}' for i in range(1,self.settings['n_factories']+1) ]
        self.model.factories = Set(initialize=self.factories)
        self.sinkholes = [ f'sink_{i}' for i in range(1,self.settings['n_sinkholes']+1) ]
        self.model.sinkholes = Set(initialize=self.sinkholes)

        # Define Resources and Products
        self.resources = [ f'resource_{i}' for i in range(1,self.settings['n_resources']+1) ]
        self.model.resources = Set(initialize=self.resources)
        self.products = [ f'product_{i}' for i in range(1,self.settings['n_products']+1) ]
        self.model.products = Set(initialize=self.products)
        
        self.model.materials = Set(initialize=self.resources + self.products)
        # define throughput
        self.min_throughput = self.settings['min_bottleneck']
        self.max_throughput = self.settings['max_bottleneck']

        # Debug info
        # print("Sources:")
        # print(self.model.sources.data())
        # print("Factories:")
        # print(self.model.factories.data())
        # print("Sinkholes:")
        # print(self.model.sinkholes.data())

    def setup_modelSpecialParams(self):
        """ Define Parameters
            - Source Positions (row,col)
            - Factory Positions (row,col)
            - Sinkhole Positions (row,col)
        """
        
        # Source Positions
        source_positions_dict = {
            self.model.sources.data()[i]: (self.quarter_div_row, i * self.quarter_div_col)
            for i in range(len(self.model.sources))
        }
        self.model.source_positions = Param(self.model.sources, initialize=source_positions_dict, within=Any)
        self.settings['source_positions'] = source_positions_dict

        # Sinkhole Positions
        sinkhole_positions_dict = {
            self.model.sinkholes.data()[i]: (self.R - 1 - self.quarter_div_row, i * self.quarter_div_col)
            for i in range(len(self.model.sinkholes))
        }
        self.model.sinkhole_positions = Param(self.model.sinkholes, initialize=sinkhole_positions_dict, within=Any)
        self.settings['sinkhole_positions'] = sinkhole_positions_dict


        # Factory Positions
        factory_positions_dict = {
            self.model.factories.data()[i]: (self.R // 2, (i + 1) * self.quarter_div_col)
            for i in range(len(self.model.factories))
        }
        self.model.factory_positions = Param(self.model.factories, initialize=factory_positions_dict, within=Any)
        self.settings['factory_positions'] = factory_positions_dict

        # Source Supply (negative for outflow)
        source_output_matrix = self.settings['source_output_matrix']
        source_supply_dict = {
            (self.model.materials.data()[r], self.model.sources.data()[c]): -source_output_matrix[r][c] if r < source_output_matrix.shape[0] else 0
            for r in range(len(self.model.materials))
            for c in range(len(self.model.sources))
        }
        self.model.source_supply = Param(self.model.materials, self.model.sources, initialize=source_supply_dict, within=Reals)
        self.source_supply_dict = source_supply_dict
        # Sinkhole Demand (positive for inflow)
        sinkhole_input_matrix = self.settings['sinkhole_input_matrix']
        sinkhole_demand_dict = {
            (self.model.materials.data()[p], self.model.sinkholes.data()[c]): sinkhole_input_matrix[-len(self.model.materials) + len(self.model.products) + p][c] if -len(self.model.materials) + len(self.model.products) + p >= 0 else 0
            for p in range(len(self.model.materials))
            for c in range(len(self.model.sinkholes))
        }
        self.model.sinkhole_demand = Param(self.model.materials, self.model.sinkholes, initialize=sinkhole_demand_dict, within=Reals)
        self.sinkhole_demand_dict = sinkhole_demand_dict
        # Factory Resource Demand (+input) and Supply (-output)
        factory_io_resource_matrix = self.settings['factory_io_resource_matrix']
        factory_io_product_matrix = self.settings['factory_io_products_matrix']
        factory_demand_dict = {
            (self.model.materials.data()[r], self.model.factories.data()[f]): factory_io_resource_matrix[r][0][f] if r < factory_io_resource_matrix.shape[0] else factory_io_product_matrix[r - factory_io_resource_matrix.shape[0]][0][f]
            for r in range(len(self.model.materials))
            for f in range(len(self.model.factories))
        }
        self.factory_demand_dict = factory_demand_dict
        self.model.factory_demand = Param(self.model.materials, self.model.factories, initialize=factory_demand_dict, within=Reals)
        factory_supply_dict = {
            (self.model.materials.data()[p], self.model.factories.data()[f]): -factory_io_resource_matrix[p][1][f] if p < factory_io_resource_matrix.shape[0] else -factory_io_product_matrix[p - factory_io_resource_matrix.shape[0]][1][f]
            for p in range(len(self.model.materials))
            for f in range(len(self.model.factories))
        }
        self.factory_supply_dict = factory_supply_dict
        self.model.factory_supply = Param(self.model.materials, self.model.factories, initialize=factory_supply_dict, within=Reals)

        

        # Debug info
        # Check source supply constraints

    def setup_modelPathParams(self):
        self.unique_paths = generate_neighbor_pairings_2d(self.R, self.C, onedir=True)
        rand_throughput = np.random.randint(self.min_throughput, self.max_throughput + 1, size=len(self.unique_paths))
        # TODO: Add time_interval and time_offset parameters later
        path_params_dict = {
            (self.unique_paths[i][0], self.unique_paths[i][1]): rand_throughput[i]
            for i in range(len(self.unique_paths))
        }
        path_params_dict.update({
            (self.unique_paths[i][1], self.unique_paths[i][0]): rand_throughput[i]
            for i in range(len(self.unique_paths))
        })
        # Convert np.int32 values to int for JSON serialization
        path_params_dict = {k: int(v) for k, v in path_params_dict.items()}
        self.settings['path_params'] = tuple_key_to_str(path_params_dict)

        self.model.path_throughput = Param(
            self.model.connections,
            initialize=path_params_dict,
            within=Reals)

    def setup_modelFlowParams(self):
        """ Define flow limits for non-source/sink/factory nodes """
        flows = {(x, y): 0 for (x, y) in self.model.nodes}
        for s, pos in self.model.source_positions.items():
            flows.pop(pos, None)
        for sk, pos in self.model.sinkhole_positions.items():
            flows.pop(pos, None)
        for f, pos in self.model.factory_positions.items():
            flows.pop(pos, None)

        self.model.flow_limits = Param(
            self.model.nodes,
            initialize=flows,
            within=Reals
        )

    def setup_modelVariables(self):
        """ Define Variables
            - Path Flows between neighboring nodes
        """
        self.model.path_flows = Var(
            self.model.connections,
            self.model.products | self.model.resources,
            within=NonNegativeReals,
            initialize=0
        )

    def setup_modelFlowConstraints(self):
        """ Define Constraints
            - Source Supply Constraints
            - Factory Resource/Product Demand and Supply Constraints
            - Sinkhole Demand Constraints
            - Intermediate Node Flow Limits
            - Flow Conservation Constraints
        """

        self.model.flow_conservation_constraints = ConstraintList()
        for n in self.model.nodes:
            for m in self.model.materials:
                inflow = compute_net_flow(self, n, m, return_inflow=True, row_major=False)
                outflow = compute_net_flow(self, n, m, return_inflow=False, row_major=False)
                net_flow = inflow - (outflow)

                
                if n in self.model.source_positions.values():
                    # Find which source this is
                    for s, pos in self.model.source_positions.items():
                        if n == pos:
                            self.model.flow_conservation_constraints.add(
                                net_flow == self.model.source_supply[m, s]
                            )
                            break
                elif n in self.model.factory_positions.values():
                    for f, pos in self.model.factory_positions.items():
                        if n == pos:
                            self.model.flow_conservation_constraints.add(
                                net_flow == self.model.factory_demand[m, f] + self.model.factory_supply[m, f]
                            )
                            break
                elif n in self.model.sinkhole_positions.values():
                    for sk, pos in self.model.sinkhole_positions.items():
                        if n == pos:
                            self.model.flow_conservation_constraints.add(
                                net_flow == self.model.sinkhole_demand[m, sk]
                            )
                            break
                
                else:  # intermediate
                    if n in self.model.flow_limits:
                        self.model.flow_conservation_constraints.add(
                            net_flow == self.model.flow_limits[n]
                        )
    
    def setup_modelThroughputConstraints(self):
        """ Define Constraints
            - Path Throughput Constraints
        """
        self.model.throughput_constraints = ConstraintList()
        for connection in self.model.connections:
            (ax, ay, bx, by) = connection
            total_flow = sum(self.model.path_flows[ax, ay, bx, by, m] + self.model.path_flows[bx, by, ax, ay, m] 
                             for m in self.model.materials)
            self.model.throughput_constraints.add(
                total_flow <= self.model.path_throughput[ax, ay, bx, by]
            )
    
    def setup_modelObjective(self):
        """ Define Objective
            - Maximum Throughput vs Total Flow
        """
        def obj_function1(model):
            # Sum the difference between max throughput and actual flow for all connections and materials
            return sum(model.path_throughput[connection] - compute_total_throughput(self, connection)
                       for connection in self.model.connections)
        self.model.objective = Objective(rule=obj_function1, sense=maximize)

    def solve_model(self):
        results = self.solver.solve(self.model, tee=False)
        return results
    
    def setup_model(self):
        self.setup_modelSets()
        self.setup_modelSpecialParams()
        self.setup_modelPathParams()
        self.setup_modelFlowParams()
        self.setup_modelVariables()
        self.setup_modelFlowConstraints()
        self.setup_modelThroughputConstraints()
        self.setup_modelObjective()

    def save_results(self, save_path='results.out', solution=''):
        rows = []
        for key in self.model.path_flows.keys():
            (ax, ay, bx, by, mat) = key
            a, b = (ax, ay), (bx, by)
            flow = self.model.path_flows[key].value
            rows.append({'time': 0, 'from_node': a, 'to_node': b, 'material': mat, 'flow': flow})
        df = pd.DataFrame(rows)
        df.to_csv(rf"./out_files/{save_path}", index=False)

    def save_model_object(self, model_name='model_object.pkl'):
        with open(rf"./out_files/models/{model_name}", mode='wb') as file:
            cpickle.dump(self.model, file)

    def load_model_object(self, model_name='model_object.pkl'):
        with open(rf"./out_files/models/{model_name}", mode='rb') as file:
            self.model = cpickle.load(file)

    def save_settings(self, save_path='settings.json'):
        with open(rf'./out_files/{save_path}', 'w') as f:
            settings_dict = self.settings.copy()
            # Remove keys safely
            settings_dict.pop('factory_io_resource_matrix', None)
            settings_dict.pop('factory_io_products_matrix', None)
            settings_dict.pop('sinkhole_input_matrix', None)
            settings_dict.pop('source_output_matrix', None)

            # Add new keys

            settings_dict['source_supply_matrix'] = tuple_key_to_str(self.source_supply_dict)
            settings_dict['factory_supply_matrix'] = tuple_key_to_str(self.factory_supply_dict)
            settings_dict['factory_demand_matrix'] = tuple_key_to_str(self.factory_demand_dict)
            settings_dict['sinkhole_demand_matrix'] = tuple_key_to_str(self.sinkhole_demand_dict)

            print(settings_dict)

            json.dump(settings_dict, f, indent=4)

    def load_settings(self, load_path='settings.json'):
        with open(rf'./out_files/{load_path}', 'r') as f:
            self.settings = json.load(f)

        self.model.source_supply = np.array(self.settings['source_supply_matrix'])
        self.model.factory_supply = np.array(self.settings['factory_supply_matrix'])
        self.model.factory_demand = np.array(self.settings['factory_demand_matrix'])
        self.model.sinkhole_demand = np.array(self.settings['sinkhole_demand_matrix'])


    def __str__(self):

        printty = f"Source Matrix\n: {self.settings['source_output_matrix']}\nFactory IO Resource Matrix\n: {self.settings['factory_io_resource_matrix']}\nFactory IO Products Matrix\n: {self.settings['factory_io_products_matrix']}\nSinkhole I Products Matrix\n: {self.settings['sinkhole_input_matrix']}"
        printty += "\n"
        printty += "\n[SOURCE SUPPLY]\n"
        for s in self.model.sources:
            for m in self.model.materials:
                supply = self.model.source_supply[m, s]
                printty += f"  {s} | {m}: supply = {supply}\n"

            printty += "\n"
        
        printty += "\n[FACTORY]\n"
        for f in self.model.factories:
            for m in self.model.materials:
                demand = self.model.factory_demand[m, f]
                supply = self.model.factory_supply[m, f]
                printty += f"  {f} | {m}: demand = {demand}, supply = {supply}\n"
            printty += "\n"

        printty += "\n[SINKHOLE DEMAND]\n"
        for sk in self.model.sinkholes:
            for m in self.model.materials:
                demand = self.model.sinkhole_demand[m, sk]
                printty += f"  {sk} | {m}: demand = {demand}\n"
            printty += "\n"
        
        printty += "\n[POSITIONS]\n"
        printty += f"  Sources: {dict(self.model.source_positions.items())}\n"
        printty += f"  Factories: {dict(self.model.factory_positions.items())}\n"
        printty += f"  Sinkholes: {dict(self.model.sinkhole_positions.items())}\n"
        
        printty += "\n[PATH PARAMETERS]\n"
        for tups in self.model.path_throughput:
            (ax, ay, bx, by) = tups
            throughput = self.model.path_throughput[tups]
            printty += f"  Path ({ax}, {ay} -> {bx}, {by}): throughput = {throughput}\n"
        
        printty += "\n[FLOW LIMITS]\n"
        for n in self.model.flow_limits:
            limit = self.model.flow_limits[n]
            printty += f"  Node {n}: flow limit = {limit}\n"

        printty += "\n[VARIABLES]\n"
        for tups in self.model.path_flows:
            (ax, ay, bx, by, mat) = tups
            flow = self.model.path_flows[tups].value
            printty += f"  Path ({ax}, {ay} -> {bx}, {by}) | Material: {mat}: flow = {flow}\n"

        

        return printty
if __name__ == "__main__":
    solver = SolverV2(solver_name='glpk', problem_dir='default.in')
    ### SETUP TEST
    solver.setup_model()
    sol = solver.solve_model()
    solver.save_results(save_path='results.out', solution=sol)
    solver.save_settings()
    solver.save_model_object(model_name='model_object.pkl')

    # ### LOADING TEST
    # solver.load_model_object(model_name='model_object.pkl')
    # solver.solve_model()