from xml.parsers.expat import model
from pyomo.environ import SolverFactory, ConcreteModel, RangeSet, Set, Var, Param, Constraint, Objective, minimize
from pyomo.environ import NonNegativeReals, Any, Reals
import os
import random
import numpy as np

from read_data import read_data
from utils import generate_neighbor_pairings_row_major, neighbors, all_neighbors, get_desc_variables, node_to_row_index
from utils import compute_net_flow

class Solver:
    def __init__(self, solver_name='glpk', problem_dir='default.in'):
        self.solver_name = solver_name
        self.solver = SolverFactory(solver_name)

        self.problem_dir = rf'./in_files/{problem_dir}'
        self.settings = read_data(self.problem_dir)

        self.models = {}
        self.solutions = np.zeros((self.settings['rows'], self.settings['cols'], int(self.settings['sim_time'])))


    def solve(self):
        for t in range(self.time_steps):
            results = self.solver.solve(self.model, tee=True)
        return results
    
    def get_components(self):
        self.rows = self.settings['rows']
        self.cols = self.settings['cols']
        self.time_steps = self.settings['sim_time']
        self.total_materials = self.settings['n_resources'] + self.settings['n_products']

        # For simple solution
        self.quarter_div_col = self.cols // 4
        self.quarter_div_row = self.rows // 3
        
        self.sources = [ f'source_{i}' for i in range(1,self.settings['n_sources']+1) ]
        self.factories = [ f'fact_{i}' for i in range(1,self.settings['n_factories']+1) ]
        self.sinkholes = [ f'sink_{i}' for i in range(1,self.settings['n_sinkholes']+1) ]

        # Settings
        self.min_throughput = self.settings['min_bottleneck']
        self.max_throughput = self.settings['max_bottleneck']

        self.min_time_interval = self.settings['min_time_interval']
        self.max_time_interval = self.settings['max_time_interval']

        self.min_time_offset = self.settings['min_time_offset']
        self.max_time_offset = self.settings['max_time_offset']

        # I/O matrices
        self.source_O = self.settings['source_output_matrix']

        self.factory_IO_resource = self.settings['factory_io_resource_matrix']
        self.factory_IO_products = self.settings['factory_io_products_matrix']

        self.sinkhole_I = self.settings['sinkhole_input_matrix']

        self.nodes = [i for i in range(self.rows * self.cols)]

        # Connections --- ROW MAJOR
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
        print(f"Problem Size: {len(self.connections)*self.time_steps*self.total_materials}\n")
    
    def model_variables(self):
        # Sets:
        #   - Nodes
        self.model.nodes = Set(initialize=self.nodes)
        #   - Connections (node a -> b)
        self.model.connections = Set(initialize=self.connections)
        #   - Sources
        self.model.sources = Set(initialize=self.sources)
        #   - Factories
        self.model.factories = Set(initialize=self.factories)
        #   - Sinkholes
        self.model.sinkholes = Set(initialize=self.sinkholes)

        #   - Resources
        self.model.resources = Set(initialize=lambda i : [f'resource_{i}' for i in range(self.settings['n_resources'])])
        #   - Products
        self.model.products = Set(initialize=lambda i : [f'product_{i}' for i in range(self.settings['n_products'])])

        # Decision Variables:
        #   * Amount of material m moved from node a to node b at time step t ((ax,ay) -> (bx,by), t, m)
        self.model.distributed_amounts = Var(self.connections, RangeSet(0, self.total_materials - 1), domain=Reals)

    def model_params(self):
        """ SOURCE/FACTORY/SINKHOLE POSITION PARAMETERS
                - SOURCE
                    - Positions"""
        
        positions = [node_to_row_index(i,self.quarter_div_col, self.cols) for i in range(self.rows)]
        source_pos_dict = {}
        for s in self.model.sources:
            source_pos_dict[s] = random.choice(positions)
            positions.remove(source_pos_dict[s])

        self.model.source_positions = Param(self.model.sources,
                                        initialize=source_pos_dict,
                                        domain=Any, mutable=False)

        """         - Supply"""
        param_dict = {
            (self.model.resources.data()[i], self.model.sources.data()[j]): self.source_O[i][j]
            for i in range(self.source_O.shape[0])
            for j in range(self.source_O.shape[1])
        }
        self.model.source_resource_supply = Param(self.model.resources, self.model.sources, 
                                             initialize = param_dict,
                                             domain = NonNegativeReals,
                                             mutable=False
        )
        """     - FACTORY
                    - Positions"""
        positions = [node_to_row_index(i,self.quarter_div_col*2,self.cols) for i in range(self.rows)]

        factory_pos_dict = {}
        for s in self.model.factories:
            factory_pos_dict[s] = random.choice(positions)
            positions.remove(factory_pos_dict[s])
        self.model.factory_positions = Param(self.model.factories, initialize=factory_pos_dict, domain=Any, mutable=False)

        """       - Resources"""
        """         - Supply"""
        param_dict = {
            (self.model.resources.data()[i], self.model.factories.data()[j]): self.factory_IO_resource[i,0,j]
            for i in range(self.factory_IO_resource.shape[0])
            for j in range(self.factory_IO_resource.shape[2])
        }
        self.model.factory_resource_demand = Param(self.model.resources, self.model.factories, 
                                             initialize = param_dict,
                                             domain = NonNegativeReals,
                                             mutable=False
        )
        """         - Demand"""
        param_dict = {
            (self.model.resources.data()[i], self.model.factories.data()[j]): self.factory_IO_resource[i,1,j]
            for i in range(self.factory_IO_resource.shape[0])
            for j in range(self.factory_IO_resource.shape[2])
        }
        self.model.factory_resource_supply = Param(self.model.resources, self.model.factories, 
                                             initialize = param_dict,
                                             domain = NonNegativeReals,
                                             mutable=False
        )


        """       - Products"""
        """         - Supply"""
        param_dict = {
            (self.model.products.data()[i], self.model.factories.data()[j]): self.factory_IO_products[i,0,j]
            for i in range(self.factory_IO_products.shape[0])
            for j in range(self.factory_IO_products.shape[2])
        }
        self.model.factory_product_supply = Param(self.model.products, self.model.factories, 
                                             initialize = param_dict,
                                             domain = NonNegativeReals,
                                             mutable=False
        )
        """         - Demand"""
        param_dict = {
            (self.model.products.data()[i], self.model.factories.data()[j]): self.factory_IO_products[i,1,j]
            for i in range(self.factory_IO_products.shape[0])
            for j in range(self.factory_IO_products.shape[2])
        }
        self.model.factory_product_demand = Param(self.model.products, self.model.factories, 
                                             initialize = param_dict,
                                             domain = NonNegativeReals,
                                             mutable=False
        )

        """      - SINKHOLE
                    - Positions"""
        positions = [node_to_row_index(i,self.quarter_div_col*3, self.cols) for i in range(self.rows)]

        sinkhole_pos_dict = {}
        for s in self.model.sinkholes:
            sinkhole_pos_dict[s] = random.choice(positions)
            positions.remove(sinkhole_pos_dict[s])
        self.model.sinkhole_positions = Param(self.model.sinkholes, initialize=sinkhole_pos_dict, domain=Any, mutable=False)

        """         - Products
                        - Demand"""
        param_dict = {
            (self.model.products.data()[i], self.model.sinkholes.data()[j]): self.sinkhole_I[i][j]
            for i in range(self.sinkhole_I.shape[0])
            for j in range(self.sinkhole_I.shape[1])
        }
        self.model.sinkhole_product_demand = Param(self.model.products, self.model.sinkholes, 
                                             initialize = param_dict,
                                             domain = NonNegativeReals,
                                             mutable=False
        )
        


        # self.model.source_positions.display()
        # self.model.factory_positions.display()
        # self.model.sinkhole_positions.display()
        # self.model.source_resource_supply.display()
        # self.model.factory_resource_demand.display()
        # self.model.factory_resource_supply.display()
        # self.model.sinkhole_product_demand.display()


        """PATH PARAMETERS"""
        rand_throughput = np.random.randint(self.min_throughput,self.max_throughput,size=(len(self.model.connections.data())))
        rand_time_intervals = np.random.randint(self.min_time_interval,self.max_time_interval,size=(len(self.model.connections.data())))
        rand_time_offsets = np.random.randint(self.min_time_offset,self.max_time_offset,size=(len(self.model.connections.data())))
        
        """     - Throughput"""
        self.model.max_bottleneck = Param(
            self.model.connections,
            initialize={self.model.connections.data()[i]:rand_throughput[i] for i in range(len(self.model.connections.data()))},
            mutable=False
        )

        # self.model.max_bottleneck.display()

        """     - Temporal (ON)"""
        self.model.time_intervals = Param(
            self.model.connections,
            initialize={self.model.connections.data()[i]:rand_time_intervals[i] for i in range(len(self.model.connections.data()))},
            mutable=False
        )

        """     - Temporal (OFF)"""
        self.model.time_offsets = Param(
            self.model.connections,
            initialize={self.model.connections.data()[i]:rand_time_offsets[i] for i in range(len(self.model.connections.data()))},
            mutable=False
        )

        """ NODE
                - Flow"""
        flows = {i: 0 for i in self.model.nodes.data()}
        for factory, pos in self.model.factory_positions.items():
            in_flows = sum(self.model.factory_resource_demand[r, factory] for r in self.model.resources) + \
                        sum(self.model.factory_product_demand[p, factory] for p in self.model.products)
            out_flows = sum(self.model.factory_resource_supply[r, factory] for r in self.model.resources) + \
                        sum(self.model.factory_product_supply[p, factory] for p in self.model.products)
            
            flows[pos] = in_flows-out_flows

        self.model.flow_limits = Param(
            self.model.nodes,
            initialize=flows,
            mutable=False
        )
        

    def model_constraints(self):
        """ CONSTRAINTS: 

                * Flow Equillibrium : All non source/sink nodes must be in equillibrium
                * Flow Direction : The flow must obey the direction of the arch
                * Flow Max : The flow of all flows through a path must not go over the throughput limit

                * Temporal Closed: The archs are assumed open at t=0, but after t_open time the arch must be closed
                * Temporal Open: The archs are assumed close first after at t=t_open, but at t = t_open+t_open time the arch must be Open

                * Source Supply : The sum of the flows for every material out of a source must be less than source_sup
                * Sink Demand : The sum of the flows for every material into a sink must be more than sink_dem

                * Factory Demand : The sum of the flows for every material into a factory must be greater than factory_dem
                * Factory Supply : The sum of the flows for every material out of a factory must be less than factory_sup

        """

        """ FLOW 
            - Equillibrium"""
        def flow_equilibrium(model, i):
            net_flow = compute_net_flow(self, i)

            if model.flow_limits[i] == 0:
                return net_flow == 0
            # If the node is a source (negative flow limit)
            elif model.flow_limits[i] > 0:
                return net_flow >= model.flow_limits[i]
            else:
                return net_flow <= model.flow_limits[i]

        """"    - Direction"""
        # TODO: Implement the flow_direction constraint logic here
        # def flow_direction(model, i):
        #     for j in model.connections[i]:
        #         if model.distributed_amounts[(i, j), m] > 0:
        #             return model.flow_directions[(i, j)] == 1
        #         elif model.distributed_amounts[(i, j), m] < 0:
        #             return model.flow_directions[(i, j)] == -1
        #     return Constraint.Skip

        """   - Max Throughput"""
        def max_throughput(model, path):
            flow = sum(
                abs(model.distributed_amounts[path, m])
                for m in range(self.total_materials)
            )
            return flow <= model.max_bottleneck[path]
        
        """ TEMPORAL
            - Closed/Open"""
        def temporal(model, i, t):
            if t == 0:
                return Constraint.Skip
            else:
                if t >= model.time_intervals[i] + model.time_offsets[i] < model.time_intervals[i]:
                    for m in range(self.total_materials):
                        self.model.distributed_amounts[i, m] = 0
                else:
                    return Constraint.Skip
                
        """ SOURCE
            - Supply"""
        def source_supply(model, s, m):
            outflow = sum(
                model.distributed_amounts[(model.source_positions[s], k), m]
                for k in neighbors(model.source_positions[s], self.rows, self.cols)
            )
            return outflow <= model.source_resource_supply[m, s]
        
        """ SINKHOLE
            - Demand"""
        def sinkhole_demand(model, sk, m):
            inflow = sum(
                model.distributed_amounts[(k, model.sinkhole_positions[sk]), m]
                for k in neighbors(model.sinkhole_positions[sk], self.rows, self.cols)
            )
            return inflow >= model.sinkhole_product_demand[m, sk]
        
        """ FACTORY
            - Demand"""
        def factory_demand(model, f, m):
            inflow = sum(
                model.distributed_amounts[(k, model.factory_positions[f]), m]
                for k in neighbors(model.factory_positions[f], self.rows, self.cols)
            )
            if m in model.resources:
                return inflow >= model.factory_resource_demand[m, f]
            elif m in model.products:
                return inflow >= model.factory_product_demand[m, f]
            else:
                raise ValueError("Material not found in resources or products.")
        
        """ - Supply"""
        def factory_supply(model, f, m):
            outflow = sum(
                model.distributed_amounts[(model.factory_positions[f], k), m]
                for k in neighbors(model.factory_positions[f], self.rows, self.cols)
            )
            if m in model.resources:
                return outflow <= model.factory_resource_supply[m, f]
            elif m in model.products:
                return outflow <= model.factory_product_supply[m, f]
            else:
                raise ValueError("Material not found in resources or products.")

        self.model.flow_equilibrium = Constraint(self.model.nodes, rule=flow_equilibrium)
        # model.flow_direction = Constraint(model.nodes, rule=flow_direction)
        self.model.max_throughput = Constraint(self.model.nodes, rule=max_throughput)
        self.model.source_supply = Constraint(self.model.sources, self.model.resources, rule=source_supply)
        self.model.sinkhole_demand = Constraint(self.model.sinkholes, self.model.products, rule=sinkhole_demand)
        self.model.factory_demand = Constraint(self.model.factories, self.model.resources | self.model.products, rule=factory_demand)
        self.model.factory_supply = Constraint(self.model.factories, self.model.resources | self.model.products, rule=factory_supply)

    def build_model(self):

        self.model = ConcreteModel()

        # Get all the components
        self.get_components()

        # Define the Variables
        self.model_variables()

        # Define the Parameters
        self.model_params()

        # Define the Constraints
        self.model_constraints()

        def obj_function(model):
            """Simply how many connections are being activated in total"""
            return sum(
                1 if model.distributed_amounts[i, m] > 0 else 0
                for i in model.connections
                for m in range(model.total_materials)
            )

        self.model.objective = Objective(rule=obj_function, sense=minimize)
    
    def display_results(self):
        raise NotImplementedError
    
if __name__ == '__main__':
    solver = Solver(solver_name='glpk', problem_dir='default.in')
    solver.build_model()
    