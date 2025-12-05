import json
from xml.parsers.expat import model
from pyomo.environ import SolverFactory, ConcreteModel, RangeSet, Set, Var, Param, Constraint, Objective, minimize, maximize
from pyomo.environ import NonNegativeReals, Any, Reals, Binary
import os
import random
import numpy as np

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd

from read_data import read_data
from utils import generate_neighbor_pairings_row_major, neighbors, all_neighbors, get_desc_variables, node_to_row_index
from utils import compute_net_flow, compute_required_flow, compute_net_flow_total, compute_total_throughput


class Solver:
    def __init__(self, solver_name='glpk', problem_dir='default.in'):
        self.solver_name = solver_name
        self.solver = SolverFactory(solver_name)

        self.problem_dir = rf'./in_files/{problem_dir}'
        self.settings = read_data(self.problem_dir)

        self.models = {}
        self.solutions = np.zeros((self.settings['rows'], self.settings['cols'], int(self.settings['sim_time'])))


    def solve_problem(self):
        for t in range(int(self.time_steps)):
            results = self.solver.solve(self.model, tee=False)
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
        #   - Connections (node a -> b and b -> a)
        self.model.connections = Set(initialize=self.connections)
        #   - Sources (-)
        self.model.sources = Set(initialize=self.sources)
        #   - Factories
        self.model.factories = Set(initialize=self.factories)
        #   - Sinkholes (+)
        self.model.sinkholes = Set(initialize=self.sinkholes)

        #   - Resources
        self.model.resources = Set(initialize=[f'resource_{i}' for i in range(self.settings['n_resources'])])
        #   - Products
        self.model.products = Set(initialize=[f'product_{i}' for i in range(self.settings['n_products'])])

        # Decision Variables:
        #   * Amount of material m moved from node a to node b at time step t ((ax,ay) -> (bx,by), t, m)
        self.model.distributed_amounts = Var(self.connections, self.model.products | self.model.resources, domain=NonNegativeReals)
        # self.model.distributed_amounts.display()

            # """Objective Function 1 : Minimize the number of active connections used in the solution"""
            # self.model.is_active = Var(self.model.connections, domain=Binary)
            # self.M = 1e6

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
        neighbor_pairs = generate_neighbor_pairings_row_major(self.rows, self.cols, onedir=True)  # Only (a, b) where a < b
        
        rand_throughput = np.random.randint(self.min_throughput,self.max_throughput,size=(len(neighbor_pairs)))
        rand_time_intervals = np.random.randint(self.min_time_interval,self.max_time_interval,size=(len(neighbor_pairs)))
        rand_time_offsets = np.random.randint(self.min_time_offset,self.max_time_offset,size=(len(neighbor_pairs)))

        path_params = {}
        for idx, (a,b) in enumerate(neighbor_pairs):
            path_params[(a,b)] = (rand_throughput[idx], rand_time_intervals[idx], rand_time_offsets[idx])
            path_params[(b,a)] = (rand_throughput[idx], rand_time_intervals[idx], rand_time_offsets[idx])


        """     - Throughput"""
        self.model.max_bottleneck = Param(
            self.model.connections,
            initialize={connection :path_params[connection][0] for connection in self.model.connections.data()},
            mutable=False
        )

        """     - Temporal (ON)"""
        self.model.time_intervals = Param(
            self.model.connections,
            initialize={connection :path_params[connection][1] for connection in self.model.connections.data()},
            mutable=False
        )

        """     - Temporal (OFF)"""
        self.model.time_offsets = Param(
            self.model.connections,
            initialize={connection :path_params[connection][2] for connection in self.model.connections.data()},
            mutable=False
        )

        """ NODE
                - Flow"""
        # Set all net_flows to 0
        flows = {i: 0 for i in self.model.nodes.data()}
        # Remove special positions
        for source, pos in self.model.source_positions.items():
            flows.pop(pos)
        for sinkhole, pos in self.model.sinkhole_positions.items():
            flows.pop(pos)
        for factory, pos in self.model.factory_positions.items():
            flows.pop(pos)

        # # Set source flows (negative)
        #     net_flow = compute_required_flow(self, source, node_type='source')
        #     flows[pos] = net_flow
        # # Set sinkhole flows (positive)
        # for sinkhole, pos in self.model.sinkhole_positions.items():
        #     net_flow = compute_required_flow(self, sinkhole, node_type='sinkhole')
        #     flows[pos] = net_flow
        # # Set factory flows (can be positive or negative)
        # for factory, pos in self.model.factory_positions.items():
        #     net_flow = compute_required_flow(self, factory, node_type='factory')
        #     flows[pos] = net_flow

        self.model.flow_limits = Param(
            self.model.nodes,
            initialize=flows,
            mutable=False
        )

        # self.model.flow_limits.display()
        # self.model.factory_positions.display()
        # self.model.sinkhole_positions.display()
        # self.model.source_positions.display()
        
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
        def flow_equilibrium(model, path):
            # If the node is a source/sink/factory, skip
            if path not in model.flow_limits:
                return Constraint.Skip

            # Compute net flow
            net_flow = compute_net_flow_total(self, path)

            # Non -source/sink/factory nodes must have net flow of 0
            if model.flow_limits[path] == 0:
                return net_flow == 0

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

        def max_throughput(model, a, b):
            path = (a, b)
            # Avoid double counting by only considering one direction
            if a > b :
                return Constraint.Skip
            flow = sum(
                # Sum the amounts in both directions for material m
                model.distributed_amounts[(a,b), m] + model.distributed_amounts[(b,a), m]
                for m in self.model.resources | self.model.products
            )
            return flow <= model.max_bottleneck[path]
        
        # """ TEMPORAL
        #     - Closed/Open"""
        # def temporal(model, i, t):
        #     if t == 0:
        #         return Constraint.Skip
        #     else:
        #         if t >= model.time_intervals[i] + model.time_offsets[i] < model.time_intervals[i]:
        #             for m in range(self.total_materials):
        #                 self.model.distributed_amounts[i, m] = 0
        #         else:
        #             return Constraint.Skip
                
        """ SOURCE
            - Supply (Per Material)"""
        # Given its supply the net flow is negative (-) and must be larger than (-)supply
        def source_supply(model, s, m):
            net_flow = compute_net_flow(self, model.source_positions[s], m)
            return net_flow >= -model.source_resource_supply[m, s]
        
        """ SINKHOLE
            - Demand"""
        # Given its demand the net flow is positive (+) and must be greater than demand 
        def sinkhole_demand(model, sk, m):
            net_flow = compute_net_flow(self, model.sinkhole_positions[sk], m)
            return net_flow >= model.sinkhole_product_demand[m, sk]

        """ FACTORY
            - Demand"""
        # Given its demand the net flow is positive (+) and must be greater than demand
        def factory_demand(model, f, m):
            net_flow = compute_net_flow(self, model.factory_positions[f], m)
            
            if m in model.resources:
                return net_flow >= model.factory_resource_demand[m, f]
            elif m in model.products:
                return net_flow >= model.factory_product_demand[m, f]
            else:
                raise ValueError("Material not found in resources or products.")
            
        
        """ - Supply"""
        # Given its supply the net flow is negative (-) and must be more than (-)supply
        def factory_supply(model, f, m):
            net_flow = compute_net_flow(self, model.factory_positions[f], m)

            if m in model.resources:
                return net_flow >= -model.factory_resource_supply[m, f]
            elif m in model.products:
                return net_flow >= -model.factory_product_supply[m, f]
            else:
                raise ValueError("Material not found in resources or products.")

        self.model.flow_equilibrium = Constraint(self.model.nodes, rule=flow_equilibrium)
        # model.flow_direction = Constraint(model.nodes, rule=flow_direction)
        self.model.max_throughput = Constraint(self.model.connections, rule=max_throughput)
        self.model.source_supply = Constraint(self.model.sources, self.model.resources, rule=source_supply)
        self.model.sinkhole_demand = Constraint(self.model.sinkholes, self.model.products, rule=sinkhole_demand)
        self.model.factory_demand = Constraint(self.model.factories, self.model.resources | self.model.products, rule=factory_demand)
        self.model.factory_supply = Constraint(self.model.factories, self.model.resources | self.model.products, rule=factory_supply)

        # """ Objective Function 1 : Minimize the number of active connections used in the solution"""
        # def activation_constraint_upper(model, a, b, m):
        #     i = (a, b)
        #     # If flow > 0, then is_active must be 1.
        #     return model.distributed_amounts[i, m] <= self.M * model.is_active[i]

        # self.model.activation_upper = Constraint(self.model.connections, self.model.products | self.model.resources, rule=activation_constraint_upper)
        # # self.model.activation_lower = Constraint(self.model.connections, self.model.products | self.model.resources, rule=activation_constraint_lower)
    
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

        def obj_function1(model):
            paths = generate_neighbor_pairings_row_major(self.rows, self.cols, onedir=True)
        # Sum the difference between max bottleneck and actual flow for all connections and materials
            return sum(model.max_bottleneck[connection] - compute_total_throughput(self, connection)
                       for connection in paths)

        self.model.objective = Objective(rule=obj_function1, sense=maximize)

    def display_results(self, solution_dir='results.csv', load_model_dir=None):
        # Read the results from CSV
        df = pd.read_csv(rf"./out_files/{solution_dir}")

        df['time'] = df['time'].astype(int)
        
        # Define colors for materials
        materials = df['material'].unique()
        colors = {}
        color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, material in enumerate(materials):
            colors[material] = color_list[i % len(color_list)]

        paths = generate_neighbor_pairings_row_major(self.rows, self.cols, onedir=True)
        # Convert node indices to (x, y) coordinates
        paths_coords = {}
        for path in paths:
            a, b = path[0], path[1]
            from_x, from_y = a % self.cols, a // self.cols
            to_x, to_y = b % self.cols, b // self.cols
            paths_coords[(a, b)] = (from_x, from_y, to_x, to_y)    
        
        def offset_parallel_line(x1, y1, x2, y2, offset=0.15):
            """Create a parallel line offset perpendicular to the original line"""
            # Calculate perpendicular direction
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length == 0:
                return x1, y1, x2, y2
            # Perpendicular unit vector
            perp_x = -dy / length
            perp_y = dx / length
            # Offset points
            new_x1 = x1 + perp_x * offset
            new_y1 = y1 + perp_y * offset
            new_x2 = x2 + perp_x * offset
            new_y2 = y2 + perp_y * offset
            return new_x1, new_y1, new_x2, new_y2
        
        def create_arrow(x1, y1, x2, y2, offset=0):
            """Create arrow points with optional offset"""
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length == 0:
                return [x1, x2], [y1, y2]
            
            # Apply offset if needed
            if offset != 0:
                perp_x = -dy / length
                perp_y = dx / length
                x1 += perp_x * offset
                y1 += perp_y * offset
                x2 += perp_x * offset
                y2 += perp_y * offset
            
            # Recalculate after offset
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            # Arrow head parameters
            arrow_size = 0.1
            angle = np.pi / 6  # 30 degrees
            
            # Arrow head points
            ax1 = x2 - arrow_size * length * (np.cos(np.arctan2(dy, dx) - angle))
            ay1 = y2 - arrow_size * length * (np.sin(np.arctan2(dy, dx) - angle))
            ax2 = x2 - arrow_size * length * (np.cos(np.arctan2(dy, dx) + angle))
            ay2 = y2 - arrow_size * length * (np.sin(np.arctan2(dy, dx) + angle))
            
            # Return line with arrow head
            return [x1, x2, ax1, x2, ax2], [y1, y2, ay1, y2, ay2]
        
        def plot_connections(from_x,from_y,to_x,to_y, name,
                             mode='lines',default=True, legend=True):
            if mode == 'arrows':
                x,y = create_arrow(from_x, from_y, to_x, to_y, offset=0)
            else:
                x = [from_x, to_x]
                y = [from_y, to_y]
            if name in colors:
                color = colors[name]
                name = f'Flow of {name}'
            else:
                name = 'Connection'
                color = 'gray'
            return go.Scatter(
                x=x,
                y=y,
                mode=mode,
                line=dict(color=color, width=2),
                name=name,
                showlegend=legend,
                hoverinfo='text',
                visible=default
                # hovertext=[name for _ in x]
            )
        
        def default_figure(path_coords):
            fig = go.Figure()

            for (a, b), (from_x, from_y, to_x, to_y) in path_coords.items():
                fig.add_trace(plot_connections(from_x, from_y, to_x, to_y, name='Connection', default=True, legend=False))
            
            # Add sources (diamonds)
            source_x = []
            source_y = []
            for source, pos in self.model.source_positions.items():
                x_coord = pos % self.cols
                y_coord = pos // self.cols
                source_x.append(x_coord)
                source_y.append(y_coord)
                print(f'Source {source}: position={pos}, coords=({x_coord}, {y_coord})')
            
            if source_x:
                fig.add_trace(go.Scatter(
                    x=source_x,
                    y=source_y,
                    mode='markers',
                    marker=dict(symbol='diamond', color='green', size=15),
                    name='Sources',
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=['Source' for _ in source_x]
                ))
            
            # Add factories (triangles)
            factory_x = []
            factory_y = []
            for factory, pos in self.model.factory_positions.items():
                x_coord = pos % self.cols
                y_coord = pos // self.cols
                factory_x.append(x_coord)
                factory_y.append(y_coord)
                print(f'Factory {factory}: position={pos}, coords=({x_coord}, {y_coord})')
            
            if factory_x:
                fig.add_trace(go.Scatter(
                    x=factory_x,
                    y=factory_y,
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='orange', size=15),
                    name='Factories',
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=['Factory' for _ in factory_x]
                ))
            
            # Add sinkholes (octagons)
            sinkhole_x = []
            sinkhole_y = []
            for sinkhole, pos in self.model.sinkhole_positions.items():
                x_coord = pos % self.cols
                y_coord = pos // self.cols
                sinkhole_x.append(x_coord)
                sinkhole_y.append(y_coord)
                print(f'Sinkhole {sinkhole}: position={pos}, coords=({x_coord}, {y_coord})')
            
            if sinkhole_x:
                fig.add_trace(go.Scatter(
                    x=sinkhole_x,
                    y=sinkhole_y,
                    mode='markers',
                    marker=dict(symbol='octagon', color='red', size=15),
                    name='Sinkholes',
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=['Sinkhole' for _ in sinkhole_x]
                ))

            fig.update_layout(
                title='Network Flow at Time',
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                clickmode='event+select',
                hovermode='closest'
            )

            fig.update_xaxes(showgrid=False, range=[-0.5, self.cols-0.5], visible=False)
            fig.update_yaxes(showgrid=False, range=[-0.5, self.rows-0.5], visible=False)
            
            return fig
        
        def update_figure(selected_time, path_coords):
            fig = default_figure(selected_time)

            # Filter data for the selected time
            df_time = df[df['time'] == selected_time]

            # For each path we draw the flow lines for each material is not 0
            # We update 

            # Add flow connections for the selected time
            for _, row in df_time.iterrows():
                from_node = row['from_node']
                to_node = row['to_node']
                material = row['material']
                flow = row['flow']

                from_x, from_y = from_node % self.cols, from_node // self.cols
                to_x, to_y = to_node % self.cols, to_node // self.cols

                if flow > 0:
                    fig.add_trace(plot_connections(from_x, from_y, to_x, to_y, name=material, mode='arrows', default=True))
            
            return fig
        import plotly.io as pio
        pio.show(default_figure(paths_coords))

    def save_results(self, save_path='results.out', solution=''):
        rows = []
        for key in self.model.distributed_amounts.keys():
            (a, b, mat) = key
            flow = self.model.distributed_amounts[(a, b), mat].value
            rows.append({'time': 0, 'from_node': a, 'to_node': b, 'material': mat, 'flow': flow})
        df = pd.DataFrame(rows)
        df.to_csv(rf"./out_files/{save_path}", index=False)

        self.display_results(solution_dir=save_path)

if __name__ == '__main__':
    solver = Solver(solver_name='glpk', problem_dir='default.in')
    solver.build_model()
    sol = solver.solve_problem()
    solver.save_results(solution=sol)