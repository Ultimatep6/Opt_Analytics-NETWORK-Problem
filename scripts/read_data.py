# This script takes in an input.in file and generates an output.out 
# file which the solver.py will use to read the parameter matrix

import random
import os
import numpy as np

def get_parts(line):
    return line.strip().split()

def read_data(file_input=r'./in_files/default.in'):
    print("\n")
    print(f"Reading data from {file_input}...")
    print("*"*40)
    print("\n")
    with open(file_input, 'r') as f:
        lines = f.readlines()
    # print(f"Total lines read: {len(lines)}")

    for i,line in enumerate(lines):

        if line.startswith('#') or line.strip() == '':
            continue

        # First row with NxN dimensions
        if i == 0:
            parts = get_parts(line)
            rows,cols = int(parts[0]), int(parts[1])
            problem_mesh = np.zeros((rows, cols), dtype=float)


        elif i == 1:
            parts = get_parts(line)
            prop_one_way, prop_two_way = float(parts[0]), float(parts[1])

        elif i == 3:
            parts = get_parts(line)
            n_sources, n_factories, n_sinkholes = int(parts[0]), int(parts[1]), int(parts[2])


        elif i == 4:
            parts = get_parts(line)
            n_resources, n_products = int(parts[0]), int(parts[1])

        elif i == 6:
            source_output_matrix = np.zeros((n_resources, n_sources), dtype=float)
            for r in range(n_resources):
                parts = get_parts(lines[i + r])
                for s in range(n_sources):
                    source_output_matrix[r][s] = float(parts[s])
            # Move index forward
            i += n_sources + 1
        
        elif i == 6 + n_resources + 1:
            factory_io_resource_matrix = np.zeros((n_resources, 2, n_factories), dtype=float)
            for f in range(n_factories):
                for r in range(n_resources):
                    parts = get_parts(lines[i + r + f * (n_resources+1)])
                    factory_io_resource_matrix[r][0][f] = float(parts[0])  # Input
                    factory_io_resource_matrix[r][1][f] = float(parts[1])  # Output

            # Move index forward
            i += n_factories * (n_resources + 1)
        elif i == 6 + n_resources + 1 + n_factories * (n_resources + 1):
            factory_io_products_matrix = np.zeros((n_products, 2, n_factories), dtype=float)
            for f in range(n_factories):
                for r in range(n_products):
                    parts = get_parts(lines[i + r + f * (n_products+1)])
                    factory_io_products_matrix[r][0][f] = float(parts[0])  # Input
                    factory_io_products_matrix[r][1][f] = float(parts[1])  # Output
            # Move index forward
            i += n_factories * (n_products + 1)

        elif i == 6 + n_resources + 1 + n_factories * (n_resources + 1) + n_factories * (n_products + 1):
            print("Row-idx :", i)
            sinkhole_i_products_matrix = np.zeros((n_sinkholes, n_products), dtype=float)
            for r in range(n_sinkholes):
                parts = get_parts(lines[i + r])
                for c in range(n_products):
                    sinkhole_i_products_matrix[r][c] = float(parts[c])
            # Move index forward
            i += n_sinkholes + 1
            print("New Row-idx :", i)

        elif i == 6 + n_resources + 1 + n_factories * (n_resources + 1) + n_factories * (n_products + 1) + n_sinkholes + 1:
            print(f"Row idx {i}")
            parts = get_parts(line)
            max_bottleneck, min_bottleneck = float(parts[0]), float(parts[1])
        
        elif i == 7 + n_resources + 1 + n_factories * (n_resources + 1) + n_factories * (n_products + 1) + n_sinkholes + 1:
            parts = get_parts(line)
            max_time_interval, min_time_interval = float(parts[0]), float(parts[1])
        
        elif i == 8 + n_resources + 1 + n_factories * (n_resources + 1) + n_factories * (n_products + 1) + n_sinkholes + 1:
            parts = get_parts(line)
            max_time_offset, min_time_offset = float(parts[0]), float(parts[1])
            

        elif i == 10 + n_resources + 1 + n_factories * (n_resources + 1) + n_factories * (n_products + 1) + n_sinkholes + 1:
            parts = get_parts(line)
            sim_time = float(parts[0])
            break

        
    print("Data read successfully.")
    print("Mesh Dimensions: {} x {}".format(rows, cols))
    print(f"Proportion One-Way: {prop_one_way}, Proportion Two-Way: {prop_two_way}")
    print("*"*40)
    print("Sources {}, Factories {}, Sinkholes {}".format(n_sources, n_factories, n_sinkholes))
    print("Resources {}, Products {}".format(n_resources, n_products))
    print("*"*40)
    print(f"Source Output Matrix Shape {source_output_matrix.shape}:")
    for r in range(n_sinkholes):
        print(f"Source {r+1}:")
        for c in range(n_sources):
            print(f"  Resource {c+1}: Output {source_output_matrix[r][c]}")
    print("*"*40)
    print(f"Factory I/O Resource Matrix Shape {factory_io_resource_matrix.shape}:")
    for f in range(n_factories):
        print(f"Factory {f+1}:")
        for r in range(n_resources):
            print(f"  Resource {r+1}: Input {factory_io_resource_matrix[r][0][f]}, Output {factory_io_resource_matrix[r][1][f]}")
    print("*"*40)
    print(f"Factory I/O Products Matrix Shape {factory_io_products_matrix.shape}:")
    for f in range(n_factories):
        print(f"Factory {f+1}:")
        for r in range(n_products):
            print(f"  Products {r+1}: Input {factory_io_products_matrix[r][0][f]}, Output {factory_io_products_matrix[r][1][f]}")
    print("*"*40)
    print(f"Sinkhole Input Matrix Shape {sinkhole_i_products_matrix.shape}:")
    for r in range(n_sinkholes):
        print(f"Sinkhole {r+1}:")
        for c in range(n_products):
            print(f"  Products {c+1}: Output {sinkhole_i_products_matrix[r][c]}")
    print("*"*40)
    print(f"Bottleneck Limits: Max {max_bottleneck}, Min {min_bottleneck}")
    print(f"Time Intervals: Max {max_time_interval}, Min {min_time_interval}")
    print(f"Time Offsets: Max {max_time_offset}, Min {min_time_offset}")
    print("*"*40)
    print(f"Simulation Time: {sim_time}")
    print("\n")

    problem_settings = {
        'rows': rows,
        'cols': cols,
        'prop_one_way': prop_one_way,
        'prop_two_way': prop_two_way,
        'n_sources': n_sources,
        'n_factories': n_factories,
        'n_sinkholes': n_sinkholes,
        'n_resources': n_resources,
        'n_products': n_products,
        'source_output_matrix': source_output_matrix,
        'factory_io_resource_matrix': factory_io_resource_matrix,
        'factory_io_products_matrix': factory_io_products_matrix,
        'sinkhole_input_matrix' : sinkhole_i_products_matrix,
        'max_bottleneck': max_bottleneck,
        'min_bottleneck': min_bottleneck,
        'max_time_interval': max_time_interval,
        'min_time_interval': min_time_interval,
        'max_time_offset': max_time_offset,
        'min_time_offset': min_time_offset,
        'sim_time' : sim_time
    }

    return problem_settings


if __name__ == "__main__":
    read_data()