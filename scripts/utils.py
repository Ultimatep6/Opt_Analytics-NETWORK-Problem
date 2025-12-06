def generate_neighbor_pairings_row_major(n, m, onedir=False):
    pairings = []
    for i in range(n):
        for j in range(m):
            node_a = i * m + j
            # Right neighbor
            if j < m - 1:
                node_b = i * m + (j + 1)
                pairings.append((node_a, node_b))
            # Down neighbor
            if i < n - 1:
                node_b = (i + 1) * m + j
                pairings.append((node_a, node_b))
            if not onedir:
                # Up neighbor
                if i > 0:
                    node_b = (i - 1) * m + j
                    pairings.append((node_a, node_b))
                # Left neighbor
                if j > 0:
                    node_b = i * m + (j - 1)
                    pairings.append((node_a, node_b))
    return pairings

def generate_neighbor_pairings_2d(rows, cols, onedir=False):
    pairings = []
    for i in range(rows):
        for j in range(cols):
            node_a = (i, j)
            # Right neighbor
            if j < cols - 1:
                node_b = (i, j + 1)
                pairings.append((node_a, node_b))
            # Down neighbor
            if i < rows - 1:
                node_b = (i + 1, j)
                pairings.append((node_a, node_b))
            if not onedir:
                # Up neighbor
                if i > 0:
                    node_b = (i - 1, j)
                    pairings.append((node_a, node_b))
                # Left neighbor
                if j > 0:
                    node_b = (i, j - 1)
                    pairings.append((node_a, node_b))
    return pairings


def node_to_row_index(row, col, cols):
    return row * cols + col

def row_index_to_node(index, cols):
    return index // cols, index % cols

def neighbors(node, rows, cols):
    """
    Given a node (row-major index), return its right and down neighbors in the grid.
    """
    i, j = node // cols, node % cols
    neighbor_list = []
    # Down neighbor
    if i < rows - 1:
        neighbor_list.append(node_to_row_index(i + 1, j, cols))
    # Right neighbor
    if j < cols - 1:
        neighbor_list.append(node_to_row_index(i, j + 1, cols))
    return neighbor_list

def all_neighbors(node, rows, cols, row_major=True):
    """
    Given a node (i, j), return its valid neighbors in the grid.
    Neighbors are up, down, left, right (no diagonals) and given in Row-major order.
    """
    if row_major:
        i,j = node[0] // cols, node[0] % cols
    else:
        i,j = node
    neighbor_list = []
    
    # Up
    if i > 0:
        if row_major:
            neighbor_list.append(node_to_row_index(i - 1, j, cols))
        neighbor_list.append((i - 1, j))

    # Down
    if i < rows - 1:
        if row_major:
            neighbor_list.append(node_to_row_index(i + 1, j, cols))
        neighbor_list.append((i + 1, j))

    # Left
    if j > 0:
        if row_major:
            neighbor_list.append(node_to_row_index(i, j - 1, cols))
        neighbor_list.append((i, j - 1))
    # Right
    if j < cols - 1:
        if row_major:
            neighbor_list.append(node_to_row_index(i, j + 1, cols))
        neighbor_list.append((i, j + 1))

    return neighbor_list


def get_desc_variables(model, time_step: int = None, material: int = None):
    """
    Extract all decision variables from the model for further processing.
    If time_step and/or material are provided, filter accordingly.
    """
    result = []
    for idx in model.all_desc_var:
        i, j, k, m = idx
        if time_step is not None and k != time_step:
            continue
        if material is not None and m != material:
            continue
        result.append(model.all_desc_var[i, j, k, m])
    return result

    
def compute_net_flow_total(solver, node, row_major=True):
    """Net flow at a node: inflows minus outflows (all materials)."""
    inflow = sum(
        solver.model.path_flows[k[0], k[1], node[0], node[1], m]
        for k in all_neighbors(node, solver.rows, solver.cols, row_major=row_major)
        for m in solver.model.materials
    )
    outflow = sum(
        solver.model.path_flows[node[0], node[1], j[0], j[1], m]
        for j in all_neighbors(node, solver.rows, solver.cols, row_major=row_major)
        for m in solver.model.materials
    )
    return inflow - outflow

def compute_net_flow(solver, node, material, return_inflow=None, row_major=True): # + is inflows, - is outflows
    """ Computed the net flow at a given node by summing inflows and outflows of a given material"""
    inflow = sum(
        solver.model.path_flows[k[0], k[1], node[0], node[1], material]
        for k in all_neighbors(node, solver.R, solver.C, row_major=row_major)
    )
    outflow = sum(
        solver.model.path_flows[node[0], node[1], j[0], j[1], material]
        for j in all_neighbors(node, solver.R, solver.C, row_major=row_major)
    )
    if return_inflow:
        return inflow
    elif not return_inflow:
        return outflow
    else:
        return inflow - outflow

def compute_required_flow(solver, object, node_type='source'):
    if node_type == 'source':
        out_flows = sum(solver.model.source_resource_supply[r, object] for r in solver.model.resources)
        return -out_flows
    
    elif node_type == 'sinkhole':
        in_flows = sum(solver.model.sinkhole_product_demand[m, object] for m in solver.model.products)
        return in_flows

    elif node_type == 'factory':
        in_flows = sum(solver.model.factory_resource_demand[r, object] for r in solver.model.resources) + \
                    sum(solver.model.factory_product_demand[p, object] for p in solver.model.products)
        out_flows = sum(solver.model.factory_resource_supply[r, object] for r in solver.model.resources) + \
                    sum(solver.model.factory_product_supply[p, object] for p in solver.model.products)

        return in_flows - out_flows
    else:
        raise ValueError("Invalid node type. Must be 'source', 'sinkhole', or 'factory'.")
    

def compute_total_throughput(solver, path):
    ax, ay, bx, by = path
    
    if node_to_row_index(ax, ay, solver.C) > node_to_row_index(bx, by, solver.C):
        return 0
    total_throughput = sum(
        solver.model.path_flows[ax, ay, bx, by, m] + solver.model.path_flows[bx, by, ax, ay, m]
        for m in solver.model.materials
    )
    return total_throughput

if __name__ == "__main__":
    r,c = 5,5
    # print(generate_neighbor_pairings_row_major(r, c),len(generate_neighbor_pairings_row_major(r, c)))
    print(generate_neighbor_pairings_2d(r, c),len(generate_neighbor_pairings_2d(r, c)))