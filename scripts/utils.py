def generate_neighbor_pairings_row_major(n, m):
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
    return pairings


def node_to_row_index(row, col, cols):
    return row * cols + col


def neighbors(node, rows, cols):
    """
    Given a node (i, j), return its valid neighbors in the grid.
    Neighbors are up, down, left, right (no diagonals) and given in Row-major order.
    """
    i,j = node // cols, node % cols
    neighbor_list = []
    
    # Up
    if i > 0:
        neighbor_list.append(node_to_row_index(i - 1, j, cols))
    
    # Down
    if i < rows - 1:
        neighbor_list.append(node_to_row_index(i + 1, j, cols))
    
    # Left
    if j > 0:
        neighbor_list.append(node_to_row_index(i, j - 1, cols))

    # Right
    if j < cols - 1:
        neighbor_list.append(node_to_row_index(i, j + 1, cols))

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

    