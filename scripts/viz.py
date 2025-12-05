import plotly.graph_objects as go
import numpy as np
import pandas as pd

from utils import generate_neighbor_pairings_row_major, row_index_to_node

color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

def create_grid_points(rows, cols):
    x_coords, y_coords = np.meshgrid(range(cols), range(rows))
    x_flat, y_flat = x_coords.flatten(), y_coords.flatten()

    return go.Scatter(
        x=x_flat,
        y=y_flat,
        mode='markers',
        marker=dict(size=8, color='white'),
        name = 'Nodes'
    )

def create_line_trace(x, y, name='Connection', default=False, legend=True, mode ='lines+markers'):
    line_style = dict(color='lightgray', width=1, dash='solid') if default else dict(color='blue', width=2)
    return go.Scatter(
        x=x,
        y=y,
        mode=mode,
        line=line_style,
        name=name,
        showlegend=legend
    )

def draw_default_lines(rows, cols):
    traces = []
    x,y = [], []
    pairings = generate_neighbor_pairings_row_major(rows, cols, onedir=True)
    for (a, b) in pairings:
        from_x, from_y = a % cols, a // cols
        to_x, to_y = b % cols, b // cols
        x.extend([from_x, to_x, None])
        y.extend([from_y, to_y, None])
    traces.append(create_line_trace(x, y, name='Connection', default=True))
    return traces

def draw_arrows(fig, x1, y1, x2, y2, material='', color='RoyalBlue'):
    for x1, y1, x2, y2 in zip(x1, y1, x2, y2):
        fig.add_shape(
            type="line",
            x0=x1, y0=y1, x1=x2, y1=y2,
            line=dict(color=color, width=3),
            name = f'Flow of {material}',
            # Arrowhead at the end
            arrowhead=3
        )

fig = go.Figure()
N,M = 5,5
fig.add_trace(create_grid_points(N, M))
for trace in draw_default_lines(N, M):
    fig.add_trace(trace)

materials = ['', 'oil']  # Example materials
flows = {
    'water': { (0, 1): 10, (0, 5): 5 },
    'oil': { (0, 1): 3, (2, 3): 8 }
}
colors = {'water': 'cyan', 'oil': 'orange'}

df = pd.read_csv(r'./out_files/results.out')  # Example CSV file
material_flows = {}
for material in df['material'].unique():
    material_flows[material] = {}
    for _, row in df.iterrows():
        if row['material'] == material and row['flow'] > 0:
            key = (row_index_to_node(row['from_node'], M), row_index_to_node(row['to_node'], M))
            material_flows[material][key] = row['flow']

for material, paths in material_flows.items():
    x = []
    y = []
    color = colors.get(material, 'blue')
    for (a, b), flow in paths.items():
        from_x, from_y = a[0], a[1]
        to_x, to_y = b[0], b[1]
        x.extend([from_x, to_x, None])
        y.extend([from_y, to_y, None])
    f
        

fig.update_layout(
    template = 'plotly_dark',
    title='NxM Grid of Points',
    xaxis=dict(title='X', range=[-1, N]),
    yaxis=dict(title='Y', range=[-1, M]),
)
fig.show()
