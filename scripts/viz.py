import random
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from utils import generate_neighbor_pairings_row_major, row_index_to_node

COLOR_LIST = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

def create_grid_points(rows, cols):
    """Create scatter plot of grid points."""
    x_coords, y_coords = np.meshgrid(range(cols), range(rows))
    x_flat, y_flat = x_coords.flatten(), y_coords.flatten()
    
    return go.Scatter(
        x=x_flat, y=y_flat,
        mode='markers',
        marker=dict(size=8, color='white'),
        name='Nodes'
    )

def create_line_trace(x, y, name='Connection', default=False):
    """Create a line trace for connections."""
    line_style = dict(color='lightgray', width=1, dash='solid') if default else dict(color='blue', width=2)
    return go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        line=line_style,
        name=name,
        showlegend=not default
    )   

def draw_default_lines(rows, cols):
    """Draw default grid connections."""
    traces = []
    x, y = [], []
    pairings = generate_neighbor_pairings_row_major(rows, cols, onedir=True)
    
    for a, b in pairings:
        from_x, from_y = a % cols, a // cols
        to_x, to_y = b % cols, b // cols
        x.extend([from_x, to_x, None])
        y.extend([from_y, to_y, None])
    
    traces.append(create_line_trace(x, y, name='Connection', default=True))
    return traces

def draw_arrows(fig, x_starts, y_starts, x_ends, y_ends, material='material', color='blue'):
    """Draw arrows as annotations for material flows."""
    for xs, ys, xe, ye in zip(x_starts, y_starts, x_ends, y_ends):
        fig.add_annotation(
            x=xe, y=ye,
            ax=xs, ay=ys,
            xref='x', yref='y',
            axref='x', ayref='y',
            arrowhead=1,
            arrowsize=2,
            arrowwidth=1.5,
            arrowcolor=color,
            showarrow=True,
            name=material
        )

# Initialize figure
fig = go.Figure()
N, M = 5, 5
# Add grid points and default lines
fig.add_trace(create_grid_points(N, M))
for trace in draw_default_lines(N, M):
    fig.add_trace(trace)

# Add source, factories and sinks
special_nodes = {'Source': [(0, 0)], 'Sink': [(N-1, M-1)], 'Factory': [(N//2, M//2)]}
for label, positions in special_nodes.items():
    if label == 'Factory':
        x, y = zip(*positions)
        fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='orange', size=15),
                    name='Factories',
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=['Factory' for _ in x]
                ))
    elif label == 'Source':
        x, y = zip(*positions)
        fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(symbol='diamond', color='green', size=15),
                    name='Sources',
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=['Source' for _ in x]
                ))
    elif label == 'Sink':
        x, y = zip(*positions)
        fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(symbol='octagon', color='red', size=15),
                    name='Sinkholes',
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=['Sinkhole' for _ in x]
                ))

# Load and process material flows
colors = COLOR_LIST.copy()
df = pd.read_csv(r'./out_files/results.out')

material_flows = {}
for material in df['material'].unique():
    material_flows[material] = {}
    for _, row in df.iterrows():
        if row['material'] == material and row['flow'] > 0:
            key = (row_index_to_node(row['from_node'], M), row_index_to_node(row['to_node'], M))
            material_flows[material][key] = row['flow']

# Draw arrows and track material colors
material_colors = {}
annotation_counts = {}

for material, paths in material_flows.items():
    color = random.choice(colors)
    colors.remove(color)
    material_colors[material] = color
    annotation_counts[material] = len(paths)
    
    x_starts, y_starts, x_ends, y_ends = [], [], [], []
    for (a, b) in paths.keys():
        x_starts.append(a[0])
        y_starts.append(a[1])
        x_ends.append(b[0])
        y_ends.append(b[1])
    
    draw_arrows(fig, x_starts, y_starts, x_ends, y_ends, material=material, color=color)

# Create toggle buttons
buttons = []
num_annotations_so_far = 0

for material in material_flows.keys():
    color = material_colors[material]
    num_arrows = annotation_counts[material]
    
    relayout_args = {}
    for i in range(len(fig.layout.annotations)):
        is_visible = num_annotations_so_far <= i < num_annotations_so_far + num_arrows
        relayout_args[f"annotations[{i}].visible"] = is_visible
    
    num_annotations_so_far += num_arrows
    
    buttons.append(
        dict(
            label=f"<span style='color:{color}'>◼</span> {material}",
            method="relayout",
            args=[relayout_args]
        )
    )

# Add "Show All" button
show_all_args = {f"annotations[{i}].visible": True for i in range(len(fig.layout.annotations))}
buttons.insert(0, dict(
    label='◼ Show All',
    method="relayout",
    args=[show_all_args]
))

# Update layout
fig.update_layout(
    template='plotly_dark',
    title='NxM Grid of Points',
    xaxis=dict(title='X', range=[-1, N]),
    yaxis=dict(title='Y', range=[-1, M]),
    updatemenus=[
        dict(
            buttons=buttons,
            direction="up",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.0,
            xanchor="right",
            y=0.0,
            yanchor="bottom"
        )
    ]
)

fig.show()
