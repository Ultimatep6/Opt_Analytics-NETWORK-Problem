import random
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import ast
import json
import dash
from dash import dcc, html, Input, Output, State, callback_context

from utils import generate_neighbor_pairings_row_major

COLOR_LIST = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

def create_grid_points(rows, cols):
    """Create scatter plot of grid points."""
    x_coords, y_coords = np.meshgrid(range(cols), range(rows))
    x_flat, y_flat = x_coords.flatten(), y_coords.flatten()
    
    return go.Scatter(
        x=x_flat, y=y_flat,
        mode='markers',
        marker=dict(size=8, color='white'),
        name='Nodes',
        hoverinfo='skip'
    )

def create_line_trace(x, y, name='Connection', default=False):
    """Create a line trace for connections."""
    line_style = dict(color='lightgray', width=1, dash='solid') if default else dict(color='blue', width=2)
    return go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        line=line_style,
        name=name,
        showlegend=not default,
        hoverinfo='skip'
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

def draw_arrows(fig, x_starts, y_starts, x_ends, y_ends, material='material', color='blue', offset=0):
    """Draw arrows as annotations for material flows.
    
    Args:
        offset: perpendicular offset from the line to allow parallel arrows for bidirectional flows
    """
    for xs, ys, xe, ye in zip(x_starts, y_starts, x_ends, y_ends):
        # Calculate perpendicular offset vector
        dx = xe - xs
        dy = ye - ys
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # Perpendicular unit vector
            perp_x = -dy / length
            perp_y = dx / length
        else:
            perp_x, perp_y = 0, 0
        
        # Apply offset to start and end points
        offset_xs = xs + offset * perp_x
        offset_ys = ys + offset * perp_y
        offset_xe = xe + offset * perp_x
        offset_ye = ye + offset * perp_y
        
        fig.add_annotation(
            x=offset_xe, y=offset_ye,
            ax=offset_xs, ay=offset_ys,
            xref='x', yref='y',
            axref='x', ayref='y',
            arrowhead=1,
            arrowsize=2,
            arrowwidth=1.5,
            arrowcolor=color,
            showarrow=True,
            name=material,
            visible=False  # Hidden by default
        )

def load_settings(settings_path='settings.json'):
    """Load settings from a JSON file."""
    import json
    with open(rf'./out_files/{settings_path}', 'r') as f:
        settings = json.load(f)
    return settings

# Initialize figure
fig = go.Figure()

# Load settings
settings = load_settings(settings_path='settings.json')
N = settings['rows']
M = settings['cols']

# Add grid points and default lines
fig.add_trace(create_grid_points(N, M))
for trace in draw_default_lines(N, M):
    fig.add_trace(trace)

# Get factory, source, sink positions
sources = {
    source: np.array(settings['source_positions'][source]) for source in settings['source_positions']
}
factories = {
    factory: np.array(settings['factory_positions'][factory]) for factory in settings['factory_positions']
}
sinkholes = {
    sinkhole: np.array(settings['sinkhole_positions'][sinkhole]) for sinkhole in settings['sinkhole_positions']
}
# Add source, factories and sinks
special_nodes = {'Sources': sources, 'Sink': sinkholes, 'Factory': factories}
for label, bodies in special_nodes.items():
    if label == 'Factory':
        name, pos = zip(*bodies.items())
        x, y = zip(*pos)
        fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='orange', size=15),
                    name='Factories',
                    showlegend=True,
                    hoverinfo='skip'
                ))
    elif label == 'Sources':
        name, pos = zip(*bodies.items())
        x, y = zip(*pos)
        fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(symbol='diamond', color='green', size=15),
                    name='Sources',
                    showlegend=True,
                    hoverinfo='skip'
                ))
    elif label == 'Sink':
        name, pos = zip(*bodies.items())
        x, y = zip(*pos)
        fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(symbol='octagon', color='red', size=15),
                    name='Sinkholes',
                    showlegend=True,
                    hoverinfo='skip'
                ))

# Load and process material flows
colors = COLOR_LIST.copy()
df = pd.read_csv(r'./out_files/results.out')

# Parse string representations of tuples to actual tuples
df['from_node'] = df['from_node'].apply(ast.literal_eval)
df['to_node'] = df['to_node'].apply(ast.literal_eval)

# Load path throughput from settings
with open(r'./out_files/settings.json', 'r') as f:
    settings = json.load(f)
    path_throughput = settings.get('path_throughput', {})

material_flows = {}
for material in df['material'].unique():
    material_flows[material] = {}
    for _, row in df.iterrows():
        if row['material'] == material and row['flow'] > 0:
            key = (row['from_node'], row['to_node'])
            material_flows[material][key] = row['flow']

# Build special node info for tooltips (supply/demand)
special_node_info = {}

# Get all sources
for source_name, source_pos in sources.items():
    source_pos_tuple = tuple(source_pos)
    special_node_info[source_pos_tuple] = {
        'name': source_name,
        'type': 'Source',
        'supplies': {}
    }
    # Get supply info from settings
    for material in material_flows.keys():
        for (a, b), flow in material_flows[material].items():
            if a == source_pos_tuple:
                if material not in special_node_info[source_pos_tuple]['supplies']:
                    special_node_info[source_pos_tuple]['supplies'][material] = 0
                special_node_info[source_pos_tuple]['supplies'][material] += flow

# Get all factories
for factory_name, factory_pos in factories.items():
    factory_pos_tuple = tuple(factory_pos)
    special_node_info[factory_pos_tuple] = {
        'name': factory_name,
        'type': 'Factory',
        'consumes': {},
        'produces': {}
    }
    # Get supply/demand info
    for material in material_flows.keys():
        for (a, b), flow in material_flows[material].items():
            if a == factory_pos_tuple:
                if material not in special_node_info[factory_pos_tuple]['produces']:
                    special_node_info[factory_pos_tuple]['produces'][material] = 0
                special_node_info[factory_pos_tuple]['produces'][material] += flow
            elif b == factory_pos_tuple:
                if material not in special_node_info[factory_pos_tuple]['consumes']:
                    special_node_info[factory_pos_tuple]['consumes'][material] = 0
                special_node_info[factory_pos_tuple]['consumes'][material] += flow

# Get all sinkholes
for sinkhole_name, sinkhole_pos in sinkholes.items():
    sinkhole_pos_tuple = tuple(sinkhole_pos)
    special_node_info[sinkhole_pos_tuple] = {
        'name': sinkhole_name,
        'type': 'Sinkhole',
        'consumes': {}
    }
    # Get consumption info
    for material in material_flows.keys():
        for (a, b), flow in material_flows[material].items():
            if b == sinkhole_pos_tuple:
                if material not in special_node_info[sinkhole_pos_tuple]['consumes']:
                    special_node_info[sinkhole_pos_tuple]['consumes'][material] = 0
                special_node_info[sinkhole_pos_tuple]['consumes'][material] += flow

# Draw arrows and track material colors
material_colors = {}
annotation_counts = {}

# Group flows by connection to handle bidirectional flows
connection_flows = {}  # (from, to, material) -> direction

for material, paths in material_flows.items():
    for (a, b), flow in paths.items():
        # Track flows with their actual direction
        conn_key = (a, b)
        if conn_key not in connection_flows:
            connection_flows[conn_key] = []
        connection_flows[conn_key].append((material, a, b))

# Pre-calculate offsets for all connections to ensure consistency
connection_offsets = {}  # (from, to, material) -> offset

for conn_key, flows in connection_flows.items():
    num_flows = len(flows)
    for idx, (material, a, b) in enumerate(flows):
        # Distribute offsets evenly around the centerline
        if num_flows == 1:
            offset = 0
        else:
            # Spread them evenly: -0.15, -0.05, 0.05, 0.15 for 4 flows, etc.
            offset = (idx - (num_flows - 1) / 2) * 0.1
        
        connection_offsets[(a, b, material)] = offset

# Draw arrows for each material
for material, paths in material_flows.items():
    color = random.choice(colors)
    colors.remove(color)
    material_colors[material] = color
    annotation_counts[material] = len(paths)
    
    x_starts, y_starts, x_ends, y_ends, offsets = [], [], [], [], []
    
    for (a, b) in paths.keys():
        x_starts.append(a[0])
        y_starts.append(a[1])
        x_ends.append(b[0])
        y_ends.append(b[1])
        
        # Get the pre-calculated offset for this specific flow
        offset = connection_offsets[(a, b, material)]
        offsets.append(offset)
    
    for xs, ys, xe, ye, offset in zip(x_starts, y_starts, x_ends, y_ends, offsets):
        draw_arrows(fig, [xs], [ys], [xe], [ye], material=material, color=color, offset=offset)

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

# Create Dash app for interactivity
app = dash.Dash(__name__)

# Store for selected nodes
app.layout = html.Div([
    dcc.Store(id='selected-nodes-store', data={'start': None, 'end': None}),
    html.Div([
        dcc.Graph(
            id='network-graph',
            figure=fig,
            style={'width': '80%', 'display': 'inline-block', 'verticalAlign': 'top'},
            config={'displayModeBar': False}
        ),
        html.Div([
            html.Div(
                id='tooltip-panel',
                style={
                    'width': '100%',
                    'padding': '15px',
                    'backgroundColor': '#222',
                    'color': 'white',
                    'borderRadius': '5px',
                    'fontFamily': 'monospace',
                    'fontSize': '12px',
                    'marginBottom': '10px'
                }
            ),
            html.Button(
                'Reset Selection',
                id='reset-button',
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '8px',
                    'backgroundColor': '#444',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '3px',
                    'cursor': 'pointer',
                    'fontSize': '11px'
                }
            )
        ], style={
            'width': '18%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'marginLeft': '2%',
            'maxHeight': '90vh',
            'overflowY': 'auto'
        })
    ], style={'display': 'flex'}),
    html.Div(
        id='instructions',
        style={
            'padding': '10px',
            'backgroundColor': '#111',
            'color': '#888',
            'fontSize': '11px',
            'marginTop': '10px'
        }
    )
])

@app.callback(
    Output('tooltip-panel', 'children'),
    Input('selected-nodes-store', 'data')
)
def update_tooltip(stored_data):
    """Update tooltip with connection flow information"""
    try:
        start_node = stored_data.get('start')
        end_node = stored_data.get('end')
        
        if start_node is None or end_node is None:
            return html.Div([
                html.H3("📍 Flow Info", style={'color': '#4CAF50', 'marginTop': 0}),
                html.P("Click a node to select start", style={'fontSize': '12px'}),
                html.P("Then click another node to select end", style={'fontSize': '12px'}),
                html.Hr(),
                html.P(f"Total materials: {len(material_flows)}", style={'fontSize': '11px', 'color': '#999'})
            ])
        
        # Ensure nodes are tuples (not lists)
        if isinstance(start_node, (list, tuple)):
            start_node = tuple(start_node)
        if isinstance(end_node, (list, tuple)):
            end_node = tuple(end_node)
        
        a, b = start_node, end_node
        conn_str = f"({a[0]},{a[1]}) → ({b[0]},{b[1]})"
        rev_str = f"({b[0]},{b[1]}) → ({a[0]},{a[1]})"
        
        # Get max throughput for this connection
        # Try both directions
        path_key_forward = f"({a[0]}, {a[1]}, {b[0]}, {b[1]})"
        path_key_backward = f"({b[0]}, {b[1]}, {a[0]}, {a[1]})"
        max_throughput_forward = path_throughput.get(path_key_forward, 0)
        max_throughput_backward = path_throughput.get(path_key_backward, 0)
        
        children = [
            html.H3("📍 Connection", style={'color': '#4CAF50', 'marginTop': 0}),
            html.Div([
                html.Div(conn_str, style={'color': '#90CAF9'}),
                html.Div(rev_str, style={'color': '#EF5350'})
            ], style={'marginBottom': '15px'}),
            html.Hr()
        ]
        
        # Add max throughput info
        children.append(html.Div(
            "⚡ Max Throughput", 
            style={'color': '#FFD700', 'fontWeight': 'bold', 'marginBottom': '8px'}
        ))
        children.append(html.Div(
            f"  {conn_str}: {max_throughput_forward}",
            style={'paddingLeft': '10px', 'color': '#90CAF9', 'marginBottom': '4px'}
        ))
        children.append(html.Div(
            f"  {rev_str}: {max_throughput_backward}",
            style={'paddingLeft': '10px', 'color': '#EF5350', 'marginBottom': '12px'}
        ))
        children.append(html.Hr())
        
        # Add flows in forward direction
        children.append(html.Div(f"→ {conn_str}", style={'color': '#90CAF9', 'fontWeight': 'bold', 'marginBottom': '8px'}))
        forward_flows = []
        for material, paths in material_flows.items():
            if (a, b) in paths:
                flow_val = paths[(a, b)]
                forward_flows.append(
                    html.Div(
                        f"  {material}: {flow_val:.2f}",
                        style={'paddingLeft': '10px', 'color': material_colors.get(material, 'white')}
                    )
                )
        if forward_flows:
            children.extend(forward_flows)
        else:
            children.append(html.Div("  (no flows)", style={'paddingLeft': '10px', 'color': '#999'}))
        
        children.append(html.Br())
        # Add flows in reverse direction
        children.append(html.Div(f"← {rev_str}", style={'color': '#EF5350', 'fontWeight': 'bold', 'marginBottom': '8px'}))
        reverse_flows = []
        for material, paths in material_flows.items():
            if (b, a) in paths:
                flow_val = paths[(b, a)]
                reverse_flows.append(
                    html.Div(
                        f"  {material}: {flow_val:.2f}",
                        style={'paddingLeft': '10px', 'color': material_colors.get(material, 'white')}
                    )
                )
        if reverse_flows:
            children.extend(reverse_flows)
        else:
            children.append(html.Div("  (no flows)", style={'paddingLeft': '10px', 'color': '#999'}))
        
        return html.Div(children)
    
    except Exception as e:
        return html.Div(f"Error: {str(e)}")

@app.callback(
    [Output('selected-nodes-store', 'data'),
     Output('instructions', 'children')],
    [Input('network-graph', 'clickData'),
     Input('reset-button', 'n_clicks')],
    State('selected-nodes-store', 'data'),
    prevent_initial_call=False
)
def handle_all_interactions(clickData, reset_clicks, stored_data):
    """Handle all graph interactions in one callback"""
    # Determine which input triggered this callback
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # If reset button was clicked
    if trigger_id == 'reset-button':
        return {'start': None, 'end': None}, "Click to select start node"
    
    # If it's a graph click
    if not clickData:
        return dash.no_update, dash.no_update
    
    try:
        points = clickData.get('points', [])
        if not points:
            return stored_data, "Click to select start node"
        
        point = points[0]
        
        # Extract node coordinates - handle both int and list types
        x_coord = point.get('x')
        y_coord = point.get('y')
        
        # Convert to scalar if they're lists
        if isinstance(x_coord, (list, tuple)):
            x_coord = float(x_coord[0]) if x_coord else 0
        if isinstance(y_coord, (list, tuple)):
            y_coord = float(y_coord[0]) if y_coord else 0
        
        x_coord = int(round(float(x_coord)))
        y_coord = int(round(float(y_coord)))
        clicked_node = (x_coord, y_coord)
        
        # First click = start, second click = end
        if stored_data['start'] is None:
            new_data = {'start': clicked_node, 'end': None}
            instruction = f"Start: {clicked_node}. Click another node for end."
        else:
            new_data = {'start': stored_data['start'], 'end': clicked_node}
            instruction = f"Connection: {stored_data['start']} → {clicked_node}"
        
        return new_data, instruction
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return stored_data, f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

