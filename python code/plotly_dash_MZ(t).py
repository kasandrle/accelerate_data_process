import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load all .txt files
folder_path = '/home/kas/Projects/accelerate_data_process/data example small/Analysis_results-ascii/MS(t)_averaged/'
dfs = []
file_options = []

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, sep='\t')
        df['source_file'] = filename
        dfs.append(df)
        file_options.append(filename)

df_all = pd.concat(dfs, ignore_index=True)

# Dash app setup
app = dash.Dash(__name__)
app.title = "MZ Colormap Viewer"

app.layout = html.Div([
    html.H1("MZ Colormap Viewer", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select File:"),
        dcc.Dropdown(
            id='file-dropdown',
            options=[{'label': f, 'value': f} for f in file_options],
            value=file_options[0]
        ),
    ], style={'width': '40%', 'display': 'inline-block'}),

    html.Div(id='mz-slider-container', style={'width': '90%', 'padding': '20px'}),

    html.Div([
        html.Label("Top Line Plot: Select MZ index(es)"),
        dcc.Dropdown(id='mz-line-dropdown', multi=True)
    ], style={'width': '60%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Side Line Plot: Select Time(s)"),
        dcc.Dropdown(id='time-line-dropdown', multi=True)
    ], style={'width': '60%', 'display': 'inline-block'}),

    dcc.Graph(id='colormap-plot')
])

@app.callback(
    Output('mz-slider-container', 'children'),
    Output('mz-line-dropdown', 'options'),
    Output('mz-line-dropdown', 'value'),
    Output('time-line-dropdown', 'options'),
    Output('time-line-dropdown', 'value'),
    Input('file-dropdown', 'value')
)
def update_controls(selected_file):
    df = df_all[df_all['source_file'] == selected_file]
    mz_cols = [col for col in df.columns if col.startswith("MZ") and "(Torr)" in col]
    mz_indices = sorted([int(col[2:-6]) for col in mz_cols])
    time_values = sorted(df['Time(s)'].unique())

    mz_slider = html.Div([
        html.Label("MZ Index Range:"),
        dcc.RangeSlider(
            id='mz-slider',
            min=min(mz_indices),
            max=max(mz_indices),
            step=1,
            value=[min(mz_indices), max(mz_indices)],
            marks={i: str(i) for i in range(min(mz_indices), max(mz_indices)+1, 10)}
        )
    ])

    mz_dropdown = [{'label': f"MZ{i}", 'value': i} for i in mz_indices]
    time_dropdown = [{'label': f"{t:.2f}", 'value': t} for t in time_values]

    return mz_slider, mz_dropdown, [mz_indices[0]], time_dropdown, [time_values[0]]

@app.callback(
    Output('colormap-plot', 'figure'),
    Input('file-dropdown', 'value'),
    Input('mz-slider', 'value'),
    Input('mz-line-dropdown', 'value'),
    Input('time-line-dropdown', 'value')
)
def update_plot(selected_file, mz_range, mz_lines, time_lines):
    df = df_all[df_all['source_file'] == selected_file]
    time = df['Time(s)'].values

    # Filter MZ columns
    mz_cols = [col for col in df.columns if col.startswith("MZ") and "(Torr)" in col]
    mz_indices = sorted([int(col[2:-6]) for col in mz_cols if mz_range[0] <= int(col[2:-6]) <= mz_range[1]])
    mz_subset = [f"MZ{i}(Torr)" for i in mz_indices]
    matrix = np.array([df[col].values for col in mz_subset])

    # Create subplot layout: 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.75, 0.25],
        row_heights=[0.3, 0.7],
        specs=[[{"colspan": 2}, None], [{"type": "heatmap"}, {"type": "xy"}]],
        subplot_titles=["Top Line Plot", f"Colormap — {selected_file}", "Side Line Plot"]
    )

    # Top line plot: intensity at selected MZ indices
    for mz_line in mz_lines:
        mz_col = f"MZ{mz_line}(Torr)"
        if mz_col in df.columns:
            fig.add_trace(go.Scatter(
                x=time,
                y=df[mz_col].values,
                mode='lines',
                name=f"MZ {mz_line}"
            ), row=1, col=1)

    # Colormap
    fig.add_trace(go.Heatmap(
        z=np.abs(matrix),
        x=time,
        y=mz_indices,
        colorscale='Viridis',
        zmin=1e-12,
        zmax=np.max(np.abs(matrix)),
        colorbar=dict(title="Intensity")
    ), row=2, col=1)

    # Vertical lines on colormap
    fig.add_vline(x=50, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_vline(x=300, line_dash="dash", line_color="red", row=2, col=1)

    # Side line plot: intensity at selected time(s)
    for t in time_lines:
        time_idx = np.argmin(np.abs(time - t))
        side_x = [df[col].values[time_idx] for col in mz_subset]
        side_y = mz_indices
        fig.add_trace(go.Scatter(
            x=side_x,
            y=side_y,
            mode='lines',
            name=f"Time {t:.2f}"
        ), row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis1=dict(title="Time (s)"),
        yaxis1=dict(title="Intensity"),
        xaxis2=dict(title="Time (s)"),
        yaxis2=dict(title="MZ index (Torr)"),
        xaxis3=dict(title="Intensity"),
        yaxis3=dict(title="MZ index (Torr)")
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)