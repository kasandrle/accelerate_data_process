import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
import nistchempy as nist
import numpy as np

import plotly.colors as pc

def get_distinct_colors(n):
    colors = pc.qualitative.Dark24  # 24 distinct colors
    if n <= len(colors):
        return colors[:n]
    else:
        # Repeat palette if not enough colors (or use another strategy)
        return [colors[i % len(colors)] for i in range(n)]


def parse_jcamp_and_return_spectrum(jcamp_str):
    peak_table_match = re.search(r"##PEAK TABLE=\(XY..XY\)(.*?)##END=", jcamp_str, re.DOTALL)
    if peak_table_match:
        peak_data = peak_table_match.group(1).strip()
        pairs = re.findall(r"(\d+),(\d+)", peak_data)
        mz = [int(m) for m, _ in pairs]
        intensity = [int(i) for _, i in pairs]
        return mz, intensity
    return [], []

# Load and parse data
path = './dummydata/outgassing_data/'
onlyfiles_outgassing = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('_outgassing_data_mean_std.txt')]
onlyfiles_TEY = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('_TEY_normalized.txt')]

#df_nist = nist.get_all_data()
#df_nist = df_nist.dropna(subset=['Mass spectrum (electron ionization)'])
df_nist = pd.read_csv('filtered_names.csv')


colors = get_distinct_colors(len(df_nist['name']))

def load_data(files, suffix):
    data = []
    for file in files:
        label = file.replace(suffix, '')
        df = pd.read_csv(join(path, file), sep='\t', skiprows=1)
        df['name'] = label
        data.append(df)
    return pd.concat(data, ignore_index=True)

df_outgassing = load_data(onlyfiles_outgassing, '_outgassing_data_mean_std.txt')
df_TEY = load_data(onlyfiles_TEY, '_TEY_normalized.txt')

# Start Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Outgassing Spectrum Viewer"),
    dcc.Dropdown(
        id='sample-dropdown',
        options=[{'label': name, 'value': name} for name in sorted(df_outgassing['name'].unique())],
        value=[df_outgassing['name'].unique()[0]],  # Now a list
        multi=True
    ),
    dcc.Dropdown(
        id='compound-dropdown',
        options=[{'label': name, 'value': name} for name in sorted(df_nist['name'].unique())],  # You can dynamically populate this or leave it empty for manual entry
        value=[],
        placeholder='Enter compound names...',
        multi=True,
        style={'marginBottom': '20px'}
    ),
    #dcc.Input(
    #    id='nist-scaling-factor',
    #    type='number',
    #    value=1.0,
    #   min=0.0,
    #    step=0.1,
    #    placeholder='Scale NIST intensity...',
    #    debounce=True,
    #    style={'marginBottom': '20px'}
    #),
    html.Div(id='scaling-inputs-container'),
    dcc.Store(id='compound-scaling-map', data={}),
    dcc.Graph(id='outgassing-plot'),
    html.H2("TEY-Shutter Data"),
    dcc.Dropdown(
        id='sample-dropdown_TEY',
        options=[{'label': name, 'value': name} for name in sorted(df_TEY['name'].unique())],
        value=[df_TEY['name'].unique()[0]],
        multi=True
    ),
    dcc.Graph(id='tey-plot')  # New graph
])

@app.callback(
    Output('scaling-inputs-container', 'children'),
    Input('compound-dropdown', 'value')
)
def display_scaling_inputs(compound_queries):
    if not compound_queries:
        return []

    return [
        html.Div([
            html.Label(f"Scale for {compound.capitalize()} (e.g. 1e-2)"),
            dcc.Input(
                id={'type': 'scale-input', 'index': compound},
                type='number',
                value=1,
                step=0.01,
                placeholder='Enter scaling (e.g. 1e-2)',
                style={'marginBottom': '10px'}
            )

        ]) for compound in compound_queries
    ]

@app.callback(
    Output('outgassing-plot', 'figure'),
    Input('sample-dropdown', 'value'),
    Input('compound-dropdown', 'value'),
    Input({'type': 'scale-input', 'index': dash.ALL}, 'value')
)
def update_figure(selected_samples, compound_queries, scaling_factors):

    fig = go.Figure()

    # Existing sample plotting logic
    if selected_samples:
        for sample in selected_samples:
            sub_df = df_outgassing[df_outgassing['name'] == sample]
            avg = sub_df['Avg Values (Torr)'].clip(lower=0)
            std = sub_df['Std Values (Torr)']
            std = std.where(std <= avg, 0)

            #avg = avg/np.max(avg)
            #std =std/np.max(avg)

            fig.add_trace(go.Bar(
                x=list(range(1, len(avg) + 1)),
                y=avg,
                error_y=dict(type='data', array=std, visible=True),
                name=sample,
                opacity=0.6
            ))

    # Plot NIST spectra for each compound
    if compound_queries:

        for i, compound_query in enumerate(compound_queries):
            try:
                s = nist.run_search(compound_query, 'name')
                X = s.compounds[0]
                X.get_ms_spectra()
                jcamp_str = X.ms_specs[0].jdx_text
                mz, intensity = parse_jcamp_and_return_spectrum(jcamp_str)

                intensity = intensity/np.max(intensity)

                factor = scaling_factors[i] if i < len(scaling_factors) else 1.0
                scaled_intensity = [v * factor for v in intensity]

                fig.add_trace(go.Bar(
                    x=mz,
                    y=scaled_intensity,
                    name=f"{compound_query.capitalize()} (×{factor})",
                    marker=dict(
                        color='rgba(0,0,0,0)',
                        line=dict(color=colors[i], width=1.5)
                    ),
                    yaxis='y2',
                    width=0.4,
                    opacity=1.0
                ))

            except Exception as e:
                print(f"Failed to fetch NIST data for '{compound_query}':", e)


    # Layout stays the same
    fig.update_layout(
        barmode='group',
        yaxis=dict(title="Ion signal (Torr)", 
                   showgrid=False,
                   #range=[0, 1.05]
        ),
        yaxis2=dict(
            title="Relative Intensity (NIST)",
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, 1.05]
        )
    )

    return fig

@app.callback(
    Output('tey-plot', 'figure'),
    Input('sample-dropdown_TEY', 'value')
)
def update_tey_plot(selected_samples):
    fig = go.Figure()

    if not selected_samples:
        return fig

    for sample in selected_samples:
        sub_df = df_TEY[df_TEY['name'] == sample]
        fig.add_trace(go.Scatter(
            x=sub_df['Time(s)'],
            y=sub_df['Normalized_TEY'],
            mode='lines',
            name=sample
        ))

    fig.update_layout(
        title="TEY-Shutter Data",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (A)",
        template="plotly_white"
    )
    return fig

# Run app
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True, port=8081)