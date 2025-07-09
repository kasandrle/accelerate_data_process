import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
import nistchempy as nist

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
    dcc.Input(
        id='compound-input',
        type='text',
        value='anthracene',
        placeholder='Enter compound name...',
        debounce=True,  # Optional: update on enter/blur rather than every keystroke
        style={'marginBottom': '20px'}
    ),
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
    Output('outgassing-plot', 'figure'),
    Input('sample-dropdown', 'value'),
    Input('compound-input', 'value')  # New input
)
def update_figure(selected_samples, compound_query):
    fig = go.Figure()

    if selected_samples:
        for sample in selected_samples:
            sub_df = df_outgassing[df_outgassing['name'] == sample]
            avg = sub_df['Avg Values (Torr)'].clip(lower=0)
            std = sub_df['Std Values (Torr)']
            std = std.where(std <= avg, 0)

            fig.add_trace(go.Bar(
                x=list(range(1, len(avg) + 1)),
                y=avg,
                error_y=dict(type='data', array=std, visible=True),
                name=sample,
                opacity=0.6
            ))

    # Try fetching the NIST spectrum with dynamic query
    try:
        s = nist.run_search(compound_query,'name')
        X = s.compounds[0]
        X.get_ms_spectra()
        jcamp_str = X.ms_specs[0].jdx_text
        mz, intensity = parse_jcamp_and_return_spectrum(jcamp_str)#

        fig.add_trace(go.Bar(
            x=mz,
            y=intensity,
            name=f"NIST Ref: {compound_query.capitalize()}",
            marker=dict(
                color='rgba(0,0,0,0)',           # transparent fill
                line=dict(color='indigo', width=1.5)  # visible outline
            ),
            yaxis='y2',
            width=0.4,
            opacity=1.0
        ))

    except Exception as e:
        print(f"Failed to fetch NIST data for '{compound_query}':", e)

    fig.update_layout(
        barmode='group',
        yaxis=dict(
            title="Ion signal (Torr)",
            showgrid=False
        ),
        yaxis2=dict(
            title="Relative Intensity (NIST)",
            overlaying='y',
            side='right',
            showgrid=False
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