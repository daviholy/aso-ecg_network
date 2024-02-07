from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_signal_with_labels_and_predictions(signal, pred, target=None, save_path=None, name=None):
    # Function to add traces for either prediction or target
    def add_traces_to_subplot(labels, row, subplot_title):
        start_idx = 0
        for idx in range(1, len(labels)):
            if labels[idx] != labels[idx - 1]:
                fig.add_trace(go.Scatter(
                    x=np.arange(start_idx, idx), 
                    y=signal[start_idx:idx], 
                    mode='lines+markers',
                    line=dict(color=colors[labels[idx-1].item()]),
                    showlegend=False), row=row, col=1)
                start_idx = idx
        fig.add_trace(go.Scatter(
            x=np.arange(start_idx, len(labels)), 
            y=signal[start_idx:len(labels)], 
            mode='lines+markers',
            line=dict(color=colors[labels[-1].item()]),
            showlegend=False), row=row, col=1)
        fig.update_layout(title=subplot_title)

    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'violet'}
    
    # Determine the number of rows based on whether target is provided
    num_rows = 1 if target is None else 2
    
    # Create subplot(s)
    titles = ['Prediction'] if target is None else ['Prediction', 'Target']
    fig = make_subplots(rows=num_rows, cols=1, subplot_titles=titles)

    # Add signal line to the prediction subplot always, and to the target subplot if it exists
    fig.add_trace(go.Scatter(
        x=np.arange(len(signal)),
        y=signal,
        mode='lines',
        line=dict(color='black'),
        showlegend=False), row=1, col=1)
    
    if target is not None:
        fig.add_trace(go.Scatter(
            x=np.arange(len(signal)),
            y=signal,
            mode='lines',
            line=dict(color='black'),
            showlegend=False), row=2, col=1)

    # Add traces for prediction
    add_traces_to_subplot(pred, 1, 'Prediction')
    
    # Add traces for target if it exists
    if target is not None:
        add_traces_to_subplot(target, 2, 'Target')  # Adjusted to use only one column

    fig.update_layout(title='Signal Values Colored by Class',
                      xaxis_title='X Axis',
                      yaxis_title='Signal Value',
                      legend_title='Legend')

    fig.show()

    if save_path:
        if name:
            fig.write_html(f"{save_path}/{name}.html")
        else:
            fig.write_html(f"{save_path}/no_name_provided.html")

def plot_jaccard_barplot(jaccards_mean: dict, jaccards_std: dict,save_path: Path | None = None):
    
    # Create a bar chart
    fig = go.Figure(data=[go.Bar(
        x=list(jaccards_mean.keys()), 
        y=list(jaccards_mean.values()),
        error_y=dict(
            type='data',  # or 'percent' for percentage-based errors
            array=[val for val in jaccards_std.values()],
            visible=True
        ),
        marker_color=['magenta', 'green', 'cyan', 'orange'],
        )])

    # Updating layout for title and y-axis label
    fig.update_layout(
        title='Test dataset - jaccard indexes',
        yaxis=dict(
            title='Jaccard index'
        )
    )
    # Show the plot
    fig.show()

    if save_path:
        fig.write_html(save_path / "jaccards.html")
