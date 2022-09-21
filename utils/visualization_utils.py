import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from stl import mesh  # pip install numpy-stl
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def visualize_accuracy_across_time_bins(
    accuracies, 
    pre_interval, 
    post_interval, 
    interval_size, 
    ax,
    label=None,
    right_align=False,
    color=None
):
    """Plots accuracies across time bins as a shaded line plot

    Args:
        accuracies: num_bins x num_runs np.array
        pre_interval: int, in miliseconds
        post_interval: int, in miliseconds
        interval_size: int, in miliseconds
    """
    means = np.nanmean(accuracies, axis=1)
    stds = np.nanstd(accuracies, axis=1)
    x = np.arange(-pre_interval, post_interval, interval_size)
    if right_align:
        # every x timepoint indicates the right of the bin
        x = x + interval_size
    mean_line, = ax.plot(x, means, label=label, linewidth=5)
    std_line = ax.fill_between(x, means - stds, means + stds, alpha=0.5)
    if color:
        mean_line.set_color(color)
        std_line.set_color(color)


def visualize_accuracy_bars(accuracies, labels, ax):
    sns.barplot(data=accuracies, capsize=.1, ci="sd", ax=ax)
    sns.swarmplot(data=accuracies, color="0", alpha=.35, ax=ax)
    ax.set_xticklabels(labels)



def plot_hist_of_selections(feature_selections, feature_dim, ax):
    dist = feature_selections[feature_dim]
    ax.hist(dist)


# def plot_values_by_trial(trial_numbers, )
#     """
#     Plots values by trial as a color grid, 
#     """

def plotly_add_glass_brain(fs, fig1, subject, areas=['brain'], show_axis=False):
    '''
    Adds a glass brain to a plotly figure
    Code for rendering stl file is from here:
    https://chart-studio.plotly.com/~empet/15276/converting-a-stl-mesh-to-plotly-gomes/#/
    
    Parameters
    ----------------
    fig1 : plotly figure
    subject : subject of whom's glass brain to plot
    areas : list of strings indicating areas to print. List must limited to
        'brain', 'fef', 'dlpfc', 'mpfc', 'hippocampus'
        Use areas='all' if you want to plot all areas
    show_axis : whether or not to include axis
    '''
    def stl2mesh3d(stl_mesh):
        # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles 
        # (i.e. three 3d points) 
        # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
        p, q, r = stl_mesh.vectors.shape #(p, 3, 3)
        # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
        # extract unique vertices from all mesh triangles
        vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
        I = np.take(ixr, [3*k for k in range(p)])
        J = np.take(ixr, [3*k+1 for k in range(p)])
        K = np.take(ixr, [3*k+2 for k in range(p)])
        return vertices, I, J, K
    
    fig_data = fig1.data
    
    if areas=='all':
        areas = ['brain', 'fef', 'dlpfc', 'mpfc', 'hippocampus']
    
    for area in areas:
        if area=='brain':
            filename = 'glass_brain.stl'
            colorscale= [[0, 'whitesmoke'], [1, 'whitesmoke']]
        elif area=='fef':
            filename = 'FEF_interaural.stl'
            colorscale= [[0, 'darkblue'], [1, 'darkblue']]
        elif area=='dlpfc':
            filename = 'dlPFC_interaural.stl'
            colorscale= [[0, 'lightgreen'], [1, 'lightgreen']]
        elif area=='mpfc':
            filename = 'mPFC_interaural.stl'
            colorscale= [[0, 'darkgreen'], [1, 'darkgreen']]
        elif area=='hippocampus':
            filename = 'Hippocampal_interaural.stl'
            colorscale= [[0, 'coral'], [1, 'coral']]
            
        stl_file = 'nhp-lfp/wcst-preprocessed/rawdata/sub-'+subject+'/anatomy/'+filename
        
        if not fs.exists(stl_file):
            print(stl_file + ' does not exist, not including...')
            continue
    
        tmp = tempfile.NamedTemporaryFile()
        with open(tmp.name, mode='wb') as f:
            fs.get_file(stl_file, tmp.name)
            my_mesh = mesh.Mesh.from_file(tmp.name)
    
        vertices, I, J, K = stl2mesh3d(my_mesh)
        x, y, z = vertices.T
    
        mesh3D = go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, \
                           opacity=0.3, colorscale=colorscale, intensity=z, showscale=False, name=area)
    
        layout = go.Layout(
            scene_xaxis_visible=False, 
            scene_yaxis_visible=False,
            scene_zaxis_visible=False,
            showlegend=True
        )
    
        fig2 = go.Figure(
            data=[mesh3D], 
            layout=layout
        )
        fig_data = fig_data + fig2.data
        
    fig3 = go.Figure(data=fig_data, layout=fig1.layout)
    if not show_axis:
        fig3.update_layout(layout)
    
    return(fig3)