# creates spike by trials, firing rates data for a specific interval

import numpy as np
import pandas as pd
from spike_tools import general as spike_general

import utils.visualization_utils as visualization_utils

import plotly.express as px


import json
import os

PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 100
EVENT = "FeedbackOnset"

name_to_color = {
    "Occipital_Lobe (Occipital)": '#1F77B4', 
    "Parietal_Lobe (Parietal" : '#FF7F0E', 
    "diencephalon (di)": '#2CA02C', 
    "telencephalon (tel)": '#D62728', 
    "Temporal_Lobe (Temporal)": '#9467BD',
    "Frontal_Lobe (Frontal)": '#8C564B',
    "metencephalon (met)": '#E377C2',
    None: '#7F7F7F',
}



def get_neuron_positions(row):
    session = row.session_name
    # For the cases like 201807250001
    sess_day = session[:8]
    info_path = f"/data/rawdata/sub-SA/sess-{sess_day}/session_info/sub-SA_sess-{sess_day}_sessioninfo.json"
    with open(info_path, 'r') as f:
        data = json.load(f)
    locs = data['electrode_info']
    locs_df = pd.DataFrame.from_dict(locs)
    electrode_pos_not_nan = locs_df[~locs_df['x'].isna() & ~locs_df['y'].isna() & ~locs_df['z'].isna()]
    units = spike_general.list_session_units(None, "SA", session, species_dir="/data")
    unit_pos = pd.merge(units, electrode_pos_not_nan, left_on="Channel", right_on="electrode_id", how="left")
    unit_pos = unit_pos.astype({"UnitID": int})
    locs_df["session"] = session
    return locs_df

def plot_positions(sess_name, positions):
    fig1 = px.scatter_3d(
        positions, x="x", y="y", z="z", 
        color="structure_level1", 
        labels={"structure_level1": "Unit Region"},
        color_discrete_map=name_to_color)
    fig = visualization_utils.plotly_add_glass_brain(None, fig1, "SA", areas='all', show_axis=False)
    fig.update_layout(
        autosize=True,
        width=750,
        height=700,
    )
    camera = dict(
        eye=dict(x=1.3, y=1, z=0.1)
    )
    temp_grid = dict(
        showgrid=False, 
        zeroline=False, 
        showticklabels=False
    )
    fig.update_layout(
        autosize=False,
        width=700,
        height=700,
        xaxis=temp_grid,
        yaxis=temp_grid,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        scene_camera=camera
    )
    fig.write_html(f"/data/patrick_scratch/multi_sess/{sess_name}/figures/unit_positions.html")


def process_session(row):
    # plot positions
    print(f"Processing {row.session_name}")
    positions = get_neuron_positions(row)
    plot_positions(row.session_name, positions)


def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
    valid_sess.apply(process_session, axis=1)

if __name__ == "__main__":
    main()