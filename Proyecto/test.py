import Definitions
import folium
import folium.plugins as fp
import os.path as osp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_folium import st_folium

from Parameters import Parameters
from back.DataController import DataController

# Variables
controller = DataController()

#temp_df = controller.query_temp_station_values(station_sel, initial_date_sel, ending_date_sel)
#temp_df = controller.query_mint_values()
#print(temp_df)
hum_df = controller.query_data()
print(hum_df)

