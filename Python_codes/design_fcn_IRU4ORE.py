# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:08:21 2026
Module name: [Design_indicators_functions]
Author: [Ander Martinez, Markel Penalba]
Affiliation: [Mondragon University]
Contact: [amartinezpe@mondragon.edu]
"""

'--> Dependencies'

# Python libraries
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LinearSegmentedColormap
import os
import math
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString, MultiPoint
from shapely.ops import polygonize, unary_union
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
'''
------------------------------------------------------------------------------ 
                            PLOTTING AND FONT SETTINGS
------------------------------------------------------------------------------
''' 

'--> Font settings'
# Example font dictionary
tick_font = {'family': 'Times New Roman', 'size': 26,'size_2':24}
# Apply font settings globally (optional)
rcParams['xtick.labelsize'] = tick_font['size']
rcParams['ytick.labelsize'] = tick_font['size']
rcParams['legend.fontsize'] = tick_font['size_2']
rcParams['font.family'] = tick_font['family']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'Times New Roman'

'--> Plotting settings'
 # Define the colors for plotting
rcParams['legend.fontsize'] = 20
cmap = cm.get_cmap("viridis")
color_range = np.linspace(0.2, 0.8, 3)  # Select colors between 20% and 80% of the colormap
colors = [cmap(value) for value in color_range]  # Extract specific colors


'''
------------------------------------------------------------------------------ 
                            POWER PRODUCTION RELATED FUNCTIONS
------------------------------------------------------------------------------
''' 
 

def read_ORE_spec(FOWT_name,WEC_name,root_dir):
    
    path_ORE = root_dir+"/Reference_ORE/"
    
    ORE_data = {}
    ORE_data = {"FOWT":{},
                
                "WEC":{}}   
    
    '--> Read FOWT specifications'    
    if FOWT_name == "IEA-15-240-MW-FOWT":
        # Load data
        with open(path_ORE+"FOWT_IEA_15MW_178.json", 'r') as f:
            data = json.loads(f.read())        
        turbine_data = pd.json_normalize(data, record_path=['power_curve_data'])   
        # Save specifications in .dic file
        ORE_data["FOWT"]["hub_height"] = 150
        ORE_data["FOWT"]["cut_in_speed"] = turbine_data['cut_in_speed_in_ms'][0]
        ORE_data["FOWT"]["rated_speed"] = turbine_data['rated_speed_in_ms'][0]
        ORE_data["FOWT"]["cut_out_speed"] = turbine_data['cut_out_speed_in_ms'][0]
        ORE_data["FOWT"]["nominal_power_in_kw"] = turbine_data['nominal_power_in_kw'][0]
        ORE_data["FOWT"]["polyn_coeff"] = turbine_data['polynomial_coefficients'][0]
        
    elif FOWT_name == "DTIU-10-MW":
        # Load data
        with open(path_ORE+"FOWT_DTU_10MW_178.json", 'r') as f:
            data = json.loads(f.read())
        turbine_data = pd.json_normalize(data, record_path=['power_curve_data'])  
        # Save specifications in .dic file
        ORE_data["FOWT"]["hub_height"] = 119
        ORE_data["FOWT"]["cut_in_speed"] = turbine_data['cut_in_speed_in_ms'][0]
        ORE_data["FOWT"]["rated_speed"] = turbine_data['rated_speed_in_ms'][0]
        ORE_data["FOWT"]["cut_out_speed"] = turbine_data['cut_out_speed_in_ms'][0]
        ORE_data["FOWT"]["nominal_power_in_kw"] = turbine_data['nominal_power_in_kw'][0]
        ORE_data["FOWT"]["polyn_coeff"] = turbine_data['polynomial_coefficients'][0]
        
        
        
    '--> Read WEC specifications'
    if WEC_name == "RM3": 
        ORE_data["WEC"]["rated_power_in_kw"] = 286        
        ORE_data["WEC"]["cut_in_Tp"] = 4.5
        ORE_data["WEC"]["cut_out_Tp"] = 17.5
        ORE_data["WEC"]["cut_in_Hs"] = 0.25
        ORE_data["WEC"]["cut_out_Hs"] = 9.75
        
        discretisation_Tp = 0.25
        discretisation_Hs = 0.25
        wave_peak_period_values = np.arange(ORE_data["WEC"]["cut_in_Tp"],ORE_data["WEC"]["cut_out_Tp"],discretisation_Tp)
        wave_peak_period_values[-1] = ORE_data["WEC"]["cut_out_Tp"]
        wave_height_values = np.arange(ORE_data["WEC"]["cut_in_Hs"],ORE_data["WEC"]["cut_out_Hs"],discretisation_Hs)
        wave_height_values[-1] = ORE_data["WEC"]["cut_out_Hs"]       
        
        # WEC power matrix 
        elec_pow_orig = pd.read_excel(path_ORE+'RM3_power_matrix.xlsx',sheet_name='Electrical_PM2')
        elec_pow_orig = elec_pow_orig.set_index(['Hs']) 
        Hs_pow = elec_pow_orig.index.values.astype(float)      # 0.5 m bins
        Tp_pow = elec_pow_orig.columns.values.astype(float)    # 1 s bins
        P_pow  = elec_pow_orig.values 
        
        # Interpolation of the Power Matrix
        interp_func = RegularGridInterpolator((Hs_pow, Tp_pow), P_pow, method='linear')

        # New refined grid
        Hs_new = np.arange(Hs_pow.min(), Hs_pow.max()+0.25, 0.25)
        Tp_new = np.arange(Tp_pow.min(), Tp_pow.max()+0.25, 0.25)
        
        Hs_mesh, Tp_mesh = np.meshgrid(Hs_new, Tp_new, indexing='ij')
        points_new = np.stack([Hs_mesh.ravel(), Tp_mesh.ravel()], axis=-1)
        
        # Interpolate
        elec_pow_int = interp_func(points_new).reshape(len(Hs_new), len(Tp_new))
        
        # New dataframe
        elec_pow = pd.DataFrame(elec_pow_int, index=Hs_new, columns=Tp_new)
        elec_pow.index.name = "Hs"
        
        ORE_data["WEC"]["power_matrix"] = elec_pow
        
    
    
    
    '--> Plot turbine-specific parameters'
    
    # FOWT power curve         
    wind_speed = np.arange(0,25,0.25)
    power = np.zeros_like(wind_speed)  
    partial_load = (wind_speed > ORE_data["FOWT"]["cut_in_speed"]) & (wind_speed < ORE_data["FOWT"]["rated_speed"])
    rated_load = (wind_speed >= ORE_data["FOWT"]["rated_speed"]) & (wind_speed < ORE_data["FOWT"]["cut_out_speed"])    
    power[rated_load] = ORE_data["FOWT"]["nominal_power_in_kw"]
    
    if len(ORE_data["FOWT"]["polyn_coeff"]) == 5: # DTIU-10-MW
        # Original polynomial with constant term
        u = wind_speed[partial_load]
        power[partial_load] = np.polyval(ORE_data["FOWT"]["polyn_coeff"], u)
    
    elif len(ORE_data["FOWT"]["polyn_coeff"]) == 4: # IEA-15-240-MW-FOWT
        # Shifted polynomial without constant term
        u = wind_speed[partial_load] - ORE_data["FOWT"]["cut_in_speed"]
        power[partial_load] = np.polyval(np.r_[ORE_data["FOWT"]["polyn_coeff"], 0.0], u)
        
    power = np.clip(power, 0, ORE_data["FOWT"]["nominal_power_in_kw"])
    
    fig1_FOWT_pc, ax1_FOWT_pc = plt.subplots(1, 1, figsize=(11, 8))
    fig1_FOWT_pc.subplots_adjust(right=0.9)
    ax1_FOWT_pc.plot(wind_speed, power,'o-', color=colors[0],markerfacecolor = colors[1], 
                     markeredgecolor = colors[0],linewidth = 2,label= FOWT_name+' power curve')
    

    ax1_FOWT_pc.axvline(ORE_data["FOWT"]["cut_in_speed"], color="grey", linestyle="--", alpha=0.5)
    ax1_FOWT_pc.axvline(ORE_data["FOWT"]["cut_out_speed"], color="grey", linestyle="--", alpha=0.5)
    ax1_FOWT_pc.axhline(ORE_data["FOWT"]["nominal_power_in_kw"], color="grey", linestyle="--", alpha=0.5)

    ax1_FOWT_pc.set_xlabel("Wind speed (m/s)",fontsize = 24)
    ax1_FOWT_pc.set_ylabel("Power output (kW)",fontsize = 24)    
    ax1_FOWT_pc.legend()
    
    # WEC power matrix
    
    cmap_WEC = LinearSegmentedColormap.from_list("viridis_truncated", plt.cm.viridis(np.linspace(0, 0.85, 256)))    
    fig_WEC_pm, ax_WEC_pm = plt.subplots(figsize=(10, 7))
    pcm = ax_WEC_pm.pcolormesh(Tp_mesh.T, Hs_mesh.T, elec_pow.T, shading="auto", cmap=cmap_WEC)
    # Isolines
    levels = np.linspace(np.nanmin(elec_pow), np.nanmax(elec_pow), 12)
    cs = ax_WEC_pm.contour(Tp_mesh.T, Hs_mesh.T, elec_pow.T, levels=levels, colors="white", linewidths=1.0)
    ax_WEC_pm.clabel(cs, inline=True, fontsize=14, fmt="%.0f")
    
    # Labels
    ax_WEC_pm.set_xlabel(r"$T_p$ [s]",fontsize =24, fontweight="bold")
    ax_WEC_pm.set_ylabel(r"$H_s$ [m]",fontsize =24, fontweight="bold")
    
    # Colour bar
    cbar = fig_WEC_pm.colorbar(pcm, ax=ax_WEC_pm)
    cbar.set_label("Electrical power [kW]",fontsize =24, fontweight="bold")
    
    plt.tight_layout()
    plt.show()
    
    # Save figures 
    path_fig = root_dir+"/Results/Figures/ORE_tech_spec/"
    os.makedirs(path_fig, exist_ok=True)
    fig1_FOWT_pc.savefig(path_fig+FOWT_name+"_power_curve.svg", format='svg')    
    fig_WEC_pm.savefig(path_fig+"RM3_power_matrix.svg", format='svg')
    
    return ORE_data  

def wind_speed_extrapolator(extrap_height,ref_height,wind_speeds):
    alpha_IEC = 0.14  
    extrap_wind = wind_speeds.to_numpy() * (extrap_height / ref_height)**alpha_IEC    
    return extrap_wind


def ORE_prod(met_data,location,root_dir):
    
       
    '--> Define ORE technical specifications '
    # Select the reference turbine
    FOWT_name = "DTIU-10-MW"#"IEA-15-240-MW-FOWT"                                                                  # or "DTIU-10-MW"
    WEC_name = "RM3"
    
    ORE_data = read_ORE_spec(FOWT_name,WEC_name,root_dir)
    databases = ["Obs","Sim","Calib"]
    databases_v2 = ["Buoy","ERA5", "CERA5"]
    var_pair = {"WEC": ["Hs", "Tp"], "FOWT": ["Hs", "Uw"]}
    ORE_AEP_df = pd.DataFrame(index=databases_v2, columns=var_pair.keys(), dtype=float)    
    ORE_prod_df = pd.DataFrame(columns=[ORE+"_"+dat  for ORE in var_pair.keys() for dat in databases_v2], dtype=float) 
    
    
    for i_dat, data in enumerate(databases):
        
        '--> Metocean data'
        Hs = met_data["Hs_"+data]
        Hs = np.asarray(Hs, dtype=float)
        Tp = met_data["Tp_"+data]
        Tp = np.asarray(Tp, dtype=float)
        if data == "Obs" or data=="Calib":
            ref_wind_speed = met_data["Uv10_"+data]
        else:
            ref_wind_speed = met_data["Uw_"+data]
        # Extrapolate wind speed to hugh height 
        ref_height = 10                                                                               # reference height at which the wind speed is measured
        U_hub = np.array(wind_speed_extrapolator(ORE_data["FOWT"]["hub_height"],ref_height,ref_wind_speed), dtype=np.float64)
        
        '--> FOWT power production'
        partial_load = (U_hub > ORE_data["FOWT"]["cut_in_speed"]) & (U_hub < ORE_data["FOWT"]["rated_speed"])
        rated_load = (U_hub >= ORE_data["FOWT"]["rated_speed"]) & (U_hub < ORE_data["FOWT"]["cut_out_speed"])
        pow_FOWT = np.zeros_like(U_hub)
        pow_FOWT[rated_load] = ORE_data["FOWT"]["nominal_power_in_kw"]
        
        if len(ORE_data["FOWT"]["polyn_coeff"]) == 5: # DTIU-10-MW
            # Original polynomial with constant term
            u = U_hub[partial_load]
            pow_FOWT[partial_load] = np.polyval(ORE_data["FOWT"]["polyn_coeff"], u)
        
        elif len(ORE_data["FOWT"]["polyn_coeff"]) == 4: # IEA-15-240-MW-FOWT
            # Shifted polynomial without constant term
            u = U_hub[partial_load] - ORE_data["FOWT"]["cut_in_speed"]
            pow_FOWT[partial_load] = np.polyval(np.r_[ORE_data["FOWT"]["polyn_coeff"], 0.0], u)
            
        pow_FOWT[pow_FOWT > ORE_data["FOWT"]["nominal_power_in_kw"]] = ORE_data["FOWT"]["nominal_power_in_kw"]      
        hours = np.arange(len(U_hub))
        date = ref_wind_speed.index
        datetime = pd.to_datetime(date)
        operational_horizon_in_years = (datetime[-1].year-datetime[0].year)+1
        FOWT_generated_energy_in_MWh = np.trapz(pow_FOWT, hours) / 1000
        FOWT_AEP_in_MWh = FOWT_generated_energy_in_MWh / operational_horizon_in_years
        capacity_factor_wind = (FOWT_AEP_in_MWh) / (10*8760) * 100
        
        ORE_AEP_df.loc[databases_v2[i_dat],"FOWT"] = FOWT_AEP_in_MWh
        ORE_prod_df["FOWT_"+databases_v2[i_dat]] = pow_FOWT
        
        
        '--> WEC power production'        
        # Operational mask (inside the WEC operating envelope)
        mask = (Hs >= ORE_data["WEC"]["cut_in_Hs"]) & (Hs <= ORE_data["WEC"]["cut_out_Hs"]) & (Tp >= ORE_data["WEC"]["cut_in_Tp"]) & (Tp <= ORE_data["WEC"]["cut_out_Tp"])

        PM_WEC = ORE_data["WEC"]["power_matrix"].to_numpy()
        Hs_grid = np.asarray(ORE_data["WEC"]["power_matrix"].index, dtype=float)
        Tp_grid = np.asarray(ORE_data["WEC"]["power_matrix"].columns, dtype=float)

        # Nearest neighbour indices (vectorised) for monotonic grids
        # For each sample, find insertion position, then choose closer of left/right
        j = np.searchsorted(Tp_grid, Tp, side="left")
        j = np.clip(j, 1, len(Tp_grid) - 1)
        j_left = j - 1
        j_right = j
        j = np.where(np.abs(Tp - Tp_grid[j_left]) <= np.abs(Tp - Tp_grid[j_right]), j_left, j_right)
        i = np.searchsorted(Hs_grid, Hs, side="left")
        i = np.clip(i, 1, len(Hs_grid) - 1)
        i_left = i - 1
        i_right = i
        i = np.where(np.abs(Hs - Hs_grid[i_left]) <= np.abs(Hs - Hs_grid[i_right]), i_left, i_right)
        # Vectorised extraction
        pow_WEC = np.zeros_like(Tp, dtype=float)
        pow_WEC[mask] = PM_WEC[i[mask], j[mask]]
        
        # Device Generated Energy: E = Integral(P(t)dt)
        hours = np.arange(len(Hs))
        generated_energy_in_MWh = np.trapz(pow_WEC, hours)/ 1000
        wave_AEP_in_MWh = generated_energy_in_MWh / operational_horizon_in_years
        capacity_factor_wave = (wave_AEP_in_MWh) / (ORE_data["WEC"]["rated_power_in_kw"]/1000*8760) * 100
        
        ORE_AEP_df.loc[databases_v2[i_dat],"WEC"] = wave_AEP_in_MWh
        ORE_prod_df["WEC_"+databases_v2[i_dat]] = pow_WEC
        
    return ORE_AEP_df, ORE_prod_df


def intervals(nums):
    """
    Calculate the intervals of consecutive sets.
    
    Args:
      nums: list of a set
          
    Return:
     list of tuples indicating intervals
    
    """
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def accessibility_hs_uw(wave_height_timeseries, wind_speed_timeseries, Op_lim, Rep_time, discretisation):
    """
    Calculates various metrics related to the accessibility of a FOWT.

    Arguments:
    - wave_height_timeseries: A list or array containing wave height values over a period of time.
    - wave_height_limit_in_meters: The threshold wave height value of the vessel in meters that determines accessibility.
    - wind_speed_timeseries: A list or array containing wind speed values over a period of time.
    - wind_speed_limit_in_meters: The threshold wind speed value of the vessel in meters per second that determines accessibility.   
    - required_weather_window_in_hours: The minimum duration of a weather window required for performing offshore operation in hours.
    - discretisation: The time-series discretisation.

    Returns:
    - approachability: The ratio of time periods where the wave height is below threshold over total time.
    - total_accessible_time: The total duration of time when the wave height is below threshold.
    - accessibility: The ratio of weather windows over total time.
    - total_weather_windows_number: The total number of weather windows found in the wave height timeseries.
    - mean_duration_of_a_ww: The average duration of a weather window in hours.
    - total_waiting_time: The total waiting time for accessing the FOWT farm based on weather windows.
    - waiting_times_intervals: A list of waiting time intervals between consecutive weather windows.
    - mean_time_to_wait: The average waiting time between weather windows in hours.
    """
    mean_duration_of_a_ww = np.zeros(len(Rep_time[:].iloc[0]))
    total_waiting_time = np.zeros(len(Rep_time[:].iloc[0]))
    waiting_times_intervals = np.zeros(len(Rep_time[:].iloc[0]))
    mean_time_to_wait = np.zeros(len(Rep_time[:].iloc[0]))
    total_weather_windows_number = np.zeros(len(Rep_time[:].iloc[0]))
    approachability = np.zeros(len(Rep_time[:].iloc[0]))
    total_accessible_time = np.zeros(len(Rep_time[:].iloc[0]))
    accessibility = np.zeros(len(Rep_time[:].iloc[0]))
    Vessel_type = ["CTV","FSV","HLV"]
    Repair_type = ["minor","medium","major"]
    waiting_time_dic = {}
    
    
    for tt in range(len(np.zeros(len(Rep_time[:].iloc[0])))):
        
        # Time steps definition
        time_steps =  np.arange(0, len(wave_height_timeseries), 1)
        
        # Required WW in time steps
        weather_window_time_steps = np.ceil(Rep_time["Time_"+Repair_type[tt]+'_'+Vessel_type[tt]].values[0] / discretisation)  
        weather_window_time_steps = weather_window_time_steps * discretisation
        
        # Calculate approachability
        is_wave_below_limit = wave_height_timeseries < Op_lim["Hs_lim_"+Vessel_type[tt]].values[0]
        is_wind_below_limit = wind_speed_timeseries < Op_lim["Uw_lim_"+Vessel_type[tt]].values[0]
        is_wave_and_wind_below_limit = is_wave_below_limit * is_wind_below_limit
        approachability[tt] = sum(is_wave_and_wind_below_limit) / len(wave_height_timeseries)
        
        # Calculate Weather Window Intervals
        approachability_time_steps = np.where(is_wave_and_wind_below_limit)[0]
        approachability_time_steps_intervals = intervals(approachability_time_steps)
        
        # Create a list of tuples defining weather windows intervals
        weather_window_interval = []
        for i in range(len(approachability_time_steps_intervals)):        
            if (approachability_time_steps_intervals[i][1] - approachability_time_steps_intervals[i][0] + 1) >= weather_window_time_steps:
                weather_window_interval.append(approachability_time_steps_intervals[i])  
                
        # New strict intervals
        new_intervals = []
        for i in range(len(weather_window_interval)):
            first_index = int(weather_window_interval[i][0])
            last_index = int(weather_window_interval[i][1])
            difference = last_index - first_index + 1
            if difference == Rep_time["Time_"+Repair_type[tt]+'_'+Vessel_type[tt]].values[0]:
                new_intervals.append(weather_window_interval[i])
            else:
                quantity_of_wws = math.floor(difference/Rep_time["Time_"+Repair_type[tt]+'_'+Vessel_type[tt]].values[0])
                for j in range(quantity_of_wws):
                    first_new_index = int(weather_window_interval[i][0] + j * (Rep_time["Time_"+Repair_type[tt]+'_'+Vessel_type[tt]].values[0]))
                    last_new_index = int(first_new_index + Rep_time["Time_"+Repair_type[tt]+'_'+Vessel_type[tt]].values[0] - 1)
                    new_intervals.append((first_new_index,last_new_index))         
        weather_window_interval = new_intervals   
        
        # Check if there is no time interval
        if not weather_window_interval:
            total_accessible_time[tt] = 0
            accessibility[tt] = 0
            final_weather_window_intervals = 0
            waiting_times_intervals[tt] = 0
            total_waiting_time[tt] = (time_steps[-1] * discretisation) + 1
            mean_time_to_wait[tt] = total_waiting_time[tt]
            total_weather_windows_number[tt] = 0
            mean_duration_of_a_ww[tt] = 0
        else:                          
            # Calculate accessibility    
            final_weather_window_intervals = []
            accessibility_counter = 0
            for i in range(len(weather_window_interval)):
                counter = 0
                first_element = weather_window_interval[i][0]
                last_element = weather_window_interval[i][1] + 1
                local_wave_height_ts = wave_height_timeseries[first_element:last_element]
                local_wind_speed_ts = wind_speed_timeseries[first_element:last_element]
                is_wave_below_limit = np.array(local_wave_height_ts) < Op_lim["Hs_lim_"+Vessel_type[tt]].values[0]
                is_wind_below_limit = np.array(local_wind_speed_ts) < Op_lim["Uw_lim_"+Vessel_type[tt]].values[0]
                is_wave_and_wind_below_limit = is_wave_below_limit * is_wind_below_limit
                counter = sum(is_wave_and_wind_below_limit)
                if counter >= Rep_time["Time_"+Repair_type[tt]+'_'+Vessel_type[tt]].values[0]:
                    final_weather_window_intervals.append(weather_window_interval[i])
                    accessibility_counter = accessibility_counter + counter
            accessibility[tt] = accessibility_counter / len(wave_height_timeseries)
            total_accessible_time[tt] = accessibility_counter
                
            # Calculate Waiting times
            waiting_time = []
            for i in range(len(final_weather_window_intervals) - 1):
                waiting_time.append(final_weather_window_intervals[i+1][0] - final_weather_window_intervals[i][1])
                
            # Waiting time between weather windows
            waiting_time_steps_between_weather_windows = np.sum(waiting_time)
            waiting_time_between_weather_windows = waiting_time_steps_between_weather_windows * discretisation
            
            # Waiting time until first weather window
            waiting_time_steps_until_first_weather_window = final_weather_window_intervals[0][0]
            waiting_time_until_first_weather_window = waiting_time_steps_until_first_weather_window * discretisation
            
            # Waiting time from last weather window until finally
            waiting_time_steps_from_last_weather_window_until_finally = time_steps[-1] - final_weather_window_intervals[-1][-1]
            waiting_time_from_last_weather_window_until_finally = waiting_time_steps_from_last_weather_window_until_finally * discretisation
            
            # For total number of waiting times 
            if waiting_time_until_first_weather_window == 0 and waiting_time_from_last_weather_window_until_finally == 0:
                waiting_time_counter = 0
            elif waiting_time_until_first_weather_window != 0 and waiting_time_from_last_weather_window_until_finally == 0:
                waiting_time_counter = 1
            elif waiting_time_until_first_weather_window == 0 and waiting_time_from_last_weather_window_until_finally != 0:
                waiting_time_counter = 1
            else:
                waiting_time_counter = 2
            
            # Total waiting time calculation
            total_waiting_time[tt] = waiting_time_until_first_weather_window + waiting_time_between_weather_windows + waiting_time_from_last_weather_window_until_finally     
            waiting_times_intervals[tt] = len(waiting_time) + waiting_time_counter 
            mean_time_to_wait[tt] = np.sum(total_waiting_time[tt]) / waiting_times_intervals[tt]
        
            # Total weather windows calculation
            total_weather_windows_number[tt] = len(final_weather_window_intervals)
            
            # Mean duration of a WW
            mean_duration_of_a_ww[tt] = total_accessible_time[tt] / total_weather_windows_number[tt]
        
        waiting_time_dic[Vessel_type[tt]] = waiting_time    
    WT_parameters = pd.DataFrame({"Mean_duration_of_WW":mean_duration_of_a_ww,"Total_waiting_time":total_waiting_time,"Waiting_time_intervals":waiting_times_intervals,"Mean_time_to_wait":mean_time_to_wait,
                                      "Total_weather_windows_number":total_weather_windows_number})
    ACC_parameters = pd.DataFrame({"Approachability":approachability,"total_accessible_time":total_accessible_time,"Accessibility_percent":accessibility*100})

    
    return WT_parameters,ACC_parameters,waiting_time_dic

    
        
def ORE_access(met_data,des_param,locations,root_dir):
    
    
    databases = ["Obs","Sim","Calib"]
    databases_v2 = ["Buoy","ERA5", "CERA5"]
    ORE_WT_df = pd.DataFrame(index = databases_v2,columns = des_param["Design"]["Access"]["Op_type"], dtype=float)                  
    ORE_WT_dic = {"Buoy":{},
                  "ERA5":{},
                  "CERA5":{},
                  }
    #ORE_prod_df = pd.DataFrame(columns=[ORE+"_"+dat  for ORE in var_pair.keys() for dat in databases_v2], dtype=float) 
    
    discretisation = 1
    for i_dat, data in enumerate(databases):
        Hs = met_data["Hs_"+data]
        if data == "Obs" or data == "Calib":
            Uw = met_data["Uv10_"+data]
            
        else:
            Uw = met_data["Uw_"+data]

        WT_params,ACC_params,ORE_WT_dic[databases_v2[i_dat]] = accessibility_hs_uw(Hs, Uw, des_param["Design"]["Access"]["Op_lim_vessel"], des_param["Design"]["Access"]["Rep_time"], discretisation)
        
        for i_opt, Op_type in enumerate(des_param["Design"]["Access"]["Op_type"]):
            ORE_WT_df.loc[databases_v2[i_dat],Op_type] = WT_params["Mean_time_to_wait"][i_opt]
            
    return ORE_WT_df,ORE_WT_dic

def alpha_shape(points: np.ndarray, alpha: float):
    """
    Concave hull (alpha-shape). Smaller alpha -> tighter/more concave.
    """
    if len(points) < 4:
        return MultiPoint(points).convex_hull

    tri = Delaunay(points)
    edges = set()

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]

        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)

        s = (a + b + c) / 2.0
        area2 = s * (s - a) * (s - b) * (s - c)
        if area2 <= 0:
            continue
        area = np.sqrt(area2)
        R = (a * b * c) / (4.0 * area)  # circumradius

        if R < 1.0 / alpha:
            for i, j in [(ia, ib), (ib, ic), (ic, ia)]:
                if i > j:
                    i, j = j, i
                edges.add((i, j))

    mls = MultiLineString([(points[i], points[j]) for i, j in edges])
    polygons = list(polygonize(mls))
    return unary_union(polygons)

'''
------------------------------------------------------------------------------ 
                     FOWT FLS ASSESSMENT RELATED FUNCTIONS
------------------------------------------------------------------------------
''' 

###############################################################################




def ORE_FLS(met_data,des_param,location,root_dir):
    
    "--> Metocean clustering"
    # Load fatigue damage estimates for WEC
    fat_dam_WEC = pd.read_excel(root_dir+'/Results/Design_indicators/FLS/Fatigue_damage_WEC_sites.xlsx',sheet_name =location,header=[0, 1], index_col=0)
    fat_dam_buoy_WEC = fat_dam_WEC["Buoy"]
    
    # Load fatigue damage estimates for FOWT      
    fat_dam_FOWT = pd.read_excel(root_dir+'/Results/Design_indicators/FLS/'+location+"_fatigue_damage_FOWT.xlsx", sheet_name=None, index_col=0)
    # Components of the FOWT
    tech_components = ["Blade root","Mooring","Tower base"]
    
    "--> Construct the outer envelope of buoy Hs&Tp combinations"
    # Maximum damage in buoy related results is extracted, considering that the Mooring 1 is the critical one
    max_dam_cond_b = fat_dam_buoy_WEC [ fat_dam_buoy_WEC["dam_Moor1"]==max(fat_dam_buoy_WEC ["dam_Moor1"])]
    # Create the outer envelope of the Hs&Tp combinations derived from buoy centroids
    t = fat_dam_buoy_WEC["Tp_Obs"].to_numpy()
    h = fat_dam_buoy_WEC["Hs_Obs"].to_numpy()
    pts = np.column_stack([t, h])              
    alpha = 0.18
    poly = alpha_shape(pts, alpha=alpha)    
    if poly.geom_type == "Polygon":
        bx, by = poly.exterior.xy        
    else:  # MultiPolygon
        for g in poly.geoms:
            bx, by = g.exterior.xy
            
    #**********************************************************************************************        
    "--> WEC FLS assessment"
    databases = ["Obs","Sim","Calib"]
    databases_v2 = ["Buoy","ERA5", "CERA5"]    
    vmin = 0 
    vmax = 1.0*1e-8
    tot_dam = []
    for i_dat, data in enumerate(databases):
        
        fat_dam_dat_spec_WEC = fat_dam_WEC[databases_v2[i_dat]]
        Hs_cent = fat_dam_dat_spec_WEC["Hs_"+data].to_numpy()
        Tp_cent = fat_dam_dat_spec_WEC["Tp_"+data].to_numpy()
        max_dam_cond = fat_dam_dat_spec_WEC[ fat_dam_dat_spec_WEC["dam_Moor1"]==max(fat_dam_dat_spec_WEC["dam_Moor1"])]
        
        
        Moor_lines = [4,5,6]
        tot_dam.append(np.sum(fat_dam_dat_spec_WEC["dam_Moor2"]))       
        
        '--> Plot centroids and corresponding damage'
        
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        im1 = ax1.scatter(Tp_cent,Hs_cent,c = fat_dam_dat_spec_WEC["dam_Moor1"],cmap='viridis',vmin=vmin,
        vmax=vmax, marker='o',label = databases_v2[i_dat])                
        ax1.plot(bx, by, linewidth=2.0,linestyle = '--',color = 'k',label = "Buoy contour")            
        ax1.scatter(max_dam_cond["Tp_"+data],max_dam_cond["Hs_"+data],s = 140,color = "tomato",edgecolor = 'k',linewidths=2,label = "Max dam. "+databases_v2[i_dat])
        ax1.scatter(max_dam_cond_b["Tp_Obs"],max_dam_cond_b["Hs_Obs"],marker = '^',s = 140,color = "orange",edgecolor = 'k',linewidths=2,label = "Max dam. Buoy")
        
        divider = make_axes_locatable(ax1)
        
        cax1 = divider.append_axes('right', size='2%', pad=0.1)
        cb1 = fig1.colorbar(im1, cax=cax1, orientation='vertical')
        cb1.set_label('Damage [-]', fontname = "Times New Roman",fontsize=24,fontweight = 'bold')
        #cb1.set_label('Probability [-]', fontname = "Times New Roman",fontsize=24,fontweight = 'bold')
        for t in cb1.ax.get_yticklabels():
            t.set_fontname("Times New Roman")
            t.set_fontsize(22)
        ax1.set_xlabel("Tp [s]",fontname = "Times New Roman",fontweight="bold",fontsize = 26)
        ax1.set_ylabel("Hs [m]",fontname = "Times New Roman",fontweight="bold",fontsize = 26)
        ax1.legend(fontsize=24, loc="upper left", frameon=True)
        ax1.set_xlim([0, 26])
        ax1.set_ylim([0, 14])
        ax1.grid(linestyle='--')
    
        
        # Save figures 
        path_fig = root_dir+"/Results/Figures/Design_assessment_indicators/"
        os.makedirs(path_fig, exist_ok=True)
        fig1.savefig(path_fig+location+"_WEC_fatigue_damage_"+databases_v2 [i_dat]+".svg", format='svg')      
    
    
    #**********************************************************************************************
    '--> FOWT FLS assessment'           
    # Extract fatigue damage arrays
    TB = {db: {comp: df[db].dropna().to_numpy() for comp, df in fat_dam_FOWT.items()} for db in databases}    
    # Calculate relative errors with respect to observations
    #summary_rows = [{"Location": location, "Component": comp, "DataSource": db, "MeanDamage": abs(df[db].sum()-df["Obs"].sum())/df["Obs"].sum()*100} for comp, df in fat_dam_FOWT.items() for db in databases]
    summary_rows = [{"Location": location, "Component": comp, "DataSource": db, "MeanDamage": df[db].sum()} for comp, df in fat_dam_FOWT.items() for db in databases]
   
    # Normalise fatigue damage by mean observed damage
    TB_norm = {db: {comp: TB[db][comp]/np.mean(TB["Obs"][comp]) for comp in tech_components} for db in databases}

    
    '--> Visualize FOWT FLS assessment results'
    # Plotting settings
    comp_gap, db_spacing = 1.0, 0.9
    db_colors = {"Obs": colors[0], "Calib": colors[1], "Sim": colors[2]}
    db_map = {"Obs": "Buoy", "Calib": "CERA5", "Sim": "ERA5"}
    x_positions = {(comp, db): i*(len(databases)*db_spacing+comp_gap)+j*db_spacing for i, comp in enumerate(tech_components) for j, db in enumerate(databases)}
    xticks = list(x_positions.values())
    xticklabels = [db_map[db] for comp, db in x_positions]
    
    fig2, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(1, color="k", lw=2, label="Buoy")
    
    for (comp, db), x in x_positions.items():
        y = TB_norm[db][comp]
        ax.scatter(np.full_like(y, x, dtype=float), y, s=60, edgecolor='k', color=db_colors[db], zorder=3)
        if db in ["Sim", "Calib"]:
            ax.hlines(np.mean(y), x-0.52, x+0.52, linestyles=(0, (5, 4)), lw=4, color=db_colors[db], zorder=2)
        # ax.scatter(x, np.max(y), s=200, marker='*', facecolors="gold", edgecolors="orangered", lw=1.5, zorder=4)
    
    # Component labels
    for comp in tech_components:
        x_center = np.mean([x_positions[(comp, db)] for db in databases])
        ax.text(x_center, -0.12, comp, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=24, fontweight="bold")
    
    # Formatting
    ax.set(yscale="log", ylabel="Norm. Damage [-]", xticks=xticks, xticklabels=xticklabels)
    ax.tick_params(axis="x", labelsize=18)
    ax.yaxis.label.set(fontsize=24, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=8, markeredgecolor='k', label='Individual damage samples'),
        Line2D([0], [0], linestyle=(0, (5, 4)), color='black', lw=3, label='Mean fatigue damage'),
        Line2D([0], [0], linestyle='-', color='black', lw=3, label='Buoy reference')
    ]
    
    ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=18)
    plt.tight_layout()
    
    # Save figures 
    path_fig = root_dir+"/Results/Figures/Design_assessment_indicators/"
    os.makedirs(path_fig, exist_ok=True)
    fig2.savefig(path_fig+location+"_FOWT_fatigue_damage.svg", format='svg')      

    
    
    '--> Save fatigue results'    
    tot_dam_arr = np.array(tot_dam)  
    tot_dam_df_WEC =   pd.DataFrame([tot_dam_arr.T],columns = databases_v2)  
    tot_dam_df_FOWT = pd.DataFrame(summary_rows)  
    
    
    
    
    
    return tot_dam_df_WEC,tot_dam_df_FOWT
        
def format_subscript_upright(label):
    #var_part, unit_part = label.split(" [")
    #unit_part = "[" + unit_part
    main, sub = label.split("_", 1)
    return rf"${main}_{{\mathrm{{{sub}}}}}$"
def ORE_ULS(des_param,location,root_dir):  
    
    "--> Load ULS analysis results"
    # Load and merge ULS results
    path = root_dir + '/Results/Design_indicators/ULS/'
    data = pd.concat([pd.read_csv(path+f'All_sites_ULS_{tech}.csv') for tech in ['FOWT', 'WEC']], ignore_index=True)
    data = data[data['location'] == location]

    # Extract variables and statistics
    variables = data[data['data_source'] == 'Buoy']['variable'].unique()
    rename_vars = {'BladeR_InpBenMom [kN.m]':'BLA_In-Ben', 'BladeR_OutpBenMom [kN.m]':'BLA_Out-Ben', 
                   'Moor1_EffTen [kN]':'MOOR_Fair-Ten','TowerB_InpBenMom [kN.m]':'TOW_In-Ben', 
                   'TowerB_OutpBenMom [kN.m]':'TOW_Out-Ben', 'Moor1_EffTen [N]':'MOOR_Fair-Ten'}
    variables_new = [rename_vars.get(v, v) for v in variables]
    new_cols = [format_subscript_upright(v) for v in variables_new]
    col_names = ['max', 'mean', 'std']

    # Mean statistics for each dataset and variable
    stats = data.groupby(['data_source', 'variable'])[col_names].mean()
    matrices = {db: stats.loc[db].reindex(variables).to_numpy() for db in ['Buoy', 'ERA5', 'CERA5']}

    Buoy_mat, ERA5_mat, CERA5_mat = (matrices[db] for db in ['Buoy', 'ERA5', 'CERA5'])
    ERA5_norm, CERA5_norm = ERA5_mat/Buoy_mat, CERA5_mat/Buoy_mat
    
    import matplotlib.colors as mcolors
    n_var, n_stat = len(variables), len(col_names)
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap, norm = plt.cm.coolwarm, mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
    
    for i in range(n_var):
        for j in range(n_stat):
            for val, coords, hatch in [(ERA5_norm[i, j], [[j, i+1], [j, i], [j+1, i+1]], "..."), (CERA5_norm[i, j], [[j+1, i], [j+1, i+1], [j, i]], "///")]:
                ax.add_patch(Polygon(coords, closed=True, facecolor=cmap(norm(val)), edgecolor="k", hatch=hatch, linewidth=0.5))
    
    ax.set(xlim=(0, n_stat), ylim=(0, n_var), aspect="equal")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_stat)+0.5)
    ax.set_xticklabels(col_names)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_yticks(np.arange(n_var)+0.5)
    ax.set_yticklabels(new_cols)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised values [-]", rotation=90, fontsize=25, fontweight="bold")
    for t in cbar.ax.get_yticklabels(): t.set_fontname("Times New Roman"); t.set_fontsize(22)
    
    ax.legend(handles=[Patch(facecolor="white", edgecolor="k", hatch="...", label="ERA5"), Patch(facecolor="white", edgecolor="k", hatch="///", label="CERA5")], bbox_to_anchor=(0.3, -0.01), loc="upper left", borderaxespad=0.)
    fig.subplots_adjust(right=0.8)
    plt.show()

    # Save figures 
    path_fig = root_dir+"/Results/Figures/Design_assessment_indicators/"
    os.makedirs(path_fig, exist_ok=True)
    fig.savefig(path_fig+location+"_USL_heatmap.svg", format='svg')  

    return stats    
       
    
    
def uncert_PP(des_param,locations,root_dir):
    tech = ["WEC", "FOWT"]
    pow_data = des_param["Design"]["GenPow"]["Pow"]
    power_boxplot_dic = {}
    
    for t in tech:
        fig, ax = plt.subplots(figsize=(12, 8))
        summary = {"Q1_ERA5": [], "Q3_ERA5": [], "median_ERA5": [], "Q1_CERA5": [], "Q3_CERA5": [], "median_CERA5": []}
    
        power_boxplot_dic[t] = {}

        for i, loc in enumerate(locations):
            df = pow_data[loc]
            ref = df[f"{t}_Buoy"]
            mask = ref > 0.0
    
            data_power = []
    
            for db in ["ERA5", "CERA5"]:
                sim = df[f"{t}_{db}"]
    
                # Original calculation: operational conditions only
                error_operational = ((sim[mask] - ref[mask]) / ref[mask]).to_numpy()
    
                # Original calculation: all conditions, replacing undefined values with zero
                error_all = np.nan_to_num(((sim - ref) / ref).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    
                # Fraction of conditions where both datasets predict zero power
                zero_fraction = np.sum((ref == 0) & (sim == 0)) / len(ref)
                #print(f"{t}_{db}_{loc}_{zero_fraction}")
    
                # Store both distributions
                power_boxplot_dic[t][f"{db}_{loc}_operational"] = error_operational
                power_boxplot_dic[t][f"{db}_{loc}_all"] = error_all
    
                # Select operational uncertainty for boxplot
                data_power.append(error_operational)
    
            positions = [i+0.8, i+1.2]
    
            bp = ax.boxplot(data_power, positions=positions, widths=0.2, patch_artist=True, showfliers=False, medianprops=dict(color="gold", linewidth=2), boxprops=dict(edgecolor="k", linewidth=1.5), whiskerprops=dict(color="k", linewidth=1.2), capprops=dict(color="k", linewidth=1.2))
    
            for j, db in enumerate(["ERA5", "CERA5"]):
                summary[f"Q1_{db}"].append(bp["whiskers"][2*j].get_ydata()[1])
                summary[f"Q3_{db}"].append(bp["whiskers"][2*j+1].get_ydata()[1])
                summary[f"median_{db}"].append(bp["medians"][j].get_ydata()[0])
                bp["boxes"][j].set_facecolor(colors[j+1])
    
        power_boxplot_dic[t] = pd.DataFrame(summary, index=locations)
    
        # Uncertainty envelopes
        for j, db in enumerate(["ERA5", "CERA5"]):
            x = np.arange(1, len(locations)+1)+(-0.2 if db=="ERA5" else 0.2)
            ax.fill_between(x, power_boxplot_dic[t][f"Q1_{db}"], power_boxplot_dic[t][f"Q3_{db}"], color=colors[j+1], alpha=0.4)
    
        # Formatting
        ax.set_xticks(np.arange(1, len(locations)+1))
        ax.set_xticklabels(locations, fontsize=24)
        ax.set_ylabel("Relative power error [-]", fontname="Times New Roman", fontweight="bold", fontsize=26)
        ax.set_title(f"{t} power bias", fontname="Times New Roman", fontweight="bold", fontsize=18)
        ax.legend(handles=[Patch(facecolor=colors[j+1], label=db) for j, db in enumerate(["ERA5", "CERA5"])], loc="upper right")
        ax.set_ylim(-1.25, 1.5)
    
        plt.tight_layout()
        plt.show()
        
        # Save figures 
        path_fig = root_dir+"/Results/Figures/Design_assessment_indicators/"
        os.makedirs(path_fig, exist_ok=True)
        fig.savefig(path_fig+"All_sites_normPP_"+t+".svg", format='svg') 
        
        plt.close("all")
        
        
     
    
def uncert_MWT(des_param,locations,root_dir):
    
    # Normalize waiting times
    des_param["Design"]["Access"]["WT_norm"] = {loc: df.div(df.loc["Buoy"]) for loc, df in des_param["Design"]["Access"]["WT"].items()}
    WT_data = des_param["Design"]["Access"]["WT"]
    locations = list(WT_data.keys())
    operations = ["CTV", "FSV", "HLV"]
    marker_map = {"CTV": "o", "FSV": "*", "HLV": "D"}
    
    cmap = plt.colormaps["viridis"]
    colors = [cmap(v) for v in np.linspace(0.2, 0.8, 3)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(locations))
    
    for i, loc in enumerate(locations):
        df = WT_data[loc].div(WT_data[loc].loc["Buoy"])
        for op in operations:
            ax.scatter(x[i]-0.08, df.loc["ERA5", op], s=140, marker=marker_map[op], edgecolor="k", color=colors[1], label=f"ERA5 - {op}" if i == 0 else "")
            ax.scatter(x[i]+0.08, df.loc["CERA5", op], s=140, marker=marker_map[op], edgecolor="k", color=colors[2], label=f"CERA5 - {op}" if i == 0 else "")
    
    ax.axhline(1.0, ls="--", color="k", lw=2, label="Buoy")
    ax.set_xticks(x)
    ax.set_xticklabels(locations)
    ax.tick_params(axis="x", pad=15)
    ax.set_ylabel("Normalised MWT [-]", fontname="Times New Roman", fontsize=26, fontweight="bold")
    ax.set_ylim(0, 2)
    legend_markers = [Line2D([0], [0], marker=marker_map[op], color="w", markerfacecolor="white", markeredgecolor="k", markersize=8, linestyle="None", label=op) for op in operations]
    legend_datasets = [Patch(facecolor=colors[1], edgecolor="k", label="ERA5"), Patch(facecolor=colors[2], edgecolor="k", label="CERA5"), Line2D([0], [0], color="k", ls="--", label="Buoy")]
    
    legend1 = ax.legend(handles=legend_markers, loc="upper left", framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=legend_datasets, loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
             
    # Save figures 
    path_fig = root_dir+"/Results/Figures/Design_assessment_indicators/"
    os.makedirs(path_fig, exist_ok=True)
    fig.savefig(path_fig+"All_sites_normMWT.svg", format='svg') 
    plt.close("all")
    
    return des_param
    
        
   
      