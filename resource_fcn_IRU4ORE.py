# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:31:46 2026

Module name: [Resource_indicators_fcns]
Author: [Ander Martinez, Markel Penalba]
Affiliation: [Mondragon University]
Contact: [amartinezpe@mondragon.edu]
"""
'--> Dependencies'

# Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from scipy.stats import spearmanr
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from virocon import (
    read_ec_benchmark_dataset,
    get_OMAE2020_Hs_Tz,
    GlobalHierarchicalModel,
    IFORMContour,
    plot_2D_contour,
    LogNormalDistribution,
    ExponentiatedWeibullDistribution,
    WeibullDistribution,
    DependenceFunction,
    WidthOfIntervalSlicer,
    HighestDensityContour,
    plot_marginal_quantiles,
    plot_dependence_functions,
    IFORMContour,ISORMContour,
    get_OMAE2020_V_Hs,get_DNVGL_Hs_U
    )



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
                            PARAMETER ESTIMATION FOR PDF
------------------------------------------------------------------------------
''' 
'-->FUNCTIONS FOR CONDITIONAL DISTRIBUTIONS'
# A 3-parameter power function, which will be used as a dependence function.
def _power3(x, a, b, c):
    return a + b * x**c
# A 3-parameter power function, which will be used as a dependence function.
def _exp3(x, a, b, c):
    return a + b * np.exp(c * x)
# A 3- parameter alpha function, which will be used as a dependence function.
def _alpha3(x, a, b, c, d_of_x):
    return (a + b * x**c) / 2.0445 ** (1 / d_of_x(x))


# A 4- parameter logistic function, which will be used as a dependence function.
def _logistics4(x, a=1, b=1, c=-1, d=1):
    return a + b / (1 + np.exp(-c * (x - d)))
# A 4- parameter logistic function, which will be used as a dependence function.
def _constant(x,a=1):
    return a+x

def _linear(x,a=1,b = 1):
    return a+b*x

def _quadratic(x,a=1,b = 1,c = 1):
    return a+b*x**c




'--> Spearmans Rank correlation between env. variables'
def rank_cor(met_data,location):
    
    databases = ["Sim","Calib"]       
    var_pair = {"Hs&Tp":["Hs","Tp"],"Hs&Uw":["Hs","Uw"]}
    site_data = met_data[location]
    
    rho_rank = pd.DataFrame(index=databases, columns=var_pair.keys(),dtype=float)
    for i_dat,data in enumerate(databases):
        for var_p in var_pair:  
            
            if data == "Calib" and var_p=="Hs&Uw":
                select_var = ["Hs","Uv10"]
            else:
                select_var = var_pair[var_p]
            
            rho_value, _ = spearmanr(site_data[select_var[0]+"_"+data].values, site_data[select_var[1]+"_"+data].values)
            rho_rank.loc[data, var_p] = rho_value
            
    return rho_rank
   
   
    
    
    
    
    
    
    

'--> Virocon toolbox based IFORM construction'

def OMAE2020(data,tr,ts,buoy_name,data_source,plot_flag):
    
    # Define the structure of the joint distribution model.
    # Wind speed and Hs conditional distribution.
    
       
    
    title_column = data.columns
    labels = [title_column [i].split('_')[0] for i in range(len(title_column ))]
    if  all(string in labels for string in ["Hs", "Tp"]):
        data.columns = ["Hs","Tp"]
        variables = data.columns
        data_cleaned = data[(data != 0).all(axis=1)]
        
        # plt.figure()
        # plt.plot(data_cleaned["Hs"].values)
        # plt.plot(data_cleaned["Tp"].values)
        
    elif all(string in labels for string in ["Hs", "Uw"]):
        data.columns = ["Hs","Vhub"]
        variables = data.columns
        data_cleaned = data[(data != 0).all(axis=1)]
        data_cleaned = data_cleaned[~((data_cleaned["Vhub"] > 0) & (data_cleaned["Vhub"] < 0.2))]
        
    elif all(string in labels for string in ["Hs", "Uv10"]):
        data.columns = ["Hs","Vhub"]
        variables = data.columns
        data_cleaned = data[(data != 0).all(axis=1)]
        data_cleaned = data_cleaned[~((data_cleaned["Vhub"] > 0) & (data_cleaned["Vhub"] < 0.2))]
        
    if "Vhub" in variables:    
        
       data_arrang = pd.DataFrame({"Vhub":data_cleaned["Vhub"].values,'Hs':data_cleaned["Hs"].values})
       data_arrang[data_arrang <= 0] = 0.1
       
       
       bounds = [(0, None), (0, None), (None, None)]
       bounds_constant = [(0,None)]
       bounds_linear = [(0, None), (0, None)]
       logistics_bounds = [(0, None), (0, None), (None, 0), (0, None)]
       #logistics_bounds = [(None, None), (None, None), (None, None), (None, None)]
       power3 = DependenceFunction(_power3, bounds, latex="$a + b * x^c$")
       exp3 = DependenceFunction(_exp3, bounds, latex="$a + b * \exp(c * x)$")
       constant = DependenceFunction(_constant, bounds_constant, latex="$a+x$")
       linear = DependenceFunction(_linear, bounds_linear, latex="$a+b*x$")
       quadratic = DependenceFunction(_quadratic, bounds, latex="$a+b*x^c$")
       
       def _logistics4(x, a=1, b=1, c=-1, d=1):
           return a + b / (1 + np.exp(c * (x - d)))

       def _alpha3(x, a, b, c, d_of_x):
           return (a + b * x**c) / 2.0445 ** (1 / d_of_x(x))

       logistics_bounds = [(0, None), (0, None), (None, 0), (0, None)]

       alpha_bounds = [(0, None), (0, None), (None, None)]
       logistics4 = DependenceFunction(
           _logistics4,
           logistics_bounds,
           weights=lambda x, y: y,
           latex="$a + b / (1 + \exp[-c * (x -d)])$",
       )
       alpha3 = DependenceFunction(
           _alpha3,
           bounds,
           d_of_x=logistics4,
           weights=lambda x, y: y,
           latex="$(a + b * x^c) / 2.0445^{1 / F()}$",
       )
       #Wind speed  
       
       try:          
           
           dist_descriptions, fit_descriptions, semantics = get_OMAE2020_V_Hs()
           fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
           fit_descriptions = [fit_description_hs, None]
           
           model = GlobalHierarchicalModel(dist_descriptions)   
           
           #Estimate the model's parameter values with the default method (MLE).
           model = GlobalHierarchicalModel(dist_descriptions)
           #model.fit(data_arrang, fit_descriptions=fit_descriptions)
           
           
           
           
           beta_dep = DependenceFunction(
               _logistics4,
               logistics_bounds,
               weights=lambda x, y: y,
               latex="$a + b / (1 + \exp[c * (x -d)])$",
           )
           alpha_dep = DependenceFunction(
               _alpha3,
               alpha_bounds,
               d_of_x=beta_dep,
               weights=lambda x, y: y,
               latex="$(a + b * x^c) / 2.0445^{1 / F()}$",
           )
           
           dist_description_v = {
               "distribution": ExponentiatedWeibullDistribution(),
               "intervals": WidthOfIntervalSlicer(1, min_n_points=300),
           }

           dist_description_hs = {
               "distribution": ExponentiatedWeibullDistribution(f_delta=5),
               "conditional_on": 0,
               "parameters": {
                   "alpha": alpha_dep,
                   "beta": beta_dep,
               },
           }

           dist_descriptions = [dist_description_v, dist_description_hs]

           fit_description_v = {"method": "wlsq", "weights": "quadratic"}
           fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
           fit_descriptions = [fit_description_v, fit_description_hs]

           semantics = {
               "names": ["Wind speed", "Significant wave height"],
               "symbols": ["V", "H_s"],
               "units": [
                   "m s$^{-1}$",
                   "m",
               ],
           }
           model = GlobalHierarchicalModel(dist_descriptions)
           model.fit(data_arrang, fit_descriptions=fit_descriptions)
           
                  
       except:
           print("\n\n SECOND APPROACH:"+buoy_name+' '+data_source+'\n\n')
           

           beta_dep = DependenceFunction(
               _logistics4,
               logistics_bounds,
               weights=lambda x, y: y,
               latex="$a + b / (1 + \exp[c * (x -d)])$",
           )
           alpha_dep = DependenceFunction(
               _alpha3,
               alpha_bounds,
               d_of_x=beta_dep,
               weights=lambda x, y: y,
               latex="$(a + b * x^c) / 2.0445^{1 / F()}$",
           )
           
           dist_description_v = {
               "distribution": ExponentiatedWeibullDistribution(),
               "intervals": WidthOfIntervalSlicer(1, min_n_points=100),
           }

           dist_description_hs = {
               "distribution": ExponentiatedWeibullDistribution(f_delta=5),
               "conditional_on": 0,
               "parameters": {
                   "alpha": alpha_dep,
                   "beta": beta_dep,
               },
           }

           dist_descriptions = [dist_description_v, dist_description_hs]

           fit_description_v = {"method": "wlsq", "weights": "quadratic"}
           fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
           fit_descriptions = [fit_description_v, fit_description_hs]

           semantics = {
               "names": ["Wind speed", "Significant wave height"],
               "symbols": ["V", "H_s"],
               "units": [
                   "m s$^{-1}$",
                   "m",
               ],
           }
           model = GlobalHierarchicalModel(dist_descriptions)
           model.fit(data_arrang, fit_descriptions=fit_descriptions)
           
                
        
    elif "Tp" in variables:
        
        data_arrang = pd.DataFrame({"Hs":data_cleaned["Hs"].values,'Tp':data_cleaned["Tp"].values})
        data_arrang[data_arrang <= 0] = 0.1
        dist_descriptions, fit_descriptions, semantics = get_OMAE2020_Hs_Tz()
        fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
        fit_descriptions = [fit_description_hs, None]
        
        model = GlobalHierarchicalModel(dist_descriptions)   
        
        #Estimate the model's parameter values with the default method (MLE).
        model = GlobalHierarchicalModel(dist_descriptions)
        model.fit(data_arrang, fit_descriptions=fit_descriptions)
    #model.fit(data_file)
    
    # Compute an IFORM contour with a return period of 50 years.    
    alpha = 1 / (tr * 365.25 * 24 / ts)
    contour = IFORMContour(model, alpha)
    coor_cont = contour.coordinates
    
    
    
    cmap = cm.get_cmap("viridis")
    color_range = np.linspace(0.2, 0.9, 4)  # Select colors between 20% and 80% of the colormap
    colors = [cmap(value) for value in color_range]  # Extract specific colors
    
    if "Vhub" in variables:  
        
        contour_sec = pd.DataFrame(coor_cont,columns = ["Vhub","Hs"])
        w = (coor_cont[:,1]**2 + coor_cont[:,0]**2)
        max_p = np.where(coor_cont[:,1]== max(coor_cont[:,1]))
        # max_Hs =  coor_cont[max_p[0],1]
        # max_Vhub =  coor_cont[max_p[0],0]   
        
        max_Hs = max(coor_cont[:,1])
        max_Vhub = max(coor_cont[:,0])
        max_p_Hs = np.where(coor_cont[:,1]== max(coor_cont[:,1]))  
        tangent_Hsmax = coor_cont[max_p_Hs[0],0]
        max_p_Vhub = np.where(coor_cont[:,0]== max(coor_cont[:,0]))
        tangent_Vhubmax = coor_cont[max_p_Vhub[0],1]
        
        EDP = pd.DataFrame({"Hs_extrem":[max_Hs],"Uw_extrem":[max_Vhub]})
        
        
               
    elif "Tp" in variables:
        
        
        contour_sec = pd.DataFrame(coor_cont,columns = ["Hs","Tp"])
        w = (coor_cont[:,1]**2 + coor_cont[:,0]**2)
        # Extreme design point
        max_p = np.where(coor_cont[:,0]== max(coor_cont[:,0]))
        max_Hs =  coor_cont[max_p[0],0]
        max_Tp =  coor_cont[max_p[0],1]
        EDP = pd.DataFrame({"Hs_extrem":max_Hs,"Tp_extrem":max_Tp})
        
        
        
          
    return EDP,contour_sec





def env_contour_plot(res_param,location,rooth_path):       
    
    
    "--> Load Extreme Design Point values"
    # EDP for Hs vs Tp
    EDP_HsTp_Buoy = res_param["Resource"]["extremes"]["Waves"]["EDP"]["Buoy"][location]
    EDP_HsTp_ERA5 = res_param["Resource"]["extremes"]["Waves"]["EDP"]["ERA5"][location]
    EDP_HsTp_CERA5 = res_param["Resource"]["extremes"]["Waves"]["EDP"]["CERA5"][location]
    
    # EDP for Hs vs Tp
    EDP_HsUw_Buoy = res_param["Resource"]["extremes"]["Wind"]["EDP"]["Buoy"][location]
    EDP_HsUw_ERA5 = res_param["Resource"]["extremes"]["Wind"]["EDP"]["ERA5"][location]
    EDP_HsUw_CERA5 = res_param["Resource"]["extremes"]["Wind"]["EDP"]["CERA5"][location]
    
    
    "--> Load IFORM contour values"
    # Contour for Hs vs Tp
    Contour_HsTp_Buoy = res_param["Resource"]["extremes"]["Waves"]["Contour"]["Buoy"][location]
    Contour_HsTp_ERA5 = res_param["Resource"]["extremes"]["Waves"]["Contour"]["ERA5"][location]
    Contour_HsTp_CERA5 = res_param["Resource"]["extremes"]["Waves"]["Contour"]["CERA5"][location]
    
    # Contour for Hs vs Tp
    Contour_HsUw_Buoy = res_param["Resource"]["extremes"]["Wind"]["Contour"]["Buoy"][location]
    Contour_HsUw_ERA5 = res_param["Resource"]["extremes"]["Wind"]["Contour"]["ERA5"][location]
    Contour_HsUw_CERA5 = res_param["Resource"]["extremes"]["Wind"]["Contour"]["CERA5"][location]
    
    "--> Load Extreme density/occurrency values corresponding to buoy data"
    # Hs&Tp based density values
    density_HsTp_df = pd.read_csv(rooth_path+'/Metocean_data/'+location+"_density_HsTp.csv.gz", compression="gzip")
    # Hs&Uw based density values
    density_HsUw_df = pd.read_csv(rooth_path+'/Metocean_data/'+location+"_density_HsUw.csv.gz", compression="gzip")

    
    '--> HsTp IFORM  plotting'
    fig1_HsTp, ax1_HsTp = plt.subplots(1, 1, figsize=(10, 8))
    fig1_HsTp.subplots_adjust(right=0.9)
    im1 = ax1_HsTp.scatter(density_HsTp_df["Tp_Obs"],density_HsTp_df["Hs_Obs"],c = density_HsTp_df["Density"],cmap='viridis', marker='o',label = "Buoy data")
    ax1_HsTp.plot(Contour_HsTp_CERA5["Tp"],Contour_HsTp_CERA5["Hs"],color = colors[0],linewidth = 3,label = "CERA5")
    ax1_HsTp.plot(Contour_HsTp_Buoy["Tp"],Contour_HsTp_Buoy["Hs"],color = colors[1],linewidth = 3,label = "Buoy")
    ax1_HsTp.plot(Contour_HsTp_ERA5["Tp"],Contour_HsTp_ERA5["Hs"],color = colors[2],linewidth = 3, label = "ERA5")
    
    ax1_HsTp.scatter(EDP_HsTp_CERA5["Tp_extrem"],EDP_HsTp_CERA5["Hs_extrem"],s = 150, color = "tomato" ,label = "EDP",marker='o',edgecolor = "k",zorder=5)
    ax1_HsTp.scatter(EDP_HsTp_ERA5["Tp_extrem"],EDP_HsTp_ERA5["Hs_extrem"],s = 150, color = "tomato" ,marker='o',edgecolor = "k",zorder=5)
    ax1_HsTp.scatter(EDP_HsTp_Buoy["Tp_extrem"],EDP_HsTp_Buoy["Hs_extrem"],s = 150, color = "tomato" ,marker='o',edgecolor = "k",zorder=5)
    ax1_HsTp.legend(loc="upper right", frameon=True)
    
    divider = make_axes_locatable(ax1_HsTp)
    cax1_HsTp = divider.append_axes('right', size='2%', pad=0.1)
    cb1_HsTp = fig1_HsTp.colorbar(im1, cax=cax1_HsTp, orientation='vertical')
    cb1_HsTp.set_label('Density [-]', fontname = "Times New Roman",fontsize=24,fontweight = 'bold')
    for t in cb1_HsTp.ax.get_yticklabels():
        t.set_fontname("Times New Roman")
        t.set_fontsize(22)
    ax1_HsTp.set_xlabel("$T_p$ [s]",fontname = "Times New Roman",fontweight="bold",fontsize = 26)
    ax1_HsTp.set_ylabel("$H_s$ [m]",fontname = "Times New Roman",fontweight="bold",fontsize = 26)
        
        
    '--> HsUw IFORM  plotting'
    
    fig1_HsUw, ax1_HsUw = plt.subplots(1,1, figsize=(10, 8))
    fig1_HsUw.subplots_adjust(right=0.9)
    im1 = ax1_HsUw.scatter(density_HsUw_df["Vhub_Obs"],density_HsUw_df["Hs_Obs"],c = density_HsUw_df["Density"],cmap='viridis', marker='o',label = "Buoy data")
    ax1_HsUw.plot(Contour_HsUw_CERA5["Vhub"],Contour_HsUw_CERA5["Hs"],color = colors[0],linewidth = 3,label = "CERA5")
    ax1_HsUw.plot(Contour_HsUw_Buoy["Vhub"],Contour_HsUw_Buoy["Hs"],color = colors[1],linewidth = 3,label = "Buoy")
    ax1_HsUw.plot(Contour_HsUw_ERA5["Vhub"],Contour_HsUw_ERA5["Hs"],color = colors[2],linewidth = 3, label = "ERA5")
    
    ax1_HsUw.axvline(x=EDP_HsUw_CERA5["Uw_extrem"].values, linestyle='--',color=colors[0])
    ax1_HsUw.axhline(y=EDP_HsUw_CERA5["Hs_extrem"].values,linestyle='--',color=colors[0])
    ax1_HsUw.scatter(EDP_HsUw_CERA5["Uw_extrem"],EDP_HsUw_CERA5["Hs_extrem"],s = 150, color = "tomato" ,marker='o',edgecolor = "k",zorder=5)
    
    
    ax1_HsUw.axvline(x=EDP_HsUw_ERA5["Uw_extrem"].values, linestyle='--',color=colors[2])
    ax1_HsUw.axhline(y=EDP_HsUw_ERA5["Hs_extrem"].values,linestyle='--',color=colors[2])
    ax1_HsUw.scatter(EDP_HsUw_ERA5["Uw_extrem"],EDP_HsUw_ERA5["Hs_extrem"],s = 150, color = "tomato" ,marker='o',edgecolor = "k",zorder=5)
    
    ax1_HsUw.axvline(x=EDP_HsUw_Buoy["Uw_extrem"].values, linestyle='--',color=colors[1])
    ax1_HsUw.axhline(y=EDP_HsUw_Buoy["Hs_extrem"].values,linestyle='--',color=colors[1])
    ax1_HsUw.scatter(EDP_HsUw_Buoy["Uw_extrem"],EDP_HsUw_Buoy["Hs_extrem"],s = 150, color = "tomato",label = "EDP" ,marker='o',edgecolor = "k",zorder=5)
    ax1_HsUw.legend(loc="upper left", frameon=True)
        
    divider = make_axes_locatable(ax1_HsUw)
    cax1_HsUw = divider.append_axes('right', size='2%', pad=0.1)
    cb1_HsUw = fig1_HsUw.colorbar(im1, cax=cax1_HsUw, orientation='vertical')
    cb1_HsUw.set_label('Density [-]', fontname = "Times New Roman",fontsize=24,fontweight = 'bold')
    for t in cb1_HsUw.ax.get_yticklabels():
        t.set_fontname("Times New Roman")
        t.set_fontsize(22)
    ax1_HsUw.set_xlabel("$U_{w-10}$ [m/s]",fontname = "Times New Roman",fontweight="bold",fontsize = 26)
    ax1_HsUw.set_ylabel("$H_s$ [m]",fontname = "Times New Roman",fontweight="bold",fontsize = 26)
    
    
    '--> Save plots'
    path_fig = rooth_path+"/Results/Figures/Resource_assessment_indicators/"
    os.makedirs(path_fig, exist_ok=True)
    
    fig1_HsTp.savefig(path_fig+location+"_IFORM_HsUw.png", format='png',dpi=500,transparent = True)     
    fig1_HsUw.savefig(path_fig+location+"_IFORM_HsUw.png", format='png',dpi=500,transparent = True)
        

def wawi_energy_flux(met_data,location,rooth_dir,plot_flag):
    
    '--> Adjust plotting settings'
    cmap = cm.get_cmap("viridis")
    color_range = np.linspace(0.2, 1, 3)  # Select colors between 20% and 80% of the colormap
    colors = [cmap(value) for value in color_range]  # Extract specific colors
    
    '--> Load metocean data'
    #Buoy related
    Udir_obs =  met_data[location]["Udir_Obs"].values;Uv10_obs = met_data[location]["Uv10_Obs"].values
    Wdir_obs = met_data[location]["Wdir_Obs"].values;Hs_obs = met_data[location]["Hs_Obs"].values;Tp_obs = met_data[location]["Tp_Obs"].values
    # ERA5 related
    Udir_sim =  met_data[location]["Udir_Sim"].values;Uv10_sim = met_data[location]["Uw_Sim"].values
    Wdir_sim = met_data[location]["Wdir_Sim"].values;Hs_sim = met_data[location]["Hs_Sim"].values;Tp_sim = met_data[location]["Tp_Sim"].values
    #CERA5 related
    Udir_calib =  met_data[location]["Udir_Obs"].values;Uv10_calib = met_data[location]["Uv10_Calib"].values    
    Wdir_calib = met_data[location]["Wdir_Obs"].values;Hs_calib = met_data[location]["Hs_Calib"].values;Tp_calib = met_data[location]["Tp_Calib"].values
    
    
    "--> Directional flux calculation"
    # Wave flux
    flux_wave_obs = 0.49*Hs_obs**2*Tp_obs*0.9
    flux_wave_sim = 0.49*Hs_sim**2*Tp_sim*0.9
    flux_wave_calib = 0.49*Hs_calib**2*Tp_calib*0.9
    # Wind flux
    rho = 1.225                                                                                       # Air density [kg/m3] 
    R = 120                                                                                           # Rotor radious                                                                   
    A = np.pi*R**2                                                                                    # Turbine rotor area
    flux_wind_obs = 0.5*rho*Uv10_obs**3/1000
    flux_wind_sim = 0.5*rho*Uv10_sim**3/1000
    flux_wind_calib = 0.5*rho*Uv10_calib**3/1000
    
    
    " Mean directional energy flux calculation"
    def mean_direction(weight, direction):
        theta = np.deg2rad(direction)
        return np.degrees(np.arctan2(np.sum(weight*np.sin(theta)), np.sum(weight*np.cos(theta))))%360

    wave_data = {"obs": (Hs_obs, Tp_obs, Wdir_obs), "sim": (Hs_sim, Tp_sim, Wdir_sim), "calib": (Hs_calib, Tp_calib, Wdir_calib)}
    wind_data = {"obs": (Uv10_obs, Wdir_obs), "sim": (Uv10_sim, Wdir_sim), "calib": (Uv10_calib, Wdir_calib)}
    
    flux_en_wave = {key: mean_direction(Hs**2*Tp, direction) for key, (Hs, Tp, direction) in wave_data.items()}
    flux_en_wind = {key: mean_direction(0.5*rho*A*U**3, direction) for key, (U, direction) in wind_data.items()}
    
    flux_wawi_df = pd.DataFrame({"MEFD_wind_obs": [flux_en_wind["obs"]], "MEFD_wind_sim": [flux_en_wind["sim"]], "MEFD_wind_calib": [flux_en_wind["calib"]], "MEFD_wave_obs": [flux_en_wave["obs"]], "MEFD_wave_sim": [flux_en_wave["sim"]], "MEFD_wave_calib": [flux_en_wave["calib"]]})
        
    
    '--> Accumulated energy per 15º directional sector'
    
    sector_width = 15                                                                                 # Width of each sector [degree]                                                                              
    edges = np.arange(0, 360 + sector_width, sector_width)
    centres = edges[:-1] + sector_width / 2    
    # Ensure directions are within [0, 360)
    Wdir_sim = np.asarray(Wdir_sim) % 360
    Wdir_obs = np.asarray(Wdir_obs) % 360
    
    sector_wave_energy_sim = np.zeros(len(centres))
    sector_wave_energy_obs = np.zeros(len(centres))
    sector_wind_energy_sim = np.zeros(len(centres))
    sector_wind_energy_obs = np.zeros(len(centres))
    for j in range(len(centres)):
        mask_wave_sim = (Wdir_sim >= edges[j]) & (Wdir_sim < edges[j+1])
        mask_wave_obs = (Wdir_obs >= edges[j]) & (Wdir_obs < edges[j+1])
        
        mask_wind_sim = (Udir_sim >= edges[j]) & (Udir_sim < edges[j+1])
        mask_wind_obs = (Udir_obs >= edges[j]) & (Udir_obs < edges[j+1])
        
        sector_wave_energy_sim[j] = np.sum(flux_wave_sim[mask_wave_sim])        
        sector_wave_energy_obs[j] = np.sum(flux_wave_obs[mask_wave_obs])
        sector_wind_energy_sim[j] = np.sum(flux_wind_sim[mask_wind_sim])        
        sector_wind_energy_obs[j] = np.sum(flux_wind_obs[mask_wind_obs])
    
    sector_wave_energy_sim =  sector_wave_energy_sim/1000       
    sector_wave_energy_obs = sector_wave_energy_obs/1000
    sector_wind_energy_sim = sector_wind_energy_sim/1000        
    sector_wind_energy_obs = sector_wind_energy_obs/1000
    
    '--> Create the Rose diagram of the accumulated energy flux'
    theta = np.deg2rad(centres)
    width = np.deg2rad(sector_width)  
    
    
    # Rose diagram of the accumulated wave flux
    fig_wave, ax_wave = plt.subplots(figsize=(8, 8),subplot_kw={"projection": "polar"})   
    ax_wave.bar(theta,sector_wave_energy_obs,width=width,align="center",alpha=0.4,color= colors [1],
    edgecolor="none",label="Observations")
    
    ax_wave.bar(theta,sector_wave_energy_sim,width=width,align="center",alpha=0.4, color= colors [2],
    edgecolor="none",label="CERA5")
    
    # Sector boundaries in radians
    theta_edges = np.deg2rad(edges)
    
    # Repeat first value to close the polar curve
    obs_outline_wave = np.r_[sector_wave_energy_obs, sector_wave_energy_obs[0]]
    sim_outline_wave = np.r_[sector_wave_energy_sim, sector_wave_energy_sim[0]]
    
    # Outer envelopes
    ax_wave.step(theta_edges,obs_outline_wave, where="post",linewidth=3,color= colors [1],
    label="_nolegend_")    
    ax_wave.step(theta_edges,sim_outline_wave,where="post",linewidth=3,color= colors [2],
    label="_nolegend_")
    
    ax_wave.set_theta_zero_location("N")   # 0° at North
    ax_wave.set_theta_direction(-1)        # clockwise
    ax_wave.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax_wave.set_xticklabels(["N", "E", "S", "W"])   
        
    locations = ["Bilbao Vizcaya","Cape Silleiro","Gulf of Cadiz","Cape Begur"]
    site_spec_yticks = {**{site: [100, 250, 500, 1000, 1700] for site in locations[:2]},
        "Gulf of Cadiz": [100, 250, 500],"Cape Begur": [100, 250, 400]}    
    
    ax_wave.set_ylim(0, max(site_spec_yticks[location]))
    ax_wave.set_yticks(site_spec_yticks[location])
    ax_wave.tick_params(axis="y", labelsize=34)
    ax_wave.legend(loc="upper right", bbox_to_anchor=(1.15, 1.10))
   
    for label in ax_wave.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(38)
    
    plt.tight_layout()
    plt.show()
    
    
    
    # Rose diagram of the accumulated wave flux    
    fig_wind, ax_wind = plt.subplots(figsize=(8, 8),subplot_kw={"projection": "polar"})
    ax_wind.bar(theta,sector_wind_energy_obs,width=width,align="center",alpha=0.4,color= colors [1],
    edgecolor="none",label="Observations")
    
    ax_wind.bar(theta,sector_wind_energy_sim,width=width,align="center",alpha=0.4, color= colors [2],
    edgecolor="none",label="CERA5")    
    # Sector boundaries in radians
    theta_edges = np.deg2rad(edges)    
    # Repeat first value to close the polar curve
    obs_outline_wind = np.r_[sector_wind_energy_obs, sector_wind_energy_obs[0]]
    sim_outline_wind = np.r_[sector_wind_energy_sim, sector_wind_energy_sim[0]]
    
    # Outer envelopes
    ax_wind.step(theta_edges,obs_outline_wind,where="post",linewidth=3,color= colors [1],
    label="_nolegend_")
    
    ax_wind.step(theta_edges,sim_outline_wind,where="post",linewidth=3,color= colors [1],
    label="_nolegend_")  
    ax_wind.set_theta_zero_location("N")   # 0° at North
    ax_wind.set_theta_direction(-1)        # clockwise
    ax_wind.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax_wind.set_xticklabels(["N", "E", "S", "W"])
    
    site_spec_yticks_wind = {"Bilbao Vizcaya": [2.5, 5, 7.5], "Cape Silleiro": [2.5,5,7.5,10],
        "Gulf of Cadiz": [1,2.5,5],"Cape Begur": [2.5,5,10,15,20,25]} 
    
    ax_wind.set_ylim(0, max(site_spec_yticks_wind[location]))    
    ax_wind.set_yticks(site_spec_yticks_wind[location])
    ax_wind.tick_params(axis="y", labelsize=34)
    ax_wind.legend(loc="upper right", bbox_to_anchor=(1.15, 1.10))   
    for label in ax_wind.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(38)
    
    plt.tight_layout()
    plt.show()  
    
    '--> Save plots'
    if plot_flag == True:
        path_fig = rooth_dir+"/Results/Figures/Resource_assessment_indicators/"
        os.makedirs(path_fig, exist_ok=True)
    
        fig_wave.savefig(path_fig+location+'_accum_waveflux_HsUw.svg', format='svg')   
        fig_wind.savefig(path_fig+location+'_accum_windflux_HsUw.svg', format='svg') 
        
    return flux_wawi_df





def scatter_diagram(met_data,res_param,init_y,end_y,location,plot_flag):
    
    
    '--> Hs&Tp scatter diagram'
    
    
    databases = ["Obs","Sim","Calib"]
    databases_v2 = ["Buoy", "ERA5","CERA5"]    
    var_pair = {"Wave":["Hs","Tp"],"WaveWind":["Hs","Uw"]}
    labels = {"Wave":["Wave significant height [m]","Peak wave period [s]"],"WaveWind":["Wave significant height [m]","Mean wind speed [m/s]"]}
    step_vec_var_wawi = {"Wave":1,"WaveWind":1}
    
    
    max_lim_Tp = max(met_data[["Tp_Obs"]].values)[0]-0.5
    max_lim_Hs = max(met_data[["Hs_Obs"]].values)[0]-0.5
    max_lim_Uv10 = max(met_data[["Uv10_Obs"]].values)[0]-0.5
    
    max_met_val = {"Wave":[max_lim_Hs,max_lim_Tp],"WaveWind":[max_lim_Hs,max_lim_Uv10]}
    min_met_val = {"Wave": [0.25,0.5],"WaveWind":[0,3.5]}
    
    years = end_y-init_y
    
    for i_dat,data in enumerate(databases):
        for var_p in var_pair:
            if var_p == "WaveWind" and data == "Obs":                
                select_var = ["Hs","Uv10"]
                
            elif var_p == "WaveWind" and data == "Calib":
                select_var = ["Hs","Uv10"]
                
            else:
                select_var = var_pair[var_p]
                
            var_1_max = max_met_val[var_p][1]
            var_2_max = max_met_val[var_p][0]
            var_1 = met_data[select_var[1]+'_'+data].values
            var_2 = met_data[select_var[0]+'_'+data].values      
    
    
            step_vec_var_1 = 1
            step_vec_var_2 = 0.5
    
    
            '---> Avoid 0 values in discretized arrays '
            min_var_1 = round(min(var_1))
            if min_var_1 == 0:
                min_var_1 = min_var_1+step_vec_var_1
            else:
                min_var_1 = min_var_1            
            min_var_2 = round(min(var_2))    
            
            if min_var_2 == 0:
                min_var_2 = min_var_2+step_vec_var_2
            else:
                min_var_2 = min_var_2
            
            min_var_1 = min_met_val[var_p][1]
            min_var_2 = min_met_val[var_p][0]
        
            '---> Discretise the variables '
            
            bins_var_1 = np.arange(min_var_1, var_1_max+step_vec_var_1, step_vec_var_1)
            ind_var_1 = np.digitize(var_1, bins_var_1, right=True)
            ind_var_1 = np.clip(ind_var_1, 1, len(bins_var_1))
            var_1_disc = bins_var_1[ind_var_1 - 1]   
            var_1_array = np.arange(min(var_1_disc),max(var_1_disc)+step_vec_var_1,step_vec_var_1)
            
            #bins_var_2 = np.arange(min_var_2, round(max(var_2),1)+step_vec_var_2, step_vec_var_2)
            bins_var_2 = np.arange(min_var_2, var_2_max+step_vec_var_2, step_vec_var_2)
            ind_var_2 = np.digitize(var_2, bins_var_2,right = True)
            ind_var_2 = np.clip(ind_var_2, 1, len(bins_var_2))
            var_2_disc = bins_var_2[ind_var_2 - 1]
            var_2_array = np.arange(min(var_2_disc),max(var_2_disc)+step_vec_var_2,step_vec_var_2)
        
    
            '---> Limit maximum values in discretized values '
            user_defined_limit = True
            if user_defined_limit == True:
                
                var_1_contour_axis = np.arange(0, round(var_1_max,1)+step_vec_var_1, step_vec_var_1)
                var_2_contour_axis = np.arange(0, round(var_2_max,1)+step_vec_var_2, step_vec_var_2)
            
            elif user_defined_limit == False:
                
                var_1_contour_axis = np.arange(0, round(max(var_1),1)+step_vec_var_1, step_vec_var_1)
                var_2_contour_axis = np.arange(0, round(max(var_2),1)+step_vec_var_2, step_vec_var_2)
        
            '--> Construct the scatter diagram'
            histogram, xedges, yedges = np.histogram2d(var_1_disc, var_2_disc, bins=[var_1_contour_axis, var_2_contour_axis])
            x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
            y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
            pos_max = np.unravel_index(np.argmax(histogram),histogram.shape)
            Proba = np.transpose(histogram)/len(var_2)
            idx = len(yedges)-2    
            Z = []
            while (idx >= 0):
              Z.append(Proba[idx])
              idx = idx - 1
              
            if plot_flag == True:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                plt.imshow(Z, extent=(np.amin(var_1), np.amax(var_1), np.amin(var_2), np.amax(var_2)),
                          norm=LogNorm(), interpolation = 'gaussian')     
                plt.colorbar().set_label('Probability')
                #plt.scatter(Tp_a, Hs_a, marker = "d", color="w")
                plt.xlabel(labels[var_p][1])
                plt.ylabel(labels[var_p][0])
                
            scatter_values = {}
            scatter_values["Prob_matrix"] = Proba
            scatter_values[var_pair[var_p][0]+"_disc_array"] = y_bin_centers
            scatter_values[var_pair[var_p][1]+"_disc_array"] = x_bin_centers
            
            res_param["Resource"]["Scatter_"+var_p][databases_v2[i_dat]][location] = scatter_values    
           
         
    return res_param



def uncert_EDP(res_param,location):
    
    '--> Calculate percentual relative errors related to EDP'
    databases = ["ERA5", "CERA5"]
    var_pair = {"Wave": ["Hs", "Tp"], "WaveWind": ["Hs", "Uw"]}
    
    columns = ["Wave_Hs", "Wave_Tp", "WaveWind_Hs", "WaveWind_Uw"]
    PRE_EDP_df = pd.DataFrame(index=databases, columns=columns, dtype=float)
    
    for data in databases:
        for var_p, variables in var_pair.items():
    
            # Buoy-related Extreme Design Point
            EDP_buoy = res_param["Resource"]["extremes"][var_p]["EDP"]["Buoy"][location]
    
            # ERA5- or CERA5-related Extreme Design Point
            EDP_assess_data = res_param["Resource"]["extremes"][var_p]["EDP"][data][location]
    
            # Percentage relative error
            PRE_EDP = abs(EDP_buoy - EDP_assess_data) / EDP_buoy * 100
    
            # Convert the two EDP-coordinate errors into a 1D array
            PRE_EDP_values = np.asarray(PRE_EDP).flatten()
    
            # Store each coordinate separately
            for i_var, variable in enumerate(variables):
                PRE_EDP_df.at[data, f"{var_p}_{variable}"] = PRE_EDP_values[i_var]
            
    return PRE_EDP_df


def uncert_flux(res_param,location):
    
    '--> Calculate percentual relative errors related to EDP'
    databases = ["sim","calib"]
    databases_v2 = ["ERA5", "CERA5"]
    var_pair = {"wave": ["Hs", "Tp"], "wind": ["Hs", "Uw"]}
    
    # Dataframe containing Percentual Directional Discrepancy
    PDD_flux_df = pd.DataFrame(index=databases_v2, columns=var_pair.keys(), dtype=float)
        
    for i_dat, data in enumerate(databases):
        for var_p, variables in enumerate(var_pair.items()):
    
            # Buoy-related Extreme Design Point
            flux_buoy = res_param["Resource"]["flux_energy"][location]["MEFD_"+variables[0]+"_obs"]
    
            # ERA5- or CERA5-related Extreme Design Point
            flux_assess_data = res_param["Resource"]["flux_energy"][location]["MEFD_"+variables[0]+'_'+data]
    
            # Percentage relative error
            delta = np.deg2rad(flux_assess_data- flux_buoy)
            error_deg = abs(np.degrees(np.arctan2(np.sin(delta), np.cos(delta))))   

            # Directional percentual discrepancy           
            PDD_flux_df.loc[databases_v2[i_dat], variables[0]] = error_deg[0] / 180 * 100
          
    return PDD_flux_df

def uncert_occur(res_param,location,plot_flag,root_dir):
    
    
    
    '--> Calculate percentual relative errors related to EDP'
    databases = ["ERA5", "CERA5"]
    databases_v2 = ["ERA5", "CERA5"]
    var_pair = {"Wave": ["Hs", "Tp"], "WaveWind": ["Hs", "Uw"]}
    var_pair_v2 = {"Wave": ["Hs [m]", "Tp [s]"], "WaveWind": ["Hs [m]", "Uw [m/s]"]}
    # Dataframe containing Percentual Directional Discrepancy
    Hell_percent_df = pd.DataFrame(index=databases_v2, columns=var_pair.keys(), dtype=float)
        
    for i_dat, data in enumerate(databases):
        for var_p, variables in enumerate(var_pair.items()):
            
            buoy_wave_scat = res_param["Resource"]["Scatter_"+variables[0]]["Buoy"][location]["Prob_matrix"]
            buoy_wave_scat_nan = buoy_wave_scat.copy()
            buoy_wave_scat_nan[buoy_wave_scat_nan == 0.0] = np.nan
            
            assess_data_scat = res_param["Resource"]["Scatter_"+variables[0]][data][location]["Prob_matrix"]
            
            error_with_zero = assess_data_scat - buoy_wave_scat
            error_with_nan = assess_data_scat - buoy_wave_scat_nan
            
            # Joint PDF discrepancies based on Hellinger distance            
            diff_Hell = np.sqrt(buoy_wave_scat) - np.sqrt(assess_data_scat)
            Hell = (1.0 / np.sqrt(2.0)) * np.sqrt(np.sum(diff_Hell**2))
            Hell_per = Hell*100   
            
            Hell_percent_df.loc[databases_v2[i_dat], variables[0]] = Hell_per
            
            if plot_flag == True: 
                
                idx_wave = len(res_param["Resource"]["Scatter_"+variables[0]][data][location]["Hs_disc_array"])+1-2    
                Z_matrix_zero = [];Z_matrix_nan = []
                
                while (idx_wave >= 0):
                  Z_matrix_zero.append(error_with_zero[idx_wave]);Z_matrix_nan.append(error_with_nan[idx_wave])              
                  idx_wave = idx_wave - 1
                  
                X_array = res_param["Resource"]["Scatter_"+variables[0]][data][location][variables[1][1]+"_disc_array"]
                Y_array = res_param["Resource"]["Scatter_"+variables[0]][data][location]["Hs_disc_array"]
                X_label = variables[1][1]
                Y_label = variables[1][0]         
                
                cmap = plt.cm.coolwarm.copy()
                cmap.set_bad(color='lightgrey')   # colour for NaN regions   
                
                fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
                fig1.subplots_adjust(right=0.85)
                im1 = ax1.imshow(Z_matrix_zero,cmap = cmap, extent=(np.amin(X_array), np.amax(X_array), 
                                 np.amin(Y_array), np.amax(Y_array)), norm=None, interpolation = 'gaussian',label = data+'-Buoy')  
                im1.set_clim(0.03, -0.03)
                divider = make_axes_locatable(ax1)
                ax1.set_aspect(aspect = 'auto')  
                cax1 = divider.append_axes('right', size='3%', pad=0.2)
                cb1 = fig1.colorbar(im1, cax=cax1, orientation='vertical')
                cb1.set_label('Probability [-]', fontname = "Times New Roman",fontsize=24,fontweight = 'bold')
                for t in cb1.ax.get_yticklabels():
                    t.set_fontname("Times New Roman")
                    t.set_fontsize(26)
                ax1.set_xlabel(var_pair_v2[variables[0]][1],fontname = "Times New Roman",fontweight="bold",fontsize = 26)
                ax1.set_ylabel(var_pair_v2[variables[0]][0],fontname = "Times New Roman",fontweight="bold",fontsize = 26) 
                
                nan_mask = np.isnan(Z_matrix_nan)
                nan_mask_plot = np.flipud(nan_mask)
                plt.rcParams['hatch.color'] = 'black'
                ax1.contourf( X_array,Y_array,nan_mask_plot, levels=[0.5, 1],colors='white',alpha = 1)
                
            
                path_fig = root_dir+"/Results/Figures/Resource_assessment_indicators/"
                os.makedirs(path_fig, exist_ok=True)            
                fig1.savefig(path_fig+location+'_'+data+'Buoy_occurrence_error_'+variables[0]+'.svg', format='svg')   
    return Hell_percent_df
                
                

                 
              
    
    

             
            
            
    
    
    
       
    
    