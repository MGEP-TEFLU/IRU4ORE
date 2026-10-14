# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:52:02 2026
Module name: [Metocean_uncertainty_impact_assessor]
Author: [Ander Martinez, Markel Penalba]
Affiliation: [Mondragon University]
Contact: [amartinezpe@mondragon.edu]



Description:
    
This module assesses how metocean-data uncertainty propagates into the resource
and design assessment of Offshore Renewable Energy (ORE) systems.

ERA5 reanalysis data and buoy observations from Puertos del Estado are considered.
The buoy data are used as reference observations to bias-correct ERA5 through
Directionally Adjusted Gumbel Quantile Mapping (DAGQM), producing the corrected
dataset CERA5.

ERA5, buoy and CERA5 data are then used to estimate resource, performance and
design metrics, including joint metocean distributions, wave and wind energy
fluxes, extreme design points, power generation and mean waiting time. Results
from ERA5 and CERA5 are compared against the buoy-based estimates to quantify
discrepancies and evaluate the improvements achieved through DAGQM correction.

The case study considers four locations around the Iberian Peninsula and two ORE
technologies: the RM3 wave energy converter (WEC) and the IEA-15-240-RWT floating
offshore wind turbine (FOWT).
  
"""


'--> Dependencies'

# Python libraries
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from windrose import WindroseAxes
# import matplotlib.cm as cm
# import statistics
# from matplotlib import rcParams
# from tqdm import tqdm

# Personal functions
from resource_main_IRU4ORE import *
from design_main_IRU4ORE import *
plt.close("all")

'*****************************************************************************'
'                       LOAD & ARRANGE METOCEAN DATA                          '
'Metocean data availabel at "Metocean_data" folder containinig data from all  '
'the locations and buoy, ERA5 and CERA5 data is uploaded to met_data dic      '
'variable.'
'*****************************************************************************'

'--> Define the directory'
def real_case_path(path):
    path = Path(path)
    parts = path.parts
    current = Path(parts[0])

    for part in parts[1:]:
        matches = [p for p in current.iterdir() if p.name.lower() == part.lower()]
        current = matches[0] if matches else current / part

    return current

Py_cod_dir = real_case_path(os.path.dirname(os.path.abspath(__file__)))
root_dir = Py_cod_dir.parent
root_dir = str(root_dir)


'--> Define the buoy names and temporal coverage'
locations = ["Bilbao Vizcaya","Cape Silleiro"]#,"Gulf of Cadiz","Cape Begur"]     # Selected buoy names
init_y = [1990,2003,2003,2003]                                                  # Initial year of the temporal coverage of each buoy
end_y = [2020,2023,2023,2023]                                                  # Initial year of the temporal coverage of each buoy


'--> Load metocean data'
met_data = get_met_data(init_y,end_y,locations,root_dir)


'*****************************************************************************'
'                         RESOURCE INDICATORS                                 '
'Metocean data uploaded into met_data is input into the res_in_calc to compute'
'the resource indicators.                                                     '
'*****************************************************************************'

'Indicators calculation'
#res_param = res_in_calc(met_data,locations,init_y,end_y,root_dir)

'Uncertainty in resource indicators calculation'
#res_param = res_ind_uncert_cal(res_param,locations,root_dir)

'Save resource indicators and uncertainty values'
#save_res_ind(res_param,locations,root_dir)
#save_un_res(res_param,locations,root_dir)





'*****************************************************************************'
'                         DESIGN INDICATORS                                   '
'Metocean data saved into met_data and resource indicators calculated previo- '
'-usly are inputed into des_in_calc to compute the design indicators.         '
'*****************************************************************************'

'Indicators calculation'
des_param = des_in_calc(met_data,0,locations,init_y,end_y,root_dir)

'Uncertainty in design indicators calculation'
des_param = des_in_uncert_calc(des_param,locations,root_dir)
            
