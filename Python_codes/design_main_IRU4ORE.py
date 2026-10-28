# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:49:44 2026
Module name: [Resource_indicators_calculator]
Author: [Ander Martinez, Markel Penalba]
Affiliation: [Mondragon University]
Contact: [amartinezpe@mondragon.edu]
"""


'--> Dependencies'
# Python libraries
import pandas as pd
import matplotlib.pyplot as plt
from design_fcn_IRU4ORE import *
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pickle
from pathlib import Path

def des_in_calc(met_data,res_param,locations,init_y,end_y,root_dir):   

    '--> Dic file where resource indicators are saved'    
    des_param = {}
    sources = ["ERA5", "Buoy", "CERA5"]
    ORE = ["WEC", "FOWT"]  
    des_param = {
        "Design": {
            "GenPow":{ "AEP":{},
                       "Pow":{},
                },
            "Access":{ 
                     "Op_type": ["CTV","FSV","HLV"],                                                                 # Types of vessesls considered
                     "Op_lim_vessel": pd.DataFrame({"Hs_lim_CTV":[2.5],"Hs_lim_FSV":[1.8],"Hs_lim_HLV":[1.5],        # Operational limits for the considered vessels 
                                  "Uw_lim_CTV":[30],"Uw_lim_FSV":[30],"Uw_lim_HLV":[25]}),                     
                     "Rep_time": pd.DataFrame({"Time_minor_CTV":[7],"Time_medium_FSV":[18],"Time_major_HLV":[116]}), # Repair time for each operation
                     'WT':{},
                     'WT_dic':{},
                      },
            "FLS":{"WEC":{},
                   "FOWT":{}},
            "ULS":{},
            
        },
        "Design_unc":{
            "EDP_unc":{},
            "Joint_prob_unc":{},
            "flux_unc":{},            
            
            }
    }
    
    '--> Per site and resource indicator calculation' 
    for i in range(len(locations)):
        
        '--> O.- Power production'
        des_param["Design"]["GenPow"]["AEP"][locations[i]],des_param["Design"]["GenPow"]["Pow"][locations[i]] = ORE_prod(met_data[locations[i]],locations[i],root_dir)
        '--> 1.- Accessibility'
        des_param["Design"]["Access"]["WT"][locations[i]],des_param["Design"]["Access"]["WT_dic"][locations[i]] = ORE_access(met_data[locations[i]],des_param,locations[i],root_dir)
        '--> 2.- Fatigue Limit State Analysis'
        des_param["Design"]["FLS"]["WEC"][locations[i]],des_param["Design"]["FLS"]["FOWT"][locations[i]] = ORE_FLS(met_data[locations[i]],des_param,locations[i],root_dir)
        '--> 3.- Ultimate Limit State Analysis'
        des_param["Design"]["ULS"][locations[i]] = ORE_ULS(des_param,locations[i],root_dir)
        
        plt.close("all")
    
    return des_param


def des_in_uncert_calc(des_param,locations,root_dir): 
    
    '--> Per site and design indicator uncertainty calculation'    
    '1.- Power production discrepancy'
    uncert_PP(des_param,locations,root_dir)   
        
    '2.- Waiting time discrepancy'
    des_param = uncert_MWT(des_param,locations,root_dir) 
        
   
    return des_param


# ============================================================
# Functions for saving design parameters related results
# ============================================================


def find_section(design, keys):
    for key in keys:
        design = design[key]
    return design

def prepare(df, kind):
    datasets = ['Buoy', 'ERA5', 'CERA5']
    if kind in ('Access_WT', 'GenPow_AEP'):
        return df.T.reindex(columns=datasets)
    if kind == 'FLS_WEC':
        return df.reindex(columns=datasets)
    if kind == 'FLS_FOWT':
        df = df.rename(columns={'DataSource': 'Data source', 'MeanDamage': 'Total damage'}).copy()
        df['Data source'] = df['Data source'].replace({'Obs': 'Buoy', 'Sim': 'ERA5', 'Calib': 'CERA5'})

        needed = ['Component', 'Data source', 'Total damage']
        if not set(needed).issubset(df.columns):
            raise ValueError(f'Missing FLS columns: {set(needed) - set(df.columns)}')

        result = df.pivot(index='Component', columns='Data source', values='Total damage')
        return result.reindex(columns=datasets)
    if kind == 'ULS':
        return df.reindex(columns=['max', 'mean', 'std'])
    raise ValueError(kind)

def save_des_ind(des_param,locations,root_dir):
    path_res = root_dir+"/Results/Design_indicators/"
    os.makedirs(path_res, exist_ok=True) 
    output_path = Path(path_res+"Design_indicators.xlsx")
    
    datasets = ['Buoy', 'ERA5', 'CERA5']
    sections = {'Access_WT': ('Access', 'WT'), 'FLS_FOWT': ('FLS', 'FOWT'), 'FLS_WEC': ('FLS', 'WEC'), 'GenPow_AEP': ('GenPow', 'AEP'), 'ULS': ('ULS',)}
    wb = Workbook()
    wb.remove(wb.active)
    dark_fill = PatternFill('solid', fgColor='183153')
    light_fill = PatternFill('solid', fgColor='DCE8F2')
    alternate_fill = PatternFill('solid', fgColor='F4F7FB')
    for sheet_name, keys in sections.items():
        data = find_section(des_param['Design'], keys)
        sites = [s for s in locations if s in data]
        ws = wb.create_sheet(sheet_name)
        if not sites:
            ws['A1'] = 'No data available'
            continue
        transformed = {s: prepare(data[s], sheet_name) for s in sites}
        row_index = transformed[sites[0]].index
        for site in sites[1:]:
            row_index = row_index.union(transformed[site].index, sort=False)
        header_cols = ['max', 'mean', 'std'] if sheet_name == 'ULS' else datasets
        row_names = list(row_index.names) if isinstance(row_index, pd.MultiIndex) else [row_index.name or 'Indicator']
        row_names = [name if name is not None else f'Identifier {j+1}' for j, name in enumerate(row_names)]
        nlabels = len(row_names)
        ws.cell(1, 1, sheet_name)
        for j, name in enumerate(row_names, 1):
            ws.cell(2, j, name)
        for site_i, site in enumerate(sites):
            first = nlabels + 1 + site_i * len(header_cols)
            last = first + len(header_cols) - 1
            ws.cell(1, first, site)
            if last > first:
                ws.merge_cells(start_row=1, start_column=first, end_row=1, end_column=last)
            for offset, col in enumerate(header_cols):
                ws.cell(2, first + offset, col)
        aligned = {site: transformed[site].reindex(row_index) for site in sites}
        for row_i, ix in enumerate(row_index, 3):
            labels = list(ix) if isinstance(ix, tuple) else [ix]
            for col_i, label in enumerate(labels, 1):
                ws.cell(row_i, col_i, str(label))
            for site_i, site in enumerate(sites):
                for col_i, col in enumerate(header_cols):
                    value = aligned[site].loc[ix, col]
                    if pd.isna(value):
                        continue
                    cell = ws.cell(row_i, nlabels + 1 + site_i * len(header_cols) + col_i, float(value))
                    cell.number_format = '0.0000E+00' if sheet_name.startswith('FLS_') else '#,##0.00'
                    cell.alignment = Alignment(horizontal='right')
        max_col = nlabels + len(sites) * len(header_cols)
        for row in ws.iter_rows(min_row=1, max_row=2, max_col=max_col):
            for cell in row:
                cell.fill = dark_fill if cell.row == 1 else light_fill
                cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF' if cell.row == 1 else '183153')
                cell.alignment = Alignment(horizontal='center', vertical='center')
        for row in ws.iter_rows(min_row=3, max_row=ws.max_row, max_col=max_col):
            for cell in row:
                if cell.row % 2 == 0:
                    cell.fill = alternate_fill
                if cell.column <= nlabels:
                    cell.font = Font(name='Calibri', size=10, color='183153')
        for col_i in range(1, max_col + 1):
            ws.column_dimensions[get_column_letter(col_i)].width = 26 if col_i <= nlabels else 18
        ws.row_dimensions[1].height = 30
        ws.row_dimensions[2].height = 25
        ws.freeze_panes = f'{get_column_letter(nlabels+1)}3'
        ws.sheet_view.showGridLines = False
        
    wb.save(output_path)
    
# ============================================================
# Functions for saving uncertainty results associated with design parameters
# ============================================================
def relative_percent(sim, ref):
    """Signed percentage discrepancy, undefined for a zero buoy reference."""
    sim, ref = float(sim), float(ref)
    return np.nan if not (np.isfinite(sim) and np.isfinite(ref)) or ref == 0 else 100.0 * abs(sim - ref) / ref

def access_unc(df):
    datasets = ['Buoy', 'ERA5', 'CERA5']
    comparisons = ['ERA5', 'CERA5']
    df = df.reindex(index=datasets, columns=['CTV', 'FSV', 'HLV'])
    result = {}
    for source in comparisons:
        errors = [relative_percent(df.at[source, vessel], df.at['Buoy', vessel]) for vessel in ['CTV', 'FSV', 'HLV']]
        # Require all three valid vessel categories to avoid a misleading partial average.
        result[source] = float(np.mean(np.abs(errors))) if np.all(np.isfinite(errors)) else np.nan
    return pd.DataFrame([result], index=pd.Index(['MWT'], name='Indicator'))

def fls_fowt_unc(df):
    datasets = ['Buoy', 'ERA5', 'CERA5']
    comparisons = ['ERA5', 'CERA5']

    # Adapt actual DataFrame structure
    df = df.rename(columns={'DataSource': 'Data source', 'MeanDamage': 'Total damage'}).copy()
    df['Data source'] = df['Data source'].replace({'Obs': 'Buoy', 'Sim': 'ERA5', 'Calib': 'CERA5'})

    needed = {'Component', 'Data source', 'Total damage'}
    if not needed.issubset(df.columns):
        raise ValueError(f'FLS FOWT missing columns: {needed - set(df.columns)}')

    # Retain Component only; Location is excluded
    table = df.pivot(index='Component', columns='Data source', values='Total damage').reindex(columns=datasets)

    # Signed percentage deviation relative to buoy measurements
    return pd.DataFrame({s: [relative_percent(v, r) for v, r in zip(table[s], table['Buoy'])] for s in comparisons}, index=table.index)

def fls_wec_unc(df):
    datasets = ['Buoy', 'ERA5', 'CERA5']
    comparisons = ['ERA5', 'CERA5']
    df = df.reindex(columns=datasets)
    result = pd.DataFrame({s: [relative_percent(v, r) for v, r in zip(df[s], df['Buoy'])] for s in comparisons}, index=df.index)
    result.index = pd.Index(['Moor_Fair'], name='Indicator')
    return result

def aep_unc(df):
    datasets = ['Buoy', 'ERA5', 'CERA5']
    comparisons = ['ERA5', 'CERA5']
    df = df.reindex(index=datasets, columns=['WEC', 'FOWT'])
    return pd.DataFrame({s: [relative_percent(df.at[s, tech], df.at['Buoy', tech]) for tech in ['WEC', 'FOWT']] for s in comparisons}, index=pd.Index(['WEC', 'FOWT'], name='Technology'))

def uls_unc(df):
    datasets = ['Buoy', 'ERA5', 'CERA5']
    comparisons = ['ERA5', 'CERA5']
    df = df.copy()

    # Convert index levels into columns if necessary
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name is not None:
        df = df.reset_index()

    # Standardise column names
    df.columns = df.columns.astype(str).str.strip()

    needed = {'data_source', 'variable', 'max'}
    if not needed.issubset(df.columns):
        raise ValueError(
            f'ULS missing columns: {needed - set(df.columns)}. '
            f'Available columns: {df.columns.tolist()}'
        )

    # Rearrange maximum loads by variable and dataset
    mx = df.pivot(index='variable', columns='data_source', values='max')
    mx = mx.reindex(columns=datasets)

    # Calculate signed relative uncertainty (%)
    result = pd.DataFrame({
        s: [relative_percent(v, r) for v, r in zip(mx[s], mx['Buoy'])]
        for s in comparisons
    }, index=mx.index)
    
    uls_labels = {
    'BladeR_InpBenMom [kN.m]': 'BLA_In-Ben',
    'BladeR_OutpBenMom [kN.m]': 'BLA_Out-Ben',
    'Moor1_EffTen [N]': 'MOOR_Fair-Ten (WEC)',
    'Moor1_EffTen [kN]': 'MOOR_Fair-Ten (FOWT)',
    'TowerB_InpBenMom [kN.m]': 'TOW_In-Ben',
    'TowerB_OutpBenMom [kN.m]': 'TOW_Out-Ben'
    }
    result.index = result.index.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    result = result.rename(index=uls_labels)
    result.index.name = 'Variable'

    return result

SECTION_FUNCTIONS = {
    'Access_WT_unc': (('Access', 'WT'), access_unc),
    'FLS_FOWT_unc': (('FLS', 'FOWT'), fls_fowt_unc),
    'FLS_WEC_unc': (('FLS', 'WEC'), fls_wec_unc),
    'GenPow_AEP_unc': (('GenPow', 'AEP'), aep_unc),
    'ULS_unc': (('ULS',), uls_unc),
}

def get_section(design, keys):
    for key in keys:
        design = design[key]
    return design

def make_uncertainty_tables(des_param,locations):
    tables = {}
    for sheet_name, (keys, function) in SECTION_FUNCTIONS.items():
        source = get_section(des_param['Design'], keys)
        transformed = {loc: function(source[loc]) for loc in locations if loc in source and source[loc] is not None}
        if not transformed:
            tables[sheet_name] = pd.DataFrame()
            continue
        tables[sheet_name] = pd.concat(transformed, axis=1)
    return tables

def save_un_des(des_param,locations, root_dir):
    
    path_res = root_dir+"/Results/Design_indicators/"
    os.makedirs(path_res, exist_ok=True) 
    output_path = Path(path_res+"Uncertainty_design_indicators.xlsx")
    
    comparisons = ['ERA5', 'CERA5']
    components = [('Blade root', 'Flapwise moment'), ('Tower base', 'Fore-aft moment'), ('Fairlead', 'Line 1')]
    
    tables = make_uncertainty_tables(des_param,locations)
    wb = Workbook()
    wb.remove(wb.active)
    navy = PatternFill('solid', fgColor='183153')
    pale = PatternFill('solid', fgColor='DCE8F2')
    for sheet_name, df in tables.items():
        ws = wb.create_sheet(sheet_name)
        if df.empty:
            ws['A1'] = 'No data available'
            continue
        nlabels = df.index.nlevels
        labels = list(df.index.names) if isinstance(df.index, pd.MultiIndex) else [df.index.name or 'Indicator']
        ws.cell(1, 1, sheet_name)
        for j, label in enumerate(labels, 1):
            ws.cell(2, j, label or f'Label {j}')
        for j, (site, dataset) in enumerate(df.columns, nlabels + 1):
            ws.cell(1, j, site)
            ws.cell(2, j, dataset)
        for row_i, (index, series) in enumerate(df.iterrows(), 3):
            keys = index if isinstance(index, tuple) else (index,)
            for col_i, key in enumerate(keys, 1):
                ws.cell(row_i, col_i, str(key))
            for col_i, value in enumerate(series, nlabels + 1):
                cell = ws.cell(row_i, col_i, None if pd.isna(value) else float(value))
                cell.number_format = '0.00"%"'  # Values already expressed in percentage points.
        for row_i, fill in ((1, navy), (2, pale)):
            ws.row_dimensions[row_i].height = 27 if row_i == 1 else 24
            for cell in ws[row_i]:
                cell.fill = fill
                cell.font = Font(bold=True, color='FFFFFF' if row_i == 1 else '183153')
                cell.alignment = Alignment(horizontal='center', vertical='center')
        site_start = nlabels + 1
        for site in dict.fromkeys(df.columns.get_level_values(0)):
            count = sum(c[0] == site for c in df.columns)
            if count > 1:
                ws.merge_cells(start_row=1, start_column=site_start, end_row=1, end_column=site_start + count - 1)
            site_start += count
        for j in range(1, nlabels + 1):
            ws.column_dimensions[get_column_letter(j)].width = 29
        for j in range(nlabels + 1, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(j)].width = 18
        ws.freeze_panes = f'{get_column_letter(nlabels + 1)}3'
        
    wb.save(output_path)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

