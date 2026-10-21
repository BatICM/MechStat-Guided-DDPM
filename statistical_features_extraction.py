"""Extract statistical features for the 24-week DDPM workflow.

Run this script first, followed by mechanistic_features_extraction.py and DDPM.py.
Set RAW_DATA_DIR to the folder with the 20 vehicle CSV files. DATA_DIR stores:
    All_20vehs_data.pkl              Charging segments and capacity estimates.
    complete_point_features_base.pkl Full weekly capacity/feature trajectories.
    point_features_base.pkl         Selected features for the first 24 weeks.

The complete trajectories are retained as DDPM targets; only the conditioning
features are limited to 24 weeks. No other history-length variants are generated.
An existing segment cache is reused; remove it to reprocess changed source data.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import re
from datetime import timedelta
from scipy.integrate import simpson
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# %% Paths and settings
PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# Put the 20 source CSV files directly in RAW_DATA_DIR, or edit this path.
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
DATA_DIR = PROJECT_DIR / "data"
HISTORY_WEEKS = 24
EXPECTED_VEHICLES = 20
SELECTED_FEATURE_COUNT = 15
PLOT_FEATURES = False
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% Extract charging segments and capacity
def find_samples_in_file(file):
    cha = file.copy()
    cha.reset_index(drop=True, inplace=True)
    cha['record_time'] = pd.to_datetime(cha['record_time'].astype(str), errors='coerce')
    time_delta = cha['record_time'].diff().fillna(pd.Timedelta(seconds=0))
    interval = timedelta(seconds=60)
    rest_index = time_delta[time_delta > interval].index
    cha_list = []
    prev_index = 0
    for idx in rest_index:
        cha_list.append(cha.iloc[prev_index:idx])
        prev_index = idx
    cha_list.append(cha.iloc[prev_index:])
    Ca_list = []
    cha_list_out = []
    for j in range(len(cha_list)):
        cha_cut = cha_list[j]
        cha_cut.reset_index(drop=True, inplace=True)
        if len(cha_cut)<100:
            continue
        dif_soc = cha_cut['soc'][1:] - cha_cut['soc'][:-1]
        if np.sum(dif_soc>2) or np.sum(dif_soc<-0.1):
            continue
        if np.min(cha_cut.soc)>50 or np.max(cha_cut.soc)<95:
            continue
        cha_cut=cha_cut[cha_cut.soc>50]

        time = cha_cut['record_time'].values
        current = cha_cut['current'].values
        soc = cha_cut['soc'].values
        Tmax= np.mean(cha_cut['max_temperature'])
        Tmin= np.mean(cha_cut['min_temperature'])
        label_Ca = real_capacity_cal(time,current,soc)
        if label_Ca<=0:
            continue
        Ca_list.append([time[0],time[-1],soc[0],soc[-1],label_Ca,Tmax,Tmin])
        cha_list_out.append(cha_list[j])
    return Ca_list,cha_list_out

def real_capacity_cal(time_data,current,SOC_data):
    if np.isnan(current).mean() > 0.1:
        return 0
    current = np.where(np.isnan(current), np.nan_to_num(current, nan=np.nan), current)
    current = np.where(np.isnan(current), np.roll(current, 1), current)
    time_sec=(time_data - time_data[0])/np.timedelta64(1, 's')
    accumulated_Q=simpson(current, x=time_sec)/3600*(-1)
    delta_SOC=SOC_data[-1]-SOC_data[0]
    if delta_SOC==0:
        return 0
    label_Ca = accumulated_Q/delta_SOC*100
    return label_Ca

def func(file_path):
    src_path = file_path
    file = pd.read_csv(src_path)
    file.columns = ['number','record_time','soc','pack_voltage','current','max_cell_voltage',
                    'min_cell_voltage','max_temperature','min_temperature','available_energy','available_capacity']
    file = file.sort_values(by='record_time')
    file.reset_index(drop=True, inplace=True)
    ca_list,cha_data_list = find_samples_in_file(file)
    return ca_list,cha_data_list

def load_or_process_data(main_path, save_path=None, force_reload=False):
    """Load the segment cache or extract data from naturally ordered CSV files."""
    save_path = Path(save_path) if save_path is not None else DATA_DIR / "All_20vehs_data.pkl"
    if save_path.exists() and not force_reload:
        print("Loading preprocessed data from pkl file...")
        with open(save_path, 'rb') as f:
            data_list = pickle.load(f)
        return data_list

    csv_paths = sorted(
        Path(main_path).glob("*.csv"),
        key=lambda p: [int(part) if part.isdigit() else part.casefold()
                       for part in re.split(r"(\d+)", p.name)],
    )
    if len(csv_paths) != EXPECTED_VEHICLES:
        raise ValueError(
            f"Expected {EXPECTED_VEHICLES} CSV files in {main_path}, found {len(csv_paths)}. "
            "Set RAW_DATA_DIR to the folder containing the vehicle CSV files."
        )
    data_list = []
    for index, file_path in enumerate(csv_paths):
        print(f"Loading vehicle {index + 1}: {file_path.name}")
        vehicle_data = func(file_path)
        if not vehicle_data[0]:
            raise ValueError(f"No usable charging segments found in {file_path.name}.")
        data_list.append(vehicle_data)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data_list, f)
    print("Saved processed data to", save_path)
    return data_list

data_list = load_or_process_data(RAW_DATA_DIR)
if len(data_list) != EXPECTED_VEHICLES:
    raise ValueError(f"Expected {EXPECTED_VEHICLES} vehicles, found {len(data_list)} in the cache.")

# %% Extract statistical features for each charging segment
def voltage_monotonicity(input_data):
    for i in range(1,len(input_data)):
        if input_data[i] <= input_data[i-1]:
            input_data[i] = input_data[i-1] + 1e-10
    return input_data

def voltage_monotonicity_de(input_data):
    for i in range(len(input_data)-1):
        if input_data[-1-i-1] >= input_data[-1-i]:
            input_data[-1-i-1] = input_data[-1-i] - 1e-5
    return input_data

def custom_interp1d(x, y):

    linear_interp = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")

    def interp_with_nearest(x_new):

        y_new = linear_interp(x_new)

        x_min, x_max = np.min(x), np.max(x)

        y_new[x_new < x_min] = y[0]
        y_new[x_new > x_max] = y[-1]
        return y_new
    return interp_with_nearest

def HIs_extract(cha_list):
    Vend = 4.15
    Vstart = 4.05
    HIs = []
    Num = 101
    V_seq = np.linspace(Vstart, Vend, Num)
    for cha_seg in cha_list:
        bat_V = voltage_monotonicity(cha_seg.pack_voltage.values / 90)

        cha_seg['record_time'] = pd.to_datetime(cha_seg['record_time'])
        t_tmp = cha_seg['record_time'].values[1:] - cha_seg['record_time'].values[:-1]
        t = t_tmp.astype('timedelta64[s]').astype(float)
        bat_Q = np.zeros(len(bat_V))
        bat_Q[1:] = np.cumsum(cha_seg.current.values[:-1] * t * (-1) / 3600)

        fun1 = custom_interp1d(bat_V, bat_Q)
        cha_Q = np.asarray(fun1(V_seq), 'float64')

        fun2 = custom_interp1d(bat_V, (cha_seg.max_temperature.values + cha_seg.min_temperature.values) / 2)
        cha_T = np.asarray(fun2(V_seq), 'float64')

        fun3 = custom_interp1d(bat_V, cha_seg.max_cell_voltage.values - cha_seg.min_cell_voltage.values)
        cha_Vdiff = np.asarray(fun3(V_seq), 'float64')

        fun4 = custom_interp1d(bat_V, cha_seg.max_temperature.values - cha_seg.min_temperature.values)
        cha_Tdiff = np.asarray(fun4(V_seq), 'float64')

        I_mean = np.mean(cha_seg.current.values)
        I_std = np.std(cha_seg.current.values)
        I_max = np.max(cha_seg.current.values)
        I_min = np.min(cha_seg.current.values)

        Vpack_mean = np.mean(cha_seg.pack_voltage.values)
        Vpack_std = np.std(cha_seg.pack_voltage.values)
        Vpack_sum = np.sum(cha_seg.pack_voltage.values)

        Vmax_mean = np.mean(cha_seg.max_cell_voltage.values)
        Vmax_std = np.std(cha_seg.max_cell_voltage.values)
        Vmax_sum = np.sum(cha_seg.max_cell_voltage.values)

        Vmin_mean = np.mean(cha_seg.min_cell_voltage.values)
        Vmin_std = np.std(cha_seg.min_cell_voltage.values)
        Vmin_sum = np.sum(cha_seg.min_cell_voltage.values)

        Tmax_mean = np.mean(cha_seg.max_temperature.values)
        Tmax_std = np.std(cha_seg.max_temperature.values)
        Tmax_sum = np.sum(cha_seg.max_temperature.values)

        SOC_mean = np.mean(cha_seg.soc.values)
        SOC_std = np.std(cha_seg.soc.values)
        SOC_sum = np.sum(cha_seg.soc.values)

        if len(cha_seg.soc.values) > 1:
            SOC_rate = (cha_seg.soc.values[-1] - cha_seg.soc.values[0]) / len(cha_seg.soc.values)
        else:
            SOC_rate = 0

        HIs.append([

            np.mean(cha_Q - cha_Q[0]), np.std(cha_Q - cha_Q[0]), np.mean(cha_T), np.std(cha_T),
            np.mean(cha_T - cha_T[0]), np.std(cha_T - cha_T[0]),
            np.mean(cha_Vdiff), np.std(cha_Vdiff), np.mean(cha_Tdiff), np.std(cha_Tdiff),
            np.mean(cha_Tdiff - cha_Tdiff[0]), np.std(cha_Tdiff - cha_Tdiff[0]),

            I_mean, I_std, I_max, I_min,
            Vpack_mean, Vpack_std, Vpack_sum,
            Vmax_mean, Vmax_std, Vmax_sum,
            Vmin_mean, Vmin_std, Vmin_sum,
            Tmax_mean, Tmax_std, Tmax_sum,
            SOC_mean, SOC_std, SOC_sum, SOC_rate
        ])
    return HIs

# %% Aggregate all statistical features by week
columns = ['Ca','Q_mean','Q_std','T_mean','T_std','Tr_mean','Tr_std','Vd_mean','Vd_std','Td_mean','Td_std','Tdr_mean','Tdr_std',
           'I_mean','I_std','I_max','I_min',
           'Vpack_mean','Vpack_std','Vpack_sum',
           'Vmax_mean','Vmax_std','Vmax_sum',
           'Vmin_mean','Vmin_std','Vmin_sum',
           'Tmax_mean','Tmax_std','Tmax_sum',
           'SOC_mean','SOC_std','SOC_sum','SOC_rate']

veh_list = []

for veh in range(len(data_list)):

    veh_data = data_list[veh]
    ca_list = veh_data[0]
    cha_list = veh_data[1]

    HIs = HIs_extract(cha_list)

    veh_df = pd.DataFrame(data=ca_list, columns=['time_s', 'time_e', 'SOC_s', 'SOC_e', 'charge_capacity', 'Tmax', 'Tmin'])
    veh_df.reset_index(drop=True, inplace=True)

    veh_df['year'] = veh_df['time_e'].dt.year
    veh_df['week'] = veh_df['time_e'].dt.isocalendar().week

    veh_df['time'] = veh_df['year'].astype(str) + '-' + veh_df['week'].apply(lambda x: f"{x:02d}")

    veh_df['index'] = veh_df.index
    grouped = veh_df.groupby('time')

    HIs = np.array(HIs)

    ca_week = []
    HIs_week = []
    time_keys = []

    for time_key, group in grouped:
        ca_week.append(group['charge_capacity'].mean())
        week_indices = group.index
        HIs_week_data = HIs[week_indices]
        HIs_week.append([np.mean(HIs_week_data, axis=0), np.std(HIs_week_data, axis=0)])
        time_keys.append(time_key)

    HIs_ave = np.array([h[0] for h in HIs_week])
    ca_HIs = np.hstack([np.array(ca_week).reshape(-1, 1), HIs_ave])
    HIs_df = pd.DataFrame(ca_HIs, columns=columns)
    HIs_df.insert(0, 'time', time_keys)

    veh_list.append(HIs_df)

with open(DATA_DIR / "complete_point_features_base.pkl", 'wb') as f:
    pickle.dump(veh_list, f)

with open(DATA_DIR / "complete_point_features_base.pkl", 'rb') as f:
   veh_list = pickle.load(f)

columns = ['Ca',
           'I_mean','I_std','I_max','I_min',
           'Q_mean','Q_std',
           'SOC_mean','SOC_std','SOC_sum','SOC_rate',
           'T_mean','T_std','Tr_mean','Tr_std','Td_mean','Td_std','Tdr_mean','Tdr_std','Tmax_mean','Tmax_std','Tmax_sum',
           'Vd_mean','Vd_std','Vpack_mean','Vpack_std','Vpack_sum','Vmax_mean','Vmax_std','Vmax_sum','Vmin_mean','Vmin_std','Vmin_sum']

all_data = []
for i, veh_df in enumerate(veh_list):
    veh_df_copy = veh_df.copy()
    veh_df_copy['vehicle_id'] = i
    all_data.append(veh_df_copy)

combined_df = pd.concat(all_data, ignore_index=True)

combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
combined_df = combined_df.dropna()
feature_cols = [col for col in columns if col != 'Ca']

# Rank candidate features by absolute Pearson correlation.
correlations = {}
p_values = {}
capacity = combined_df['Ca'].values

for feature in feature_cols:
    feature_values = combined_df[feature].values
    corr_coef, p_val = pearsonr(capacity, feature_values)
    correlations[feature] = corr_coef
    p_values[feature] = p_val

corr_results = pd.DataFrame({
    'Feature': list(correlations.keys()),
    'Correlation': list(correlations.values()),
    'P_value': list(p_values.values()),
    'Abs_Correlation': [abs(x) for x in correlations.values()]
}).sort_values('Abs_Correlation', ascending=False)

selected_features = corr_results.head(SELECTED_FEATURE_COUNT)['Feature'].tolist()
print(f"\nSelected statistical features: {selected_features}")

pcc_abs_range = (corr_results.head(SELECTED_FEATURE_COUNT)['Abs_Correlation'].min(), corr_results.head(SELECTED_FEATURE_COUNT)['Abs_Correlation'].max())
print(f"Absolute PCC range of selected features: {pcc_abs_range}")

features_list = []
selected_columns = ['time', 'Ca'] + selected_features

for veh_df in veh_list:

    selected_df = veh_df[selected_columns].copy()
    features_list.append(selected_df)

print(f"\nCreated feature tables for {len(features_list)} vehicles.")
print(f"First feature table shape: {features_list[0].shape}")
print(f"Columns: {list(features_list[0].columns)}")

# %% Save the 24-week inputs for mechanistic feature extraction
veh_choose_week_list = [df.iloc[:HISTORY_WEEKS].copy() for df in features_list]
with open(DATA_DIR / "point_features_base.pkl", 'wb') as f:
    pickle.dump(veh_choose_week_list, f)

# %% Optional feature visualization
if PLOT_FEATURES:
    with open(DATA_DIR / "complete_point_features_base.pkl", 'rb') as f:
       veh_list = pickle.load(f)

    # %% Aggregate all statistical features by week
    columns = ['Ca','Q_mean','Q_std','T_mean','T_std','Tr_mean','Tr_std','Vd_mean','Vd_std','Td_mean','Td_std','Tdr_mean','Tdr_std',
               'I_mean','I_std','I_max','I_min',
               'Vpack_mean','Vpack_std','Vpack_sum',
               'Vmax_mean','Vmax_std','Vmax_sum',
               'Vmin_mean','Vmin_std','Vmin_sum',
               'Tmax_mean','Tmax_std','Tmax_sum',
               'SOC_mean','SOC_std','SOC_sum','SOC_rate']

    fig, axes = plt.subplots(9, 4, figsize=(20, 25))
    axes = axes.flatten()

    selected_vehicles = veh_list[:5]
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, feature in enumerate(columns):
        if i >= len(axes):
            break

        for veh_id, veh_df in enumerate(selected_vehicles):
            if feature in veh_df.columns:
                x = np.arange(len(veh_df))
                axes[i].plot(x, veh_df[feature], 'o-',
                            linewidth=2, markersize=3, alpha=0.8,
                            color=colors[veh_id], label=f'Vehicle {veh_id+1}')

        axes[i].set_xlabel('Week', fontsize=10)
        axes[i].set_ylabel(feature, fontsize=10)
        axes[i].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

        if i == 0:
            axes[i].legend(fontsize=8)

    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
