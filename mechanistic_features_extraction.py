"""Extract mechanistic features for the 24-week DDPM workflow.

Run statistical_features_extraction.py first. This script reads only:
    data/All_20vehs_data.pkl
    data/point_features_base.pkl

It writes the two 24-week inputs consumed by DDPM.py:
    data/point_features_rescombined_24weeks.pkl
    data/ic_tensor_24weeks.pt

Edit DATA_DIR to change the shared input/output folder. Diagnostic figures are
optional and disabled by default. No other history-length variants are required.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import torch
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter

# %% Paths and 24-week inputs
PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
HISTORY_WEEKS = 24
PLOT_FEATURES = False
DATA_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "All_20vehs_data.pkl", "rb") as f:
    data_list = pickle.load(f)
with open(DATA_DIR / "point_features_base.pkl", "rb") as f:
    veh_choose_week_list = pickle.load(f)
if len(data_list) != len(veh_choose_week_list):
    raise ValueError("Charging data and statistical features must contain the same vehicles.")
for vehicle_id, frame in enumerate(veh_choose_week_list):
    if len(frame) != HISTORY_WEEKS or frame.shape[1] != 17:
        raise ValueError(
            f"Vehicle {vehicle_id}: expected 24 rows and 17 columns (time, capacity, "
            "15 statistical features). Run statistical_features_extraction.py first."
        )

# %% Estimate resistance and construct compensated voltage curves
def calculate_resis_pack(current,voltage,step):
    """Return the resistance at the first qualifying current transition."""
    resis = []

    for i in range(len(current)-step):
        if abs(current[i]-current[i+1])>20 and abs(current[i])>60:
            resis=-1*(voltage[i+step]-voltage[i])/(current[i+step]-current[i])
            break
        else:
            resis=[]
    return resis

cha_time_list=[]
cha_cell_ca_list=[]
cha_cell_V_list=[]
Res_df_list=[]

for veh_num in range(len(data_list)):
    print(f"Processing vehicle {veh_num}")
    cha_list=data_list[veh_num][1].copy()
    ca_data = pd.DataFrame(data=data_list[veh_num][0].copy(),columns=['time_s','time_e','SOC_s','SOC_e','capacity','Tmax','Tmin'])
    Res_list=[]

    for cha_seg in cha_list:
        Res_tmp=[]
        Res_tmp.append(calculate_resis_pack(cha_seg.current.values,cha_seg.max_cell_voltage.values,step=5))
        cha_seg['pack_voltage'] = pd.to_numeric(cha_seg['pack_voltage'], errors='coerce')
        Res_tmp.append(calculate_resis_pack(cha_seg.current.values,cha_seg.pack_voltage.values/90,step=5))
        Res_tmp.append(calculate_resis_pack(cha_seg.current.values,cha_seg.min_cell_voltage.values,step=5))
        Res_list.append(np.array(Res_tmp))

    indexes = [index for index, item in enumerate(Res_list) if not (item.size == 0 or np.all(np.abs(item) == float('inf')))]
    Res_list =[Res_list[i] for i in indexes]

    Res_df=pd.DataFrame(data=(np.array(Res_list)*1e3).astype(float),columns=['Max','Ave','Min'])
    Res_df['Time']=ca_data.time_s[indexes].values.reshape(-1,1)
    Res_df['Tave']=(ca_data.Tmax[indexes].values.reshape(-1,1)+ca_data.Tmax[indexes].values.reshape(-1,1))/2
    Res_df=Res_df[Res_df.Max<1]
    Res_df=Res_df[Res_df.Ave>0]
    Res_df=Res_df[Res_df.Min>0]
    Res_df=Res_df[Res_df.Max>0]

    cha_time=[]
    cha_cell_ca=[]
    cha_cell_V=[]

    for i in range(len(cha_list)):
        cha_seg=cha_list[i].copy()
        Res_df["Time"] = pd.to_datetime(Res_df["Time"])
        target_time = cha_seg.record_time.values[0]
        target_time = pd.to_datetime(target_time)

        resistance_avg = np.interp(
            target_time.timestamp(),
            Res_df["Time"].apply(lambda x: x.timestamp()),
            Res_df["Ave"]
        )

        resistance_max = np.interp(
            target_time.timestamp(),
            Res_df["Time"].apply(lambda x: x.timestamp()),
            Res_df["Max"]
        )

        resistance_min = np.interp(
            target_time.timestamp(),
            Res_df["Time"].apply(lambda x: x.timestamp()),
            Res_df["Min"]
        )

        avg_ocv = cha_seg['pack_voltage'].values + cha_seg.current.values * resistance_avg * 90/1e3
        avg_ocv_per_cell = avg_ocv / 90

        max_ocv = cha_seg['max_cell_voltage'].values + cha_seg.current.values * resistance_max/1e3

        min_ocv = cha_seg['min_cell_voltage'].values + cha_seg.current.values * resistance_min/1e3

        if len(cha_seg)<10:
            continue

        t_tmp=cha_seg['record_time'].values-cha_seg['record_time'].values[0]
        t=t_tmp/ np.timedelta64(1, 's')
        ca_tmp=np.zeros([len(cha_seg)])

        for j in range(1,len(cha_seg)):
            ca_tmp[j]=ca_tmp[j-1]+cha_seg.current.values[j]*(t[j]-t[j-1])/3600*(-1)

        cha_time.append(cha_seg.record_time.values[0])
        cha_cell_ca.append(ca_tmp)

        cha_cell_V.append([avg_ocv_per_cell, max_ocv, min_ocv])

    cha_time_list.append(cha_time)
    cha_cell_ca_list.append(cha_cell_ca)
    cha_cell_V_list.append(cha_cell_V)
    Res_df_list.append(Res_df)

# %% Weekly resistance features
weekly_resistance_data = []

for df in Res_df_list:

    df['Time'] = pd.to_datetime(df['Time'])

    df['Week'] = df['Time'].dt.isocalendar().year.astype(str) + '-' + df['Time'].dt.isocalendar().week.astype(str).str.zfill(2)

    weekly_df = df.groupby('Week').agg({
        'Max': 'mean',
        'Ave': 'mean',
        'Min': 'mean',

    }).reset_index()

    weekly_resistance_data.append(weekly_df)

resistance_choose_week_list = [df.iloc[:HISTORY_WEEKS].copy() for df in weekly_resistance_data]

# %% Optional resistance visualization
if PLOT_FEATURES:
    plt.rcParams['font.family'] = 'Times New Roman'

    selected_vehicles = [10, 18]
    vehicle_labels = [11, 19]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4),dpi=600)

    for i, veh_idx in enumerate(selected_vehicles):
       ax = axes[i]

       if veh_idx < len(weekly_resistance_data):
           df = weekly_resistance_data[veh_idx]

           if len(df) > 0:

               week_indices = range(len(df))

               ax.scatter(week_indices, df['Max'], c='#97ceff', s=60, alpha=0.8,
                         marker='o', label='Max', edgecolors='black', linewidth=0.8)
               ax.scatter(week_indices, df['Ave'], c='#fecee6', s=60, alpha=0.8,
                         marker='o', label='Ave', edgecolors='black', linewidth=0.8)
               ax.scatter(week_indices, df['Min'], c='#E91E63', s=60, alpha=0.8,
                         marker='o', label='Min', edgecolors='black', linewidth=0.8)

               ax.set_xlabel('Week', fontsize=12, family='Times New Roman')
               ax.set_ylabel('Resistance (mΩ)', fontsize=12, color='black', family='Times New Roman')

               ax.tick_params(axis='both', labelsize=10)
               ax.tick_params(axis='y', labelcolor='black')

               ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

               x_margin = max(len(df) * 0.05, 2)
               ax.set_xlim(-x_margin, len(df) - 1 + x_margin)

               y_data = pd.concat([df['Max'], df['Ave'], df['Min']])
               y_min, y_max = y_data.min(), y_data.max()
               y_margin = (y_max - y_min) * 0.1
               ax.set_ylim(y_min - y_margin, y_max + y_margin)

               x_ticks = [0, 20, 40, 60, 80, 100, 120]

               valid_ticks = [tick for tick in x_ticks if tick < len(df)]
               ax.set_xticks(valid_ticks)

               ax.spines['top'].set_visible(True)
               ax.spines['right'].set_visible(True)
               ax.spines['left'].set_visible(True)
               ax.spines['bottom'].set_visible(True)
               ax.spines['top'].set_linewidth(1.2)
               ax.spines['right'].set_linewidth(1.2)
               ax.spines['left'].set_linewidth(1.2)
               ax.spines['bottom'].set_linewidth(1.2)

           else:
               ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                      transform=ax.transAxes, fontsize=14, color='gray', family='Times New Roman')
               ax.set_title(f'Vehicle {vehicle_labels[i]}', fontsize=14, fontweight='bold',
                           family='Times New Roman')
       else:
           ax.text(0.5, 0.5, 'Vehicle Index Out of Range', ha='center', va='center',
                  transform=ax.transAxes, fontsize=14, color='red', family='Times New Roman')
           ax.set_title(f'Vehicle {vehicle_labels[i]}', fontsize=14, fontweight='bold',
                       family='Times New Roman')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.3)
    plt.show()

# %% Merge 24-week point features and resistance
def merge_veh_resistance_data(veh_data_list, resistance_data_list, output_filename):
    points_res_list = []

    for i in range(len(veh_data_list)):

        # Replace the three lowest-ranked statistical features with resistance.
        veh_df = veh_data_list[i].copy().iloc[:, :-3].reset_index(drop=True)
        resistance_df = resistance_data_list[i].copy()

        time_index = veh_df['time'].tolist()

        aligned_resistance = pd.DataFrame(index=range(len(time_index)))
        aligned_resistance['Week'] = time_index

        for col in ['Max', 'Ave', 'Min']:

            original_data = dict(zip(resistance_df['Week'], resistance_df[col]))

            aligned_values = []
            for week in time_index:
                if week in original_data:
                    aligned_values.append(original_data[week])
                else:
                    aligned_values.append(np.nan)

            aligned_resistance[col] = pd.Series(aligned_values).interpolate(method='linear')

            aligned_resistance[col] = aligned_resistance[col].ffill().bfill()

        resistance_cols = aligned_resistance[['Max', 'Ave', 'Min']]

        merged_df = pd.concat([veh_df, resistance_cols], axis=1)

        points_res_list.append(merged_df)

    with open(output_filename, 'wb') as f:
        pickle.dump(points_res_list, f)

    print(f"Saved merged features for {len(points_res_list)} vehicles to {output_filename}")
    return points_res_list

points_res_24weeks = merge_veh_resistance_data(
    veh_choose_week_list,
    resistance_choose_week_list,
    DATA_DIR / "point_features_rescombined_24weeks.pkl"
)

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

weekly_IC_data = []
V_min = 3.6
V_max = 4.15

for cha_time, cha_cell_ca, cha_cell_V in zip(cha_time_list, cha_cell_ca_list, cha_cell_V_list):
    cha_time = pd.to_datetime(cha_time)

    data_by_week = {}
    for t, Q, V_types in zip(cha_time, cha_cell_ca, cha_cell_V):
        week_key = f"{t.year}-{t.isocalendar()[1]:02d}"
        if week_key not in data_by_week:
            data_by_week[week_key] = []
        data_by_week[week_key].append((Q, V_types))

    weekly_IC = {}
    for week, charging_data in data_by_week.items():

        Q_list_avg = []
        Q_list_max = []
        Q_list_min = []

        V_uniform = np.linspace(V_min, V_max, num=1000)
        dV = np.gradient(V_uniform)

        for Q, V_types in charging_data:

            V_avg_raw = np.array(V_types[0]) if isinstance(V_types[0], (list, np.ndarray)) else np.array([V_types[0]])
            V_max_raw = np.array(V_types[1]) if isinstance(V_types[1], (list, np.ndarray)) else np.array([V_types[1]])
            V_min_raw = np.array(V_types[2]) if isinstance(V_types[2], (list, np.ndarray)) else np.array([V_types[2]])

            V_avg = np.array(voltage_monotonicity_de(gaussian_filter1d(V_avg_raw, sigma=5)))
            V_max_ocv = np.array(voltage_monotonicity_de(gaussian_filter1d(V_max_raw, sigma=5)))
            V_min_ocv = np.array(voltage_monotonicity_de(gaussian_filter1d(V_min_raw, sigma=5)))

            Q = np.array(Q)

            if len(V_avg) > 0 and len(Q) > 0:
                try:
                    interp_func_avg = custom_interp1d(V_avg, Q)
                    Q_uniform_avg = interp_func_avg(V_uniform)
                    Q_uniform_avg = Q_uniform_avg - Q_uniform_avg[0]
                    Q_list_avg.append(Q_uniform_avg)
                except Exception as e:
                    print(f"Interpolation error (average OCV): {e}")

            if len(V_max_ocv) > 0 and len(Q) > 0:
                try:
                    interp_func_max = custom_interp1d(V_max_ocv, Q)
                    Q_uniform_max = interp_func_max(V_uniform)
                    Q_uniform_max = Q_uniform_max - Q_uniform_max[0]
                    Q_list_max.append(Q_uniform_max)
                except Exception as e:
                    print(f"Interpolation error (maximum OCV): {e}")

            if len(V_min_ocv) > 0 and len(Q) > 0:
                try:
                    interp_func_min = custom_interp1d(V_min_ocv, Q)
                    Q_uniform_min = interp_func_min(V_uniform)
                    Q_uniform_min = Q_uniform_min - Q_uniform_min[0]
                    Q_list_min.append(Q_uniform_min)
                except Exception as e:
                    print(f"Interpolation error (minimum OCV): {e}")

        if Q_list_avg:
            Q_array_avg = np.array(Q_list_avg)
            Q_mean_avg = np.nanmean(Q_array_avg, axis=0)

            dQ_avg = np.gradient(Q_mean_avg)
            dV = np.gradient(V_uniform)

            if isinstance(dQ_avg, tuple):
                dQ_avg = dQ_avg[0]
            if isinstance(dV, tuple):
                dV = dV[0]

            dQ_dV_avg = dQ_avg / dV
        else:
            Q_mean_avg = np.array([])
            dQ_dV_avg = np.array([])

        if Q_list_max:
            Q_array_max = np.array(Q_list_max)
            Q_mean_max = np.nanmean(Q_array_max, axis=0)

            dQ_max = np.gradient(Q_mean_max)
            if isinstance(dQ_max, tuple):
                dQ_max = dQ_max[0]
            if isinstance(dV, tuple):
                dV = dV[0]

            dQ_dV_max = dQ_max / dV
        else:
            Q_mean_max = np.array([])
            dQ_dV_max = np.array([])

        if Q_list_min:
            Q_array_min = np.array(Q_list_min)
            Q_mean_min = np.nanmean(Q_array_min, axis=0)

            dQ_min = np.gradient(Q_mean_min)
            if isinstance(dQ_min, tuple):
                dQ_min = dQ_min[0]
            if isinstance(dV, tuple):
                dV = dV[0]

            dQ_dV_min = dQ_min / dV
        else:
            Q_mean_min = np.array([])
            dQ_dV_min = np.array([])

        weekly_IC[week] = {
            'avg': (V_uniform, Q_mean_avg, dQ_dV_avg),
            'max': (V_uniform, Q_mean_max, dQ_dV_max),
            'min': (V_uniform, Q_mean_min, dQ_dV_min)
        }

    weekly_IC_data.append(weekly_IC)

# %% Optional IC and resistance visualizations
if PLOT_FEATURES:
    def color_fun(L):
        pink=[255,179,204]
        lightblue=[0,115,189]
        colordata = np.linspace(lightblue,pink, L)/255
        return colordata

    plt.figure(figsize=(10, 8), dpi=600)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.82, wspace=0.6, hspace=0.3)

    handles = []
    labels = []
    for k in range(len(weekly_IC_data)):
        plt.subplot(5, 5, k + 1)
        weekly_IC = weekly_IC_data[k]
        weekly_averages = {}
        for week, data_dict in weekly_IC.items():
            if 'avg' in data_dict:
                V_uniform, Q_mean, dQ_dV = data_dict['avg']

                if len(dQ_dV) > 0:
                    weekly_averages[week] = (V_uniform, Q_mean, dQ_dV)

        sorted_weeks = sorted(weekly_averages.keys())
        if len(sorted_weeks) > 124:

            indices = np.linspace(0, len(sorted_weeks)-1, 124, dtype=int)
            selected_weeks = [sorted_weeks[i] for i in indices]
        else:
            selected_weeks = sorted_weeks

        selected_weekly_averages = {week: weekly_averages[week] for week in selected_weeks}

        colordata = color_fun(len(selected_weekly_averages))
        for i, (week, (V_uniform, Q_mean, averaged_dQ_dV)) in enumerate(selected_weekly_averages.items()):

            handle, = plt.plot(V_uniform,gaussian_filter1d(averaged_dQ_dV, sigma=5),label=f'{week}',color=colordata[i])

            if k == 0:
                handles.append(handle)
                labels.append(week)

        plt.xlabel('Voltage [V]')
        plt.ylabel('dQ/dV [Ah/V]')

    plt.figlegend(handles, labels, loc='upper center', ncol=5, fontsize='small', frameon=False, bbox_to_anchor=(0.5, 0.9))
    plt.show()

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import ScalarMappable

    def color_fun(L):

        pink_hex = "#97CEFF"
        blue_hex = "#E91E63"

        pink = np.array([int(pink_hex[i:i+2], 16) for i in (1, 3, 5)])
        blue = np.array([int(blue_hex[i:i+2], 16) for i in (1, 3, 5)])

        colordata = np.linspace(blue, pink, L) / 255
        return colordata

    selected_vehicles = [13, 17]
    vehicle_labels = [14, 18]
    num_weeks = 124

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=600)
    axes = axes.flatten()

    colordata = color_fun(num_weeks)

    for idx, veh_idx in enumerate(selected_vehicles):
        ax = axes[idx]
        weekly_IC = weekly_IC_data[veh_idx]
        weekly_averages = {}

        for week, data_dict in weekly_IC.items():
            if 'avg' in data_dict:
                V_uniform, Q_mean, dQ_dV = data_dict['avg']
                if len(dQ_dV) > 0:
                    weekly_averages[week] = (V_uniform, Q_mean, dQ_dV)

        sorted_weeks = sorted(weekly_averages.keys())
        if len(sorted_weeks) > num_weeks:
            indices = np.linspace(0, len(sorted_weeks) - 1, num_weeks, dtype=int)
            selected_weeks = [sorted_weeks[i] for i in indices]
        else:
            selected_weeks = sorted_weeks
            num_weeks = len(sorted_weeks)
            colordata = color_fun(num_weeks)

        selected_weekly_averages = {week: weekly_averages[week] for week in selected_weeks}

        for i, (week, (V_uniform, Q_mean, averaged_dQ_dV)) in enumerate(selected_weekly_averages.items()):
            ax.plot(
                V_uniform,
                gaussian_filter1d(averaged_dQ_dV, sigma=10),
                color=colordata[i]
            )

        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('dQ/dV [Ah/V]')

    cmap = ListedColormap(colordata)
    norm = plt.Normalize(vmin=0, vmax=num_weeks)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes[1], orientation='vertical',
                        fraction=0.046, pad=0.03)
    cbar.set_label('Week', fontsize=10, fontname='Times New Roman')
    cbar.set_ticks([0, 25, 55, 85, 115])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

    plt.rcParams['font.family'] = 'Times New Roman'
    from matplotlib.cm import ScalarMappable

    from matplotlib.colors import to_rgb

    def color_fun(L):

        colors_hex = ['#001C5C', '#003B8E', '#3A7BD5', '#8CBFFF', '#DCEBFF']

        colors_rgb = np.array([to_rgb(c) for c in colors_hex])

        x = np.linspace(0, 1, len(colors_rgb))
        xi = np.linspace(0, 1, L)

        r = np.interp(xi, x, colors_rgb[:, 0])
        g = np.interp(xi, x, colors_rgb[:, 1])
        b = np.interp(xi, x, colors_rgb[:, 2])

        colordata = np.vstack([r, g, b]).T

        return colordata

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=600)

    selected_vehicles_resistance = [10, 18]
    vehicle_labels_resistance = [11, 19]

    for i, veh_idx in enumerate(selected_vehicles_resistance):
        ax = axes[0, i]

        if veh_idx < len(weekly_resistance_data):
            df = weekly_resistance_data[veh_idx]

            if len(df) > 0:

                week_indices = range(len(df))
                ax.scatter(week_indices, df['Max'], c='#003B8E', s=60, alpha=0.8,
                          marker='o', label='Max', edgecolors='black', linewidth=0.8)
                ax.scatter(week_indices, df['Ave'], c='#3A7BD5', s=60, alpha=0.8,
                          marker='o', label='Ave', edgecolors='black', linewidth=0.8)
                ax.scatter(week_indices, df['Min'], c='#DCEBFF', s=60, alpha=0.8,
                          marker='o', label='Min', edgecolors='black', linewidth=0.8)
                ax.set_xlabel('Week', fontsize=20, family='Times New Roman')
                ax.set_ylabel('Resistance (mΩ)', fontsize=20, color='black', family='Times New Roman')

                ax.tick_params(axis='both', labelsize=18)
                ax.tick_params(axis='y', labelcolor='black')

                ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

                x_margin = max(len(df) * 0.05, 2)
                ax.set_xlim(-x_margin, len(df) - 1 + x_margin)

                y_data = pd.concat([df['Max'], df['Ave'], df['Min']])
                y_min, y_max = y_data.min(), y_data.max()
                y_margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - y_margin, y_max + y_margin)

                x_ticks = [0, 20, 40, 60, 80, 100, 120]
                valid_ticks = [tick for tick in x_ticks if tick < len(df)]
                ax.set_xticks(valid_ticks)

                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['top'].set_linewidth(1.2)
                ax.spines['right'].set_linewidth(1.2)
                ax.spines['left'].set_linewidth(1.2)
                ax.spines['bottom'].set_linewidth(1.2)

            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14, color='gray', family='Times New Roman')
                ax.set_title(f'Vehicle {vehicle_labels_resistance[i]} - Resistance',
                            fontsize=14, fontweight='bold', family='Times New Roman')
        else:
            ax.text(0.5, 0.5, 'Vehicle Index Out of Range', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='red', family='Times New Roman')
            ax.set_title(f'Vehicle {vehicle_labels_resistance[i]} - Resistance',
                        fontsize=14, fontweight='bold', family='Times New Roman')

    selected_vehicles_IC = [13, 17]
    vehicle_labels_IC = [14, 18]
    num_weeks = 124

    colordata = color_fun(num_weeks)

    for idx, veh_idx in enumerate(selected_vehicles_IC):
        ax = axes[1, idx]

        weekly_IC = weekly_IC_data[veh_idx]
        weekly_averages = {}

        for week, data_dict in weekly_IC.items():
            if 'avg' in data_dict:
                V_uniform, Q_mean, dQ_dV = data_dict['avg']
                if len(dQ_dV) > 0:
                    weekly_averages[week] = (V_uniform, Q_mean, dQ_dV)

        sorted_weeks = sorted(weekly_averages.keys())

        if len(sorted_weeks) > num_weeks:
            indices = np.linspace(0, len(sorted_weeks) - 1, num_weeks, dtype=int)
            selected_weeks = [sorted_weeks[i] for i in indices]
        else:
            selected_weeks = sorted_weeks
            current_num_weeks = len(sorted_weeks)
            colordata_current = color_fun(current_num_weeks)

        selected_weekly_averages = {week: weekly_averages[week] for week in selected_weeks}

        for i, (week, (V_uniform, Q_mean, averaged_dQ_dV)) in enumerate(selected_weekly_averages.items()):
            if len(sorted_weeks) > num_weeks:
                color = colordata[i]
            else:
                color = colordata_current[i]

            ax.plot(
                V_uniform,
                gaussian_filter1d(averaged_dQ_dV, sigma=10),
                color=color
            )

        ax.set_xlabel('Voltage (V)', fontsize=20, family='Times New Roman')
        ax.set_ylabel('dQ/dV (Ah/V)', fontsize=20, family='Times New Roman')
        ax.tick_params(axis='both', labelsize=18)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    cmap = ListedColormap(colordata)
    norm = plt.Normalize(vmin=0, vmax=num_weeks)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.35])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Week', fontsize=12, fontname='Times New Roman')
    cbar.set_ticks([0, 25, 55, 85, 115])
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.88, hspace=0.35, wspace=0.35)
    plt.show()

# %% Prepare 24-week IC inputs
def preprocess_ic_curves_dict(weekly_IC_data, num_weeks=24, num_points=100, v_range=(3.7, 4.1)):
    """Interpolate avg/max/min IC channels for the first num_weeks weeks. Repeat the final week when necessary, using distinct ordinal keys."""

    processed_data = {}

    v_uniform_target = np.linspace(v_range[0], v_range[1], num_points)

    ocv_types = ['avg', 'max', 'min']

    for vehicle_idx, vehicle_data in enumerate(weekly_IC_data):
        vehicle_id = f"vehicle_{vehicle_idx}"
        processed_data[vehicle_id] = {}

        weeks = sorted(vehicle_data.keys())
        if not weeks:
            raise ValueError(f"Vehicle {vehicle_idx} has no IC weeks.")

        selected_weeks = weeks[:num_weeks]

        if len(selected_weeks) < num_weeks:
            print(f"Warning: vehicle {vehicle_idx} has {len(weeks)} weeks; repeating its final week to reach {num_weeks}.")

            selected_weeks = selected_weeks + [weeks[-1]] * (num_weeks - len(weeks))

        for week_idx, week in enumerate(selected_weeks):
            week_id = f"{week_idx:02d}:{week}"
            processed_data[vehicle_id][week_id] = {}

            for ocv_type in ocv_types:

                v_uniform, q_mean, dq_dv = vehicle_data[week][ocv_type]

                valid_indices = (v_uniform >= v_range[0]) & (v_uniform <= v_range[1])

                if not np.any(valid_indices):
                    print(f"Warning: vehicle {vehicle_idx}, week {week}, channel {ocv_type}: no data within {v_range}.")

                    processed_data[vehicle_id][week_id][ocv_type] = [0.0] * num_points
                    continue

                v_valid = v_uniform[valid_indices]
                q_valid = q_mean[valid_indices]
                dq_valid = dq_dv[valid_indices]

                if len(v_valid) <= 1 or len(q_valid) <= 1 or len(dq_valid) <= 1:
                    print(f"Warning: vehicle {vehicle_idx}, week {week}, channel {ocv_type}: too few points to interpolate.")
                    processed_data[vehicle_id][week_id][ocv_type] = [0.0] * num_points
                    continue

                try:

                    dq_interp = interpolate.interp1d(v_valid, dq_valid, bounds_error=False, fill_value="extrapolate")

                    processed_data[vehicle_id][week_id][ocv_type] = dq_interp(v_uniform_target).tolist()

                except Exception as e:
                    print(f"Interpolation error: vehicle {vehicle_idx}, week {week}, channel {ocv_type}: {e}")

                    processed_data[vehicle_id][week_id][ocv_type] = [0.0] * num_points

    return processed_data

raw_ic_24 = preprocess_ic_curves_dict(weekly_IC_data, num_weeks=HISTORY_WEEKS)

def process_raw_ic_to_features(raw_ic: dict, device: str = 'cpu', max_weeks: int = 24) -> torch.Tensor:
    """Pack IC data in numeric vehicle order into a float32 tensor of shape (vehicles * max_weeks, 3, 100), with channels avg, max, and min."""
    if not raw_ic:
        raise ValueError("IC input data is empty.")

    batch_size = len(raw_ic)
    seq_length = 100
    num_channels = 3

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    ic_tensor = torch.zeros(batch_size, max_weeks, num_channels, seq_length,
                           dtype=torch.float32, device=device)

    vehicle_ids = sorted(raw_ic.keys(), key=lambda key: int(key.rsplit("_", 1)[1]))

    processed_count = 0
    error_count = 0

    for v_idx, vehicle_id in enumerate(vehicle_ids):
        weeks_data = raw_ic[vehicle_id]
        if not weeks_data:
            continue

        week_ids = sorted(weeks_data.keys())

        for w_idx, week_id in enumerate(week_ids):
            if w_idx >= max_weeks:
                break

            week_data = weeks_data[week_id]

            try:

                for channel_name in ['avg', 'max', 'min']:
                    if channel_name not in week_data:
                        raise ValueError(f"Missing channel: {channel_name}")
                    if len(week_data[channel_name]) != seq_length:
                        raise ValueError(f"{channel_name} length mismatch: expected {seq_length}, got {len(week_data[channel_name])}.")

                ic_tensor[v_idx, w_idx, 0] = torch.tensor(week_data['avg'], dtype=torch.float32, device=device)
                ic_tensor[v_idx, w_idx, 1] = torch.tensor(week_data['max'], dtype=torch.float32, device=device)
                ic_tensor[v_idx, w_idx, 2] = torch.tensor(week_data['min'], dtype=torch.float32, device=device)

                processed_count += 1

            except Exception as e:
                print(f"Error processing vehicle {vehicle_id}, week {week_id}: {e}")
                error_count += 1

                continue

    ic_tensor_reshaped = ic_tensor.view(-1, num_channels, seq_length)

    print(f"  - Vehicles: {batch_size}")
    print(f"  - Weeks per vehicle: {max_weeks}")
    print(f"  - Processed: {processed_count} samples")
    print(f"  - Errors: {error_count} samples")
    print(f"  - Original tensor shape: {ic_tensor.shape}")
    print(f"  - Output tensor shape: {ic_tensor_reshaped.shape}")

    return ic_tensor_reshaped

ic_tensor_24weeks = process_raw_ic_to_features(raw_ic_24, max_weeks=HISTORY_WEEKS)

torch.save(ic_tensor_24weeks, DATA_DIR / "ic_tensor_24weeks.pt")

expected_shape = (len(veh_choose_week_list) * HISTORY_WEEKS, 3, 100)
if tuple(ic_tensor_24weeks.shape) != expected_shape:
    raise ValueError(f"Unexpected IC tensor shape: {tuple(ic_tensor_24weeks.shape)}.")
print(f"Saved 24-week IC input to {DATA_DIR / 'ic_tensor_24weeks.pt'}")
