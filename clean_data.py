import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import xarray as xr


def read_grib_data(file, save_name):
    # load grib data from ERA5
    print('Doing grib stuff...')
    data = xr.open_dataset(file, engine="cfgrib")
    vars = data.to_array().to_numpy()  # 5-d array [variable,times,heights,lats,lons]

    num_predictors = int(vars.shape[0] * vars.shape[2] * vars.shape[3] * vars.shape[4])
    t = int(vars.shape[1])
    vars = np.rollaxis(vars, 1, 0)
    vars = vars.reshape((t, num_predictors)).T  # remove physical dimensions

    with open(save_name, 'wb') as f:
        np.save(f,arr=vars)

    return vars


def avg_obs_data(file):
    # average the observation data
    print('Processing obs data...')
    raw = pd.read_csv(file,sep=' ',header=0,skipinitialspace=True)
    dates = np.array(raw.Dates)
    times = np.array(raw.Times)  # observations every 5 mins
    speeds = np.array(raw.Wind)

    # take hourly averages
    # need to have array that has values for each hour, incl nan for missing data
    num_years = 2
    num_hours = ((num_years * 365)+1) * 24  # +1 for leap year
    arr = np.empty(num_hours)  # initialize hourly data array
    h = 0  # counter for hour number
    hourly_speeds = []  # list of 5-min speeds that will be avgd

    time_str = str(times[0])
    while len(time_str) < 6:
        time_str = str(0) + time_str
    hour_start = datetime.strptime(str(dates[0]) + time_str,'%Y%m%d%H%M%S')

    for i in range(len(speeds)):
        time_str = str(times[i])
        while len(time_str) < 6:
            time_str = str(0) + time_str
        time = datetime.strptime(str(dates[i]) + time_str,'%Y%m%d%H%M%S')

        while time - hour_start > timedelta(hours=1):  # next hour
            if len(hourly_speeds) == 0:
                arr[h] = np.nan
            else:
                arr[h] = np.mean(hourly_speeds)
            h += 1
            hourly_speeds = []
            hour_start += timedelta(hours=1)

        hourly_speeds.append(speeds[i])  # add this speed to hourly list

    if len(hourly_speeds) == 0:
        arr[h] = np.nan
    else:
        arr[h] = np.mean(hourly_speeds)
    h += 1
    print(h)
    print(num_hours)

    # save array to binary file
    print('Writing...')
    with open('hourly_data_2020-2021.npy', 'wb') as f:
        np.save(f,arr=arr)

    return arr


def index_data(obs_file, era_file):
    # delete nan obs from the era data as well so they match

    obs = np.load(obs_file)
    era = np.load(era_file)

    to_delete = []
    for i in range(len(obs)):
        if np.isnan(obs[i]):
            to_delete.append(i)

    obs = np.delete(obs, to_delete)
    obs = np.delete(obs, [-1])  # obs has 1 extra at end
    to_delete.insert(0, 0)  # era data has 1 extra at beginning
    era = np.delete(era, to_delete,axis=1)

    print(obs.shape)
    print(era.shape)
  

    print('Writing...')
    with open(obs_file, 'wb') as f:
        np.save(f,arr=obs)

    with open(era_file, 'wb') as f:
        np.save(f,arr=era)


if __name__ == '__main__':
    era_2021 = read_grib_data('C:/Users/jillp/555/2021.grib','predictors_2021.npy')  # big ass file
    era_2020 = read_grib_data('C:/Users/jillp/555/2020.grib','predictors_2020.npy')  # big ass file
    era = np.concatenate((era_2020, era_2021),axis=1)
    with open('predictors_2020-2021.npy', 'wb') as f:
        np.save(f,arr=era)

    obs = avg_obs_data('bm_obs_2020-2021.txt')
    index_data('hourly_data_2020-2021.npy','predictors_2020-2021.npy')
    print('Done!')
