import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
import math
from scipy import interpolate


# get data
print('Getting data...')
obs = np.load('hourly_data_2020-2021.npy')
x = np.load('100m_2020-2021.npy')
print(x.shape)
print(obs.shape)

w100_0 = np.sqrt(np.square(x[0,:,0,0])+np.square(x[1,:,0,0]))  # (55.75, -120.5)
w10_0 = np.sqrt(np.square(x[2,:,0,0])+np.square(x[3,:,0,0]))
w100_01 = np.sqrt(np.square(x[0,:,0,1])+np.square(x[1,:,0,1]))  # (55.75, -120.25)
w10_01 = np.sqrt(np.square(x[2,:,0,1])+np.square(x[3,:,0,1]))
w100_10 = np.sqrt(np.square(x[0,:,1,0])+np.square(x[1,:,1,0]))  # (55.5, -120.5)
w10_10 = np.sqrt(np.square(x[2,:,1,0])+np.square(x[3,:,1,0]))
w100_1 = np.sqrt(np.square(x[0,:,1,1])+np.square(x[1,:,1,1]))  # (55.5, -120.25)
w10_1 = np.sqrt(np.square(x[2,:,1,1])+np.square(x[3,:,1,1]))

print('Interpolating...')
# horizontal interp using scipy.interpolate.griddata where D=2, n=4, m=number of obs
coords = np.array([[55.5,-120.25],[55.5,-120.5],[55.75,-120.25],[55.75,-120.5]])
vals10 = np.array([w10_1, w10_10, w10_01, w10_0])
vals100 = np.array([w100_1, w100_10, w100_01, w100_0])
loc = np.array([(55.6986, -120.4306)])
w10 = interpolate.griddata(coords,vals10,loc,method='linear').squeeze()*3.6
w100 = interpolate.griddata(coords,vals100,loc,method='linear').squeeze()*3.6


# vertical interp using power law
alpha = np.log(w100/w10) / np.log(100/10)  # shear exponent
w_turbine = w100*(80/100)**alpha


# plot
t=range(80)
plt.plot(t, w_turbine[:80],label='turbine')
plt.plot(t, obs[:80],label='obs')
plt.xlabel('Time (h)')
plt.ylabel('Wind speed (km/h)')
plt.legend()
plt.show()


# calibration via linear regression


