# energyOptTset

from cvxopt import matrix, solvers
from cvxopt.modeling import op, dot, variable
import time
import pandas as pd
import numpy as np
import sys
import datetime as datetime
#start_time=time.process_time()

# Parameters for one week long simulation with 2hr horizon --------------------------------------------
heatorcool = 'heat'

# Modes:
OCCUPANCY_MODE = True
SETPOINT_MODE = 'Adaptive90'	# 'Adaptive90' or 'Fixed'

# For Fixed setpoints:
FIXED_UPPER = 24.0
FIXED_LOWER = 20.0
# Max and min for heating and cooling in adaptive setpoint control for 90% of people
HEAT_TEMP_MAX_90 = 26.2
HEAT_TEMP_MIN_90 = 18.9
COOL_TEMP_MAX_90 = 30.2
COOL_TEMP_MIN_90 = 22.9

# Other constants
n=24 # number of timesteps within prediction windows (24 x 5-min timesteps in 2 hr window)
timestep = 5*60
days = 7
totaltimesteps = days*12*24+3*12 

# get inputs from UCEF --------------------------------------------
day = int(sys.argv[1])
block = int(sys.argv[2]) +1+(day-1)*24 # block goes 0:23 (represents the hour within a day)
temp_indoor_initial = float(sys.argv[3])

# Get data from excel/csv files --------------------------------------------
# Get outdoor temps
outdoor_temp_df = pd.read_excel('../../optimization/OutdoorTemp.xlsx', sheet_name='Feb12thru19_2021_1hr',header=0)
start_date = datetime.datetime(2021,2,12)
dates = np.array([start_date + datetime.timedelta(hours=i) for i in range(8*24+1)])
outdoor_temp_df = outdoor_temp_df.set_index(dates)
outdoor_temp_df = outdoor_temp_df.resample('5min').pad()
# print(outdoor_temp_df.head())
temp_outdoor_all=matrix(outdoor_temp_df.to_numpy())
outdoor_temp_df.columns = ['column1']

# Adaptive Setpoints
if SETPOINT_MODE == 'Adaptive90':
	# use outdoor temps to get adaptive setpoints using lambda functions
	outdoor_to_cool90 = lambda x: x*0.31 + 19.8
	outdoor_to_heat90 = lambda x: x*0.31 + 15.8
	adaptive_cooling_90 = outdoor_temp_df.apply(outdoor_to_cool90)
	adaptive_heating_90 = outdoor_temp_df.apply(outdoor_to_heat90)
	# print(adaptive_heating_90)
	# When temps too low or too high set to min or max (See adaptive setpoints)
	adaptive_cooling_90.loc[(adaptive_cooling_90['column1'] < COOL_TEMP_MIN_90)] = COOL_TEMP_MIN_90
	adaptive_cooling_90.loc[(adaptive_cooling_90['column1'] > COOL_TEMP_MAX_90)] = COOL_TEMP_MAX_90
	adaptive_heating_90.loc[(adaptive_heating_90['column1'] < HEAT_TEMP_MIN_90)] = HEAT_TEMP_MIN_90
	adaptive_heating_90.loc[(adaptive_heating_90['column1'] > HEAT_TEMP_MAX_90)] = HEAT_TEMP_MAX_90
	# change from pd dataframe to matrix
	# print(adaptive_heating_90)
	adaptive_cooling_90 = matrix(adaptive_cooling_90.to_numpy())
	adaptive_heating_90 = matrix(adaptive_heating_90.to_numpy())
	# print(adaptive_heating_90)

if OCCUPANCY_MODE == True:
	# use outdoor temps to get bands where 100% of people are comfortable using lambda functions
	convertOutTemptoCool100 = lambda x: x*0.31 + 19.3   # calculated that 100% band is +/-1.5C 
	convertOutTemptoHeat100 = lambda x: x*0.31 + 16.3
	adaptive_cooling_100 = outdoor_temp_df.apply(convertOutTemptoCool100)
	adaptive_heating_100 = outdoor_temp_df.apply(convertOutTemptoHeat100)
	# print(adaptive_heating_100)
	# When temps too low or too high set to min or max (See adaptive 100)
	adaptive_cooling_100.loc[(adaptive_cooling_100['column1'] < 22.4)] = 22.4
	adaptive_cooling_100.loc[(adaptive_cooling_100['column1'] > 29.7)] = 29.7
	adaptive_heating_100.loc[(adaptive_heating_100['column1'] < 18.4)] = 18.4
	adaptive_heating_100.loc[(adaptive_heating_100['column1'] > 25.7)] = 25.7
	# change from pd dataframe to matrix
	# print(adaptive_heating_100)
	adaptive_cooling_100 = matrix(adaptive_cooling_100.to_numpy())
	adaptive_heating_100 = matrix(adaptive_heating_100.to_numpy())
	# print(adaptive_heating_100)

if OCCUPANCY_MODE == True:
	# get occupancy data
	occupancy_df = pd.read_csv('../../optimization/occupancy_1hr.csv')
	occupancy_df = occupancy_df.set_index('Dates/Times')
	occupancy_df.index = pd.to_datetime(occupancy_df.index)
	occupancy_df = occupancy_df.resample('5min').pad()
	# print(occupancy_df.head)
	occupancy_comfort_range = matrix(occupancy_df['Comfort Range'].to_numpy())

#------------------------------- Done getting data from excel sheets

#------------------------------- setting up optimization
# c matrix is hourly cost per kWh of energy (I think this can be deleted)

if OCCUPANCY_MODE == True:
	comfort_range = occupancy_comfort_range[(block-1)*12:(block-1)*12+n,0]

	adaptiveHeat = adaptive_heating_100[(block-1)*12:(block-1)*12+n,0]
	adaptiveCool = adaptive_cooling_100[(block-1)*12:(block-1)*12+n,0]


print('adaptive heating setpoints')
j = 0
while j<12:
	print(adaptiveHeat[j,0])
	j=j+1
print('adaptive cooling setpoints')
j = 0
while j<12:
	print(adaptiveCool[j,0])
	j=j+1
