# energyOptTset

from cvxopt import matrix, solvers
from cvxopt.modeling import op, dot, variable
import time
import pandas as pd
import sys
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
PRICING_MULTIPLIER = 4.0
PRICING_OFFSET = 0.10

# constant coefficients for indoor temperature equation --------------------------------------------
c1 = 1.72*10**-5
if heatorcool == 'heat':
	c2 = 0.0031
else:
	c2 = -0.0031
c3 = 3.58*10**-7

# get inputs from UCEF --------------------------------------------
day = int(sys.argv[1])
block = int(sys.argv[2]) +1+(day-1)*24 # block goes 0:23 (represents the hour within a day)
temp_indoor_initial = float(sys.argv[3])

# Get data from excel/csv files --------------------------------------------
# Get outdoor temps
outdoor_temp_df = pd.read_excel('OutdoorTemp.xlsx', sheet_name='Feb12thru19',header=0)
# print(df.head())
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
	occupancy_df = pd.read_csv('occupancy_1hr.csv')
	occupancy_comfort_range = matrix(occupancy_df['Comfort Range'].to_numpy())

# get solar radiation
q_solar_all_df = pd.read_excel('Solar.xlsx', sheet_name='Feb12thru19')
q_solar_all=matrix(q_solar_all_df.to_numpy())

# get wholesale prices
wholesaleprice_all_df = pd.read_excel('WholesalePrice.xlsx', sheet_name='Feb12thru19')
wholesaleprice_all=matrix(wholesaleprice_all_df.to_numpy())

#------------------------------- Done getting data from excel sheets

#------------------------------- setting up optimization
# c matrix is hourly cost per kWh of energy (I think this can be deleted)
Output = matrix(0.00, (totaltimesteps,2))
cost =0
c = matrix(0.20, (totaltimesteps,1))

# setting up optimization to minimize energy times price
x=variable(n)	# x is energy usage that we are predicting

# A matrix is coefficients of energy used variables in constraint equations (see PJs equations)
AA = matrix(0.0, (n*2,n))
k = 0
while k<n:
	j = 2*k
	AA[j,k] = timestep*c2
	AA[j+1,k] = -timestep*c2
	k=k+1
k=0
while k<n:
	j=2*k+2
	while j<2*n-1:
		AA[j,k] = AA[j-2,k]*-timestep*c1+ AA[j-2,k]
		AA[j+1,k] = AA[j-1,k]*-timestep*c1+ AA[j-1,k]
		j+=2
	k=k+1

## Added 12/11
# making sure energy is positive for heating
heat_positive = matrix(0.0, (n,n))
i = 0
while i<n:
	heat_positive[i,i] = -1.0 # setting boundary condition: Energy used at each timestep must be greater than 0
	i +=1
# making sure energy is negative for cooling
cool_negative = matrix(0.0, (n,n))
i = 0
while i<n:
	cool_negative[i,i] = 1.0 # setting boundary condition: Energy used at each timestep must be less than 0
	i +=1

d = matrix(0.0, (n,1))
# inequality constraints
heatineq = (heat_positive*x<=d)
coolineq = (cool_negative*x<=d)
energyLimit = matrix(0.25, (n,1)) # .4 before
heatlimiteq = (cool_negative*x<=energyLimit)

# creating S matrix to make b matrix simpler
temp_outdoor= temp_outdoor_all[(block-1)*12:(block-1)*12+n,0] # getting next two hours of data
q_solar=q_solar_all[(block-1)*12:(block-1)*12+n,0]

# get price for next two hours
cc=wholesaleprice_all[(block-1)*12:(block-1)*12+n,0]*PRICING_MULTIPLIER/1000+PRICING_OFFSET
# cc = c[(block-1)*12:(block-1)*12+n,0]
S = matrix(0.0, (n,1))
S[0,0] = timestep*(c1*(temp_outdoor[0]-temp_indoor_initial)+c3*q_solar[0])+temp_indoor_initial

i=1
while i<n:
	S[i,0] = timestep*(c1*(temp_outdoor[i]-S[i-1,0])+c3*q_solar[i])+S[i-1,0]
	i+=1
#print(S)

# b matrix is constant term in constaint equations
b = matrix(0.0, (n*2,1))

if OCCUPANCY_MODE == True:
	comfort_range = occupancy_comfort_range[(block-1)*12:(block-1)*12+n,0]

	if SETPOINT_MODE == 'Adaptive90':
		adaptiveHeat = adaptive_heating_90[(block-1)*12:(block-1)*12+n,0]
		adaptiveCool = adaptive_cooling_90[(block-1)*12:(block-1)*12+n,0]
		k = 0
		while k<n:
			b[2*k,0]=adaptiveCool[k,0]-S[k,0]+comfort_range[k,0]
			b[2*k+1,0]=-adaptiveHeat[k,0]+S[k,0]-comfort_range[k,0]
			k=k+1

	if SETPOINT_MODE == 'Fixed':
		k = 0
		while k<n:
			b[2*k,0]=FIXED_UPPER-S[k,0]+comfort_range[k,0]
			b[2*k+1,0]=-FIXED_LOWER+S[k,0]-comfort_range[k,0]
			k=k+1
else:
	if SETPOINT_MODE == 'Adaptive90':
		adaptiveHeat = adaptive_heating_90[(block-1)*12:(block-1)*12+n,0]
		adaptiveCool = adaptive_cooling_90[(block-1)*12:(block-1)*12+n,0]
		k = 0
		while k<n:
			b[2*k,0]=adaptiveCool[k,0]-S[k,0]
			b[2*k+1,0]=-adaptiveHeat[k,0]+S[k,0]
			k=k+1
	if SETPOINT_MODE == 'Fixed':
		k = 0
		while k<n:
			b[2*k,0]=FIXED_UPPER-S[k,0]
			b[2*k+1,0]=-FIXED_LOWER+S[k,0]
			k=k+1

# time to solve for energy at each timestep
#print(cc)
#print(AA)
#print(b)
ineq = (AA*x <= b)
if heatorcool == 'heat':
	lp2 = op(dot(cc,x),ineq)
	op.addconstraint(lp2, heatineq)
	op.addconstraint(lp2,heatlimiteq)
if heatorcool == 'cool':
	lp2 = op(dot(-cc,x),ineq)
	op.addconstraint(lp2, coolineq)
lp2.solve()
# ---------------------- solved

if x.value == None:
	energy = matrix(0.00, (13,1))
	print('energy consumption')
	j=0
	while j<13:
		print(energy[j])
		j = j+1
	j=0
	temp_indoor = matrix(0.0, (totaltimesteps,1))
	while j<13:
		temp_indoor[j]=adaptiveHeat[j,0]
		j=j+1
	print('indoor temp prediction')
	j = 0
	while j<13:
		print(temp_indoor[j,0])
		j = j+1

	print('pricing per timestep')
	j = 0
	while j<13:
		print(cc[j,0])
		j = j+1

	print('outdoor temp')
	j = 0
	while j<13:
		print(temp_outdoor[j,0])
		j = j+1
	print('solar radiation')
	j = 0
	while j<13:
		print(q_solar[j,0])
		j = j+1
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
	quit()

energy = x.value

# print(lp2.objective.value())
temp_indoor = matrix(0.0, (n,1))
temp_indoor[0,0] = temp_indoor_initial
p = 1
while p<n:
	temp_indoor[p,0] = timestep*(c1*(temp_outdoor[p-1,0]-temp_indoor[p-1,0])+c2*energy[p-1,0]+c3*q_solar[p-1,0])+temp_indoor[p-1,0]
	p = p+1
# Output[(block-1)*12:(block-1)*12+n,0] = energy[0:n,0] #0:12
# Output[(block-1)*12:(block-1)*12+n,1] = temp_indoor[0:n,0] #0:12
cost = cost + lp2.objective.value()	
#cost = cost + cc[0:12,0].trans()*energy[0:12,0]
# print(ii)
# print(cost)
# print(Output)
# print(temp_indoor)
# temp_indoor_initial= temp_indoor[12,0]
# print(temp_indoor_initial)

# # solve for thermostat temperature at each timestep
# thermo = matrix(0.0, (n,1))
# i = 0
# while i<n:
# 	thermo[i,0] = (-d2*temp_indoor[i,0]-d3*temp_outdoor[i]-d4*q_solar[i]+energy[i]*1000/12)/d1
# 	i = i+1

print('energy consumption')
j=0
while j<13:
	print(energy[j])
	j = j+1

print('indoor temp prediction')
j = 0
while j<13:
	print(temp_indoor[j,0])
	j = j+1

print('pricing per timestep')
j = 0
while j<13:
	print(cc[j,0])
	j = j+1

print('outdoor temp')
j = 0
while j<13:
	print(temp_outdoor[j,0])
	j = j+1
print('solar radiation')
j = 0
while j<13:
	print(q_solar[j,0])
	j = j+1
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

# print(Output)
# with pd.ExcelWriter('OutdoorTemp.xlsx', OCCUPANCY_MODE ='a') as writ # 'Occupancy',
# er:
# #	Output.to_excel(writer, sheet_name = 'Sheet2')
# 	df = pd.DataFrame(Output).T
# 	df.to_excel(writer, sheet_name = 'Sheet2')
# print("Total price =") 
# print(cost)
# print("Thermostat setup =") 
# print(thermo)
# print("Indoor Temperature =") 
# print(temp_indoor)
#print("--- %s seconds ---" % (time.process_time()-start_time))
