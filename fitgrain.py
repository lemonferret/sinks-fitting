#!/usr/bin/env python3
import numpy as np
import math as math
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, report_fit

def get_data():
	R10_r05= load_data(['sumfile_R10_r05_JL1e-3','sumfile_R10_r05_JL2e-3',
		'sumfile_R10_r05_JL5e-3', 'sumfile_R10_r05_JL1e-2', 'sumfile_R10_r05_JL2e-2',
		'sumfile_R10_r05_JL5e-2', 'sumfile_R10_r05_JL1e-1', 'sumfile_R10_r05_JL2e-1',
		'sumfile_R10_r05_JL3e-1', 'sumfile_R10_r05_JL4e-1', 'sumfile_R10_r05_JL5e-1' ])
	R10_r1= load_data(['sumfile_R10_r1_JL1e-3','sumfile_R10_r1_JL2e-3',
		'sumfile_R10_r1_JL5e-3', 'sumfile_R10_r1_JL1e-2', 'sumfile_R10_r1_JL2e-2',
		'sumfile_R10_r1_JL5e-2', 'sumfile_R10_r1_JL1e-1', 'sumfile_R10_r1_JL2e-1',
		'sumfile_R10_r1_JL3e-1', 'sumfile_R10_r1_JL4e-1', 'sumfile_R10_r1_JL5e-1' ])
	R100_r05= load_data(['sumfile_R100_r05_JL1e-3','sumfile_R100_r05_JL2e-3',
		'sumfile_R100_r05_JL5e-3', 'sumfile_R100_r05_JL1e-2', 'sumfile_R100_r05_JL2e-2',
		'sumfile_R100_r05_JL5e-2', 'sumfile_R100_r05_JL1e-1', 'sumfile_R100_r05_JL2e-1',
		'sumfile_R100_r05_JL3e-1', 'sumfile_R100_r05_JL4e-1', 'sumfile_R100_r05_JL5e-1' ])
	R100_r1= load_data(['sumfile_R100_r1_JL1e-3','sumfile_R100_r1_JL2e-3',
		'sumfile_R100_r1_JL5e-3', 'sumfile_R100_r1_JL1e-2', 'sumfile_R100_r1_JL2e-2',
		'sumfile_R100_r1_JL5e-2', 'sumfile_R100_r1_JL1e-1', 'sumfile_R100_r1_JL2e-1',
		'sumfile_R100_r1_JL3e-1', 'sumfile_R100_r1_JL4e-1', 'sumfile_R100_r1_JL5e-1' ])
	R100_r2= load_data(['sumfile_R100_r2_JL1e-3','sumfile_R100_r2_JL2e-3',
		'sumfile_R100_r2_JL5e-3', 'sumfile_R100_r2_JL1e-2', 'sumfile_R100_r2_JL2e-2',
		'sumfile_R100_r2_JL5e-2', 'sumfile_R100_r2_JL1e-1', 'sumfile_R100_r2_JL2e-1',
		'sumfile_R100_r2_JL3e-1', 'sumfile_R100_r2_JL4e-1', 'sumfile_R100_r2_JL5e-1' ])
	R1000_r05= load_data(['sumfile_R1000_r05_JL1e-3','sumfile_R1000_r05_JL2e-3',
		'sumfile_R1000_r05_JL5e-3', 'sumfile_R1000_r05_JL1e-2', 'sumfile_R1000_r05_JL2e-2',
		'sumfile_R1000_r05_JL5e-2', 'sumfile_R1000_r05_JL1e-1', 'sumfile_R1000_r05_JL2e-1',
		'sumfile_R1000_r05_JL3e-1', 'sumfile_R1000_r05_JL4e-1', 'sumfile_R1000_r05_JL5e-1' ])
	R1000_r1= load_data(['sumfile_R1000_r1_JL1e-3','sumfile_R1000_r1_JL2e-3',
		'sumfile_R1000_r1_JL5e-3', 'sumfile_R1000_r1_JL1e-2', 'sumfile_R1000_r1_JL2e-2',
		'sumfile_R1000_r1_JL5e-2', 'sumfile_R1000_r1_JL1e-1', 'sumfile_R1000_r1_JL2e-1',
		'sumfile_R1000_r1_JL3e-1', 'sumfile_R1000_r1_JL4e-1', 'sumfile_R1000_r1_JL5e-1' ])
	R1000_r2= load_data(['sumfile_R1000_r2_JL1e-3','sumfile_R1000_r2_JL2e-3',
		'sumfile_R1000_r2_JL5e-3', 'sumfile_R1000_r2_JL1e-2', 'sumfile_R1000_r2_JL2e-2',
		'sumfile_R1000_r2_JL5e-2', 'sumfile_R1000_r2_JL1e-1', 'sumfile_R1000_r2_JL2e-1',
		'sumfile_R1000_r2_JL3e-1', 'sumfile_R1000_r2_JL4e-1', 'sumfile_R1000_r2_JL5e-1' ])
	
	data = append_data(R10_r05, R10_r1, R100_r05, R100_r1, R100_r2,
		R1000_r05, R1000_r1, R1000_r2)
	return data

def append_data(R10_r05, R10_r1, R100_r05, R100_r1, R100_r2,
	R1000_r05, R1000_r1, R1000_r2):
	data = []
	data.append(R10_r05)
	data.append(R10_r1)
	data.append(R100_r05)
	data.append(R100_r1)
	data.append(R100_r2)
	data.append(R1000_r05)
	data.append(R1000_r1)
	data.append(R1000_r2)
	return np.array(data)

def load_data(name):
	k = []
	for i in name:
		data = pd.read_csv(i, delim_whitespace=True,  skiprows=0, header=None) 
		k.append(data[0][4])
	return np.asarray(k/k[0])
	
def fun3(x, A, B, C, R, r):
	vf = (((4/3)*math.pi*((R+r)**3-(R**3)))/((4/3)*math.pi*((R)**3)))
	fun= 1/(np.exp(x*vf**A)**B)
	return fun

def fun4(x, A, B, C, R, r):
	vf = (((4/3)*math.pi*((R+r)**3-(R**3)))/((4/3)*math.pi*((R)**3)))
	fun= np.exp(-A*(x)/((1-B*vf**C)))
	return fun
	
def fit_dataset(params, i, x):
    A = params['A_%i' % (i+1)]
    B = params['B_%i' % (i+1)]
    C = params['C_%i' % (i+1)]
    R = params['R_%i' % (i+1)]
    r = params['r_%i' % (i+1)]
    return fun3(x, A, B, C, R, r)	
    
def objective(params, x, data):
    ndata, _ = data.shape
    resid = 0.0*data[:]
    resid[0, :] = data[0, :] - fit_dataset(params, 0, x/0.5)
    resid[1, :] = data[1, :] - fit_dataset(params, 1, x/1)
    resid[2, :] = data[2, :] - fit_dataset(params, 2, x/0.5)
    resid[3, :] = data[3, :] - fit_dataset(params, 3, x/1)
    resid[4, :] = data[4, :] - fit_dataset(params, 4, x/2)
    resid[5, :] = data[5, :] - fit_dataset(params, 5, x/0.5)
    resid[6, :] = data[6, :] - fit_dataset(params, 6, x/1)
    resid[7, :] = data[7, :] - fit_dataset(params, 7, x/2)
    return resid.flatten()	
    
	
def main():
	data= get_data()
	x = np.asarray([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
	
	fit_params = Parameters()
	for iy, y in enumerate(data):
		fit_params.add('A_%i' % (iy+1), value=-0.0194330, min=-0.020, max=-0.019)
		fit_params.add('B_%i' % (iy+1), value=1.439857135, min=1.43, max=1.45)
		#fit_params.add('A_%i' % (iy+1), value=1.50056, min=1.5, max=1.52)
		#fit_params.add('B_%i' % (iy+1), value=-0.3116278, min=-0.32, max=-0.33)
		fit_params.add('C_%i' % (iy+1), value=2.29421695, min=2.2, max=2.4)
		if iy ==0:
			fit_params.add('R_%i' % (iy+1), value=10, min=9.99, max=10.01)
			fit_params.add('r_%i' % (iy+1), value=0.5, min=0.499, max=0.501)
		if iy ==1:
			fit_params.add('R_%i' % (iy+1), value=10, min=9.99, max=10.01)
			fit_params.add('r_%i' % (iy+1), value=1, min=0.999, max=1.001)
		if iy ==2:
			fit_params.add('R_%i' % (iy+1), value=100, min=99.99, max=100.01)
			fit_params.add('r_%i' % (iy+1), value=0.5, min=0.499, max=0.501)
		if iy ==3:
			fit_params.add('R_%i' % (iy+1), value=100, min=99.99, max=100.01)
			fit_params.add('r_%i' % (iy+1), value=1, min=0.999, max=1.001)
		if iy ==4:
			fit_params.add('R_%i' % (iy+1), value=100, min=99.99, max=100.01)
			fit_params.add('r_%i' % (iy+1), value=2, min=1.999, max=2.001)
		if iy ==5:
			fit_params.add('R_%i' % (iy+1), value=1000, min=999.99, max=1000.01)
			fit_params.add('r_%i' % (iy+1), value=0.5, min=0.499, max=0.501)
		if iy ==6:
			fit_params.add('R_%i' % (iy+1), value=1000, min=999.99, max=1000.01)
			fit_params.add('r_%i' % (iy+1), value=1, min=0.999, max=1.001)
		if iy ==7:
			fit_params.add('R_%i' % (iy+1), value=1000, min=999.99, max=1000.01)
			fit_params.add('r_%i' % (iy+1), value=2, min=1.999, max=2.001)


	for iy in (2, 3, 4, 5, 6, 7, 8):
		fit_params['A_%i' % iy].expr = 'A_1'
		fit_params['B_%i' % iy].expr = 'B_1'
		fit_params['C_%i' % iy].expr = 'C_1'
	#print(data)
	out = minimize(objective, fit_params, args=(x, data))
	report_fit(out.params)
	
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	

	y_fit0 = fit_dataset(out.params, 0, x/(10-0.5))
	ax1.plot(x/(10-0.5), data[0, :], 'o', x/(10-0.5), y_fit0, '-', label='10/0.5')
	y_fit1 = fit_dataset(out.params, 1, x/(10-1))
	ax1.plot(x/(10-1), data[1, :], 'o', x/(10-1), y_fit1, '-', label='10/1')
	y_fit2 = fit_dataset(out.params, 2, x/(100-0.5))
	ax1.plot(x/(100-0.5), data[2, :], 'o', x/(100-0.5), y_fit2, '-', label='100/0.5')
	y_fit3 = fit_dataset(out.params, 3, x/(100-1))
	ax1.plot(x/(100-1), data[3, :], 'o', x/(100-1), y_fit3, '-', label='100/1')
	y_fit4 = fit_dataset(out.params, 4, x/(100-2))
	ax1.plot(x/(100-2), data[4, :], 'o', x/(100-2), y_fit4, '-', label='100/2')
	y_fit5 = fit_dataset(out.params, 5, x/(1000-0.5))
	ax1.plot(x/(1000-0.5), data[5, :], 'o', x/(1000-0.5), y_fit5, '-', label='1000/0.5')
	y_fit6 = fit_dataset(out.params, 6, x/(1000-1))
	ax1.plot(x/(1000-1), data[6, :], 'o', x/(1000-1), y_fit6, '-', label='1000/1')
	y_fit7 = fit_dataset(out.params, 7, x/(1000-2))
	ax1.plot(x/(1000-2), data[7, :], 'o', x/(1000-2), y_fit7, '-', label='1000/2')

	chi2 =0
	y_fit =[*y_fit0, *y_fit1, *y_fit2, *y_fit3, *y_fit4, *y_fit5, *y_fit6, *y_fit7]
	i= 0
	for one in data:
		for n in one:
			chi2 = chi2 + ((y_fit[i]-n)**2)
			i +=1
			
	print(chi2) 	#fun4 = 3.709193321083706e-05 !!!
					#fun3 = 3.912769188437742e-05
	ax1.legend()
	ax1.set_yscale('log')
	ax1.set_xscale('log')
	plt.show()

		
if __name__=='__main__':
	main()
