#!/usr/bin/env python3
import numpy as np
import math as math
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, report_fit

def get_data():
	vf25_r02= load_data(['vf2.5e-4r=0.2/sumfile_JL1e-3', 
	'vf2.5e-4r=0.2/sumfile_JL2e-3', 'vf2.5e-4r=0.2/sumfile_JL5e-3', 
	'vf2.5e-4r=0.2/sumfile_JL1e-2', 'vf2.5e-4r=0.2/sumfile_JL2e-2',
	'vf2.5e-4r=0.2/sumfile_JL5e-2', 'vf2.5e-4r=0.2/sumfile_JL1e-1'])
	vf25_r05= load_data(['vf2.5e-4r=0.5/sumfile_JL1e-3', 
	'vf2.5e-4r=0.5/sumfile_JL2e-3', 'vf2.5e-4r=0.5/sumfile_JL5e-3', 
	'vf2.5e-4r=0.5/sumfile_JL1e-2', 'vf2.5e-4r=0.5/sumfile_JL2e-2',
	'vf2.5e-4r=0.5/sumfile_JL5e-2', 'vf2.5e-4r=0.5/sumfile_JL1e-1'])
	vf25_r10= load_data(['vf2.5e-4r=1.0/sumfile_JL1e-3', 
	'vf2.5e-4r=1.0/sumfile_JL2e-3', 'vf2.5e-4r=1.0/sumfile_JL5e-3', 
	'vf2.5e-4r=1.0/sumfile_JL1e-2', 'vf2.5e-4r=1.0/sumfile_JL2e-2',
	'vf2.5e-4r=1.0/sumfile_JL5e-2', 'vf2.5e-4r=1.0/sumfile_JL1e-1'])
	
	vf4_r02= load_data(['vf4e-6r=0.2/sumfile_JL1e-3', 
	'vf4e-6r=0.2/sumfile_JL2e-3', 'vf4e-6r=0.2/sumfile_JL5e-3', 
	'vf4e-6r=0.2/sumfile_JL1e-2', 'vf4e-6r=0.2/sumfile_JL2e-2',
	'vf4e-6r=0.2/sumfile_JL5e-2', 'vf4e-6r=0.2/sumfile_JL1e-1'])
	vf4_r05= load_data(['vf4e-6r=0.5/sumfile_JL1e-3', 
	'vf4e-6r=0.5/sumfile_JL2e-3', 'vf4e-6r=0.5/sumfile_JL5e-3', 
	'vf4e-6r=0.5/sumfile_JL1e-2', 'vf4e-6r=0.5/sumfile_JL2e-2',
	'vf4e-6r=0.5/sumfile_JL5e-2', 'vf4e-6r=0.5/sumfile_JL1e-1'])
	vf4_r10=load_data(['vf4e-6r=1.0/sumfile_JL1e-3', 
	'vf4e-6r=1.0/sumfile_JL2e-3', 'vf4e-6r=1.0/sumfile_JL5e-3', 
	'vf4e-6r=1.0/sumfile_JL1e-2', 'vf4e-6r=1.0/sumfile_JL2e-2',
	'vf4e-6r=1.0/sumfile_JL5e-2', 'vf4e-6r=1.0/sumfile_JL1e-1'])
	
	vf7_r02= load_data(['vf7.86e-4r=0.2/sumfile_JL1e-3', 
	'vf7.86e-4r=0.2/sumfile_JL2e-3', 'vf7.86e-4r=0.2/sumfile_JL5e-3', 
	'vf7.86e-4r=0.2/sumfile_JL1e-2', 'vf7.86e-4r=0.2/sumfile_JL2e-2',
	'vf7.86e-4r=0.2/sumfile_JL5e-2', 'vf7.86e-4r=0.2/sumfile_JL1e-1'])
	vf7_r05= load_data(['vf7.86e-4r=0.5/sumfile_JL1e-3', 
	'vf7.86e-4r=0.5/sumfile_JL2e-3', 'vf7.86e-4r=0.5/sumfile_JL5e-3', 
	'vf7.86e-4r=0.5/sumfile_JL1e-2', 'vf7.86e-4r=0.5/sumfile_JL2e-2',
	'vf7.86e-4r=0.5/sumfile_JL5e-2', 'vf7.86e-4r=0.5/sumfile_JL1e-1'])
	vf7_r10=load_data(['vf7.86e-4r=1.0/sumfile_JL1e-3',
	'vf7.86e-4r=1.0/sumfile_JL2e-3', 'vf7.86e-4r=1.0/sumfile_JL5e-3', 
	'vf7.86e-4r=1.0/sumfile_JL1e-2', 'vf7.86e-4r=1.0/sumfile_JL2e-2',
	'vf7.86e-4r=1.0/sumfile_JL5e-2', 'vf7.86e-4r=1.0/sumfile_JL1e-1'])
	

	data = append_data(vf25_r02, vf25_r05, vf25_r10, vf4_r02, vf4_r05, vf4_r10, vf7_r02, vf7_r05, vf7_r10)
	return data

def append_data(vf25_r02, vf25_r05, vf25_r10, vf4_r02, vf4_r05, vf4_r10, vf7_r02, vf7_r05, vf7_r10):
	data = []
	data.append(vf25_r02)
	data.append(vf25_r05)
	data.append(vf25_r10)
	data.append(vf4_r02)
	data.append(vf4_r05)
	data.append(vf4_r10)
	data.append(vf7_r02)
	data.append(vf7_r05)
	data.append(vf7_r10)
	return np.array(data)

def load_data(name):
	k = []
	for i in name:
		data = pd.read_csv(i, delim_whitespace=True,  skiprows=0, header=None) 
		k.append(data[0][5])
	return np.asarray(k/k[0])
	
#def fun3(x, A, B, C, r):
#	vf = ()/()
#	fun= 1/(np.exp(x*vf**A)**B)
#	return fun

def fun3(x, A, B, C, r, vf):
	fun= np.exp(-A*(x/r)/((1-B*pow(vf,C))))
	return fun
	
def fit_dataset(params, i, x, r, vf):
    A = params['A_%i' % (i+1)]
    B = params['B_%i' % (i+1)]
    C = params['C_%i' % (i+1)]
    return fun3(x, A, B, C, r, vf)	
    
def objective(params, x, data):
    ndata, _ = data.shape
    resid = 0.0*data[:]
    resid[0, :] = data[0, :] - fit_dataset(params, 0, x, 0.2, 2.5e-4)
    resid[1, :] = data[1, :] - fit_dataset(params, 1, x, 0.5, 2.5e-4)
    resid[2, :] = data[2, :] - fit_dataset(params, 2, x, 1.0, 2.5e-4)
    resid[3, :] = data[3, :] - fit_dataset(params, 3, x, 0.2, 4e-6)
    resid[4, :] = data[4, :] - fit_dataset(params, 4, x, 0.5, 4e-6)
    resid[5, :] = data[5, :] - fit_dataset(params, 5, x, 1.0, 4e-6)
    resid[6, :] = data[6, :] - fit_dataset(params, 6, x, 0.2, 7.86e-4)
    resid[7, :] = data[7, :] - fit_dataset(params, 7, x, 0.5, 7.86e-4)
    resid[8, :] = data[8, :] - fit_dataset(params, 8, x, 1.0, 7.86e-4)
    return resid.flatten()	
    
	
def main():
	data= get_data()
	x = np.asarray([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
	
	fit_params = Parameters()
	for iy, y in enumerate(data):
		fit_params.add('A_%i' % (iy+1), value=0.030094330, min=0.0, max=0.049)
		fit_params.add('B_%i' % (iy+1), value=1.199, min=1.0, max=1.45)
		fit_params.add('C_%i' % (iy+1), value=0.08566, min=0.0, max=0.1)

	for iy in (2, 3, 4, 5, 6, 7, 8, 9):
		fit_params['A_%i' % iy].expr = 'A_1'
		fit_params['B_%i' % iy].expr = 'B_1'
		fit_params['C_%i' % iy].expr = 'C_1'
	#print(data)
	out = minimize(objective, fit_params, args=(x, data))
	report_fit(out.params)
	
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	

	y_fit0 = fit_dataset(out.params, 0, x, 0.2, 2.5e-4)
	ax1.plot(x/0.2, data[0, :], 'o', x/0.2, y_fit0, '-', label='0')
	y_fit1 = fit_dataset(out.params, 1, x, 0.5, 2.5e-4)
	ax1.plot(x/0.5, data[1, :], 'o', x/0.5, y_fit1, '-', label='1')
	y_fit2 = fit_dataset(out.params, 2, x, 1.0, 2.5e-4)
	ax1.plot(x/1.0, data[2, :], 'o', x/1.0, y_fit2, '-', label='2')
	y_fit3 = fit_dataset(out.params, 3, x, 0.2, 4e-6)
	ax1.plot(x/0.2, data[3, :], 'o', x/0.2, y_fit3, '-', label='3')
	y_fit4 = fit_dataset(out.params, 4, x, 0.5, 4e-6)
	ax1.plot(x/0.5, data[4, :], 'o', x/0.5, y_fit4, '-', label='4')
	y_fit5 = fit_dataset(out.params, 5, x, 1.0, 4e-6)
	ax1.plot(x/1.0, data[5, :], 'o', x/1.0, y_fit5, '-', label='5')
	y_fit6 = fit_dataset(out.params, 6, x, 0.2, 7.86e-4)
	ax1.plot(x/0.2, data[6, :], 'o', x/0.2, y_fit6, '-', label='6')
	y_fit7 = fit_dataset(out.params, 7, x, 0.5, 7.86e-4)
	ax1.plot(x/0.5, data[7, :], 'o', x/0.5, y_fit7, '-', label='7')
	y_fit8 = fit_dataset(out.params, 8, x, 1.0, 7.86e-4)
	ax1.plot(x/1.0, data[8, :], 'o', x/1.0, y_fit8, '-', label='8')


	ax1.legend()
	ax1.set_yscale('log')
	ax1.set_xscale('log')
	plt.show()

		
if __name__=='__main__':
	main()
