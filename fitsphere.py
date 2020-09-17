#!/usr/bin/env python3
import numpy as np
import math as math
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, report_fit

def get_data():
	vf03= load_data(['sumfile_VF03-0001', 'sumfile_VF03-0005', 'sumfile_VF03-001',
		'sumfile_VF03-002', 'sumfile_VF03-005', 'sumfile_VF03-01',
		'sumfile_VF03-02', 'sumfile_VF03-03', 'sumfile_VF03-04', 'sumfile_VF03-05'])
	vf1= load_data(['sumfile_VF1-0001', 'sumfile_VF1-0005', 'sumfile_VF1-001',
		'sumfile_VF1-002', 'sumfile_VF1-005', 'sumfile_VF1-01',
		'sumfile_VF1-02', 'sumfile_VF1-03', 'sumfile_VF1-04', 'sumfile_VF1-05'])
	vf2= load_data(['sumfile_VF2-0001', 'sumfile_VF2-0005', 'sumfile_VF2-001',
		'sumfile_VF2-002', 'sumfile_VF2-005', 'sumfile_VF2-01',
		'sumfile_VF2-02', 'sumfile_VF2-03', 'sumfile_VF2-04', 'sumfile_VF2-05'])
	vf3= load_data(['sumfile_VF3-0001', 'sumfile_VF3-0005', 'sumfile_VF3-001',
		'sumfile_VF3-002', 'sumfile_VF3-005', 'sumfile_VF3-01',
		'sumfile_VF3-02', 'sumfile_VF3-03', 'sumfile_VF3-04', 'sumfile_VF3-05'])
	vf4= load_data(['sumfile_VF4-0001', 'sumfile_VF4-0005', 'sumfile_VF4-001',
		'sumfile_VF4-002', 'sumfile_VF4-005', 'sumfile_VF4-01',
		'sumfile_VF4-02', 'sumfile_VF4-03', 'sumfile_VF4-04', 'sumfile_VF4-05'])
	vf5= load_data(['sumfile_VF5-0001', 'sumfile_VF5-0005', 'sumfile_VF5-001',
		'sumfile_VF5-002', 'sumfile_VF5-005', 'sumfile_VF5-01',
		'sumfile_VF5-02', 'sumfile_VF5-03', 'sumfile_VF5-04', 'sumfile_VF5-05'])
	vf6= load_data(['sumfile_VF6-0005', 'sumfile_VF6-0005', 'sumfile_VF6-001',
		'sumfile_VF6-002', 'sumfile_VF6-005', 'sumfile_VF6-01',
		'sumfile_VF6-02', 'sumfile_VF6-03', 'sumfile_VF6-04', 'sumfile_VF6-05'])
	return vf03, vf1, vf2, vf3, vf4, vf5

def append_data(vf03, vf1, vf2, vf3, vf4, vf5):
	data = []
	data.append(vf03)
	data.append(vf1)
	data.append(vf2)
	data.append(vf3)
	data.append(vf4)
	data.append(vf5)
	return np.array(data)

def load_data(name):
	k = []
	for i in name:
		data = pd.read_csv(i, delim_whitespace=True,  skiprows=0, header=None) 
		k.append(data[0][1]) 
	return np.asarray(k/k[0])
	
def fun4(x, A, B, C, vf):
	fun= np.exp((-A*x)/((1-B*(vf**C))))
	return fun
	
def fit_dataset(params, i, x):
    A = params['A_%i' % (i+1)]
    B = params['B_%i' % (i+1)]
    C = params['C_%i' % (i+1)]
    vf = params['vf_%i' % (i+1)]
    return fun4(x, A, B, C, vf)	
    
def objective(params, x, data):
    ndata, _ = data.shape
    resid = 0.0*data[:]
    for i in range(ndata):
        resid[i, :] = data[i, :] - fit_dataset(params, i, x)
    return resid.flatten()	
    
	
def main():
	vf03, vf1, vf2, vf3, vf4, vf5= get_data()
	data = append_data(vf03, vf1, vf2, vf3, vf4, vf5)
	x = np.asarray([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

	
	fit_params = Parameters()
	for iy, y in enumerate(data):
		fit_params.add('A_%i' % (iy+1), value=0.3, min=0.2, max=0.4)
		fit_params.add('B_%i' % (iy+1), value=1.12572729, min=0.9, max=1.5)
		fit_params.add('C_%i' % (iy+1), value=0.31595050, min=0.2, max=0.5)
		if iy ==0:
			fit_params.add('vf_%i' % (iy+1), value=0.3, min=0.2999, max=0.30001)
		if iy ==1:
			fit_params.add('vf_%i' % (iy+1), value=1e-1, min=0.999e-1, max=1.001e-1)
		if iy ==2:
			fit_params.add('vf_%i' % (iy+1), value=1e-2, min=0.99e-2, max=1.01e-2)
		if iy ==3:
			fit_params.add('vf_%i' % (iy+1), value=1e-3, min=0.99e-3, max=1.01e-3)
		if iy ==4:
			fit_params.add('vf_%i' % (iy+1), value=1e-4, min=0.99e-4, max=1.01e-4)
		if iy ==5:
			fit_params.add('vf_%i' % (iy+1), value=1e-5, min=0.99e-5, max=1.01e-5)



	for iy in (2, 3, 4, 5, 6):
		fit_params['A_%i' % iy].expr = 'A_1'
		fit_params['B_%i' % iy].expr = 'B_1'
		fit_params['C_%i' % iy].expr = 'C_1'
	out = minimize(objective, fit_params, args=(x, data))
	report_fit(out.params)
	
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	
	#for i in range(6):

	y_fit = fit_dataset(out.params, 0, x)
	ax1.plot(x, data[0, :], 'o', x, y_fit, '-', label='vf=0.3')
	y_fit = fit_dataset(out.params, 1, x)
	ax1.plot(x, data[1, :], 'o', x, y_fit, '-', label='vf=0.1')
	y_fit = fit_dataset(out.params, 2, x)
	ax1.plot(x, data[2, :], 'o', x, y_fit, '-', label='vf=1e-2')
	y_fit = fit_dataset(out.params, 3, x)
	ax1.plot(x, data[3, :], 'o', x, y_fit, '-', label='vf=1e-3')
	y_fit = fit_dataset(out.params, 4, x)
	ax1.plot(x, data[4, :], 'o', x, y_fit, '-', label='vf=1e-4')
	y_fit = fit_dataset(out.params, 5, x)
	ax1.plot(x, data[5, :], 'o', x, y_fit, '-', label='vf=1e-5')
	
	ax1.legend()
	ax1.set_yscale('log')
	ax1.set_xscale('log')
	plt.show()

		
if __name__=='__main__':
	main()
