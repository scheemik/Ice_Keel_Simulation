"""
Written by: Rosalie Cormier, August 2021
"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp

#Both of these values must match those used in the simulation
dt = 5e-3
freq = 15

tasks = ['energy', 'salt'] #Can add more
task_titles = ['Total Energy', 'Total Salt']
ylabels = ['Energy (J)', 'Salt (g)']

numbers = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']
	#Modify as needed

for j in range(len(tasks)):

	task = tasks[j]
	task_title = task_titles[j]
	ylabel = ylabels[j]

	task_data = []

	for number in numbers:

		file = '{0}_tseries_{1}.txt'.format(task, number)

		y_data_i = np.loadtxt(file)
		task_data = np.concatenate([task_data, y_data_i])

	iterations = len(task_data)

	total_time = dt*freq*iterations

	t_data = np.linspace(0, total_time, iterations)

	plt.figure()
	plt.scatter(t_data, task_data, s=5, c='r')
	plt.xlabel('Time (s)')
	plt.ylabel(ylabel)
	plt.title(task_title)
	plt.savefig('{0}_fig'.format(task))
	plt.close()
