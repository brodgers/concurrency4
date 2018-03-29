import pandas as pd 
import numpy as np 
import pylab as plt

sm = 0.183645
med = 3.04395
big = 16.0363

def graph_speedup():
	seq = np.genfromtxt('out_single', delimiter='\n')
	mul = np.genfromtxt('out_multi', delimiter='\n')
	cud = np.genfromtxt('out_cuda', delimiter='\n')
	shr = np.genfromtxt('out_cuda_shared', delimiter='\n')

	small = [seq[0], mul[0], cud[0], shr[0]]
	medium = [seq[1], mul[1], cud[1], shr[1]]
	large = [seq[2], mul[2], cud[2], shr[2]]

	small = [sm/x for x in small]
	medium = [med/x for x in medium]
	large = [big/x for x in large]

	index = np.arange(4)
	bar_width = .2

	names = ['single-thread-cpu', '8-thread-spinlock', 'basic-cuda', 'cuda-shared-mem']

	plt.bar(index - bar_width, small, bar_width, label='2048')
	plt.bar(index, medium, bar_width, label='16384')
	plt.bar(index + bar_width, large, bar_width, label='65536')
	plt.xlabel('Implementation')
	plt.ylabel('Speedup')
	plt.xticks(index, names)
	plt.legend()

	plt.show()

graph_speedup()