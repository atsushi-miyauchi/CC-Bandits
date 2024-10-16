#ecoding:utf-8

import math
import random
from scipy.stats import bernoulli
from time import time
import os 
import statistics
import heapq

def choose_2(n): 
	return n * (n - 1) / 2

def tilde_log(n):
	val = 0
	for i in range(n+1): val += 1/(i+1)
	return val	

def cost(clustering, w_true): 
	n = len(w_true)
	seen = [[0] * n for i in range(n)]
	val = 0	
	for C in clustering: 
		for u in C:
			for v in C: 
				if u < v:
					val += (1 - w_true[u][v])
					seen[u][v] = 1
	for v in range(n):
		for u in range(v):
			if seen[u][v] == 0:
				val += w_true[u][v]
	return val

def Pivot(weight): 
	clustering = []
	m = choose_2(n)
	V_r_indicator = [1] * n
	V_r = range(n)
	num_nodes_removed = 0
	while(num_nodes_removed < n): 
		pivot = random.choice(V_r)
		C = [i for i in V_r if i != pivot and weight[min(pivot, i)][max(pivot, i)] > 0.5]
		C.append(pivot)
		clustering.append(C)
		for v in C: V_r_indicator[v] = 0
		V_r = [v for v in V_r if V_r_indicator[v] == 1]
		num_nodes_removed += len(C)
	return clustering

def Pivot_prev(weight):
	clustering = []
	num_nodes_removed = 0
	V_r = [1] * n
	while(num_nodes_removed < n): 
		pivot = random.choice([i for i in range(n) if V_r[i] == 1])
		C = [i for i in range(n) if V_r[i] == 1 and weight[min(pivot, i)][max(pivot, i)] > 0.5]
		C.append(pivot)
		clustering.append(C)
		for v in C: V_r[v] = 0
		num_nodes_removed += len(C)
	return clustering

def QCFB(T): 
	t1 = time()
	clustering = []
	m = choose_2(n)
	T_e = math.floor(math.floor(T) / m)
	V_r_indicator = [1] * n
	V_r = range(n)
	w_emp = {}
	num_nodes_removed = 0
	sample_complexity = 0 
	while(num_nodes_removed < n): 
		pivot = random.choice(V_r)
		for v in V_r: 
			if v != pivot: 
				rands = bernoulli.rvs(w_true[min(pivot,v)][max(pivot,v)], size=T_e)
				sample_complexity += T_e
				w_emp[min(pivot,v),max(pivot,v)] = sum(rands) / T_e
		C = [i for i in V_r if i != pivot and w_emp[min(pivot, i),max(pivot, i)] > 0.5]
		C.append(pivot)
		clustering.append(C)
		for v in C: V_r_indicator[v] = 0
		V_r = [v for v in V_r if V_r_indicator[v] == 1]
		if n - num_nodes_removed - len(C) >= 2: 
			T_e += math.floor(T_e * (choose_2(len(C)) + len(C) * (n - num_nodes_removed - len(C)) - (n - num_nodes_removed - 1)) / choose_2(n - num_nodes_removed - len(C)))
		num_nodes_removed += len(C)
	t2 = time()
	print("\nOutput of KC-FB")
	print("Clustering:", clustering)
	print("Cost of clustering:", cost(clustering, w_true))

def QCFC(delta=0.01, eps=10): 
	t1 = time()
	m = choose_2(n)
	eps_prime = eps / (12 * m)
	E_r = [[1] * n for i in range(n)]
	Good = [[0] * n for i in range(n)]
	Bad = [[0] * n for i in range(n)]
	w_emp = [[0] * n for i in range(n)]
	num_samples = [[0] * n for i in range(n)]
	for v in range(n): 
		for u in range(v):
			w_emp[u][v] = bernoulli.rvs(w_true[u][v])
			num_samples[u][v] = 1
	conf_rad = [[0] * n for i in range(n)]
	for v in range(n): 
		for u in range(v):
			conf_rad[u][v] = math.sqrt(math.log(4 * m / delta) / 2)
	num_edges_removed = 0

	h_max = []
	h_min = []

	num_max = [[1] * n for i in range(n)]
	num_min = [[1] * n for i in range(n)]

	for v in range(n): 
		for u in range(v):
			heapq.heappush(h_max, (-1*(w_emp[u][v] - conf_rad[u][v]), (u, v, 1)))
			heapq.heappush(h_min, (w_emp[u][v] + conf_rad[u][v], (u, v, 1)))

	while(num_edges_removed < m): 
		e_max = heapq.heappop(h_max)[1]
		while E_r[e_max[0]][e_max[1]] != 1 or e_max[2] != num_max[e_max[0]][e_max[1]]: e_max = heapq.heappop(h_max)[1]
		sample = bernoulli.rvs(w_true[e_max[0]][e_max[1]])
		w_emp[e_max[0]][e_max[1]] = (w_emp[e_max[0]][e_max[1]] * num_samples[e_max[0]][e_max[1]] + sample) / (num_samples[e_max[0]][e_max[1]] + 1)
		num_samples[e_max[0]][e_max[1]] += 1
		conf_rad[e_max[0]][e_max[1]] = math.sqrt(math.log(4 * m * (num_samples[e_max[0]][e_max[1]]** 2) / delta) / (2 * num_samples[e_max[0]][e_max[1]]))
		if w_emp[e_max[0]][e_max[1]] - conf_rad[e_max[0]][e_max[1]] >= 0.5 - eps_prime:
			Good[e_max[0]][e_max[1]] = 1
			E_r[e_max[0]][e_max[1]] = 0
			num_edges_removed += 1
		else: 
			heapq.heappush(h_max, (-1*(w_emp[e_max[0]][e_max[1]] - conf_rad[e_max[0]][e_max[1]]), (e_max[0], e_max[1], 1)))
			heapq.heappush(h_min, (w_emp[e_max[0]][e_max[1]] + conf_rad[e_max[0]][e_max[1]], (e_max[0], e_max[1], num_min[e_max[0]][e_max[1]] + 1)))
			num_min[e_max[0]][e_max[1]] += 1

		e_min = heapq.heappop(h_min)[1]
		while E_r[e_min[0]][e_min[1]] != 1 or e_min[2] != num_min[e_min[0]][e_min[1]]: 
			if len(h_min) > 0: e_min = heapq.heappop(h_min)[1]
			else: break
		sample = bernoulli.rvs(w_true[e_min[0]][e_min[1]])
		w_emp[e_min[0]][e_min[1]] = (w_emp[e_min[0]][e_min[1]] * num_samples[e_min[0]][e_min[1]] + sample) / (num_samples[e_min[0]][e_min[1]] + 1)
		num_samples[e_min[0]][e_min[1]] += 1
		conf_rad[e_min[0]][e_min[1]] = math.sqrt(math.log(4 * m * (num_samples[e_min[0]][e_min[1]]** 2) / delta) / (2 * num_samples[e_min[0]][e_min[1]]))
		if w_emp[e_min[0]][e_min[1]] + conf_rad[e_min[0]][e_min[1]] <= 0.5 + eps_prime:
			Bad[e_min[0]][e_min[1]] = 1
			E_r[e_min[0]][e_min[1]] = 0
			num_edges_removed += 1
		else: 
			heapq.heappush(h_min, (w_emp[e_min[0]][e_min[1]] + conf_rad[e_min[0]][e_min[1]], (e_min[0], e_min[1], 1)))
			heapq.heappush(h_max, (-1*(w_emp[e_min[0]][e_min[1]] - conf_rad[e_min[0]][e_min[1]]), (e_min[0], e_min[1], num_max[e_min[0]][e_min[1]] + 1)))
			num_max[e_min[0]][e_min[1]] += 1
	clustering = Pivot(Good)
	t2 = time()
	sample_complexity = sum([sum(num_samples[i]) for i in range(n)])
	print("\nOutput of KC-FC")
	print("Clustering:", clustering)
	print("Cost of clustering:", cost(clustering, w_true))

def Uniform_FB(T):
	t1 = time()
	m = choose_2(n)
	w_emp = [[0] * n for i in range(n)]
	num_samples = math.floor(math.floor(T) / m)
	sample_complexity = 0
	for v in range(n):
		for u in range(v):
			samples = bernoulli.rvs(w_true[u][v], size=num_samples)
			sample_complexity += num_samples
			w_emp[u][v] = sum(samples) / num_samples
	clustering = Pivot(w_emp)
	t2 = time()
	print("\nOutput of Uniform-FB")
	print("Clustering:", clustering)
	print("Cost of clustering:", cost(clustering, w_true))

def Uniform_FC(delta=0.01, eps=10): 
	m = choose_2(n)
	w_emp = [[0] * n for i in range(n)]
	num_samples = math.ceil(18 * (m ** 2) * math.log(2 * m / delta) / (eps ** 2))
	for v in range(n):
		for u in range(v):
			samples = bernoulli.rvs(w_true[u][v], size=num_samples)
			w_emp[u][v] = sum(samples) / num_samples
	clustering = Pivot(w_emp)
	print("\nOutput of Uniform-FC")
	print("Clustering:", clustering)
	print("Cost of clustering:", cost(clustering, w_true))

def Oracle():
	t1 = time()
	clustering = Pivot(w_true)
	t2 = time()
	print("\nOutput of KwikCluster with the true similarity")
	print("Clustering:", clustering)
	print("Cost of clustering:", cost(clustering, w_true))


random.seed(0)


# Run all algorithms in the FC setting for Lesmis
f = open("./instance_FC/lemis_050.txt", 'r')

for i, line in enumerate(f.readlines()):
	if i == 0: 
		line = line.strip().split()
		n = int(line[0])
		w_true = [[0] * n for i in range(n)]
		continue
	line = line.strip().split()
	u = int(line[0])
	v = int(line[1])
	w_true[min(u, v)][max(u, v)] = float(line[2])

QCFC(0.01, math.sqrt(n))
#Uniform_FC(0.01, math.sqrt(n)) # This is quite time-consuming


# Run all algorithms in the FB setting for Lesmis
f = open("./instance_FB/lemis.txt", 'r')

for i, line in enumerate(f.readlines()):
	if i == 0: 
		line = line.strip().split()
		n = int(line[0])
		w_true = [[0] * n for i in range(n)]
		continue
	line = line.strip().split()
	u = int(line[0])
	v = int(line[1])
	w_true[min(u, v)][max(u, v)] = float(line[2])

QCFB(n**2.1)
Uniform_FB(n**2.1)
Oracle()


