# Code to test how estimate error changes with node density
from ukf import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from centre import *
import signal
import sys

def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        plt.close(fig)
        sys.exit(0)

def make_pos(n):
	start = -1
	end = 750
	pos = []
	x=start
	for k in range(n+1):
		pos.append([x,20])
		x+=(end-start)/n
	# print(pos)
	return pos


signal.signal(signal.SIGINT, signal_handler)

#Defining various node densities here
pos1 = make_pos(10)
pos2 = make_pos(50)
pos3 = make_pos(25)

#Running the real and 3 estimates here
a=[]
e1=[]
e2=[]
e3=[]

num_run = 50

for f in range(num_run):
	print(f)
	init_x = [1+np.random.standard_normal(),1+np.random.standard_normal()]
	init_est_x = np.array([1,1])
	init_variance = np.diag([1,1])
	Q = np.diag([0.01,0.0001])
	R = 100
	num_iter = 150
	x = np.copy(init_x)
	real_x = []
	real_obs = []
	real_obs_low = []
	real_obs_medium = []
	real_v = []
	np.random.seed(f)
	#Generating Actual Data
	for t in range(num_iter):
		x = f_act(x,Q)
		real_x.append(x[0])
		real_v.append(x[1])
		obs = [z_act(x,pos3[i][0],pos3[i][1],R) for i in range(len(pos3))]
		obs_low = [z_act(x,pos1[i][0],pos1[i][1],R) for i in range(len(pos1))]
		obs_med = [z_act(x,pos2[i][0],pos2[i][1],R) for i in range(len(pos2))]
		real_obs.append(obs)
		real_obs_low.append(obs_low)
		real_obs_medium.append(obs_med)
	a.append(real_x)

	#Generating Estimate using imukf
	imukf = centre(len(pos3),pos3,np.copy(init_est_x),init_variance,Q,R)
	prediction_x = []
	prediction_v = []
	for t in range(num_iter):
		imukf.step(real_obs[t])
		prediction_x.append(imukf.x_predicted[0])
		prediction_v.append(imukf.x_predicted[1])
	e3.append(prediction_x)

	imukf = centre(len(pos2),pos2,np.copy(init_est_x),np.copy(init_variance),Q,R)
	prediction_x = []
	prediction_v = []
	for t in range(num_iter):
		imukf.step(real_obs_medium[t])
		prediction_x.append(imukf.x_predicted[0])
		prediction_v.append(imukf.x_predicted[1])
	e2.append(prediction_x)

	imukf = centre(len(pos1),pos1,np.copy(init_est_x),init_variance,Q,R)
	prediction_x = []
	prediction_v = []
	for t in range(num_iter):
		imukf.step(real_obs_low[t])
		prediction_x.append(imukf.x_predicted[0])
		prediction_v.append(imukf.x_predicted[1])
	e1.append(prediction_x)


assert len(a)==len(e1)
assert len(e2)==len(e2)
assert len(a)==len(e2)

n=20
real_x = mv(np.mean(a,axis=0),n=n)
x1 = mv(np.mean(e1,axis=0),n=n)
x2 = mv(np.mean(e2,axis=0),n=n)
x3 = mv(np.mean(e3,axis=0),n=n)

fig=plt.figure()
plt.plot(abs(x1-real_x))
plt.plot(abs(x2-real_x))
plt.plot(abs(x3-real_x))
plt.legend(['Low','Medium','High'])
print('Press Ctrl+C')
# signal.pause()
plt.show()