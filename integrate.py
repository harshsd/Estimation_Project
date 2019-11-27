from ukf import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from centre import *

# Main Code Starts Here
a=[]
b=[]
c=[]
av=[]
bv=[]
cv=[]
for f in range(50):
	print(f)
	init_x = [1+np.random.standard_normal(),1+np.random.standard_normal()]
	init_est_x = np.array([1,1])
	init_variance = np.diag([1,1])
	Q = np.diag([0.01,0.0001])
	R = 100
	num_iter = 300
	x = np.copy(init_x)
	real_x = []
	real_obs = []
	real_v = []
	positions = [[-1, 20], [150,20], [300, 20], [450, 20], [600, 20], [750, 20]]
	np.random.seed(0)
	#Generating Actual Data
	for t in range(num_iter):
		x = f_act(x,Q)
		real_x.append(x[0])
		real_v.append(x[1])
		obs = [z_act(x,positions[i][0],positions[i][1],R) for i in range(len(positions))]
		real_obs.append(obs)

	#Generating Estimate using imukf
	imukf = centre(len(positions),positions,np.copy(init_est_x),init_variance,Q,R)
	prediction_x = []
	prediction_v = []
	for t in range(num_iter):
		imukf.step(real_obs[t])
		prediction_x.append(imukf.x_predicted[0])
		prediction_v.append(imukf.x_predicted[1])

	#Generating estimate only from ukf of sensor 1
	ukf = UKF_node(-1,20,np.copy(init_est_x),init_variance,Q,R)
	prediction_x_node1 = []
	prediction_v_node1 = []
	for t in range(num_iter):
		ukf.predict_from_sigma(real_obs[t][0])
		prediction_x_node1.append(ukf.state[0])
		prediction_v_node1.append(ukf.state[1])

	#Converting to arrays
	prediction_x = np.array(prediction_x)
	prediction_x_node1 = np.array(prediction_x_node1)
	real_x = np.array(real_x)

	a.append(real_x)
	av.append(real_v)
	b.append(prediction_x)
	bv.append(prediction_v)
	c.append(prediction_x_node1)
	cv.append(prediction_v_node1)

real_x = np.mean(a,axis=0)
real_v = np.mean(av,axis=0)
prediction_x = np.mean(b,axis=0)
prediction_v = np.mean(bv,axis=0)
prediction_x_node1 = np.mean(c,axis=0)
prediction_v_node1 = np.mean(cv,axis=0)

markers = ['o','P','X','s','*']

#Plot Uncomment after issue sorted xD
plt.figure()
plt.title('Mean Postion Error vs Time')
plt.xlabel('Time')
plt.ylabel('Mean Position Error')
plt.plot(mv(abs(prediction_x-real_x)),marker='o',markevery=30)
plt.plot(mv(abs(prediction_x_node1-real_x)),marker='P',markevery=30)
plt.legend(['imukf','ukf'])
plt.savefig('error.png')

plt.figure()
plt.title('Velocity (Real vs Tracked)')
plt.plot(real_v,marker='o',markevery=30)
plt.plot(prediction_v,marker='P',markevery=30)
plt.plot(prediction_v_node1,marker='X',markevery=30)
plt.legend(['real_v','imukf_v','ukf_v'])
plt.savefig('vel.png')

plt.figure()
plt.title('Position (Real vs Tracked)')
plt.plot(real_x,marker='o',markevery=30)
plt.plot(prediction_x,marker='P',markevery=30)
plt.plot(prediction_x_node1,marker='x',markevery=30)
plt.legend(['real','imukf','ukf'])
plt.savefig('pos.png')

pickle.dump([real_x,prediction_x,prediction_x_node1],open('run.pickle','wb'))

plt.show()