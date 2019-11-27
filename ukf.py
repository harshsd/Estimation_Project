import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def f(x_prev):
	x = np.array([x_prev[0]+x_prev[1],x_prev[1]+0.01])
	return x

def f_act(x_prev,w_matrix):
	x = f(x_prev)
	w = np.random.multivariate_normal([0,0],w_matrix,1)[0]
	return x+w

def z(x,x_i,y_i):
	a = x[0] - x_i
	a = a*a + y_i*y_i
	a = -20 - 23*np.log(a)
	return a

def z_act(x,x_i,y_i,R):
	a = z(x,x_i,y_i)
	a = a + np.random.standard_normal()*np.sqrt(R)
	return a

class UKF_node:
	def __init__(self,x,y,init_vector,init_variance,Q,R):
		self.n_sigma = 2*len(init_vector)+1
		self.x = x
		self.y = y
		self.state = np.copy(init_vector)
		self.variance = np.copy(init_variance)
		self.Q = Q
		self.R = R
		assert np.all(np.linalg.eigvals(self.variance) > 0), np.linalg.eigvals(self.variance)

	def update_sigma(self):
		assert np.all(np.linalg.eigvals(self.variance) > 0), np.linalg.eigvals(self.variance)
		S = scipy.linalg.sqrtm(self.variance)*np.sqrt(3)
		sigma_points = []
		weights = []
		sigma_points.append(self.state)
		weights.append(1/3)
		for i in range(len(self.state)):
			sigma_points.append(self.state+S[i])
			weights.append(1/6)
			sigma_points.append(self.state-S[i])
			weights.append(1/6)
		self.sigma_points = np.array(sigma_points)
		self.sigma_weights = np.array(weights)
		assert len(self.sigma_points) == self.n_sigma

	def predict_from_sigma(self,z_observed):
		self.update_sigma()
		#Predicting new x from sigma
		for i in range(self.n_sigma):
			sigma_i = f(self.sigma_points[i])
			if(i==0):
				self.x_new_sigma = self.sigma_weights[i]*sigma_i
			else:
				self.x_new_sigma += self.sigma_weights[i]*sigma_i
		#Predictinf Kx from sigma
		for i in range(self.n_sigma):
			sigma_i = f(self.sigma_points[i])
			del_sigma = sigma_i - self.x_new_sigma
			if(i==0):
				self.Kx_new_sigma = self.sigma_weights[i]*np.outer(del_sigma,del_sigma)
			else:
				self.Kx_new_sigma += self.sigma_weights[i]*np.outer(del_sigma,del_sigma)
		self.Kx_new_sigma += self.Q
		#Predicting z from sigma
		for i in range(self.n_sigma):
			sigma_i = f(self.sigma_points[i])
			if(i==0):
				self.z_new_sigma = self.sigma_weights[i]*z(sigma_i,self.x,self.y)
			else:
				self.z_new_sigma += self.sigma_weights[i]*z(sigma_i,self.x,self.y)
		#Predicting K_z and K_xz
		for i in range(self.n_sigma):
			sigma_z = z(f(self.sigma_points[i]),self.x,self.y)
			del_z = sigma_z - self.z_new_sigma
			del_sigma = f(self.sigma_points[i]) - self.x_new_sigma
			if(i==0):
				self.Kz_new_sigma = self.sigma_weights[i]*np.outer(del_z,del_z)
				self.Kxz_new_sigma = self.sigma_weights[i]*np.outer(del_sigma,del_z)
			else:
				self.Kz_new_sigma += self.sigma_weights[i]*np.outer(del_z,del_z)
				self.Kxz_new_sigma += self.sigma_weights[i]*np.outer(del_sigma,del_z)
		self.Kz_new_sigma = self.Kz_new_sigma + self.R
		#Update step
		phi = np.matmul(self.Kxz_new_sigma,np.linalg.inv(self.Kz_new_sigma))
		z_inv = [z_observed - self.z_new_sigma]
		self.z_inv = z_inv
		self.state = self.x_new_sigma + np.matmul(phi,z_inv)
		self.variance = self.Kx_new_sigma - np.matmul(phi,np.matmul(self.Kz_new_sigma,np.transpose(phi)))

# Main Code Starts Here
# init_x = [1,1]
# init_variance = np.diag([1,1])
# Q = np.diag([0.01,0.0001])
# R = 100
# num_iter = 1000
# x = np.copy(init_x)
# real_x = []
# real_obs = []
# for t in range(num_iter):
# 	x = f_act(x,Q)
# 	real_x.append(x[0])
# 	obs = z_act(x,-1,20,R)  #Change here
# 	real_obs.append(obs)
# #Predicting using observations
# ukf = UKF_node(-1,20,init_x,init_variance,Q,R)
# prediction_x = []
# for t in range(num_iter):
# 	ukf.predict_from_sigma(real_obs[t])
# 	prediction_x.append(ukf.state[0])
# # print(real_x)
# # print(prediction_x)

# #Plot Uncomment after issue sorted xD
# plt.figure()
# plt.plot(real_x)
# plt.plot(prediction_x)
# plt.show()