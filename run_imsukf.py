import numpy as np
import matplotlib.pyplot as plt

#Constant Definitions
x_mean = np.array([1 1])  #Starting mean
x_var = np.array([[1,0],[0,1]])	#Variance of X
Q = np.array([[0.01,0],[0,0.0001]])	# Variance of state noise
R = 100	#Variance of observation noise


#Function Definitions (Chenage State Space and transiotion Matrix here)
def f(x_prev,w_matrix=None):
	x = np.array([x_prev[0]+x_prev[1],x_prev[1]+0.01])
	if(w_matrix == None):
		return x
	w = np.random.multivariate_normal([0,0],w_matrix,1)[0]
	return x+w

def z(x,x_i,y_i,R=None):
	z = -20 -23*np.log(y_i**2 + (x[0]-x_i)**2)
	if(R == None):
		return z
	v = np.random.normal(0,R,1)[0]
	return z+v

def get_sigma_points_and_weights(x,C,type_s='even'): #x is mean vector and C is covariance matrix
	#generate 2N points
	assert len(x) == 2
	S = np.sqrt(len(x)*C)
	sigma_points = []
	weights = []
	for i in range(len(x)):
		sigma_points.append(x+S[i])
		weights.append(1/(2*len(x)))
		sigma_points.append(x-S[i])
		weights.append(1/(2*len(x)))
	sigma_points = np.array(sigma_points)
	weights = np.array(weights)
	if(type_s=='even'):
		return [sigma_points,weights]
	else:
		return None #Add here to make 2n+1 sigma points

def get_tran_matrix():
	P = np.zeros((6,6))
	P[:] = 0.01
	for i in range(5):
		P[i][i] = 0.95
	return P

def cross_prod(row_matrix):
	return np.matmul(np.transpose(row_matrix),row_matrix)



#Node Definition that carries out UKF
class Node:
	def __init__(self,weights,x_node,y_node):
		self.state_vector = np.copy(x_mean)
		self.state_variance = np.copy(x_var)
		# self.x_sigma = sigma_points
		# self.w_sigma = weights
		self.z_sigma = np.zeros((len(sigma_points)))
		self.z_til = 0 #Random Initialization. Never Used
		self.K_z = None #Updated later

	def get_sigma_points():
		[self.x_sigma,self.w_sigma] = get_sigma_points_and_weights(self.state_vector,self.state_variance)

	def step(self,z_obs):
		x_predict = np.array([0,0])
		K_x_predict = np.array([0,0],[0,0])
		self.get_sigma_points()
		x_sigma = self.x_sigma
		w_sigma = self.w_sigma
		for i in range(len(self.x_sigma)):
			x_sigma[i] = f(x_sigma)
			x_predict = x_predict + w_sigma[i]*x_sigma[i]
		for i in range(len(self.x_sigma))
			x_b = x_sigma[i] - x_predict
			K_x_predict = K_x_predict + w_sigma[i]*cross_prod(x_b)
		z_predict = 0
		for i in range(len(self.x_sigma)):
			z_sigma[i] = z(x_sigma[i])
			z_predict = z_predict + w_sigma[i]*z_sigma[i]
		for i in range(len(self.x_sigma)):
			z_b = z_sigma[i] - z_predict
			x_b = x_sigma[i] - x_predict
			K_z += w_sigma[i]*z_b*z_b
			K_xz += w_sigma[i]*x_b*z_b
		phi = (1/K_z)*K_xz
		self.K_z = K_z
		self.z_til = z_obs - z_predict
		self.state_vector = x_predict + self.z_til*phi
		self.state_variance = K_x_predict - K_z*cross_prod(phi)

	def message(self):
		return [self.state_vector,self.state_variance,self.z_til,self.K_z]


#Head Node that receives estimates from all nodes and performs filtering
class Head:
	def __init__(self,node_positions,sigma_points):
		self.time = 0
		self.nodes = [Node(sigma_points,weights,node_positions[i][0],node_positions[i][1]) for i in range(len(node_positions))]
		self.likelihood = np.zeros(len(self.nodes))
		self.likelihood[:] = 1/6 #Initially all nodes have same likelihood #This is c_tilda in paper
		self.matrix = get_tran_matrix()

	def head_step(self,observations):
		self.time+=1
		messages = []
		L_i = []
		for i in range(len(self.nodes)):
			self.nodes[i].step(observations[i])
			message_i = self.nodes[i].message
			messages.append(message_i)
			L_i.append(np.random.standard_normal()*np.sqrt(message_i[3]) + message_i[2]) #Normal with mean z_til and variance K_z
		for i in range(len(self.likelihood)):
			self.likelihood[i] = self.likelihood[i]*L_i[i] #Update probability
			self.likelihood = self.likelihood/(np.sum(self.likelihood)) #Normalization
		for i in range(len(self.likelihood)):
			x_i_predict = np.array([0,0])
			K_i = np.array([[0,0],[0,0]])
			for j in range(len(self.likelihood)):
				c_j_i = self.matrix[j][i]   #c and c tilda are same
				x_i_predict += c_j_i * messages[j][0]
			for j in range(len(self.likelihood)):
				x_b = x_i_predict - messages[j][0]
				K_i += c_j_i*(messages[j][1] + cross_prod(x_b))
			#Send message and update staus here directly
			self.nodes[i].state_vector = np.copy(x_i_predict)
			self.nodes[i].state_variance = np.copy(K_i)