import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

#Function Definitions (Chenage State Space and transiotion Matrix here)
def f(x_prev,w_matrix=[[-1]]):
	x = np.array([x_prev[0]+x_prev[1],x_prev[1]+0.01])
	if(w_matrix[0][0] == -1):
		return x
	w = np.random.multivariate_normal([0,0],w_matrix,1)[0]
	w=0
	return x+w

def z(x,x_i,y_i,R=None):
	z = -20-23*np.log(y_i**2 + (x[0]-x_i)**2)
	if(R == None):
		return z
	v = np.random.standard_normal()*np.sqrt(R)
	# assert z==v, str(z)+" "+str(z+v)
	return z+v

def get_z(x,pos,R):
	obs = []
	for i in range(len(pos)):
		obs.append([z(x,pos[i][0],pos[i][1],R),pos[i]])
	return obs

def get_sigma_points_and_weights(x,C,type_s='even'): #x is mean vector and C is covariance matrix
	#generate 2N points
	assert len(x) == 2
	assert np.all(np.linalg.eigvals(C) > 0), np.linalg.eigvals(C)
	S = scipy.linalg.sqrtm(len(x)*C)
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
	for i in range(6):
		P[i][i] = 0.95
	return P

def cross_prod(row_matrix):
	assert len(row_matrix) == 2
	row_matrix = row_matrix.reshape((1,2))
	K = np.matmul(np.transpose(row_matrix),row_matrix)
	return K



#Node Definition that carries out UKF
class Node:
	def __init__(self,x_node,y_node):
		self.state_vector = np.copy(x_mean)
		self.state_variance = np.copy(x_var)
		# self.x_sigma = sigma_points
		# self.w_sigma = weights
		self.z_sigma = np.zeros((4))
		self.z_til = 0 #Random Initialization. Never Used
		self.K_z = None #Updated later
		self.xpos = x_node
		self.ypos = y_node

	def get_sigma_points(self):
		[self.x_sigma,self.w_sigma] = get_sigma_points_and_weights(self.state_vector,self.state_variance)
		# print(np.mean(self.x_sigma,axis=0))
		# print(self.state_vector)

	def step(self,z_obs):
		print("Node State:")
		print(self.state_vector)
		assert z_obs[1][0] == self.xpos
		assert z_obs[1][1] == self.ypos
		z_obs = z_obs[0]
		x_predict = np.array([0.0,0.0])
		K_x_predict = np.array([[0.0,0.0],[0.0,0.0]])
		self.get_sigma_points()
		x_sigma = self.x_sigma
		w_sigma = self.w_sigma
		for i in range(len(self.x_sigma)):
			x_sigma[i] = f(x_sigma[i])
			x_predict = x_predict + w_sigma[i]*x_sigma[i]
		print("Prediction Vector")
		print(x_predict)
		for i in range(len(self.x_sigma)):
			x_b = x_sigma[i] - x_predict
			print(x_b)
			K_x_predict = K_x_predict + w_sigma[i]*cross_prod(x_b)
		print(K_x_predict)
		z_predict = 0
		for i in range(len(self.x_sigma)):
			self.z_sigma[i] = z(x_sigma[i],self.xpos,self.ypos)
			z_predict = z_predict + w_sigma[i]*self.z_sigma[i]
		print(z_predict)
		print(self.z_sigma)
		# z_predict = z_predict[0]
		K_z = 0
		K_xz = np.array([0.0,0.0])
		print("******")
		for i in range(len(self.x_sigma)):
			z_b = self.z_sigma[i] - z_predict
			x_b = x_sigma[i] - x_predict
			K_z += w_sigma[i]*z_b*z_b
			K_xz += w_sigma[i]*x_b*z_b
			assert w_sigma[i] == 0.25
		print(K_z)
		print(K_xz)
		# phi = K_xz/K_z
		phi = K_xz
		self.K_z = K_z
		self.z_til = z_obs - z_predict
		print("Z_til")
		print(self.z_til)
		print(z_obs)
		print(phi)
		self.state_vector = x_predict + self.z_til*phi
		self.state_variance = K_x_predict - K_z*cross_prod(phi)
		print('Node vector:')
		print(self.state_vector)
		print(self.state_variance)

	def message(self):
		return [self.state_vector,self.state_variance,self.z_til,self.K_z]


#Head Node that receives estimates from all nodes and performs filtering
class Head:
	def __init__(self,node_positions):
		self.time = 0
		self.nodes = [Node(node_positions[i][0],node_positions[i][1]) for i in range(len(node_positions))]
		self.likelihood = np.zeros(len(self.nodes))
		self.likelihood[:] = 1/(len(node_positions)) #Initially all nodes have same likelihood #This is c_tilda in paper
		print(self.likelihood)
		self.matrix = get_tran_matrix()
		self.matrix = [[1]]
		self.x_predict = None

	def head_step(self,observations):
		self.time+=1
		messages = []
		L_i = []
		for i in range(len(self.nodes)):
			self.nodes[i].step(observations[i])
			print(observations[i])
			message_i = self.nodes[i].message()
			messages.append(message_i)
			p = np.random.standard_normal()*np.sqrt(message_i[3]) + abs(message_i[2])
			if(p<0): p=0
			L_i.append(p) #Normal with mean z_til and variance K_z
		for i in range(len(self.likelihood)):
			self.likelihood[i] = self.likelihood[i]*L_i[i] #Update probability
		assert(np.sum(self.likelihood>0)) , np.sum(self.likelihood)
		self.likelihood = self.likelihood/(np.sum(self.likelihood)) #Normalization
		for i in range(len(self.likelihood)):
			x_i_predict = np.array([0.0,0.0])
			K_i = np.array([[0.0,0.0],[0.0,0.0]])
			for j in range(len(self.likelihood)):
				c_j_i = self.matrix[j][i]   #c and c tilda are same
				x_i_predict += c_j_i * messages[j][0]
			for j in range(len(self.likelihood)):
				x_b = x_i_predict - messages[j][0]
				K_i += c_j_i*(messages[j][1] + cross_prod(x_b))
			#Send message and update staus here directly
			self.nodes[i].state_vector = np.copy(x_i_predict)
			self.nodes[i].state_variance = np.copy(K_i)
		self.x_predict = np.array([0.0,0.0])
		for i in range(len(self.likelihood)):
			self.x_predict += self.likelihood[i]*messages[i][0]
		print("Overall Prediction")
		print(self.x_predict)


#Main Code here
#Constant Definitions
x_mean = np.array([1,1])  #Starting mean
x_var = np.array([[1,0],[0,1]])	#Variance of X
# Q = np.array([[0.01,0],[0,0.0001]])	# Variance of state noise
# R = 100	#Variance of observation noise
Q = np.array([[0,0],[0,0]])
R = 100
num_iter = 5
# node_positions = [[-1,20],[150,20],[300,20],[450,20],[600,20],[750,20]]
node_positions = [[-1,20]]

#Initialising system
sys = Head(node_positions)
x_actual = [1+np.random.standard_normal(),1+np.random.standard_normal()]
actual = []
obs = []
# Generate actual position and z_observations
for i in range(num_iter):
	x_actual = f(x_actual)
	actual.append(x_actual[0])
	obs.append(get_z(x_actual,node_positions,R))

track = []
for i in range(len(obs)):
	observation = obs[i]
	sys.head_step(observation)
	track.append(sys.x_predict[0])


print(actual)
print(track)

# Plot Results
# plt.figure()
# plt.plot(actual)
# plt.plot(track)
# plt.legend(["Actual","Track"])
# plt.show()