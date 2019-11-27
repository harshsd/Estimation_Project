from ukf import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def get_transition_matrix(n):
	P = np.diag([1-0.01*n for i in range(n)])
	P = P+0.01
	# print(P)
	return P

def mv(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class centre:
	def __init__(self,num_nodes,node_positions,init_vector,init_variance,Q,R):
		assert len(node_positions) == num_nodes
		self.num_nodes = num_nodes
		self.nodes = [UKF_node(node_positions[i][0],node_positions[i][1],init_vector,init_variance,Q,R) for i in range(num_nodes)]
		self.P = get_transition_matrix(num_nodes)
		self.probabilities = np.zeros(num_nodes)
		self.probabilities[0] = 1

	def step(self,obs_vector):
		assert len(obs_vector) == self.num_nodes
		# print(self.probabilities)
		self.probabilities = np.matmul(self.probabilities,self.P)  #Estimate state from transition matrix
		node_values = []
		for i in range(self.num_nodes):
			self.nodes[i].predict_from_sigma(obs_vector[i])
			node = self.nodes[i]
			k = [node.state,node.variance,node.z_inv,node.Kz_new_sigma]  #state,variance,z_innovation,z_variance
			node_values.append(k)
		node_likeliness = []
		for i in range(self.num_nodes):
			random = np.random.standard_normal()*np.sqrt(node_values[i][3]) + node_values[i][2]
			# print(np.shape(random))
			node_likeliness.append(max(random[0][0],1e-5))
		# print(node_likeliness)
		self.probabilities = np.multiply(self.probabilities,node_likeliness)
		# print(self.probabilities)
		self.probabilities = self.probabilities/(np.sum(self.probabilities))
		# print(np.shape(self.probabilities))
		# print(np.shape(node_values[0][0]))
		self.x_predicted = np.sum([node_values[i][0]*self.probabilities[i] for i in range(len(self.probabilities))],axis=0)
		for i in range(len(self.probabilities)):
			if(i==0):
				self.Kx_predicted = self.probabilities[i]*(node_values[i][1] + np.outer(self.x_predicted-node_values[i][0],self.x_predicted-node_values[i][0]))
			else:
				self.Kx_predicted += self.probabilities[i]*(node_values[i][1] + np.outer(self.x_predicted-node_values[i][0],self.x_predicted-node_values[i][0]))
		# Perform smoothing here(TODO)
		next_prob = np.matmul(self.probabilities,self.P)
		c_j_i = np.zeros((self.num_nodes,self.num_nodes))
		for i in range(self.num_nodes):
			for j in range(self.num_nodes):
				c_j_i[j][i] = self.P[i][j]*self.probabilities[j]/next_prob[i]
				# print(c_j_i)
				# print(self.P[j][i]*self.probabilities[j]/next_prob[i])
		# print(c_j_i)
		for i in range(self.num_nodes):
			for j in range(self.num_nodes):
				if(j==0):
					x_i = node_values[j][0]*c_j_i[j][i]
				else:
					x_i += node_values[j][0]*c_j_i[j][i]
			for j in range(self.num_nodes):
				x_err = x_i - node_values[j][0]
				if(j==0):
					Kx_i = (node_values[j][1] + np.outer(x_err,x_err))*c_j_i[j][i]
				else:
					Kx_i += (node_values[j][1] + np.outer(x_err,x_err))*c_j_i[j][i]
			# print(Kx_i)
			self.nodes[i].state = x_i
			self.nodes[i].variance = Kx_i
			# print("****")
			# print(x_i)
			# print(node_values[i][0])