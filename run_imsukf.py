import numpy as np

def f(x_prev,w_matrix):
	x = np.array(x_prev[0]+x_prev[1],x_prev[1]+0.01)
	w = np.random.multivariate_normal([0,0],w_matrix,1)[0]
	return x+w

def z(x,x_i,y_i,R):
	z = -20 -23*np.log(y_i**2 + (x[0]-x_i)**2)
	v = np.random.normal(0,R,1)[0]