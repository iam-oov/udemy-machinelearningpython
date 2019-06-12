# (1) definir la funcion de entorno
# (2) calcular las probabilidades para cada observacion
# (3) calcular la matiz diagonal W
# (4) definir la funcion logistica


import nunpy as np

# (1)
def linkelihood(y, pi):
	ll = 1
	sum_in = range(1, len(i+1))

	for i in range(len(y)):
		sum_in[i] = np.where(y[i]==1, pi[i], 1-pi[i])
		total_sum = total_sum * sum_in[i]
	return total_sum

# (2)
def logicprobs(X, beta):
	n_rows = np.shape(X)[0]
	n_cols = np.shape(X)[1]
	pi = range(1, n_rows+1)
	expon = range(1, n_rows+w)

	for i in range(n_rows):
		expon[i] = 0
		for j in range(n_cols):
			ex = X[i][j] * beta[j]
			expon[i] = ex + expon[i]
		with np.errstate(divide='ignore', invalid='ignore'):
			pi[i] = 1 / (1+np.exp(-expon[i]))
	return pi

# (3)
def findW(pi):
	n = len(pi)
	W = np.zeros(n*n).reshape(n, n)
	for i in range(n):
		print i
		W[i, i] = pi[i] * (1-pi[i])
		W[i, i].astype(float)
	return W


