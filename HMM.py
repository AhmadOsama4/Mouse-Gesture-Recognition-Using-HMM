import numpy as np
import math

class DiscreteHMM(object):
	def __init__(self, no_of_states, no_of_emissions):
		self.num_states = no_of_states
		self.num_emissions = no_of_emissions
		#add 2 states for initial and final
		states = no_of_states + 2
		emissions = no_of_emissions

		self.trans_prob = np.random.rand(states * states).reshape(states, states)
		self.emission_prob = np.random.rand(states * emissions).reshape(states, emissions)
		
		self.normalize()
		
	def normalize(self):
		'''
		self.trans_prob[self.num_states - 1] = 0 # transition from final state to any other state
		self.trans_prob[:, 0] = 0 #transition from any state to initial state
		self.trans_prob[0][self.num_states - 1] = 0 # transtion from initial to final state
		self.trans_prob[self.num_states - 1][self.num_states - 1] = 1 #final state to final state
		'''
		#normalize probabilities
		self.trans_prob = self.trans_prob / self.trans_prob.sum(axis = 1, keepdims = True)
		self.emission_prob = self.emission_prob / self.emission_prob.sum(axis = 1, keepdims = True)

	#forward algorithm
	def forward(self, sequence):
		N = self.num_states + 2
		K = len(sequence)
		self.alpha = np.zeros((N, K))

		for i in range(1, N - 1): # initialize with trans prob from initial state
			self.alpha[i][0] = self.trans_prob[0][i] * self.emission_prob[i][sequence[0]]

		for k in range(1, K): # sequence length
			for i in range(1, N - 1): #current states
				for j in range(1, N - 1): # previous state
					self.alpha[i][k] += self.alpha[j][k - 1] * self.trans_prob[j][i] * self.emission_prob[i][sequence[k]]
					
		prob = 0.0
		for i in range(1, N - 1):
			prob += self.alpha[i][K - 1] * self.trans_prob[i][N - 1]
			#print self.alpha[i][K - 1] * self.trans_prob[i][N - 1]

		return prob

	#backward algorithm
	def backward(self, sequence):
		N = self.num_states + 2
		K = len(sequence) + 1

		self.beta = np.zeros((N, K))

		self.beta[N - 1][K - 1] = 1.0

		for i in range(1, N - 1):
			self.beta[i][K - 2] = self.trans_prob[i][N - 1] * self.emission_prob[i][sequence[K - 2]]

		for k in range(len(sequence) - 2, -1, -1):
			for i in range(1, N - 1): # current state
				for j in range(1, N - 1): #next state
					self.beta[i][k] += self.beta[j][k + 1] * self.trans_prob[i][j] * self.emission_prob[i][sequence[k]]
		prob = 0.0
		for i in range(1, N - 1):
			prob += self.beta[i][0] * self.trans_prob[0][i]

		return prob

	def buildGamma(self, P, limit):
		N = self.num_states + 2
		self.gamma = np.zeros((N, N, limit + 1))
		#print self.gamma.shape
		#EPS = 1e-18

		for k in range(1, limit + 1):
			for i in range(1, N - 1):
				for j in range(1, N):
					self.gamma[i][j][k] = self.alpha[i][k - 1] * self.trans_prob[i][j] * self.beta[j][k] / P
					'''
					if self.gamma[i][j][k] < EPS:
						print str(i) + ' ' + str(j) + ' ' + str(k)
						print str(self.alpha[i][k - 1]) + ' ' + str(self.trans_prob[i][j]) + ' ' + str(self.beta[j][k])
						print 'Gamma ' + str(self.gamma[i][j][k])
						print ''
					'''

		#gamma at k = 0
		for i in range(1, N - 1):
			self.gamma[0][i][0] = self.trans_prob[0][i] * self.beta[i][0] / P
		
		#for i in range(1, N - 1):
			#from initial state
			#self.gamma[0][i][0] = self.beta[i][0] * self.trans_prob[0][i]

	def Run(self, sequence):
		FP = self.forward(sequence)		
		BP = self.backward(sequence)
		
		print '\nForward Prob is ' + str(FP)
		print 'Backward Prob is ' + str(BP)

		self.buildGamma(FP, len(sequence))
		
		
		N = self.num_states + 2
		#update trans prob
		for i in range(N - 1):
			div = self.gamma[i, :, :].sum()			
			for j in range(1, N):
				self.trans_prob[i][j] = self.gamma[i, j, :].sum() / div
				print str(i) + ' ' + str(j) + ' ' + str(self.trans_prob[i][j])


		#update emission prob
		for i in range(1, N - 1):
			div = self.gamma[i, :, :].sum()
			for j in range(self.num_emissions):
				indexes = (sequence == j)
				indexes = np.append(indexes, np.array([False]) )				
				
				self.emission_prob[i][j] = self.gamma[i, :, indexes].sum() / div 

	#train using forward backward (Baum-Welch) algorithm
	def train(self, sequences, num_epoches = 1):
		for epoche in range(num_epoches):
			for sequence in sequences:
				for c in range(1):
					self.Run(np.array(sequence))
					print ' \n '	
				break
			break

	def predict(self, sequence):
		return self.forward(sequence)
