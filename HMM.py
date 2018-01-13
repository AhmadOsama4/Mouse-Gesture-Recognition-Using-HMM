import numpy as np
import math
import pickle

class DiscreteHMM():
	def __init__(self, no_of_states, no_of_emissions, name, trained = False):
		self.name = name
		self.num_states = no_of_states
		self.num_emissions = no_of_emissions
		self.trained = trained

		states = no_of_states
		emissions = no_of_emissions

		if trained:
			self.load_weights()
			return

		self.trans_prob = np.random.rand(states * states).reshape(states, states)
		self.emission_prob = np.random.rand(states * emissions).reshape(states, emissions)
		
		self.normalize()
		
	def normalize(self):
		self.trans_prob[0][0] = 0 #self transition in initial state
		self.trans_prob[:, 0] = 0 #transition from any state to initial state
		
		#normalize probabilities
		self.trans_prob = self.trans_prob / self.trans_prob.sum(axis = 1, keepdims = True)
		self.emission_prob = self.emission_prob / self.emission_prob.sum(axis = 1, keepdims = True)

	#forward algorithm
	def forward(self, sequence):
		N = self.num_states
		K = len(sequence)

		self.alpha = np.zeros((N, K))	
		
		self.alpha[0][0] = 1

		for i in range(1, N): # initialize with trans prob from initial state
			self.alpha[i][1] = self.trans_prob[0][i] * self.emission_prob[i][sequence[1]]

		for k in range(2, K): # sequence length
			for i in range(1, N): #current states
				for j in range(1, N): # previous state
					self.alpha[i][k] += self.alpha[j][k - 1] * self.trans_prob[j][i] * self.emission_prob[i][sequence[k]]				

		prob = self.alpha[:, K - 1].sum()
	
		return prob

	#backward algorithm
	def backward(self, sequence):
		N = self.num_states
		K = len(sequence)

		self.beta = np.zeros((N, K))

		for i in range(1, N):
			self.beta[i][K - 1] = self.emission_prob[i][sequence[K - 1]]

		for k in range(K - 2, 0, -1):
			for i in range(1, N): # current state
				for j in range(1, N): #next state
					self.beta[i][k] += self.beta[j][k + 1] * self.trans_prob[i][j] * self.emission_prob[i][sequence[k]]
		prob = 0.0
		for i in range(1, N):
			prob += self.beta[i][1] * self.trans_prob[0][i]

		return prob

	def buildGamma(self, P, limit):
		N = self.num_states
		self.gamma = np.zeros((N, N, limit))
		#print self.gamma.shape	

		for k in range(1, limit):
			for i in range(N):
				for j in range(1, N):
					self.gamma[i][j][k] = self.alpha[i][k - 1] * self.trans_prob[i][j] * self.beta[j][k] / P	


	def Run(self, sequence):
		#add initial unused emission for intial state 
		sequence = np.append(np.array([-1]) , sequence)

		FP = self.forward(sequence)		
		BP = self.backward(sequence)
		
		#print '\nForward Prob is ' , FP
		if np.isnan(FP):
			print 'NAN'
		#print 'Backward Prob is ' + str(BP)

		self.buildGamma(FP, len(sequence))
		
		N = self.num_states
		#update trans prob
		for i in range(N):
			div = self.gamma[i, :, :].sum()			
			for j in range(1, N):
				self.trans_prob[i][j] = self.gamma[i, j, :].sum() / div		
				
		#update emission prob
		for i in range(1, N - 1):
			div = self.gamma[i, :, :].sum()
			for j in range(self.num_emissions):
				indexes = (sequence == j)				
				self.emission_prob[i][j] = (self.gamma[i, :, indexes].sum() + 1) / div		

		self.normalize()

	#train using forward backward (Baum-Welch) algorithm
	def train(self, sequences, num_epoches = 4):
		if self.trained:
			return
		for epoche in range(num_epoches):
			idx = 0
			for sequence in sequences:
				self.Run(np.array(sequence))
										
		self.write_weights_to_file()

	def predict(self, sequence):
		return self.forward(sequence)

	def write_weights_to_file(self):
		filename = 'VariablesValues/' + self.name + '.pkl'
		
		f = open(filename, 'wb')
		pickle.dump(self.trans_prob, f)
		pickle.dump(self.emission_prob, f)

		f.close()

	def load_weights(self):
		filename = 'VariablesValues/' + self.name + '.pkl'

		f = open(filename, 'rb')
		self.trans_prob = pickle.load(f)
		self.emission_prob = pickle.load(f)

		f.close()
