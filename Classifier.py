from HMM import *
from Sequence import *
import numpy as np

class Classifier():
	def __init__(self, names, states, emmissions, trained = False):
		self.names = names
		self.states = states
		self.emmissions = emmissions
		self.trained = trained
		
	def train(self):
		if self.trained:
			self.load_weights()
			return

		self.ClassesHMM = []
		for name in self.names:
			sequences = getSequences(name, self.emmissions)
			model = DiscreteHMM(self.states, self.emmissions)
			model.train(sequences)
			self.ClassesHMM.append(model)

	def load_weights(self):
		print 'TODO:'

	def predict(self, sequence):
		predictions = []

		for i in range(len(self.names)):
			predictions.append(self.ClassesHMM[i].predict(sequence))

		predictions = np.array(predictions)

		idx = np.argmax(predictions)

		return self.names[idx]

		