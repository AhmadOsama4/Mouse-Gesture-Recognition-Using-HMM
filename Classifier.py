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
		self.ClassesHMM = []

		if self.trained: # load weights values if trained before
			for name in self.names:
				model = DiscreteHMM(self.states, self.emmissions, name, True)
				self.ClassesHMM.append(model)
			return

		
		for name in self.names:
			sequences = getSequences(name, self.emmissions)
			model = DiscreteHMM(self.states, self.emmissions, name)
			model.train(sequences)
			self.ClassesHMM.append(model)

	def predict(self, sequence):
		predictions = []

		for i in range(len(self.names)):
			predictions.append(self.ClassesHMM[i].predict(sequence))

		predictions = np.array(predictions)

		idx = np.argmax(predictions)

		return self.names[idx]

		