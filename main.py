from HMM import *
from Quantizer import *
from XMLParser import *

states = 8
emmissions = 6
filename = 'TESTGesture.xml'
sequences = []
X, Y = parseFile(filename)

for i in range(len(X)):
	sequence = quantize(X[i], Y[i], emmissions)
	sequences.append(sequence)


HMM = DiscreteHMM(states, emmissions)

HMM.train(sequences)

for sequence in sequences:
	print str(HMM.predict(sequence)) + '\n'