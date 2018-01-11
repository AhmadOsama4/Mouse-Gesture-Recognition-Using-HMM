from Quantizer import *
from XMLParser import *

def getSequences(filename, emmissions):
	filename = filename + '.xml'
	X, Y = parseFile(filename)
	sequences = []

	for i in range(len(X)):
		sequence = quantize(X[i], Y[i], emmissions)
		sequences.append(sequence)

	return sequences