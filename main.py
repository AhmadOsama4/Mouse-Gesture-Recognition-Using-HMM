from HMM import *
from Quantizer import *
from XMLParser import *
from Sequence import *
from Classifier import *

names = ['BottomLeftCorner', 'Tick']
#names = ['BottomLeftCorner']
states = 9
emissions = 8

classifier = Classifier(names, states, emissions)
classifier.train()

correct = 0
total = 0

#Test on test data of each class
for name in names:
	sequences = getSequences(name, emissions, 'test')
	for seq in sequences:
		result = classifier.predict(seq)
		if result == name:
			correct += 1
		total += 1

accuracy = 1.0*correct / total

print 'Model Accuracy ', accuracy
