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

seq1 = getSequences('BottomLeftCorner', emissions)

correct = 0
total = 0

for seq in seq1:
	result = classifier.predict(seq)
	if result[0] == 'B':
		print 'Class 1 Correct'
		correct += 1
	else:
		print 'Class 1 Wrong'
	total += 1

seq2 = getSequences('Tick', emissions)
for seq in seq2:
	result = classifier.predict(seq)
	#print result
	if result[0] == 'T':
		print 'Class 2 Correct'
		correct += 1
	else:
		print 'Class 2 Wrong'
	total += 1

accuracy = 1.0*correct / total

print 'Accuracy ', accuracy
