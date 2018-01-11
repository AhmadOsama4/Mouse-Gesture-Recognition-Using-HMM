from HMM import *
from Quantizer import *
from XMLParser import *
from Sequence import *
from Classifier import *

names = ['BottomLeftCornerGestures', 'TESTGesture']
states = 5
emissions = 6

classifier = Classifier(names, states, emissions)
classifier.train()

seq1 = getSequences('BottomLeftCornerGestures', emissions)

for seq in seq1:
	result = classifier.predict(seq)
	if result[0] == 'B':
		print 'Class 1 Correct'
	else:
		print 'Class 1 Wrong'

seq2 = getSequences('TESTGesture', emissions)
for seq in seq2:
	result = classifier.predict(seq)
	if result[0] == 'T':
		print 'Class 2 Correct'
	else:
		print 'Class 2 Wrong'
