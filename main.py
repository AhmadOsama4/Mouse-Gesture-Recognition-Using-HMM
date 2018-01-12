from HMM import *
from Quantizer import *
from XMLParser import *
from Sequence import *
from Classifier import *

names = ['BottomLeftCorner', 'Tick', 'UpperRightCorner', 'UpperLeftCorner', 'LessThan', 'GreaterThan', 'Square', 'Triangle']
#names = ['BottomLeftCorner', 'UpperRightCorner', 'Tick', 'Circle',  'UpperLeftCorner']
#names = ['Triangle', 'Tick', 'LessThan']
#names = ['Tick', 'UpperRightCorner', 'UpperLeftCorner', 'LessThan', 'GreaterThan', 'Square', 'Triangle']
#names = ['Tick', 'UpperRightCorner', 'UpperLeftCorner', 'GreaterThan', 'Square', 'Triangle']

states = 5
emissions = 7

classifier = Classifier(names, states, emissions, trained = False)
classifier.train()

correct = 0
total = 0

#Test on test data of each class
for name in names:
	sequences = getSequences(name, emissions, trainORtest = 'test')
	for seq in sequences:
		result = classifier.predict(seq)
		if result == name:
			correct += 1
		else:
			print 'Expected ', name , ' found ', result
		total += 1

accuracy = 1.0*correct / total

print 'Model Accuracy ', accuracy
