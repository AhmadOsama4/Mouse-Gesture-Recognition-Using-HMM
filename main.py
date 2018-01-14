from HMM import *
from Quantizer import *
from XMLParser import *
from Sequence import *
from Classifier import *
from GUI import *

names = ['BottomLeftCorner', 'Tick', 'UpperRightCorner', 'UpperLeftCorner', 'LessThan', 'GreaterThan', 'Square', 'Triangle']

states = 5
emissions = 8

classifier = Classifier(names, states, emissions, trained = True)
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

root = Tk()
root.minsize(920, 700)
root.title('Mouse Gesture Classifier')

g = GUI(classifier, root, emissions)

root.mainloop()