from Tkinter import *
import tkMessageBox
from Classifier import *
from Quantizer import *

class GUI(object):
	def __init__(self, classifier, master, levels):
		self.X = [] # x coordinates for drawn points
		self.Y = [] # y coordinates for drawn points
		self.prev_x = 0
		self.prev_y = 0
		self.levels = levels

		self.classifier = classifier
		self.canvas =  Canvas(master, borderwidth = 0, background = 'white')
		self.canvas.pack(fill="both", expand=True)
		self.canvas.bind("<ButtonPress-1>", self.mousePressed)
		self.canvas.bind("<B1-Motion>", self.draw)
		self.canvas.bind("<ButtonRelease-1>", self.mouseRealeased)

	def mousePressed(self, event):
		x = event.x
		y = event.y

		self.X.append(x)
		self.Y.append(y)

		self.prev_x = x
		self.prev_y = y

	def draw(self, event):
		x = event.x
		y = event.y
		self.X.append(x)
		self.Y.append(y)

		self.canvas.create_line(self.prev_x, self.prev_y, x, y, fill='red', width=10)

		self.prev_x = x
		self.prev_y = y

	def mouseRealeased(self, event):
		sequence = quantize(self.X, self.Y, self.levels)
		
		if len(sequence) < 2:
			tkMessageBox.showinfo('Prediction','Please Draw a shape!')
		else:
			result = self.classifier.predict(sequence)
			output = 'You have drawn a ' + result

			tkMessageBox.showinfo('Prediction',output)
			
		self.X = []
		self.Y = []
		self.canvas.delete("all")