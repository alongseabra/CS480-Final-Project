# Copyright 2016 Niek Temme.
# Authors: Anson Long-Seabra, Jay Kelner, Alec Reudi, Nick Temme
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This code was adapted from Niek Temme's blog post, which can be found at
# https://niektemme.com/2016/02/21/tensorflow-handwriting/ 
# Niek's code does most of the heavy-lifting, along with TensorFlow, a Google API
# 
# ==============================================================================


#import modules
import sys
import time
import tensorflow as tf
from PIL import Image,ImageFilter
import os
from tensorflow.examples.tutorials.mnist import input_data

"""
The ImageUtil class allows for normalization of images into 28 x 28. This is Niek's
code, but we wrapped in in an object for our use.
"""
class ImageUtil(object):	

	def imageprepare(self, argv):
		"""
		This function returns the pixel values.
		The imput is a png file location.
		"""
		im = Image.open(argv).convert('L')
		width = float(im.size[0])
		height = float(im.size[1])
		newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels

		if width > height: #check which dimension is bigger
			#Width is bigger. Width becomes 20 pixels.
			nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
			if (nheight == 0): #rare case but minimum is 1 pixel
				nheight = 1
			# resize and sharpen
			img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
			wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
			newImage.paste(img, (4, wtop)) #paste resized image on white canvas
		else:
			#Height is bigger. Heigth becomes 20 pixels.
			nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
			if (nwidth == 0): #rare case but minimum is 1 pixel
				nwidth = 1
			 # resize and sharpen
			img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
			wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
			newImage.paste(img, (wleft, 4)) #paste resized image on white canvas

		#newImage.save("sample.png")

		tv = list(newImage.getdata()) #get pixel values

		#normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
		tva = [ (255-x)*1.0/255.0 for x in tv]
		return tva
		#print(tva)

	def main(self, filename):
		"""
		Main function.
		"""
		self.filename = filename
		imvalue = self.imageprepare(filename)
		predint = self.predictint(imvalue)
		print ("The digit drawn in " + self.filename + "is a " + str(predint[0])) #first value in list


# Copyright 2016 Niek Temme.
# Adapted form the on the MNIST biginners tutorial by Google.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
Documentation at
http://niektemme.com/ @@to do

This script is based on the Tensoflow MNIST beginners tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
"""
"""
Modified by Anson Long-Seabra, Jay Kelner, Daniel Aiken, and Alec Ruedi
Our simple modification allows the gradient descent optimizer to be used as a parameter.
This allows us to see which values yield the best results.
"""
#import modules

#import tensorflow as tf
"""
The ModelCreator builds our neural net using TensorFlow. Again, a lot of Niek's code,
but we made it object-oriented and added a few tweaks.
"""
class ModelCreator(object):

	def __init__(self, opt, directory):
		self.optimizer = opt
		self.directory = directory

	def createModel(self):

		#import data
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

		# Create the model
		x = tf.placeholder(tf.float32, [None, 784])
		W = tf.Variable(tf.zeros([784, 10]), name="W")
		b = tf.Variable(tf.zeros([10]), name="b")
		y = tf.nn.softmax(tf.matmul(x, W) + b)

		# Define loss and optimizer
		#optimizer = sys.argv[1]
		y_ = tf.placeholder(tf.float32, [None, 10])
		cross_entropy = -tf.reduce_sum(y_*tf.log(y))

		#Here we changed the code to allow the optimizer to be changed
		train_step = tf.train.GradientDescentOptimizer(self.optimizer).minimize(cross_entropy)

		init_op = tf.initialize_all_variables()
		saver = tf.train.Saver()


		# Train the model and save the model to disk as a model.ckpt file
		# file is stored in the same directory as this python script is started
		"""
		The use of 'with tf.Session() as sess:' is taken from the Tensor flow documentation
		on on saving and restoring variables.
		https://www.tensorflow.org/versions/master/how_tos/variables/index.html
		"""
		with tf.Session() as sess:
			sess.run(init_op)
			for i in range(1000):
				batch_xs, batch_ys = mnist.train.next_batch(100)
				sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

			#Now the training is done and the model is ready to be tested

			#This code is exclusively added by the group
			i = 0
			j = 0

			#A dictionary to store our accuracy in
			resultsDict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
			
			#Keep track of how well our model is doing
			for filename in os.listdir(self.directory):
				fullPath = os.path.abspath(os.path.join(self.directory, filename))
				if fullPath.lower().endswith('.png'):
					firstChar = int(filename[0])
					imvalue = ImageUtil().imageprepare(fullPath)
					prediction=tf.argmax(y,1)
					print filename + " is a " + str(prediction.eval(feed_dict={x: [imvalue]}, session=sess)[0])
					i += 1

					#In our test data, the first character of the filename corresponds to the
					#correct identification of the file
					if (firstChar == prediction.eval(feed_dict={x: [imvalue]}, session=sess)[0]):
						resultsDict[j] += 1
					if (i == 700):
						i = 0
						j += 1
			total = 0

			#Print the accuracy
			for k in range(0, 10):
				print("For digit " + str(k) + " the accuracy was " + str(resultsDict[k] / 700.0))
				total += resultsDict[k]
			print("Total accuracy was " + str(total / 7000.0))

"""
The main program was written exclusively by us. It ties all of of the code together in 
an easy-to-use script
"""
def main():

	print("Welcome to Anson, Jay, Daniel, and Alec's CS480 Final Project.")
	print("Today we will be using a neural network to identify handwritten digits.")

	directory = input("Please enter the name of the directory that holds your files, in quotes: ")
	directory = str.strip(directory)

	gradient = input("What is the gradient descent you would like to use on your neural network? Ex: 0.01 (No quotes): ")


	mCreator = ModelCreator(gradient, directory)
	mCreator.createModel()

main()








