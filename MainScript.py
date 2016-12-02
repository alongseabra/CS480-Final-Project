# Copyright 2016 Niek Temme.
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

"""Predict a handwritten integer (MNIST beginners).

Script requires
1) saved model (model.ckpt file) in the same location as the script is run from.
(requried a model created in the MNIST beginners tutorial)
2) one argument (png file location of a handwritten integer)

Documentation at:
http://niektemme.com/ @@to do
"""

#import modules
import sys
import time
import tensorflow as tf
from PIL import Image,ImageFilter
import os
#import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Predictor(object):

		

	def predictint(self, imvalue):
		"""
		This function returns the predicted integer.
		The imput is the pixel values from the imageprepare() function.
		"""

		# Define the model (same as when creating the model file)
		x = tf.placeholder(tf.float32, [None, 784])
		W = tf.Variable(tf.zeros([784, 10]), name="W")
		b = tf.Variable(tf.zeros([10]), name="b")
		y = tf.nn.softmax(tf.matmul(x, W) + b)

		init_op = tf.initialize_all_variables()
		saver = tf.train.Saver()

		"""
		Load the model.ckpt file
		file is stored in the same directory as this python script is started
		Use the model to predict the integer. Integer is returend as list.

		Based on the documentatoin at
		https://www.tensorflow.org/versions/master/how_tos/variables/index.html
		"""

		#tf.print_tensors_in_checkpoint_file()
		with tf.Session() as sess:
			sess.run(init_op)
			saver.restore(sess, "model.ckpt")
			#print ("Model restored.")

			prediction=tf.argmax(y,1)
			return prediction.eval(feed_dict={x: [imvalue]}, session=sess)


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

	#if __name__ == "__main__":
	#    main(self.filename)


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
Our simple modification allows the gradient descent optimizer to be modified.
This allows us to see which values yield the best results.
"""
#import modules

#import tensorflow as tf

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

			
			i = 0
			for filename in os.listdir(self.directory):
				fullPath = os.path.abspath(os.path.join(self.directory, filename))
				if fullPath.lower().endswith('.png'):
					imvalue = Predictor().imageprepare(fullPath)
					prediction=tf.argmax(y,1)
					print "File #" + str(i) + " is a " + str(prediction.eval(feed_dict={x: [imvalue]}, session=sess)[0])
					i += 1

			#save_path = saver.save(sess, "model.ckpt")
			#print ("Model saved in file: ", save_path)
			#print("SAVING PATH IS " + os.path.abspath(save_path))
			#saved = file('model.ckpt')
			#saved.close()


def main():

	print("Welcome to Anson, Jay, Daniel, and Alec's CS480 Final Project.")
	print("Today we will be using a neural network to identify handwritten digits.")

	directory = input("Please enter the name of the directory that holds your files: ")
	directory = str.strip(directory)

	gradient = input("What is the gradient descent you would like to use on your neural network?: ")


	mCreator = ModelCreator(gradient, directory)
	mCreator.createModel()




main()








