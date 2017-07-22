from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
#matrix math operations
import numpy as np
import keras.models
#regular expresions
import re
#load the file itself
import sys
import os, cv2
import base64
#define where our model is 
#this is where our model is saved (in the mdoel folder)
sys.path.append(os.path.abspath('./model'))
#help us load the model
from load import *
#init flask app
app = Flask(__name__)
#model is our model object
#graph is our computation graph
global model, graph
model, graph = init()

#decoding an image from base64 into raw binary data
def convertImage(imgData1):
	#imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#imgstr = int(imgData1, base=2)
	#print(imgstr)
	#imgstr = numpy.unpackbits(Bytes)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgData1))

#tell our app what happens when a user goes to a certain address

@app.route('/', methods = ['GET', "POST"])
def predict():
	imgData = request.get_data()
	print("predicting...")
	convertImage(imgData)
	#x = imread('output.png', mode = 'L')
	#x = np.invert(x)
	#x = imresize(x, 128, 128)
	#xnew = imresize(x,(128,128))
	#x = cv2.resize(x,(128,128))
	#this 4d tensor is what we feed into the model
	#x = x.reshape(1, 128, 128, 3)
	x = cv2.imread('output.png')
	x = np.invert(x)
#test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
	x = cv2.resize(x,(200,200))
	x = np.array(x)
	x = x.astype('float32')
	x /= 199
	x= np.expand_dims(x, axis=0)
	#x = np.array(x)
	#x = x.astype('float32')
	#x /= 255
	#x = cv2.resize(x,(128,128))
	#x = np.array(x)
	#x = x.astype('float32')
	#x /= 255
	#x= np.expand_dims(x, axis=0)
	#x= np.expand_dims(x, axis=0)
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		#print "debug3"
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	


if __name__ == "__main__":
	#port = int(os.environ.get('PORT', 8000))
	#app.run(host='0.0.0.0', port=port)
	#app.run(threaded=True)
	app.run()
	print("running")
	#decide what port to run the app in
		#run the app locally on the givn port
	#optional if we want to run in debugging mode
	#app.run(debug=True)

