#import modules
import heterocl as hcl
from PIL import Image
import math
import os
import numpy as np
import imageio
import time

#need to initiate hcl
hcl.init(init_dtype=hcl.Float())

#path to input image
path = 'lane_fixed.png'
image = imageio.imread(path)
npimage = np.asarray(image)
imgdata = hcl.asarray(npimage)

#specify width and height of the input image
height, width, dummy = image.shape


#need to define placeholders to define kernel and create schedule
data = hcl.placeholder((height, width, dummy), "data", dtype=hcl.Float())
lowthresh = hcl.placeholder((1,), "lowthresh", dtype=hcl.Float())
highthresh = hcl.placeholder((1,), "highthresh", dtype=hcl.Float())

#define kernel for all computations done in hcl
def kernel(data, lowthresh, highthresh):
	newdata = hcl.compute((height, width), lambda x,y: data[x][y][0] + data[x][y][1] + data[x][y][2], "newdata", dtype=hcl.Float()) 
	def compute_out(data, x, y, lowthresh, highthresh):
		with hcl.if_(data[x][y] < lowthresh[0]):
			hcl.return_(0)
		with hcl.elif_(data[x][y] >= highthresh[0]):
			hcl.return_(255)
		with hcl.else_():
			hcl.return_(25)
	return hcl.compute((height,width), lambda x,y: compute_out(newdata,x,y, lowthresh, highthresh))

#create schedule and function
sched = hcl.create_schedule([data, lowthresh, highthresh], kernel)
func = hcl.build(sched)

#need to define input/output array as an hcl array
high =  0.09 * (int(npimage[..., 0].max()) + int(npimage[..., 1].max()) + int(npimage[..., 2].max()))
low = 0.05 * high

print(height)
print(width)

result = hcl.asarray(np.zeros((height, width)) ,dtype=hcl.Float())
hclhigh = hcl.asarray(np.array([high]))
hcllow = hcl.asarray(np.array([low]))

#run the function
func(imgdata, hcllow, hclhigh, result)

print(result)
#change the type of output back to numpy array
newresult = result.asnumpy().astype(int)
print(newresult)
#define array for image
newimgarry = np.zeros((height, width, 3))

#assign (length, length, length) to each pixel
for x in range (0, height):
	for y in range (0, width):
		for z in range (0, 3):
			newimgarry[x,y,z] = newresult[x,y]

#create an image with the array
imageio.imsave('lane_fixedmore.png', newimgarry)

print(time.process_time())