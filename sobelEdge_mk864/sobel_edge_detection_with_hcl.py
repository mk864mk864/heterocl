
#import modules
import heterocl as hcl
from PIL import Image
import math
import os
import numpy as np
from scipy import misc

#need to initiate hcl
hcl.init(init_dtype=hcl.Float())

#specify width and height of the input image
height = 900#225
width = 1200#225

#need to define placeholders to define kernel and create schedule
data = hcl.placeholder((height, width, 3), "data", dtype=hcl.Float())
Gx = hcl.placeholder((3,3), "Gx", dtype=hcl.Float())
Gy = hcl.placeholder((3,3), "Gy", dtype=hcl.Float())

#path to input image
path = 'lane.png'#'spongebob.png'

#need to convert all input data to hcl arrays
imgdata = hcl.asarray(np.asarray(misc.imread(path)))
Gxdata = hcl.asarray(np.array([[1,0,-1],[2,0,-2],[1,0,-1]]))
Gydata = hcl.asarray(np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))

#define kernel for all computations done in hcl
def kernel(data, Gx, Gy):
	newdata =  hcl.compute((height, width), lambda x,y: data[x][y][0] + data[x][y][1] + data[x][y][2], "newdata", dtype=hcl.Float()) 
	r = hcl.reduce_axis(0,3)
	c = hcl.reduce_axis(0,3)
	Gxresult = hcl.compute((height-2, width-2), lambda x,y: hcl.sum(newdata[x+r, y+c]*Gx[r,c], axis=[r,c]), "Gxresult", dtype=hcl.Float())
	t = hcl.reduce_axis(0,3)
	g = hcl.reduce_axis(0,3)
	Gyresult = hcl.compute((height-2, width-2), lambda x,y: hcl.sum(newdata[x+t, y+g]*Gy[t,g], axis=[t,g]), "Gyresult", dtype=hcl.Float())
	return hcl.compute((height-2, width-2), lambda x,y: hcl.sqrt((Gxresult[x][y]*Gxresult[x][y])+(Gyresult[x][y]*Gyresult[x][y]))/4328*255, dtype=hcl.Float())

#create schedule and function
sched = hcl.create_schedule([data, Gx, Gy], kernel)
func = hcl.build(sched)

#need to define output array as an hcl array
length = hcl.asarray(np.zeros((height-2, width-2)) ,dtype=hcl.Float())

#run the function
func(imgdata, Gxdata, Gydata, length)

#change the type of output back to numpy array
newlength = length.asnumpy().astype(int)

#define array for image
newimgarry = np.zeros((height-2, width-2, 3))

#assign (length, length, length) to each pixel
for x in range (0, height-2):
	for y in range (0, width-2):
		for z in range (0, 3):
			newimgarry[x,y,z] = newlength[x,y]

#create an image with the array
misc.imsave('lane_fixed.png', newimgarry)