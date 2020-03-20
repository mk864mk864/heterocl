
#Import Modules
import heterocl as hcl
from PIL import Image
import math
import os
import PIL
import numpy as np
from scipy import misc

hcl.init(init_dtype=hcl.Float())

width = 225
height = 225

path = 'spongebob.png'
img = misc.imread(path)
imgdd = np.zeros((height,width,3))
for x in range (0,height):
	for y in range (0,width):
		for z in range (0,3):
			imgdd[x, y, z] = img[x,y,z]

imgdata = hcl.asarray(imgdd)
data = hcl.placeholder((width, height, 3), "data", dtype=hcl.Float())
Gx = hcl.placeholder((3,3), "Gx", dtype=hcl.Float())
Gy = hcl.placeholder((3,3), "Gy", dtype=hcl.Float())

Gxdata1 = hcl.asarray(np.array([[1,0,-1],[2,0,-2],[1,0,-1]]))
Gydata1 = hcl.asarray(np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))

Gxdata = hcl.asarray(Gxdata1)
Gydata = hcl.asarray(Gydata1)

def kernel(data, Gx, Gy):
	newdata =  hcl.compute((width, height), lambda x,y: data[x][y][0] + data[x][y][1] + data[x][y][2], "newdata", dtype=hcl.Float()) 
	r = hcl.reduce_axis(0,3)
	c = hcl.reduce_axis(0,3)
	Gxresult = hcl.compute((width-2, height-2), lambda x,y: hcl.sum(newdata[x+r, y+c]*Gx[r,c], axis=[r,c]), "Gxresult", dtype=hcl.Float())
	t = hcl.reduce_axis(0,3)
	g = hcl.reduce_axis(0,3)
	Gyresult = hcl.compute((width-2, height-2), lambda x,y: hcl.sum(newdata[x+t, y+g]*Gy[t,g], axis=[t,g]), "Gyresult", dtype=hcl.Float())
	return hcl.compute((width-2, height-2), lambda x,y: hcl.sqrt((Gxresult[x][y]*Gxresult[x][y])+(Gyresult[x][y]*Gyresult[x][y]))/4328*255, dtype=hcl.Float())


sched = hcl.create_schedule([data, Gx, Gy], kernel)
func = hcl.build(sched)

length = hcl.asarray(np.zeros((width-2, height-2)) ,dtype=hcl.Float())

func(imgdata, Gxdata, Gydata, length)
newlength = length.asnumpy().astype(int)

newimgarry = np.zeros((width-2,height-2,3))

for x in range (0, width-2):
	for y in range (0, height-2):
		for z in range (0, 3):
			newimgarry[x,y,z] = newlength[x,y]

misc.imsave('spongebob_fixed.png', newimgarry)