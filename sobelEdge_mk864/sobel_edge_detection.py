#Import Modules
import heterocl as hcl
from PIL import Image
import math
import os
import numpy as np

width = 9
height = 9



imgdd = np.zeros((height,width,3))
for x in range (0,height):
	for y in range (0,width):
		for z in range (0,3):
			#pizel = img.getpixel((x,y))
			#imgdd[y,x,z] = pizel[z]
			imgdd[x,y,z] = (z+3 + y+x)*10


#path = 'lane.jpg'
#img = Image.open(path)
#newimg = Image.new("RGB", (width,height), "white")

newimgarray = np.zeros((height,width,3))

for x in range (1,width-1):
	for y in range (1,height-1):
		Gx = 0
		Gy = 0

		p = imgdd[x-1, y-1]
		r = p[0]
		g = p[1]
		b = p[2]

		intensity = r+g+b

		Gx += -intensity
		Gy += -intensity
	
		p = imgdd[x-1,y]
		r = p[0]
		g = p[1]
		b = p[2]

		Gx += -2 * (r+g+b)

		p = imgdd[x-1,y+1]
		r = p[0]
		g = p[1]
		b = p[2]

		Gx += -(r+g+b)
		Gy += (r+g+b)

		p = imgdd[x,y-1]
		r = p[0]
		g = p[1]
		b = p[2]

		Gy += -2*(r+g+b)

		p = imgdd[x, y+1]
		r = p[0]
		g = p[1]
		b = p[2]

		Gy += 2 * (r+g+b)

		p = imgdd[x+1, y-1]
		r = p[0]
		g = p[1]
		b = p[2]

		Gx += (r+g+b)
		Gy += -(r+g+b)

		p = imgdd[x+1, y]
		r = p[0]
		g = p[1]
		b = p[2]

		Gx +=2*(r+g+b)

		p = imgdd[x+1, y+1]
		r = p[0]
		g = p[1]
		b = p[2]

		Gx += (r+g+b)
		Gy += (r+g+b)

		length = math.sqrt((Gx*Gx) + (Gy*Gy))

		length = length / 4328*255
		length = int(length)
		newimgarray[x,y] = length
		#newimg.putpixel((x,y),(length,length,length))




#newimg.save("lane_fixed.jpg")

print(newimgarray)