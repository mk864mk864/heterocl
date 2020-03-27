#Import Modules
import heterocl as hcl
from PIL import Image
import math
import os
import numpy as np
import time

width = 1920
height = 1080

path = '172333.jpg'
img = Image.open(path)
newimg = Image.new("RGB", (width,height), "white")

for x in range (1,width-1):
	for y in range (1,height-1):
		Gx = 0
		Gy = 0

		p = img.getpixel((x-1, y-1))
		r = p[0]
		g = p[1]
		b = p[2]

		intensity = r+g+b

		Gx += -intensity
		Gy += -intensity
	
		p = img.getpixel((x-1,y))
		r = p[0]
		g = p[1]
		b = p[2]

		Gx += -2 * (r+g+b)

		p = img.getpixel((x-1,y+1))
		r = p[0]
		g = p[1]
		b = p[2]

		Gx += -(r+g+b)
		Gy += (r+g+b)

		p = img.getpixel((x,y-1))
		r = p[0]
		g = p[1]
		b = p[2]

		Gy += -2*(r+g+b)

		p = img.getpixel((x, y+1))
		r = p[0]
		g = p[1]
		b = p[2]

		Gy += 2 * (r+g+b)

		p = img.getpixel((x+1, y-1))
		r = p[0]
		g = p[1]
		b = p[2]

		Gx += (r+g+b)
		Gy += -(r+g+b)

		p = img.getpixel((x+1, y))
		r = p[0]
		g = p[1]
		b = p[2]

		Gx +=2*(r+g+b)

		p = img.getpixel((x+1, y+1))
		r = p[0]
		g = p[1]
		b = p[2]

		Gx += (r+g+b)
		Gy += (r+g+b)

		length = math.sqrt((Gx*Gx) + (Gy*Gy))

		length = length / 4328*255
		length = int(length)
		newimg.putpixel((x,y),(length,length,length))




newimg.save("172333_fixed.jpg")

print(time.process_time())