import numpy as np
import imgTools as it

DTHETA = 0.01
DR = .5
EDGE_THRESH = 40
ACCUM_THRESH = 30

def round_to(n, prec):
	corr = 0.5 if n >= 0 else -0.5
	return int(n / prec + corr) * prec

#Input (x, y) array of uint8
#Output (theta, r) array of uint
def houghTrans(arrayIn):
	if (it.testArrayShape(arrayIn, (-1, -1), np.uint8) is False):
		return np.ndarray((0, 0), np.uint)

	xSize = arrayIn.shape[0]
	ySize = arrayIn.shape[1]

	#number of steps of theta
	thetaSize = int(2 * np.pi / DTHETA)
	#largest possible R is the diagonal of the input array
	rSize = int(np.sqrt(xSize * xSize + ySize * ySize) / DR)

	accumArray = np.zeros((thetaSize, rSize), np.uint)
	print("accumArray shape: " + str(accumArray.shape))

	for x in range(xSize):
		for y in range(ySize):
			#Skip non-edge pixels
			if (arrayIn[x, y] < EDGE_THRESH):
				continue

			#For theta 0 to Pi
			theta = 0
			while (theta < (np.pi - DTHETA)):
				#Get normal distance to line passing through x,y with angle theta
				newR = x * np.cos(theta) + y * np.sin(theta)
				#Round to nearest DR
				newR = round_to(newR, DR)
				#Add new theta, R point to accumulator array
				accumArray[int(theta / DTHETA), int(newR / DR)] += 1

				theta += DTHETA

	return accumArray		

def houghTransImg(imgIn):
	if (it.testArrayShape(imgIn, (-1, -1, 3), -1) is False):
		return np.ndarray((0, 0, 3), np.uint8)

	arrayIn = it.toArray(imgIn)
	houghArray = houghTrans(arrayIn)
	imgOut = it.toBmp(houghArray)
	return imgOut

def reverseHoughTrans(accumArray, shapeOut):
	#Plan:
	#Create empty array of appropriate size, give shapeOut
	#For each theta:
		#For each R:
			#If value at theta, R is above threshold then
			#For x=0 to x=xSize:
				#solve for y 
				#draw point x, y
		#Special case theta=0:
		#For each R:
			#If value at theta, R is above threshold then
			#For y=0 to y=ySize:
				#solve for x 
				#draw point x, y
	if (it.testArrayShape(accumArray, (-1, -1), np.uint) is False):
		return np.ndarray(shapeOut, np.uint8)

	arrayOut = np.zeros(shapeOut, np.uint)

	xSize = shapeOut[0]
	ySize = shapeOut[1]
	thetaSize = accumArray.shape[0]
	rSize = accumArray.shape[1]

	for thetaStep in range(thetaSize):
		theta = thetaStep * DTHETA
		for rStep in range(rSize):
			r = rStep * DR

			if (accumArray[thetaStep, rStep] < ACCUM_THRESH):
				continue

			print("Adding line: theta=" + str(theta) + ", R=" + str(r))

			#Step x, solve for y
			if (np.abs(np.sin(theta)) > 0.05):
				for x in range(xSize):
					y = (r - x * np.cos(theta)) / np.sin(theta)
					if (y > 0 and y < ySize):
						y = round_to(y, 1)
						arrayOut[x, y] += 1

			#Step y, solve for x
			else:
				for y in range(ySize):
					x = (r - y * np.sin(theta)) / np.cos(theta)
					if (x > 0 and x < xSize):
						x = round_to(x, 1)
						arrayOut[x, y] += 1


	return it.normalizeArray(arrayOut)
