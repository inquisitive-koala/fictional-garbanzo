import numpy as np
import imgTools as it

DTHETA = 0.01
DR = .2
PIXEL_THRESH = 40

def round_to(n, prec):
	corr = 0.5 if n >= 0 else -0.5
	return int(n / prec + corr) * prec

#Input (x, y) array of uint8
#Output (theta, r) array of uint
def houghTransform(arrayIn):
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
			if (arrayIn[x, y] < PIXEL_THRESH):
				continue

			#For theta 0 to Pi
			theta = 0
			while (theta < np.pi):
				#Get normal distance to line passing through x,y with angle theta
				newR = x * np.cos(theta) + y * np.sin(theta)
				#Round to nearest DR
				round_to(newR, DR)
				#Add new theta, R point to accumulator array
				accumArray[int(theta / DTHETA), int(newR / DR)] += 1

				theta += DTHETA

	return accumArray		

#Plan:
# 1) Initialize accumulation matrix 
# 2) For each edge pixel (pixel value greater than threshold):
#	a) Set theta to 0
#	b) For each theta 0 to Pi:
#		i) Solve for R (r = x*cos(theta) + y*sin(theta))
#		ii) Increment value of accum matrix at point [theta, R]
#		iii) theta += dTheta

