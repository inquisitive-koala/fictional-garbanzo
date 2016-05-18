#Some common image manipulation tools we'll need
from scipy import misc
import numpy
import numpy as np

#Take a ndarray of shape (x, y, 3), type uint8
#Return greyscaled ndarray of shape (x, y), type uint8
def bmp2greyArray(array):
	if (isinstance(array, np.ndarray) != True):
		print("Bad input, expected numpy.ndarray")
		return np.ndarray((0, 0), np.uint8)

	if (len(array.shape) != 3) or (array.shape[2] != 3):
		print("Incorrect array shape, expected (x, y, 3)")
		return np.ndarray((0, 0), np.uint8)

	if (array.dtype != np.uint8):
		printf("Bad array type, expected uint8")
		return np.ndarray((0, 0), np.uint8)

	xSize = array.shape[0]
	ySize = array.shape[1]
	array2d = np.ndarray((xSize, ySize), np.uint8)

	for j in range(ySize):
		for i in range(xSize):
			red = array[i, j, 0]
			green = array[i, j, 1]
			blue = array[i, j, 2]
			grey = np.uint8(0.59*red + 0.30*green + 0.11*blue)
			array2d[i, j] = grey

	return array2d

#Take a (x, y), type float array
#Output all values in range 0-255, type uint8
def normalizeArray(arrayIn):
	if (isinstance(arrayIn, np.ndarray) != True):
		print("Bad input, expected numpy.ndarray")
		return np.ndarray((0, 0), np.uint8)

	if len(arrayIn.shape) != 2:
		print("Incorrect array shape, expected (x, y)")
		return np.ndarray((0, 0), np.uint8)

	if (arrayIn.dtype != np.float):
		printf("Bad array type, expected float")
		return np.ndarray((0, 0), np.float)

	maxVal = np.amax(arrayIn)
	minVal = np.amin(arrayIn)
	arrayRange = maxVal - minVal

	arrayTemp = np.copy(arrayIn)

	if (minVal < 0):
		arrayTemp -= minVal
		print(arrayTemp)

	if (arrayRange > 0):
		arrayTemp /= arrayRange
		arrayTemp *= 255
		print(arrayTemp)

	return arrayTemp.astype(np.uint8)

#Take a ndarray of shape (x, y, 1), type uint8
#Return expanded array of shape (x, y, 3), type uint8
#RGB values will all be equal
def array2bmp(array):
	if (isinstance(array, np.ndarray) != True):
		print("Bad input, expected numpy.ndarray")
		return np.ndarray((0, 0, 3), np.uint8)

	if len(array.shape) != 2:
		print("Incorrect array shape, expected (x, y)")
		return np.ndarray((0, 0, 3), np.uint8)

	if (array.dtype != np.uint8):
		printf("Bad array type, expected uint8")
		return np.ndarray((0, 0, 3), np.uint8)

	xSize = array.shape[0]
	ySize = array.shape[1]
	arrayBmp = np.ndarray((xSize, ySize, 3), np.uint8)

	for j in range(ySize):
		for i in range(xSize):
			grey = array[i, j]
			arrayBmp[i, j] = [grey, grey, grey]

	return arrayBmp


def edgeDetect(arrayIn, matEdge):
	xSize = arrayIn.shape[0]
	ySize = arrayIn.shape[1]
	arrayOut = np.zeros((xSize, ySize), dtype=np.float)

	#print("using edge matrix:\n")
	#print(matEdge)

	img3x3window = np.ndarray((3, 3), dtype=np.float)
	mat3x3edge = np.ndarray((3, 3), dtype=np.float)

	#Avoid looping all the way to the edge of the array since we will
	#read pixels up to 2 steps ahead of the current index
	for j in range(0, ySize - 2):
		if (j >= ySize):
			break
		for i in range(0, xSize - 2):
			if (i >= xSize):
				break

			#print("\n" + str(j) + ", " + str(i) + ":")
			#Construct a 3x3 kernel at this location
			for ypix in range(3):
				for xpix in range(3):
					img3x3window[ypix, xpix] = arrayIn[j + ypix, i + xpix]

			#Multiply each matrix element-wise, then sum to get a single value
			mat3x3edge = np.multiply(matEdge, img3x3window)
			arraysum = np.sum(mat3x3edge)
			#print("\n3x3 edge")
			#print(mat3x3edge)
			
			arrayOut[j, i] = arraysum
			#Add our edge-processed kernel to our output array
			#for ypix in range(3):
				#for xpix in range(3):
					#arrayOut[j + ypix, i + xpix] += mat3x3edge[ypix, xpix]

			#print("arrayOut:")
			#print(arrayOut)

	return arrayOut


#Take a ndarray of shape (x, y), type uint8
#Matrix mult each 3x3 kernel of pixels by:
#-1  0  1
#-2  0  2
#-1  0  1
#Return (x, y) array
def vertEdgeDetection(arrayIn):
	if (isinstance(arrayIn, np.ndarray) != True):
		print("Bad input, expected numpy.ndarray")
		return np.ndarray((0, 0), np.uint8)

	if len(arrayIn.shape) != 2:
		print("Incorrect array shape, expected (x, y)")
		return np.ndarray((0, 0), np.uint8)

	if (arrayIn.dtype != np.uint8):
		printf("Bad array type, expected uint8")
		return np.ndarray((0, 0), np.uint8)

	matEdge = np.ndarray((3, 3), dtype=np.float)
	matEdge[0, 0] = -1
	matEdge[0, 1] = 0
	matEdge[0, 2] = 1
	matEdge[1, 0] = -2
	matEdge[1, 1] = 0
	matEdge[1, 2] = 2
	matEdge[2, 0] = -1
	matEdge[2, 1] = 0
	matEdge[2, 2] = 1

	return edgeDetect(arrayIn, matEdge)


#Take a ndarray of shape (x, y), type uint8
#Matrix mult each 3x3 kernel of pixels by:
# 1  2  1
# 0  0  0
#-1 -2 -1
#Return (x, y) array
def horizEdgeDetection(arrayIn):
	if (isinstance(arrayIn, np.ndarray) != True):
		print("Bad input, expected numpy.ndarray")
		return np.ndarray((0, 0), np.uint8)

	if len(arrayIn.shape) != 2:
		print("Incorrect array shape, expected (x, y)")
		return np.ndarray((0, 0), np.uint8)

	if (arrayIn.dtype != np.uint8):
		printf("Bad array type, expected uint8")
		return np.ndarray((0, 0), np.uint8)

	matEdge = np.ndarray((3, 3), dtype=np.float)
	matEdge[0, 0] = 1
	matEdge[0, 1] = 2
	matEdge[0, 2] = 1
	matEdge[1, 0] = 0
	matEdge[1, 1] = 0
	matEdge[1, 2] = 0
	matEdge[2, 0] = -1
	matEdge[2, 1] = -2
	matEdge[2, 2] = -1

	return edgeDetect(arrayIn, matEdge)



def runHorizEdgeDetect(arrayIn):
	if (isinstance(arrayIn, np.ndarray) != True):
		print("Bad input, expected numpy.ndarray")
		return np.ndarray((0, 0, 3), np.uint8)

	if (len(arrayIn.shape) != 3) or (arrayIn.shape[2] != 3):
		print("Incorrect array shape, expected (x, y, 3)")
		return np.ndarray((0, 0, 3), np.uint8)

	if (arrayIn.dtype != np.uint8):
		printf("Bad array type, expected uint8")
		return np.ndarray((0, 0, 3), np.uint8)

	greyImg = bmp2greyArray(arrayIn)
	edgeImg = horizEdgeDetection(greyImg)
	normImg = normalizeArray(edgeImg)
	bmpImg = array2bmp(normImg)

	return bmpImg

def runVertEdgeDetect(arrayIn):
	if (isinstance(arrayIn, np.ndarray) != True):
		print("Bad input, expected numpy.ndarray")
		return np.ndarray((0, 0, 3), np.uint8)

	if (len(arrayIn.shape) != 3) or (arrayIn.shape[2] != 3):
		print("Incorrect array shape, expected (x, y, 3)")
		return np.ndarray((0, 0, 3), np.uint8)

	if (arrayIn.dtype != np.uint8):
		printf("Bad array type, expected uint8")
		return np.ndarray((0, 0, 3), np.uint8)

	greyImg = bmp2greyArray(arrayIn)
	edgeImg = vertEdgeDetection(greyImg)
	normImg = normalizeArray(edgeImg)
	bmpImg = array2bmp(normImg)

	return bmpImg
