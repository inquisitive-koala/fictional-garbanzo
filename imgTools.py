#Some common image manipulation tools we'll need
from scipy import misc
import numpy as np


horizKernel = np.ndarray((3, 3), dtype=np.float)
horizKernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

vertKernel = np.ndarray((3, 3), dtype=np.float)
vertKernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

avgKernel = np.ndarray((3, 3), dtype=np.float)
avgKernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

def testArrayShape(arrayIn, desiredShape, desiredType):
	#Test it is np.ndarray
	if (isinstance(arrayIn, np.ndarray) != True):
		print("Bad input, expected numpy.ndarray")
		return False

	#Test overall shape
	if (len(arrayIn.shape) != len(desiredShape)):
		print("Incorrect array shape:")
		print("Expected " + str(desiredShape))
		print("Received " + str(arrayIn.shape))
		return False

	#Test individual dimensions of shape
	shapeSize = len(desiredShape)
	for x in range(shapeSize):
		if (desiredShape[x] == -1):
			continue

		if (arrayIn.shape[x] != desiredShape[x]):
			print("Incorrect array shape:")
			print("Expected " + str(desiredShape))
			print("Received " + str(arrayIn.shape))
			return False

	#Test data type
	if (desiredType != -1):
		if (arrayIn.dtype != desiredType):
			print("Bad array type:")
			print("Expected " + str(desiredType))
			print("Received " + str(arrayIn.dtype))
			return False

	return True

#Take a ndarray of shape (x, y, 3), type uint8
#Return greyscaled ndarray of shape (x, y), type uint8
def bmp2greyArray(array):
	if (testArrayShape(array, (-1, -1, 3), np.uint8) is False):
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
	if (testArrayShape(arrayIn, (-1, -1), -1) is False):
		return np.ndarray((0, 0), np.uint8)

	arrayTemp = np.copy(arrayIn).astype(np.float)

	maxVal = np.amax(arrayTemp)
	minVal = np.amin(arrayTemp)

	if (minVal < 0):
		arrayTemp = np.absolute(arrayTemp)
		minVal = np.amin(arrayTemp)
		#print(arrayTemp)

	arrayRange = maxVal - minVal

	if (arrayRange > 0):
		arrayTemp /= arrayRange
		arrayTemp *= 255
		#print(arrayTemp)

	return arrayTemp.astype(np.uint8)

#Take a ndarray of shape (x, y), type uint8
#Return expanded array of shape (x, y, 3), type uint8
#RGB values will all be equal
def array2bmp(arrayIn):
	if (testArrayShape(arrayIn, (-1, -1), -1) is False):
		return np.ndarray((0, 0, 3), np.uint8)

	xSize = arrayIn.shape[0]
	ySize = arrayIn.shape[1]
	arrayBmp = np.ndarray((xSize, ySize, 3), np.uint8)

	for j in range(ySize):
		for i in range(xSize):
			grey = arrayIn[i, j]
			arrayBmp[i, j] = [grey, grey, grey]

	return arrayBmp


#Take a ndarray of shape (x, y), type uint8
#Run filter on each 3x3 area of pixels
def filterize(arrayIn, filter3x3):
	if (testArrayShape(arrayIn, (-1, -1), np.uint8) is False):
		return np.ndarray((0, 0, 3), np.uint8)

	xSize = arrayIn.shape[0]
	ySize = arrayIn.shape[1]
	arrayOut = np.zeros((xSize, ySize), dtype=np.float)

	#print("using kernel:\n")
	#print(3x3filter)

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
			result = np.multiply(filter3x3, img3x3window)
			arraysum = np.sum(result)
			
			arrayOut[j, i] = arraysum

	#print("arrayOut:")
	#print(arrayOut)

	return arrayOut


def runFilter(arrayIn, kernel):
	if (testArrayShape(arrayIn, (-1, -1, 3), np.uint8) is False):
		return np.ndarray((0, 0, 3), np.uint8)

	greyImg = bmp2greyArray(arrayIn)
	filteredImg = filterize(greyImg, kernel)
	normImg = normalizeArray(filteredImg)
	bmpImg = array2bmp(normImg)

	return bmpImg

