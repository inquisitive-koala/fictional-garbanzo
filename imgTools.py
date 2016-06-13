#Some common image manipulation tools we'll need
from scipy import misc
import numpy as np
import traceback as tb


horizKernel = np.ndarray((3, 3), dtype=np.float)
horizKernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

vertKernel = np.ndarray((3, 3), dtype=np.float)
vertKernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

avgKernel = np.ndarray((3, 3), dtype=np.float)
avgKernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

def print_stack():
	for line in tb.format_stack():
		print(line.strip())
	print

def testArrayShape(arrayIn, desiredShape, desiredType):
	#Test it is np.ndarray
	if (isinstance(arrayIn, np.ndarray) != True):
		print_stack()
		print("Bad input, expected numpy.ndarray")
		return False

	#Test overall shape
	if (len(arrayIn.shape) != len(desiredShape)):
		print_stack()
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
			print_stack()
			print("Incorrect array shape:")
			print("Expected " + str(desiredShape))
			print("Received " + str(arrayIn.shape))
			return False

	#Test data type
	if (desiredType != -1):
		if (arrayIn.dtype != desiredType):
			print_stack()
			print("Bad array type:")
			print("Expected " + str(desiredType))
			print("Received " + str(arrayIn.dtype))
			return False

	return True

#Take a ndarray of shape (x, y, 3), type uint8
#Return greyscaled ndarray of shape (x, y), type uint8
def toArray(array):
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
def toBmp(arrayIn):
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

def toRedBmp(arrayIn):
	#Convert to array if input is in Img form
	if (testArrayShape(arrayIn, (-1, -1, 3), -1)):
		arrayIn = toArray(arrayIn)

	if (testArrayShape(arrayIn, (-1, -1), -1) is False):
		return np.ndarray((0, 0, 3), np.uint8)

	xSize = arrayIn.shape[0]
	ySize = arrayIn.shape[1]
	arrayBmp = np.ndarray((xSize, ySize, 3), np.uint8)

	for j in range(ySize):
		for i in range(xSize):
			grey = arrayIn[i, j]
			if (grey < 255):
				arrayBmp[i, j] = [255, grey, grey]
			else:
				arrayBmp[i, j] = [255, 255, 255]

	return arrayBmp

#Element-wise addition of two equal-size arrays, followed by normalization
def addArrays(arrayIn1, arrayIn2):
	if (testArrayShape(arrayIn1, (-1, -1), -1) is False):
		return np.ndarray((0, 0), np.uint8)
	if (testArrayShape(arrayIn2, (-1, -1), -1) is False):
		return np.ndarray((0, 0), np.uint8)

	if (arrayIn1.shape[0] != arrayIn2.shape[0]):
		print("Mismatched size, axis 0")
		return np.ndarray((0, 0), np.uint8)
	if (arrayIn1.shape[1] != arrayIn2.shape[1]):
		print("Mismatched size, axis 1")
		return np.ndarray((0, 0), np.uint8)

	xSize = arrayIn1.shape[0]
	ySize = arrayIn1.shape[1]
	arrayOut = np.ndarray((xSize, ySize), np.float)

	for j in range(arrayIn1.shape[0]):
		for i in range(arrayIn1.shape[1]):
			arrayOut[i, j] = float(arrayIn1[i, j]) + float(arrayIn2[i, j])

	return normalizeArray(arrayOut)

def addImgs(imgIn1, imgIn2):
	greyArray1 = toArray(imgIn1)
	greyArray2 = toArray(imgIn2)
	sumArray = addArrays(greyArray1, greyArray2)
	bmpImg = toBmp(sumArray)
	return bmpImg

#places overlayImg "on top" of baseImg
#assumes that white is "transparent"
def overlayImg(baseImg, overlayImg):
	if (testArrayShape(baseImg, (-1, -1, 3), np.uint8) is False):
		return np.ndarray((0, 0, 3), np.uint8)
	if (testArrayShape(overlayImg, (-1, -1, 3), np.uint8) is False):
		return np.ndarray((0, 0, 3), np.uint8)

	if (baseImg.shape[0] != overlayImg.shape[0]):
		print("Mismatched size, axis 0")
		return np.ndarray((0, 0, 3), np.uint8)
	if (baseImg.shape[1] != overlayImg.shape[1]):
		print("Mismatched size, axis 1")
		return np.ndarray((0, 0, 3), np.uint8)
	

	xSize = baseImg.shape[0]
	ySize = baseImg.shape[1]
	imgOut = np.ndarray((xSize, ySize, 3), np.float)

	for x in range(xSize):
		for y in range(ySize):
			imgOut[x, y] = baseImg[x, y]

			if (overlayImg[x, y, 0] != 255):
				newVal = (imgOut[x, y, 0] + overlayImg[x, y, 0]) / 2
				imgOut[x, y, 0] = newVal
			if (overlayImg[x, y, 1] != 255):
				newVal = (imgOut[x, y, 1] + overlayImg[x, y, 1]) / 2
				imgOut[x, y, 1] = newVal
			if (overlayImg[x, y, 2] != 255):
				newVal = (imgOut[x, y, 2] + overlayImg[x, y, 2]) / 2
				imgOut[x, y, 2] = newVal

	return imgOut

#Black to white, white to black
def invGreyScaleArray(arrayIn):
	if (testArrayShape(arrayIn, (-1, -1), np.uint8) is False):
		return np.ndarray((0, 0), np.uint8)

	xSize = arrayIn.shape[0]
	ySize = arrayIn.shape[1]

	arrayOut = np.zeros(arrayIn.shape, np.uint8)

	for x in range(xSize):
		for y in range(ySize):
			arrayOut[x, y] = 255 - arrayIn[x, y]

	return arrayOut


def invGreyScaleImg(imgIn):
	greyArray = toArray(imgIn)
	invArray = invGreyScaleArray(greyArray)
	invBmp = toBmp(invArray)
	return invBmp

#Take a ndarray of shape (x, y), type uint8
#Run filter on each 3x3 area of pixels
def filterArray(arrayIn, filter3x3):
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
		for i in range(0, xSize - 2):

			#print("\n" + str(j) + ", " + str(i) + ":")
			#Construct a 3x3 kernel at this location
			for ypix in range(3):
				for xpix in range(3):
					img3x3window[xpix, ypix] = arrayIn[i + xpix, j + ypix]

			#Multiply each matrix element-wise, then sum to get a single value
			result = np.multiply(filter3x3, img3x3window)
			arraysum = np.sum(result)
			
			arrayOut[i, j] = arraysum

	#print("arrayOut:")
	#print(arrayOut)

	return arrayOut


def filterImg(arrayIn, kernel):
	if (testArrayShape(arrayIn, (-1, -1, 3), np.uint8) is False):
		return np.ndarray((0, 0, 3), np.uint8)

	greyImg = toArray(arrayIn)
	filteredImg = filterArray(greyImg, kernel)
	normImg = normalizeArray(filteredImg)
	bmpImg = toBmp(normImg)

	return bmpImg

