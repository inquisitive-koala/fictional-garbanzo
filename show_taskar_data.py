from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np

def showLetter(letter):
	plt.imshow(1-letter['image'], interpolation='nearest', cmap='Greys')
	plt.show()

def showWord(word):
	letter_images = [l['image'] for l in word]
	word_image = np.hstack(letter_images)
	plt.imshow(1 - word_image, interpolation='nearest', cmap='Greys')
	plt.show()

def getWords(letter_data):
	all_words = {}

	for i in range(1, 52153):
		key = str(i)
		letter = letter_data[key]
		word_id = letter['word_id']
		if word_id not in all_words:
			all_words[word_id] = []
		all_words[word_id].append(letter)

	for word in all_words.values():
		word.sort(key = lambda x : x['position'])

	return all_words

def wordText(word):
	return ''.join([l['letter'] for l in word])