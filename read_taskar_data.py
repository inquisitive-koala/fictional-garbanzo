import numpy as np

def readTaskarData(directory):
	letters_file = directory + '/letter.data'

	letter_data = {}
	for line in open(letters_file):
		tokens = line.split()
		dat = {}
		letter_id = tokens[0]
		dat['id'] = letter_id
		dat['letter'] = tokens[1]
		dat['next_id'] = tokens[2]
		dat['word_id'] = tokens[3]
		dat['position'] = tokens[4]
		dat['fold'] = tokens[5]

		line_data = [int(i) for i in tokens[6:]]
		dat['image'] = np.reshape(line_data, (16,8))
		letter_data[letter_id] = dat

	return letter_data

