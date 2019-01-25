from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from os import listdir
from os.path import isfile, join
import numpy as np
import gc
import pickle
from Data.analyse import affection_context

FILE_EXTENSION = './Data/Data/validation_small/'
FILES = [join(FILE_EXTENSION, file_name) for file_name in listdir(FILE_EXTENSION) if isfile(join(FILE_EXTENSION, file_name)) and file_name != '.DS_Store']

# Function for finding the largest number less than K+1 divisible by X
def largest(X, K):
    # returning ans
    return (K - (K % X))

class validationDataGenerator(keras.utils.Sequence):

    def __init__(self, tokenizer, batch_size=20, sliding_window_size=20, shuffle=False):
        self.batch_size = batch_size
        self.sliding_window_size = sliding_window_size
        self.shuffle = shuffle

        # Use existing tokenizer created for training
        self.tokenizer = tokenizer

        # Reversed tokenizer for going from index go word
        self.reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))
        # Vocabulary size
        self.vocab_size = len(self.tokenizer.word_index) + 1

        self.batch_per_epoch = self.nr_batch_per_epoch()

        self.current_file_nr = -1
        self.x_file = None
        self.y_file = None
        self.load_next_sequence()
        # To keep track where in the current sequence the last batch was taken from
        self.batch_in_file = 0

        self.affect_context = affection_context()
        self.affect_categories = self.affect_context.affect_categories

        ### Tet varible ######
        self.NEXT_FILE = True

    def load_next_sequence(self):
        self.current_file_nr = (self.current_file_nr + 1) % len(FILES)
        file_path = FILES[self.current_file_nr]
        with open(file_path) as file:
            file_content = file.read()

        file_content = file_content.lower()
        file_content = file_content.replace('\n', ' ')

        # Get an encoding of the text
        file_encoded = self.tokenizer.texts_to_sequences([file_content])[0]

        sequences = []
        for i in range(self.sliding_window_size + 1, len(file_encoded)+1):
            sequence = file_encoded[i - (self.sliding_window_size + 1):i]
            sequences.append(sequence)

        sequences = np.asarray(sequences)

        X = np.array(sequences[:, 0:self.sliding_window_size])
        y = np.array(sequences[:, -1])
        y_b = y[0:largest(self.batch_size, y.shape[0])]
        X_b = X[0:largest(self.batch_size, X.shape[0])]

        self.x_file = X_b
        self.y_file = y_b

    def batch_generator(self):
        # properly check might give errors
        # Check if there are enough words left in the sequence to create a batch
        while True:

            if self.batch_in_file == self.x_file.shape[0]/self.batch_size:
                self.load_next_sequence()
                self.batch_in_file = 0
                self.NEXT_FILE = True
                gc.collect()
            else:
                self.NEXT_FILE = False

            start = self.batch_in_file*self.batch_size
            x_batch = np.array(self.x_file[start:start+self.batch_size, :])
            affect_batch = self.affect_for_batch(x_batch)
            x_batch = keras.utils.to_categorical(x_batch, num_classes=self.vocab_size)

            self.batch_in_file += 1

            y_batch = np.array(self.y_file[start:start+self.batch_size])
            y_batch = to_categorical(y_batch, num_classes=self.vocab_size)

            yield {'input_1': x_batch, 'input_2': affect_batch}, y_batch

    def affect_for_batch(self, x_batch):
        affect_batch = np.zeros((x_batch.shape[0], self.affect_categories))
        for i, row in enumerate(x_batch):
            context = []
            for index in row:
                context.append(self.reverse_word_map[index])
            affect_batch[i,:] = self.affect_context.binary_affection_vector_for_context(context)

        return affect_batch

    def nr_batch_per_epoch(self):
        batch_per_epoch = 0
        for file_path in FILES:
            with open(file_path) as file:
                file_content = file.read()

            file_content = file_content.lower()
            file_content = file_content.replace('\n', ' ')

            # Get an encoding of the text
            file_encoded_sequence = self.tokenizer.texts_to_sequences([file_content])[0]

            # the // operator gives int values
            batch_per_epoch += ((len(file_encoded_sequence)-(self.sliding_window_size+1))//(self.batch_size))
        return batch_per_epoch

