from tensorflow.keras import models
from tensorflow.keras.models import load_model
import pickle
from numpy import array
import random
import bisect
import numpy as np
from tensorflow.keras import utils

from Models.base_line_affect_LM_model import base_line_affect_lm_model

def cumulative_distribution_function(probabilities):
    # floating point error
    total = sum(probabilities)
    cdf = []
    c_sum = 0.0
    for p in probabilities:
        c_sum += p
        cdf.append((c_sum/total))
    return cdf


def sample_index_from_distribution(probabilities):
    cdf = cumulative_distribution_function(probabilities)
    # Get a random number between 0 and 1
    x = random.random()
    # Get the index of where x can be inserted to still keep the list ordered
    index = bisect.bisect(cdf, x)
    return index




# Must send in a model with batch size 1, otherwise can not sample one word at a time
class sample_word:
    def __init__(self):
        # Load the tokenizer from file
        with open('../model/tokenizer/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer

        self.vocab_size = len(self.tokenizer.word_index) + 1
        #print(str(self.vocab_size) + ' size!!!!!!')

        #self.model = LM_Model(vocab_size, look_back=1, batch_size=1, stateful=True).model
        #self.model = base_line_affect_lm_model(self.vocab_size, look_back_steps=1, batch_size=1, state_ful=True).model
        self.model = base_line_affect_lm_model(self.vocab_size, look_back_steps=20, batch_size=1, state_ful=False).model
        self.model.load_weights('../model/best_weights.hdf5')


        # Save the emotional embedding weights as np array to file


    def sample_new_sequence(self, text_sample):
        self.model.reset_states()

        # Reset the memory cell and hidden node since a new sequence will be started
        encoded_sequence = self.tokenizer.texts_to_sequences([text_sample])[0]

        # Loop over all the word indexes in the list and predict the next word
        for word in encoded_sequence:
            encoded_word = array([word])

            affect_batch = np.zeros((1, 5))
            affect_batch[0, :] = [0, 0, 0, 0, 1]

            x_batch = utils.to_categorical(encoded_word, num_classes=self.vocab_size)
            x_batch = np.expand_dims(x_batch, axis=0)

            #print('shape: {}!!!!!'.format(x_batch.shape))

            input = {'input_1': x_batch, 'input_2': affect_batch}
            #print('Not passed prediction')

            word_prediction = self.model.predict(input, verbose=2)

        # Sample a word based on the probabilities of the words
        #sampled_index = sample_index_from_distribution(word_prediction[0])

        sampled_index = np.argmax(word_prediction[0])

        # Loop over the tokenizer and find the word corresponding to the sampled index
        sampled_word = ''
        for word, index in self.tokenizer.word_index.items():
            if sampled_index == index:
                sampled_word = word
                break

        if sampled_word == 'unk':
            ind = np.argpartition(word_prediction[0], -2)[-2:]
            best_ind = ind[0]
            if ind[0] == sampled_index:
                best_ind = ind[1]

            for word, index in self.tokenizer.word_index.items():
                if best_ind == index:
                    sampled_word = word
                    break

        return sampled_word

    def sample_words(self, words, n=15):
        sample = ''
        for i in range(n):
            sampled_word = self.sample_next_word(words)
            sample += sampled_word + ' '
            words += ' {}'.format(sampled_word)
            _, words = words.split(' ', 1)

        print(sample)
        return sample

    def sample_next_word(self, words, affect=[0,0,0,0,0]):
        'Sample new words, given 20, previous'

        # Encode input to network, initial words and effect categories
        encoded_word = array(self.tokenizer.texts_to_sequences([words])[0])
        encoded_word = array([encoded_word])

        affect_batch = np.zeros((1, 5))
        affect_batch[0, :] = affect
        x_batch = utils.to_categorical(encoded_word, num_classes=self.vocab_size)
        #x_batch = np.expand_dims(x_batch, axis=0)

        input = {'input_1': x_batch, 'input_2': affect_batch}

        # Predict next word based on input
        prediction = self.model.predict(input, verbose=0)

        # Word index
        sampled_index = np.argmax(prediction[0])

        # Loop over the tokenizer and find the word corresponding to the sampled index
        sampled_word = ''
        for word, index in self.tokenizer.word_index.items():
            if sampled_index == index:
                sampled_word = word
                # print(sampled_word)
                break

        if sampled_word == 'unk':
            # print('is unk')
            ind = np.argpartition(prediction[0], -2)[-2:]
            best_ind = ind[0]
            if ind[0] == sampled_index:
                best_ind = ind[1]

            for word, index in self.tokenizer.word_index.items():
                if best_ind == index:
                    sampled_word = word
                    break

        return sampled_word


if __name__ == '__main__':

    input_file = 'inference_input.txt'
    output_file = 'inference_output.txt'

    with open(input_file) as file:
        content = file.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    sampler = sample_word()
    for row in content:
        sample = sampler.sample_words(row)

        # Write the sample to file
        with open(output_file, "a") as file:
            file.write(row + ': ' + sample + '\n')
