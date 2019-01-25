from tensorflow.keras import models
from tensorflow.keras.layers import Dense, LSTM, Input, Add, Lambda, CuDNNLSTM, Softmax
from tensorflow.keras.activations import sigmoid
import numpy as np

class base_line_affect_lm_model:

    def __init__(self, vocab_size, batch_size, look_back_steps, embedding_matrix=None, bias_vector=None, embedding=False, state_ful=False):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.look_back_steps = look_back_steps
        # Bool to tell if a pre trained word embedding is used
        self.emb = embedding
        # Check if embedding is used
        if self.emb:
            self.embedding = np.transpose(embedding_matrix)
        self.bias_vector = bias_vector

        # Parameter for tuning the affect energy term
        self.start_weights = []
        self.model = self.define_model(look_back_steps, batch_size, state_ful=state_ful)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def define_model(self, look_back_steps, batch_size, state_ful=False):

        # The base LM model
        if state_ful:
            word_input = Input((look_back_steps, self.vocab_size), batch_size=batch_size)
        else:
            word_input = Input((look_back_steps, self.vocab_size))
        # The output vector from the LSTM layer is 200d and return_sequences=True must be set in order to use
        #lstm_layer1 = CuDNNLSTM(100, return_sequences=True, stateful=state_ful)(word_input)
        #lstm_layer2 = CuDNNLSTM(100, stateful=state_ful)(lstm_layer1)
        lstm_layer1 = LSTM(100, return_sequences=True, stateful=state_ful)(word_input)
        lstm_layer2 = LSTM(100, stateful=state_ful)(lstm_layer1)
        # Set the weights to the weights corresponding to the embedding
        dense_word_decoder = Dense(self.vocab_size, trainable=False, name='embedding_layer')(lstm_layer2)

        # The energy term for affect words
        # The input is the LIWC feature extraction which has 5 categories (5 index input vector)
        affect_input = Input((5,))
        dense_1 = Dense(50, activation=sigmoid)(affect_input)
        dense_2 = Dense(100, activation=sigmoid)(dense_1)
        # The report does not say anything about the activation function for the second layer
        dense_affect_decoder_1 = Dense(self.vocab_size)(dense_2)
        # multiply the affect vector with the beta energy term
        beta = 1.5
        dense_affect_decoder = Lambda(lambda x: x*beta)(dense_affect_decoder_1)

        # the bias term for unigram occurrences has not been added to the model
        final_layer = Add()([dense_word_decoder, dense_affect_decoder])
        prediction_layer = Softmax()(final_layer)

        model = models.Model(inputs=[word_input, affect_input], outputs=[prediction_layer])

        if self.emb:
            weights = model.get_layer('embedding_layer').get_weights()

            weights[0] = self.embedding
            weights[1] = self.bias_vector
            model.get_layer('embedding_layer').set_weights(weights)

            self.start_weights.append(weights[0])
            self.start_weights.append(weights[1])

        model.summary()

        return model

        # function for solving the problem with having to predict with the same batch size as the training batch size
        # the weights of a trained model are copied to a new model with batch size 1

    def redefine_model(self, old_model, look_back=1, state_ful=True):
        batch_size = 1
        model = self.define_model(look_back, batch_size, state_ful=state_ful)
        # copy weights from old to new model
        old_weights = old_model.get_weights()
        model.set_weights(old_weights)

        return model

