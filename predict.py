from __future__ import print_function

import nltk
import numpy as np
from utils import dataset
dataset = dataset()
import hgtk

# will use nltk.pos_tag and s_tag (StanfordPOSTagger)
# tokenization will be done by nltk.word_tokenize

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

train_source = [line[0] for line in dataset]
train_target = ['\t'+hgtk.text.decompose(line[1])+'\n' for line in dataset]

input_characters = set()
for line in train_source:
    for char in line:
        if char not in input_characters:
            input_characters.add(char)

target_characters = set()
for line in train_target:
    for char in line:
        if char not in target_characters:
            target_characters.add(char)

######

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.01
#set_session(tf.Session(config=config))

from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

import string

latent_dim = 32 # Latent dimensionality of the encoding space.
num_samples = len(train_source)  # Number of samples to train on.

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in train_source])
max_decoder_seq_length = max([len(txt) for txt in train_target])

input_token_index = input_characters
target_token_index = target_characters

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras import layers

########################################

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_bi = (Bidirectional(LSTM(latent_dim, return_sequences=True)))
encoder_seq = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder = LSTM(latent_dim, return_state=True)
encoder_output0, state_h0,state_c0= encoder_seq(encoder_bi(encoder_inputs))
encoder_outputs, state_h, state_c = encoder(encoder_output0)
# We discard `encoder_outputs` and only keep the states.
encoder_states0= [state_h0, state_c0]
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state = encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

############ INFERENCE

test_source = train_source[-100:]
test_target = train_target[-100:]

encoder_input_test_data = np.zeros(
    (len(test_source), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_test_data = np.zeros(
    (len(test_source), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_test_data = np.zeros(
    (len(test_source), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (test_source_utt, test_target_utt) in enumerate(zip(test_source, test_target)):
    for t, char in enumerate(test_source_utt):
      if char in input_token_index:
        encoder_input_test_data[i, -len(test_source_utt)+t, input_token_index[char]] = 1.
    for t, char in enumerate(test_target_utt):
      if char in target_token_index:
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_test_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_test_data[i, t - 1, target_token_index[char]] = 1.

from keras.models import load_model
model.load_weights('model/enko_seq2seq_280.hdf5')

#model = load_model('model/enko_seq2seq_280.hdf5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models

from keras import layers

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, d_state_h, d_state_c = decoder_lstm(
    decoder_inputs, initial_state= decoder_states_inputs)
decoder_states = [d_state_h, d_state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

reverse_input_token_index = input_characters
reverse_target_token_index = target_characters

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # Update states
        states_value = [h, c]
    return decoded_sentence

def make_seq(s):
    encoder_input_test_temp = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(s):
        if char in input_token_index:
            encoder_input_test_temp[0, -len(s)+t, input_token_index[char]] = 1.
    return encoder_input_test_temp

def return_trans(s):
    input_seq = make_seq(s)
    decoded_sentence = decode_sequence(input_seq)[:-1]
    return hgtk.text.compose(decoded_sentence)
