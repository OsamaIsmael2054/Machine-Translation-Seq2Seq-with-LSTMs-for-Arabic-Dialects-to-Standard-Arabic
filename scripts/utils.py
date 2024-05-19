from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
import os
import string
import re


def remove_punctuation_arabic(data):
    """remove punctuation of data

    Args:
        data (df): cleaned dataframe
    """
    def remove_punctuation_function(text):
        # Define Arabic punctuation characters
        arabic_punctuation = string.punctuation + '؛،؟ـ'

        # Remove punctuation using regex
        return re.sub('[' + re.escape(arabic_punctuation) + ']', '', text)
    
    data['source'] = data['source'].apply(remove_punctuation_function)
    data['target'] = data['target'].apply(remove_punctuation_function)

    return data



# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

def tokenization(lines):
    tokenizer = Tokenizer(split=" ")
    tokenizer.fit_on_texts(lines)

    return tokenizer

def prepare_tokenizer(data, tokenizer_save_path = None):
    """Train Tokenizer of data

    Args:
        data (list): list of text lines
        tokenizer_save_path (str, optional): path to save tokenizer in. Defaults to None.

    Returns:
        tokenizer: trained tokenizer to tokenize data
        int: vocab size of tokenizer
    """
    tokenizer = tokenization(data)#.str.cat(sep=' '))
    vocab_size = len(tokenizer.word_index) + 1

    if tokenizer_save_path:
        directory = os.path.dirname(tokenizer_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(tokenizer_save_path)
        with open(tokenizer_save_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer, vocab_size


def prepare_data(data, src_tokenizer, tg_tokenizer, src_length, tg_length):
    """_summary_

    Args:
        data (pd.dataframe): dataframe represents data
        src_tokenizer (tokenizer): tokenizer of data source
        tg_tokenizer (tokenizer): tokenizer of target source
        src_length (int): length of source sentence
        tg_length (int): length of target sentence

    Returns:
        data_X: list of tokens of source
        data_Y: list of tokens of target
    """
    data_X = encode_sequences(src_tokenizer, src_length, data['source'])
    data_Y = encode_sequences(tg_tokenizer, tg_length, data['target'])


    return data_X, data_Y

def get_predictions(model, testX, predictions_output_file_path = None):
  preds = []
  for i in range(0,testX.shape[0],512):
    end = i+512
    if end > testX.shape[0]: end = testX.shape[0]
    batch = testX[i:end]
    predict_x = model.predict(batch.reshape(batch.shape[0],batch.shape[1]))
    preds_batch = np.argmax(predict_x, axis=-1)
    preds.extend(preds_batch)
  if predictions_output_file_path:
    save_as_pickle(preds, predictions_output_file_path)
  return preds

def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None

def pred_2_text(preds, target_tokentizer):
    # To decode, we need to create a reverse mapping from indices to words
    reverse_word_index = dict([(index, word) for (word, index) in target_tokentizer.word_index.items()])

    def decode_sequence(sequence):
        return ' '.join([reverse_word_index.get(i, '') for i in sequence])

    # Decode the sequences back to text
    decoded_texts = [decode_sequence(seq).strip() for seq in preds]
    return decoded_texts

def save_as_pickle(object, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as handle:
        object = pickle.load(handle)
    return object