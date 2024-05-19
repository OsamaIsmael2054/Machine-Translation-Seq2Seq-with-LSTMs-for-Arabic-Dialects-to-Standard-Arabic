import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse

from scripts.utils import remove_punctuation_arabic, prepare_tokenizer, prepare_data
from scripts.models import Seq2SeqModel
from keras.callbacks import EarlyStopping

import wandb
from wandb.integration.keras import WandbMetricsLogger



def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path
    units = args.units
    src_length = args.src_length
    tg_length = args.tg_length
    

    wandb.init(config={"bs": 12})
    output_path = "outputs/lstm_seq2seq"

    data_train = pd.read_csv(data_path+"/train.csv").drop_duplicates().dropna()
    data_val = pd.read_csv(data_path+"/val.csv").drop_duplicates().dropna()

    data_train = remove_punctuation_arabic(data_train)
    data_val = remove_punctuation_arabic(data_val)

    src_tokenizer, src_vocab_size = prepare_tokenizer(data_train['source'], tokenizer_save_path = output_path +"/tokenizers/source_tknz.pickle")
    tg_tokenizer, tg_vocab_size = prepare_tokenizer(data_train['target'], tokenizer_save_path = output_path +"/tokenizers/target_tknz.pickle")
    
    trainX, trainY = prepare_data(data_train, src_tokenizer, tg_tokenizer, src_length, tg_length)
    valX, valY = prepare_data(data_val, src_tokenizer, tg_tokenizer, src_length, tg_length)


    checkpoint_path = output_path + "/checkpoint/"
    units = 512

    seq2seq_model = Seq2SeqModel(src_vocab_size, tg_vocab_size, src_length, tg_length, units, checkpoint_path = checkpoint_path)
    seq2seq_model.compile_model(learning_rate=0.01)

    # To use the checkpoint callback during training
    checkpoint_callback = seq2seq_model.get_checkpoint_callback()
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # train model
    history = seq2seq_model.model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                        epochs=10, batch_size=8, validation_data=(valX, valY.reshape(valY.shape[0], valY.shape[1], 1)),
                        callbacks=[WandbMetricsLogger(),checkpoint_callback,early_stopping_callback],verbose=1)
    
    
def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data', help='Data set path default value "data"')
    parser.add_argument('--output_path', type=str, default='outputs', help='outputs path default value "outputs"')
    parser.add_argument('--units', type=int, default=512, help='Embedding and LSTM units default value 512')
    parser.add_argument('--src_length', type=int, default=13, help='The length of input sequences (number of time steps in each input sequence)')
    parser.add_argument('--tg_length', type=int, default=15, help='The number of time steps in the output sequence.')
    
    # Parse the arguments and return the Namespace object
    return parser.parse_args()


if __name__ == "__main__":
    main()