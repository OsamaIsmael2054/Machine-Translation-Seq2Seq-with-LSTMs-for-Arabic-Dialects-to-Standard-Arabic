import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import argparse
from utils import load_pickle, remove_punctuation_arabic, get_predictions, prepare_data, pred_2_text
from keras.models import load_model
import pandas as pd

def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data', help='Data set path default value "data"')
    parser.add_argument('--output_path', type=str, default='outputs', help='outputs path default value "outputs"')
    parser.add_argument('--src_length', type=int, default=13, help='The length of input sequences (number of time steps in each input sequence)')
    parser.add_argument('--tg_length', type=int, default=15, help='The number of time steps in the output sequence.')
    parser.add_argument('--checkpoint_path', type=str, help='keras checkpoint path')
    parser.add_argument('--scr_tokenizer_path', type=str, help='source_tokenizer_path')
    parser.add_argument('--trg_tokenizer_path', type=str, help='target_tokenizer_path')

    # Parse the arguments and return the Namespace object
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path
    checkpoint_path = args.checkpoint_path
    src_tokenizer_path = args.scr_tokenizer_path
    tg_tokenizer_path = args.trg_tokenizer_path
    src_length = args.src_length
    tg_length = args.tg_length

    src_tokenizer = load_pickle(src_tokenizer_path)
    tg_tokenizer = load_pickle(tg_tokenizer_path)

    model = load_model(checkpoint_path)

    data_test = pd.read_csv(data_path+"/test.csv")
    # data_test = data_test.iloc[:10].copy()
    data_test = remove_punctuation_arabic(data_test)

    testX, testY = prepare_data(data_test, src_tokenizer, tg_tokenizer, src_length, tg_length)
    pred = get_predictions(model, testX, output_path +"/lstm_seq2seq/prediction/output_index.pickle")
    preds_text = pred_2_text(pred,tg_tokenizer)
    data_test['predictions'] = preds_text
    data_test.to_csv(output_path + "/predictions.csv",index=False)


if __name__ == "__main__":
    main()