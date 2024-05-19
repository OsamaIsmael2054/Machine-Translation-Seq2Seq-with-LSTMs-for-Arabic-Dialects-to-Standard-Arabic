import nltk
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import argparse

# Make sure to download the necessary NLTK data
nltk.download('punkt')

# Tokenize Arabic text
def tokenize(text):
    return nltk.word_tokenize(text)


def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_path', type=str, help='predictions path')
    # Parse the arguments and return the Namespace object
    return parser.parse_args()

def main():
    args = parse_args()
    predictions_path = args.predictions_path

    df = pd.read_csv(predictions_path)
    reference_texts = df['target']
    candidate_texts = df['predictions']
    # Tokenize the texts
    references = [[tokenize(ref)] for ref in reference_texts]
    candidates = [tokenize(cand) for cand in candidate_texts]
    corpus_score = corpus_bleu(references, candidates)
    print(f"Corpus BLEU score: {corpus_score:.4f}")

if __name__ == "__main__":
    main()