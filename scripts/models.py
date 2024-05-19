
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers

class Seq2SeqModel:
    def __init__(self, in_vocab, out_vocab, in_timesteps, out_timesteps, units, checkpoint_path = None):
        """ Init model 

        Args:
            in_vocab (int): Number of vocab size of encoder
            out_vocab (int): Number of vocab size of decoder
            in_timesteps (int): length of input sentence
            out_timesteps (int): length of output sentence
            units (int): number of hidden units
            checkpoint_path (str, optional): Path to saving directory. Defaults to None.
        """
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.units = units
        self.checkpoint_path = checkpoint_path
        self.model = self.build_model()
        self.load_checkpoint()
    
    def build_model(self):
        """Building model

        Returns:
            model: Builded model
        """
        model = Sequential()
        model.add(Embedding(self.in_vocab, self.units, input_length=self.in_timesteps, mask_zero=True))
        model.add(LSTM(self.units))
        model.add(RepeatVector(self.out_timesteps))
        model.add(LSTM(self.units, return_sequences=True))
        model.add(Dense(self.out_vocab, activation='softmax'))
        return model
    
    def compile_model(self, learning_rate=0.01):
        """compile mode

        Args:
            learning_rate (float, optional): learning rate of optimizer. Defaults to 0.01.
        """
        rms = optimizers.RMSprop(learning_rate=learning_rate)
        self.model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')
    
    def load_checkpoint(self):
        """Loading checkpoint if exists
        """
        if os.path.exists(self.checkpoint_path):
            self.model.load_weights(self.checkpoint_path)
            print(f"Checkpoint loaded from {self.checkpoint_path}")
        else:
            print("No checkpoint found, starting from scratch.")
    
    def get_checkpoint_callback(self):
        """callback to save model

        Returns:
            callback: callback of saving model 
        """
        checkpoint_path = os.path.join(self.checkpoint_path, 'model_epoch-{epoch:02d}_val_loss-{val_loss:.2f}.keras')
        return ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')