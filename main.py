import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


class Jondomo():

    def __init__(self, name, texts, labels, label_mapping, epochs):
        self.name = name
        self.texts = texts
        self.labels = labels
        self.label_mapping = label_mapping
        self.epochs = epochs
        self.label_mapping_inv = self.get_label_mapping_inv()
        self.tokenizer = Tokenizer()
        self.model = Sequential()
        self.get_model()


    def get_label_mapping_inv(self):
        return {v: k for k, v in self.label_mapping.items()}

    def get_padded_sequences(self):
        self.tokenizer.fit_on_texts(self.texts)
        sequences = self.tokenizer.texts_to_sequences(self.texts)
        padded_sequences = pad_sequences(sequences)
        return padded_sequences

    def get_numeric_label(self):
        numeric_labels = np.array([self.label_mapping[label] for label in self.labels], dtype=np.int32)
        return numeric_labels

    def get_model(self):
        padded_sequences = self.get_padded_sequences()
        numeric_labels = self.get_numeric_label()
        self.model.add(Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=16,
                                 input_length=padded_sequences.shape[1]))
        self.model.add(LSTM(32))
        self.model.add(Dense(len(self.label_mapping), activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(padded_sequences, numeric_labels, epochs=self.epochs, validation_split=0.30)

    def get_predict(self, input_text):
        new_sequences = self.tokenizer.texts_to_sequences(input_text)
        new_padded_sequences = pad_sequences(new_sequences, maxlen=self.get_padded_sequences().shape[1])
        predictions = self.model.predict(new_padded_sequences)
        predicted_labels = [self.label_mapping_inv[tf.argmax(prediction).numpy()] for prediction in predictions]
        print(f'{self.name} : {input_text[0] + predicted_labels[0]}')
