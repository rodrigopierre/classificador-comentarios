import pandas as pd
data = pd.read_csv('model/data.csv', sep=';', encoding='latin1')

import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
nltk.download('rslp')
stop_words = set(stopwords.words('portuguese'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Zá-úÁ-ÚçÇ\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['text'] = data['text'].apply(clean_text)

from sklearn.model_selection import train_test_split

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test))

loss, accuracy = model.evaluate(X_test_pad, y_test)

import os
os.system("clear")

print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')


# Salvar o modelo treinado e o tokenizer
model.save("model/sentiment_model.h5")
import pickle

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)



def predict_sentiment(comment):
    new_comments_seq = tokenizer.texts_to_sequences([comment])
    new_comments_pad = pad_sequences(new_comments_seq, maxlen=100)

    prediction = model.predict(new_comments_pad)
    print(prediction)
    return "Positivo" if prediction[0][0] > 0.5 else "Negativo"


print("Digite seus comentários para análise de sentimento (digite 'sair' para encerrar):")
while True:
    user_input = input("Comentário: ")
    if user_input.lower() == 'sair':
        print("Programa encerrado.")
        break
    sentiment = predict_sentiment(user_input)
    print(f"Sentimento detectado: {sentiment}\n")