from flask import Flask, request, render_template
import re
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Baixe os recursos do NLTK apenas na primeira execução
nltk.download('stopwords')

app = Flask(__name__)

# Carregue os recursos
stop_words = set(stopwords.words('portuguese'))
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('model/sentiment_model.h5')


def clean_text(text):
    text = re.sub(r'[^a-zA-Zá-úÁ-ÚçÇ\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def predict_sentiment(comment):
    clean_comment = clean_text(comment)
    comment_seq = tokenizer.texts_to_sequences([clean_comment])
    comment_pad = pad_sequences(comment_seq, maxlen=100)
    prediction = model.predict(comment_pad)
    return "Positivo" if prediction[0][0] > 0.5 else "Negativo"


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        comment = request.form['comment']
        sentiment = predict_sentiment(comment)
    return render_template('index.html', sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
