import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from flask import Flask, request, jsonify
import pickle
import os
import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Email extraction and preprocessing functions
def clean_text(text):
    decoded_bytes, charset = decode_header(text)[0]
    if isinstance(decoded_bytes, bytes):
        if charset:
            text = decoded_bytes.decode(charset)
        else:
            text = decoded_bytes.decode()
    return text


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    text = soup.get_text()
    cleaned_text = text.strip()
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text


def extract_clean_text_from_plain_text(text):
    cleaned_text = text.strip()
    cleaned_text = ' '.join(cleaned_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').split())
    return cleaned_text


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)


# Model training and Flask app functions
def load_data():
    data = pd.read_csv("data.csv")
    df = pd.DataFrame(data)
    return df


def preprocess_text_df(df):
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
    return df


def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred)

    return model, vectorizer


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))


def save_model(model, vectorizer):
    if not os.path.exists('model'):
        os.makedirs('model')

    with open('model/spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


def load_model():
    with open('model/spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@app.route('/predict', methods=['POST'])
def predict():
    model, vectorizer = load_model()
    data = request.get_json()
    text = data['text']
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return jsonify({'prediction': prediction[0]})


def fetch_emails():
    handshake = imaplib.IMAP4_SSL('imap.zoho.com', 993)
    handshake.login('sanjana.gadaginmath@jmtworldwidellc.com', 'bdcP6bAFr81z')
    handshake.select('Inbox')
    status, messages = handshake.search(None, 'ALL')
    email_ids = messages[0].split()

    for email_id in email_ids:
        status, msg_data = handshake.fetch(email_id, '(RFC822)')
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject = clean_text(msg["Subject"])
        email_from = clean_text(msg["From"])

        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if "attachment" not in content_disposition:
                if content_type == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    cleaned_body = extract_clean_text_from_plain_text(body)
                    predict_and_move_spam(handshake, email_id, cleaned_body, subject, email_from)
                elif content_type == "text/html":
                    html_body = part.get_payload(decode=True).decode()
                    text_body = extract_text_from_html(html_body)
                    predict_and_move_spam(handshake, email_id, text_body, subject, email_from)


def predict_and_move_spam(handshake, email_id, body, subject, email_from):
    model, vectorizer = load_model()
    text_tfidf = vectorizer.transform([body])
    prediction = model.predict(text_tfidf)[0]
    print(f"From: {email_from}")
    print(f"Subject: {subject}")
    print(f"Body: {body[:100]}")
    print(f"Prediction: {prediction}")

    if prediction == 'spam':
        move_to_spam_folder(handshake, email_id)
    print("****************************************************************")

def move_to_spam_folder(handshake, email_id):
    result = handshake.store(email_id, '+FLAGS', '\\Seen')
    result = handshake.copy(email_id, 'Spam')
    if result[0] == 'OK':
        handshake.store(email_id, '+FLAGS', '\\Deleted')
        handshake.expunge()
    print(f"Email {email_id.decode()} moved to Spam folder.")


if __name__ == '__main__':
    df = load_data()
    df = preprocess_text_df(df)
    model, vectorizer = train_model(df)
    save_model(model, vectorizer)
    fetch_emails()
    app.run(debug=True)
