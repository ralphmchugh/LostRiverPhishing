import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Correcting the file path
data = pd.read_csv('/content/spam_dataset.csv.csv')  # Double check the extension
print(data.head())

# Vectorizing the text data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])  # Assuming 'text' is the correct column name for email texts
y = data['spam']  # Adjust to the correct column name for spam labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_email(email):
    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)
    return 'spam' if prediction[0] == 1 else 'not spam'

# Example usage
sample_email = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."
print(predict_email(sample_email))

legit_email = "your password has been changed. Your phone has been hacked we need your 5 digit pin"
print(predict_email(legit_email))

