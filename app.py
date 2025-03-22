mport streamlit as st
import joblib
import re
import string
# Load model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
def clean_text(text):
text = text.lower()
text = re.sub(r'\d+', '', text)
text = text.translate(str.maketrans('', '', string.punctuation))
text = re.sub(r'\s+', ' ', text).strip()
return text
def predict_sms(message):
cleaned_message = clean_text(message)
vectorized_message = vectorizer.transform([cleaned_message])
prediction = model.predict(vectorized_message)[0]
return "Spam" if prediction == 1 else "Ham"
st.title("📩 SMS Spam Classifier")
user_input = st.text_area("Enter SMS Message Here:", "")
if st.button("Check"):
if user_input.strip():
result = predict_sms(user_input)
st.error("🚨 Spam Detected!") if result == "Spam" else st.success("✅ Safe Message")
else:
st.warning("⚠ Enter a message to classify.")
