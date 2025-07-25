import streamlit as st
import joblib

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_all_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    nb = joblib.load("naive_bayes_model.pkl")
    lr = joblib.load("logistic_regression_model.pkl")
    svm = joblib.load("svm_model.pkl")
    return tfidf, nb, lr, svm

tfidf, nb_model, lr_model, svm_model = load_all_models()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("💬 Emotion Detection from Text")

st.write("Type a sentence and select the model you want to use to detect the emotion.")

# User input
text_input = st.text_area("Enter your sentence:")

# Model selection
model = st.radio(
    "Choose a model:",
    ["Naive Bayes", "Logistic Regression", "Support Vector Machine"]
)

# Predict
if st.button("🔍 Predict Emotion"):
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        input_vector = tfidf.transform([text_input])

        if model == "Naive Bayes":
            prediction = nb_model.predict(input_vector)[0]
        elif model == "Logistic Regression":
            prediction = lr_model.predict(input_vector)[0]
        else:
            prediction = svm_model.predict(input_vector)[0]

        st.success(f"🎯 Predicted Emotion: `{prediction.upper()}`")
