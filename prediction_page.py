import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier

# Define file paths for loading the model, vectorizer, and image
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"
image_path = "tweeter and X LOGO.jpg"
data_path = "twitter_racism_parsed_dataset.csv"

def load_model_and_vectorizer():
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    return loaded_model, loaded_vectorizer

def predict_tweet(model, vectorizer, tweet):
    X = vectorizer.transform([tweet])
    prediction = model.predict(X)
    probability = model.predict_proba(X)[0]
    return "Racist" if prediction[0] == 1 else "Not Racist", probability

def prediction_page(dark_mode):
    st.title("Racism Detection")
    st.header("Predict Racism in Tweets")

    # Add Twitter logo from local file
    image = Image.open(image_path)
    st.image(image, use_column_width=True, caption='Twitter Logo')

    st.write("Enter a tweet and click 'Predict' to determine if it's racist or not.")

    # Add an input box for user to enter the tweet
    user_input = st.text_area("Enter the tweet:", "")
    
    # Add a button to trigger prediction
    if st.button('Predict'):
        model, vectorizer = load_model_and_vectorizer()
        result, probability = predict_tweet(model, vectorizer, user_input)
        st.write(f"Prediction: {result}")
        st.write(f"Probability of being 'Not Racist': {probability[0]:.2f}")
        st.write(f"Probability of being 'Racist': {probability[1]:.2f}")

        # Visualize the probability
        fig, ax = plt.subplots()
        categories = ['Not Racist', 'Racist']
        sns.barplot(x=categories, y=probability, ax=ax)
        ax.set_ylim(0, 1)
        for i, v in enumerate(probability):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
        st.pyplot(fig)

        # Add Word Cloud visualization
        data = pd.read_csv(data_path)
        st.subheader("Word Cloud of Racist Tweets")
        racist_tweets = " ".join(data[data['oh_label'] == 1]['Text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white' if not dark_mode else 'black').generate(racist_tweets)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Add download button for report
        st.subheader("Download Prediction Report")
        report = f"Prediction: {result}\nProbability of being 'Not Racist': {probability[0]:.2f}\nProbability of being 'Racist': {probability[1]:.2f}"
        st.download_button(label="Download Report", data=report, file_name="prediction_report.txt", mime="text/plain")

    # Add feedback section
    st.subheader("Feedback")
    feedback = st.text_area("Provide your feedback here:")
    if st.button('Submit Feedback'):
        with open("feedback.txt", "a") as f:
            f.write(feedback + "\n")
        st.write("Thank you for your feedback!")
