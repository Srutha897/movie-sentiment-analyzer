import streamlit as st
import pickle
import re

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Clean text function
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    return text

# Initialize session state
if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

# App title
st.title("🎬 Movie Sentiment Analyzer")
st.write("Enter a movie review and I'll predict whether it's positive or negative!")

# Example buttons
st.subheader("Try an example:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😊 Positive Example"):
        st.session_state.review_text = "This movie was absolutely amazing! The acting was brilliant and the story was incredibly moving."

with col2:
    if st.button("😞 Negative Example"):
        st.session_state.review_text = "This was the worst movie I have ever seen. Terrible acting, boring story, complete waste of time."

with col3:
    if st.button("😐 Mixed Example"):
        st.session_state.review_text = "The visuals were great but the story was confusing and the acting felt forced."

# Text input
review = st.text_area("Enter your movie review here:", value=st.session_state.review_text, height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        # Clean and vectorize
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]
        
        # Show result
        if prediction == 1:
            st.success(f"✅ POSITIVE — {confidence[1]*100:.1f}% confident")
            st.progress(confidence[1])
        else:
            st.error(f"❌ NEGATIVE — {confidence[0]*100:.1f}% confident")
            st.progress(confidence[0])