import streamlit as st
import requests
import nltk
nltk.download('punkt')
from textblob import TextBlob
from collections import Counter
import warnings
warnings.filterwarnings("ignore")



st.set_page_config(layout="wide")


st.markdown(
    """
    <style>
    .stApp {
        background-color:rgb(20, 30, 30);
        color: #FFFFFF;
    }
    .stTextArea textarea {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .stButton button {
        background-color: #7a7367;
        color: white;
    }
    .stProgress > div > div > div {
        background-color: #FAFAFA
    }
    </style>
    """,
    unsafe_allow_html=True
)

HUGGING_FACE_TOKEN = st.secrets["hugging_face"]["token"]


API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
headers = {"Authorization": f"Bearer {st.secrets['hugging_face']['token']}"}


label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def analyze_sentiment(text):
    data = query({"inputs": text})
    if isinstance(data, dict) and "error" in data:
        st.error(f"API Error: {data['error']}")
        return None  # Return None if there's an error
    
    try:
        # Check if the response is a list of lists (multiple sentences)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # Process all sentences
            all_sentiments = data
        else:
            # If the response is a single list of sentiment scores, treat it as one sentence
            all_sentiments = [data]
        
        # Initialize a dictionary to store aggregated scores
        aggregated_scores = {"LABEL_0": 0.0, "LABEL_1": 0.0, "LABEL_2": 0.0}
        
        # Iterate over all sentences and aggregate scores
        for sentiments in all_sentiments:
            for sentiment in sentiments:
                label = sentiment["label"]
                score = sentiment["score"]
                aggregated_scores[label] += score
        
        # Calculate average scores
        num_sentences = len(all_sentiments)
        average_scores = {label: score / num_sentences for label, score in aggregated_scores.items()}
        
        return average_scores
    except Exception as e:
        st.warning(f"Failed to analyse sentiment: {e}")
        return None



st.title("Sentiment Analysis App")


st.subheader("Try adding more text or changing the existing text of the examples:")

'''

'''
examples = [
    """I absolutely love this product! It's amazing and works perfectly.""",
    """This is the worst experience I've ever had. I regret buying this.""",
    """The meeting is scheduled for 3 PM today in Conference Room B. Please bring your updated reports.""",
    """Hello Tanzhu,

Thank you for your interest in Jack Link’s! We received your application for our Junior Business Analyst (m/w/d) position located in Eyber Str. 81, 91522 Ansbach, Germany!

Learn more about us and our amazing story of a humble family recipe passed down through generations to become the global brand it is today. The story of Jack Link’s is an amazing one: The Celebration of Jack Link's

Applying to the posting, just as you have done, is the best way to get your resume and contact information to the correct person for this job opening. If your work experience matches the requirements outlined in the job description for which you applied, a recruiter or hiring manager will be in contact with you. If your work experience does not match the requirements outlined in the job description for which you applied, your resume will be kept on file for future openings that may be a better match for your background. 

Thank you again for your application and for your interest in Jack Link’s and Running with Sasquatch!

Sincerely, 

Jack Link's Recruiting Team""",
    """Dear Mr Dereli,

 

Thank you for applying for the position of (Junior) Data Analyst (w/m/d) and for your interest in MSR Consulting Group.

After careful consideration, we regret to inform you that we have chosen another candidate for this position who matches the criteria even more closely.

Please be assured, however, that this decision is by no means a rejection of you and your character or your job profile,

 

We thank you for taking the time to apply and are sorry that we are not able to offer you a more positive response.

 

We wish you every success in your continued personal and professional life.

 

Kind regards,

Teresa Höhl
Talent Acquisition + Employer Branding"""
]


cols = st.columns(len(examples))
for i, example in enumerate(examples):
    if cols[i].button(f"Example {i+1}"):
        user_input = example
        st.session_state.user_input = user_input
'''
'''
# Text area for user input
user_input = st.text_area("Enter your own text here:",
                        value=st.session_state.get("user_input", ""),
                        height=200,  # Fixed height of 200 pixels
                        key="text_area")
'''

'''
if st.button("Analyze Sentiment!"):
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            average_scores = analyze_sentiment(user_input)
            if average_scores:
                st.subheader("Sentiment Analysis Results")

                # Sort the sentiment scores in descending order
                sorted_scores = sorted(
                    average_scores.items(),
                    key=lambda x: x[1],  # Sort by score (value)
                    reverse=True  # Descending order
                )

                for label, score in average_scores.items():
                    sentiment_label = label_mapping[label]
                    st.write(f"{sentiment_label}: {score * 100:.2f}%")
                    st.progress(score)
            else:
                st.error("Failed to analyze sentiment. Please try again.")
    else:
        st.write("Please enter some text!")

if st.button("Clear Results"):
    st.session_state.user_input = ""
    st.rerun()
'''

'''
with st.expander("**Click to see more details**"):
    col1, col2 = st.columns(2)

    if user_input:
         # Create a TextBlob object
        blob = TextBlob(user_input)

        # Perform text analysis
        word_count = len(blob.words)
        char_count = len(user_input)
        sentence_count = len(blob.sentences)
        most_common_words = Counter(blob.words).most_common(5)

        # Perform sentiment analysis
        sentiment = blob.sentiment
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity

        # Display text statistics
        with col1:
            st.subheader("Text Statistics")
            st.write(f"**Word Count:** {word_count}")
            st.write(f"**Character Count:** {char_count}")
            st.write(f"**Sentence Count:** {sentence_count}")
            st.write(f"**Subjectivity:** {subjectivity:.4f} (Range: 0 to 1)")
            # Interpret subjectivity
            if subjectivity > 0.66:
                st.write("**Text Type:** Subjective (Opinionated)")
            elif subjectivity < 0.33:
                st.write("**Text Type:** Objective (Fact-based)")
            else:
                st.write("**Text Type:** In-Between (Neither Subjective or Objective)")

        with col2:
            st.subheader("Most Common Words")
            for word, count in most_common_words:
                st.write(f"- {word} ({count} times)")