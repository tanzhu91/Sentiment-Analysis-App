import streamlit as st
import requests



st.set_page_config(page_title="Sentiment Analysis App", layout="wide")


st.markdown(
    """
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextArea textarea {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .stButton button {
        background-color: #1976D2;
        color: white;
    }
    .stProgress > div > div > div {
        background-color: #FAFAFA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

HUGGING_FACE_TOKEN = st.secrets["hugging_face"]["token"]


API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
headers = {"Authorization": f"Bearer {st.secrets['hugging_face']['token']}"}  # Replace with your token


label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


st.title("Sentiment Analysis App")


st.subheader("Try adding more text or changing the existing text of the examples:")
st.subheader("Try these examples:")
examples = [
    "I absolutely love this product! It's amazing and works perfectly.",
    "This is the worst experience I've ever had. I regret buying this.",
    "The product is okay, but it didn't really meet my expectations.",
    "I'm not sure how I feel about this. It's neither great nor terrible.",
    "The service was fantastic, but the product itself was disappointing."
]


cols = st.columns(len(examples))
for i, example in enumerate(examples):
    if cols[i].button(f"Example {i+1}"):
        user_input = example
        st.session_state.user_input = user_input


user_input = st.text_area("Enter your own text here:", value=st.session_state.get("user_input", ""))


if st.button("Clear Results"):
    st.session_state.user_input = "" 

if st.button("Analyze Sentiment !"):
    if user_input:
        data = query({"inputs": user_input})
        if isinstance(data, dict) and "error" in data:
            st.error(f"API Error: {data['error']}")
        else:
            try:
                sentiments = data[0] 
                for result in sentiments:
                    label = label_mapping[result['label']]
                    score = result['score']
                    st.write(f"{label}: {score * 100:.2f}%")
                    st.progress(score)
            except (KeyError, IndexError, TypeError):
                st.error("Unexpected API response format. Please check the API response.")
    else:
        st.write("Please enter some text!")