import streamlit as st
from transformers import pipeline

with st.sidebar:
    with st.container():
        # Put bio here with markdown
        st.empty()

    st.subheader('Natural Language Processing')
    st.write("[Text Classification (Emotion)](https://joshuasigma-tce-streamlit-app-0j7jjl.streamlit.app/)")
    st.empty()
    st.subheader("Imaging")
    st.write("[Fast Style Transfer](https://joshuasigma-fast-image-style-transfer-app-23kl7l.streamlit.app/)")
    st.write("[Fast Instagram Filters with Pygram](https://joshuasigma-instagram-filters-app-2aine9.streamlit.app/)")

st.title('Classify text by emotion')
st.text('Joshua Patterson  |  March 18, 2023')
st.image('https://i.ibb.co/025LsCj/bert-img-google.jpg', caption="Jochen Hartmann, 'Emotion English DistilRoBERTa-base'. https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.")

with st.container():
    st.write("Text classification is important in many applications these days. We can utitlize a model from HuggingFace, based off of Google's popular BERT design, to perform this. "
             "Using the transformers package and a version of BERT, we can perform emotion classification on text with minimal code.")

with st.container():
    st.subheader("The pretuned BERT model")
    st.write('The model and pipeline are already setup by HuggingFace. "Emotion English DistilRoBERTa-base," the name of the model, was trained using Twitter, Reddit, student reports and TV dialogue, '
             'You can find it [here](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base). It has seven emotions as labels. Anger, disgust, fear, joy, neutral, sadness and surprise.')

with st.container():
    st.subheader("Real world use cases")
    st.write("The first thing I think of when thinking of potential use cases is marketing analysis. Using text about a certain subject to determine emotion. "
             "I'm sure there are many use cases that haven't yet been considered.")

with st.container():
    st.subheader("The code")
    st.write("Make sure you have the transformers library installed and either tensorflow or pytorch. If not you can install them using pip like this (I reccomend using a venv): ")
    st.code("pip install transformers")
    st.write("and")
    st.code("pip install tensorflow")
    st.write("or")
    st.code("pip install torch")
    st.write("Once you have transformers and a framework you are ready to write and run the code.")
    st.code("""from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
classifier("This is really awesome!")
    """)
    st.code("""Output:
[[{'label': 'anger', 'score': 0.004419783595949411},
  {'label': 'disgust', 'score': 0.0026119900392368436},
  {'label': 'fear', 'score': 0.0009138521908316761},
  {'label': 'joy', 'score': 0.8691687984466553},
  {'label': 'neutral', 'score': 0.048964586851000786},
  {'label': 'sadness', 'score': 0.003092392183840275},
  {'label': 'surprise', 'score': 0.070528684265911579}]]
    """)

with st.container():
    st.subheader("Examples")
    st.write("Let's imagine that we were asked to analyze comments about Ai. Here are a few possible comments.")
    st.code("classifier('I think AI will make the world a better place.')")
    st.code('''Output:
[
  [
    {
      "label": "joy",
      "score": 0.49170994758605957
    },
    {
      "label": "neutral",
      "score": 0.45121660828590393
    ...
    }
  ]
]
    ''')
    st.code("classifier('Im so scared it will take over.')")
    st.code('''Output:
[
  [
    {
      "label": "fear",
      "score": 0.9907322525978088
    ...
    }
  ]
]

    ''')
    st.code("classifier('Its just another computer program.')")
    st.code('''Output:
[
  [
    {
      "label": "neutral",
      "score": 0.943964421749115
    ...
    }
  ]
]
    ''')
    st.write("I hope you get a good ida of whats capable from these examples. Try it out for yourself below.")

with st.spinner("Loading model, it's cool enough to wait for!"):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def get_score(emotion):
    return emotion["score"]

def get_percent(num):
    return round(num * 100)

def get_emotions(text):
    detected_emotions = []
    for emotion in classifier(text)[0]:
        if emotion['score']>0.10:
            detected_emotions.append(emotion)
    detected_emotions.sort(key=get_score, reverse=True)
    for emotion in detected_emotions:
        st.text(emotion['label'].capitalize() + "  |  " + str(get_percent(emotion['score'])) + " %")
        st.progress(emotion['score'])

st.header('Input text to classify')

# Input text
input_text = st.text_input(label="Input text to classify here.", max_chars=112)

# If input text var is not none display text
if input_text != "":
    get_emotions(input_text)
else:
    st.empty()
