import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import pickle
import os
from PIL import Image as PILImage

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
nltk.download('punkt')

MODEL_PATH = "image_caption_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

model = load_model(MODEL_PATH, compile=False)
tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
max_length = 51

base_model = InceptionV3(weights="imagenet")
cnn_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

grammar_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
grammar_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

translator = Translator()
lang_map = {"English": "en", "Telugu": "te", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn"}

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return cnn_model.predict(img, verbose=0)

def generate_caption(photo_feature):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()

def generate_captions_blip(image_path, processor, blip_model, num_captions=4): 
    raw_image = PILImage.open(image_path).convert("RGB")
    inputs = processor(images=raw_image, return_tensors="pt")
    generated_captions = set()
    while len(generated_captions) < num_captions:
        outputs = blip_model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            do_sample=True,
            top_p=0.9,
            temperature=1.3,
            num_return_sequences=1
        )
        caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        generated_captions.add(caption)
    return list(generated_captions)

def correct_grammar(text):
    input_text = "gec: " + text
    input_ids = grammar_tokenizer.encode(input_text, return_tensors='pt')
    outputs = grammar_model.generate(input_ids, max_length=64, num_beams=5, early_stopping=True)
    return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)

st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("\U0001F5BC️ Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns(2)
with col1:
    language = st.selectbox("Translate to Language", list(lang_map.keys()))
with col2:
    emotion = st.selectbox("Add Emotion", ["Normal", "Romantic", "Joke", "Happy", "Sad", "Angry"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    if st.button("Generate Captions"):
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        photo = extract_features(temp_path)
        caption_lstm = generate_caption(photo)

        blip_captions = generate_captions_blip(temp_path, blip_processor, blip_model, num_captions=4)

        all_captions = [caption_lstm] + blip_captions

        st.subheader("Generated Captions:")
        for i, cap in enumerate(all_captions, 1):
            caption = correct_grammar(cap)
            if emotion != "Normal":
                caption = f"[{emotion}] {caption}"
            if language != "English":
                try:
                    caption = translator.translate(caption, dest=lang_map[language]).text
                except:
                    st.warning("Translation failed. Showing English.")
            st.markdown(f"**\U0001F4F8 Caption {i}:** {caption}")

st.markdown("---")
st.markdown("### \U0001F4CA BLEU Score Evaluation (Manual Test)")

smoothie = SmoothingFunction().method4
reference_text = st.text_area("Enter reference captions (one per line)", height=100)
candidate_text = st.text_input("Enter generated candidate caption")

if st.button("Calculate BLEU Score") and reference_text and candidate_text:
    references = [nltk.word_tokenize(ref.strip()) for ref in reference_text.splitlines()]
    candidate = nltk.word_tokenize(candidate_text.strip())

    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    st.write(f"**BLEU-1:** {bleu1:.4f}")
    st.write(f"**BLEU-2:** {bleu2:.4f}")
    st.write(f"**BLEU-3:** {bleu3:.4f}")
    st.write(f"**BLEU-4:** {bleu4:.4f}")

st.markdown("---")
st.markdown("**Project by Group A**")
