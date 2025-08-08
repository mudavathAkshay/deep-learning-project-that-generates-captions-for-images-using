🖼️ Image Caption Generator
An AI-powered application that generates descriptive captions for images using a CNN + LSTM model trained on 118k COCO dataset images, refined with BLIP for better accuracy. The app supports grammar correction, translation, and emotion tagging, with a Streamlit frontend for real-time caption generation.

🚀 Features
Custom Trained Model – CNN + LSTM architecture on COCO dataset

BLIP Refinement – Improves caption quality and fluency

Grammar Correction – Uses NLP model to fix errors

Emotion Tagging – Add tone like Romantic, Happy, Sad, etc.

Language Translation – Supports English, Telugu, Hindi, Tamil, Bengali

BLEU Score Evaluation – Check model accuracy

Streamlit Web App – Upload an image and get instant captions

🛠️ Tech Stack
Python, TensorFlow/Keras

Transformers (BLIP, Grammar Correction Model)

Streamlit

Googletrans (Translation)

NLTK (BLEU score calculation)

📂 Project Structure
arduino
Copy
Edit
📁 image-caption-generator
│── app.py                # Streamlit application
│── image_caption_model.keras
│── tokenizer.pkl
│── requirements.txt
│── README.md
│── sample_images/
📸 Example Output
Input Image:
Couple standing on beach (Black & White)

Generated Captions:

Lost in the vastness of the ocean, finding solace in each other's embrace 🌊

Two souls, one horizon.

Finding peace on the shore, their silhouettes against the moody sky.

A grayscale love story unfolding on the sandy shore.

Embracing the quiet strength of the ocean ❤️
