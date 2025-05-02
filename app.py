import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from keras.preprocessing.sequence import pad_sequences
import uvicorn
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("lstm_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 🚀 Create FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str
    steps: int = 10  # optional: number of words to generate

@app.post("/generate")
async def generate_text(input: InputText):
    text = input.text
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
    prediction = model.predict(padded_token_text, verbose=0)[0]

    top_n = input.steps  # how many suggestions user wants
    top_indices = prediction.argsort()[-top_n:][::-1]  # get top N indices

    # Map indices to words
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    next_possible_words = [index_to_word.get(i, '') for i in top_indices]

    return {"generate": next_possible_words}
