import time
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
from keras.utils import to_categorical
from typing import List
from storage import SentenceStorage
import threading
import os
from contextlib import asynccontextmanager
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the model
model = tf.keras.models.load_model("lstm_model.keras")

# Initialize SentenceStorage (your db-like logic)
storage = SentenceStorage()

# Load the tokenizer from file
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Read old FAQ data (from faq.txt)
if len(storage.all_p_sentences()) == 0:
 with open("faq.txt", "r") as f:
    faq_data = f.readlines()

# Append old FAQs to SentenceStorage if not already stored
if len(storage.all_p_sentences()) == 0:
 for sentence in faq_data:
    storage.add_sentence(sentence.strip(), processed=True)

# Background retraining logic
RETRAIN_INTERVAL = 300  # 5 minutes in seconds
RETRAIN_THRESHOLD = 5   # trigger if unprocessed sentences >= 5

def retrain_loop():
    while True:
        time.sleep(10)  # Check every 10s
        unprocessed = storage.get_unprocessed()
        if len(unprocessed) >= RETRAIN_THRESHOLD or int(time.time()) % RETRAIN_INTERVAL < 10:
            retrain_model()



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background retrain loop thread
    threading.Thread(target=retrain_loop, daemon=True).start()
    yield  # Control returns to FastAPI
    # You can clean up here if needed on shutdown


# FastAPI app definition
app = FastAPI(lifespan=lifespan)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Threaded background retrain loop
def retrain_model():
    global model, tokenizer
    faq_data = storage.all_p_sentences()
    data = storage.get_unprocessed()
    if not data:
        return
    print("Rebuilding started with unprocessed sentences:", data)
    faq_data = [row[0] for row in faq_data if row[0].strip()]
    # Extract sentences
    sentences = [s for _, s in data] + faq_data  # Combine old (faq.txt) and new sentences
    print(len(data),len(faq_data),faq_data[0],"--",len(sentences))
    # Rebuild tokenizer to include new vocabulary
    tokenizer.fit_on_texts(sentences)
    print("Updated word_index:", tokenizer.word_index)

    # Rebuild model based on updated tokenizer vocabulary size
    new_vocab_size = len(tokenizer.word_index) + 1
    
    print("Vocab:",new_vocab_size)

    # Prepare data for retraining
    input_sequences = []
    ids = []
    
    for (sentence_id, sentence) in data:  # old + new
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
        for j in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:j+1])
            if sentence_id is not None:
                ids.append(sentence_id)
    for sentence in faq_data:
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
        for j in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:j+1])
            
    max_len = max([len(x) for x in input_sequences])
    padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')
    X = padded_input_sequences[:,:-1]
    y = padded_input_sequences[:,-1]
    print(X.shape,y.shape)

    y = to_categorical(y,num_classes=283)

    model = Sequential()
    model.add(Embedding(283, 100))
    model.add(LSTM(150))
    model.add(Dense(283, activation='softmax'))
    # Retrain model
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=0)
    model.save("lstm_model.keras")

    # Save new tokenizer
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Mark sentences as processed
    storage.mark_processed(ids)
    print("Rebuild completed")

def build_model(vocab_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 100, input_length=56))
    model.add(tf.keras.layers.LSTM(150))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



class InputText(BaseModel):
    text: str
    steps: int = 10  # Number of prediction steps

class BatchInput(BaseModel):
    texts: List[str]

@app.post("/generate")
async def generate_text(input: InputText):
    text = input.text
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
    prediction = model.predict(padded_token_text, verbose=0)[0]

    top_n = input.steps
    top_indices = prediction.argsort()[-top_n:][::-1]

    # Map indices to words
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    next_possible_words = [index_to_word.get(i, '') for i in top_indices]

    return {"generate": next_possible_words}

@app.post("/submit")
async def submit_sentence(input: InputText):
    storage.add_sentence(input.text)
    return {"status": "stored"}

@app.post("/submit-batch")
async def submit_batch(input: BatchInput):
    for sentence in input.texts:
        storage.add_sentence(sentence)
    return {"status": f"stored {len(input.texts)} sentences"}

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model)
    return {"status": "retraining started in background"}

