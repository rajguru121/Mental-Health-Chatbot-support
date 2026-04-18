import pickle
import re
import numpy as np

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model/risk_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z\' ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================
# KEYWORD LISTS
# =========================

# These words/phrases ALWAYS trigger HIGH risk (suicide/self-harm related)
HIGH_RISK_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "want to die", "wanna die",
    "end my life", "take my life", "ending it all",
    "don't want to live", "dont want to live",
    "no reason to live", "better off dead",
    "self harm", "cut myself", "hurt myself",
    "hang myself", "jump off", "overdose",
    "wish i was dead", "wish i were dead",
    "nobody would miss me", "not worth living",
    "i'm a burden", "im a burden",
    "can't take it anymore", "cant take it anymore",
    "goodbye forever", "final goodbye", "last message",
    "planning to die", "want to end it",
    "no way out", "give up on life",
]

# These words/phrases trigger MEDIUM risk (emotional distress, NOT high danger)
MEDIUM_RISK_KEYWORDS = [
    "depression", "depressed", "anxiety", "anxious",
    "lonely", "hopeless", "worthless", "helpless",
    "stressed", "overwhelmed", "exhausted", "miserable",
    "crying", "can't sleep", "cant sleep", "insomnia",
    "panic attack", "feeling low", "feeling down",
    "hate myself", "i'm sad", "im sad", "so sad",
    "broken", "empty inside", "numb", "suffering",
    "struggling", "falling apart",
]

# These words/phrases indicate LOW risk (positive sentiment)
LOW_RISK_KEYWORDS = [
    "happy", "great", "good", "amazing", "love",
    "wonderful", "fantastic", "blessed", "grateful",
    "excited", "joyful", "cheerful", "proud",
    "feeling good", "doing well", "doing great",
]

# =========================
# PREDICTION FUNCTION
# =========================
def predict_risk(text):
    text_clean = clean_text(text)

    # Rule 1: HIGH risk keywords always return HIGH
    for phrase in HIGH_RISK_KEYWORDS:
        if phrase in text_clean:
            return "HIGH"

    # Rule 2: Check for MEDIUM risk keywords
    for phrase in MEDIUM_RISK_KEYWORDS:
        if phrase in text_clean:
            return "MEDIUM"

    # Rule 3: Check for LOW risk (positive) keywords
    for word in LOW_RISK_KEYWORDS:
        if word in text_clean:
            return "LOW"

    # Rule 4: ML prediction (for everything else)
    vec = vectorizer.transform([text_clean])
    prediction_encoded = model.predict(vec)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    return prediction

# =========================
# INTERACTIVE LOOP
# =========================
print("Mental Health Risk Detection Model Ready!")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower().strip() == "exit":
        print("Goodbye!")
        break

    if len(user_input.strip()) < 2:
        print("Bot: Please type a longer message so I can help you better.\n")
        continue

    result = predict_risk(user_input)

    if result == "HIGH":
        print("Bot: [HIGH RISK] - Please seek help immediately.")
        print("     National Suicide Prevention Lifeline: 988")
        print("     Crisis Text Line: Text HOME to 741741\n")
    elif result == "MEDIUM":
        print("Bot: [MODERATE RISK] - You might be going through a tough time.")
        print("     Would you like to talk more about how you're feeling?\n")
    else:
        print("Bot: [LOW RISK] - You seem to be doing okay. Keep going!\n")