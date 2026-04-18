import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle

# =========================
# STEP 1: LOAD DATA
# =========================

# Twitter (no header row in file)
twitter_train = pd.read_csv("data/twitter_training.csv", header=None)
twitter_val   = pd.read_csv("data/twitter_validation.csv", header=None)

twitter_train.columns = ["id", "game", "sentiment", "text"]
twitter_val.columns   = ["id", "game", "sentiment", "text"]

twitter = pd.concat([twitter_train, twitter_val], ignore_index=True)

# Emotion and suicide datasets
emotion = pd.read_csv("data/goemotions_combined.csv")
suicide = pd.read_csv("data/Suicide_Detection.csv")

print("Datasets loaded successfully!")

# =========================
# STEP 2: PROCESS EMOTION DATASET (multi-label → single label)
# =========================

# Priority: HIGH > MEDIUM > LOW
emotion_high   = ['grief', 'remorse']
emotion_medium = ['sadness', 'fear', 'anger', 'disappointment', 'disgust',
                  'nervousness', 'embarrassment', 'annoyance']
emotion_low    = ['joy', 'amusement', 'love', 'excitement', 'gratitude',
                  'admiration', 'approval', 'optimism', 'caring', 'pride',
                  'relief', 'desire', 'curiosity', 'surprise', 'realization',
                  'neutral', 'confusion']

emotion['emotion'] = None
emotion['risk']    = None

for label in emotion_high:
    if label in emotion.columns:
        mask = (emotion[label] == 1) & (emotion['emotion'].isna())
        emotion.loc[mask, 'emotion'] = label
        emotion.loc[mask, 'risk']    = 'HIGH'

for label in emotion_medium:
    if label in emotion.columns:
        mask = (emotion[label] == 1) & (emotion['emotion'].isna())
        emotion.loc[mask, 'emotion'] = label
        emotion.loc[mask, 'risk']    = 'MEDIUM'

for label in emotion_low:
    if label in emotion.columns:
        mask = (emotion[label] == 1) & (emotion['emotion'].isna())
        emotion.loc[mask, 'emotion'] = label
        emotion.loc[mask, 'risk']    = 'LOW'

emotion = emotion.dropna(subset=['emotion'])
emotion = emotion[['text', 'risk']]
print(f"Emotion dataset: {len(emotion)} rows after processing")

# =========================
# STEP 3: SELECT COLUMNS
# =========================

twitter = twitter[['text', 'sentiment']]
suicide = suicide[['text', 'class']]

# =========================
# STEP 4: CLEAN TEXT
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)        # remove URLs
    text = re.sub(r'@\w+', '', text)           # remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)      # keep hashtag text
    text = re.sub(r'[^a-zA-Z\' ]', '', text)   # keep apostrophes (don't, can't)
    text = re.sub(r'\s+', ' ', text)            # collapse whitespace
    return text.strip()

twitter['text'] = twitter['text'].apply(clean_text)
emotion['text'] = emotion['text'].apply(clean_text)
suicide['text'] = suicide['text'].apply(clean_text)
print("Text cleaned!")

# =========================
# STEP 5: REMOVE NULLS & SHORT TEXT
# =========================

twitter = twitter.dropna().drop_duplicates()
emotion = emotion.dropna().drop_duplicates()
suicide = suicide.dropna().drop_duplicates()

MIN_TEXT_LEN = 10
twitter = twitter[twitter['text'].str.len() >= MIN_TEXT_LEN]
emotion = emotion[emotion['text'].str.len() >= MIN_TEXT_LEN]
suicide = suicide[suicide['text'].str.len() >= MIN_TEXT_LEN]

# Remove rows that are just numbers or single repeated characters
twitter = twitter[twitter['text'].str.contains(r'[a-zA-Z]{3,}', regex=True)]
emotion = emotion[emotion['text'].str.contains(r'[a-zA-Z]{3,}', regex=True)]
suicide = suicide[suicide['text'].str.contains(r'[a-zA-Z]{3,}', regex=True)]

print(f"After cleaning: Twitter={len(twitter)}, Emotion={len(emotion)}, Suicide={len(suicide)}")

# =========================
# STEP 6: CREATE RISK LABELS
# =========================

def map_twitter_risk(sentiment):
    s = str(sentiment).strip().lower()
    if s == 'positive':
        return 'LOW'
    elif s == 'negative':
        return 'MEDIUM'
    else:  # neutral / irrelevant
        return 'LOW'

twitter['risk'] = twitter['sentiment'].apply(map_twitter_risk)

def map_suicide_risk(label):
    return "HIGH" if str(label).strip().lower() == "suicide" else "LOW"

suicide['risk'] = suicide['class'].apply(map_suicide_risk)
print("Risk labels created!")

# =========================
# STEP 7: UNIFY FORMAT & MERGE
# =========================

twitter = twitter[['text', 'risk']]
suicide = suicide[['text', 'risk']]

final_df = pd.concat([twitter, emotion, suicide], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Data merged!")

# =========================
# STEP 8: CHECK DISTRIBUTION
# =========================

print("\nBefore balancing:")
print(final_df['risk'].value_counts())

# =========================
# STEP 9: BALANCE DATA
# =========================

min_class_count = final_df['risk'].value_counts().min()
sample_size = min(50000, min_class_count)
print(f"\nBalancing to {sample_size} samples per class...")

low    = final_df[final_df['risk'] == "LOW"].sample(sample_size, random_state=42)
medium = final_df[final_df['risk'] == "MEDIUM"].sample(sample_size, random_state=42)
high   = final_df[final_df['risk'] == "HIGH"].sample(sample_size, random_state=42)

final_df = pd.concat([low, medium, high])
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter balancing:")
print(final_df['risk'].value_counts())

# =========================
# STEP 10: TRAIN/TEST SPLIT (stratified)
# =========================

X = final_df['text']
y = final_df['risk']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# =========================
# STEP 11: TF-IDF VECTORIZER
# Settings per requirements:
#   max_features=10000  → large vocabulary for good coverage
#   ngram_range=(1,2)   → unigrams + bigrams ("kill myself", "want die")
#   stop_words="english"→ removes common filler words
#   sublinear_tf=True   → log normalization reduces impact of very frequent terms
# =========================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"TF-IDF features: {X_train_vec.shape[1]}")

# =========================
# STEP 12: TRAIN MODEL
# Settings per requirements:
#   class_weight='balanced' → gives extra weight to HIGH risk class
#                             so the model is penalized more for missing
#                             HIGH risk messages (better recall on crisis text)
# =========================

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',   # Boosts HIGH risk recall — critical for safety
    C=1.0,
    solver='lbfgs',
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_vec, y_train)
print("Model trained!")

# =========================
# STEP 13: EVALUATE
# =========================

y_pred   = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}")

target_names = label_encoder.classes_
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=target_names, columns=target_names))

# HIGH risk specific metrics (most important for safety)
high_idx       = list(target_names).index("HIGH")
high_precision = cm[high_idx, high_idx] / cm[:, high_idx].sum() if cm[:, high_idx].sum() > 0 else 0
high_recall    = cm[high_idx, high_idx] / cm[high_idx, :].sum() if cm[high_idx, :].sum() > 0 else 0
f1             = (2 * high_precision * high_recall / (high_precision + high_recall)
                  if (high_precision + high_recall) > 0 else 0)

print(f"\n[!] HIGH Risk Detection:")
print(f"    Precision: {high_precision:.4f}")
print(f"    Recall:    {high_recall:.4f}")
print(f"    F1-Score:  {f1:.4f}")

# =========================
# STEP 14: SAVE MODEL
# =========================

pickle.dump(model,         open("model/risk_model.pkl",     "wb"))
pickle.dump(vectorizer,    open("model/vectorizer.pkl",     "wb"))
pickle.dump(label_encoder, open("model/label_encoder.pkl",  "wb"))

print("\nModel saved successfully!")
