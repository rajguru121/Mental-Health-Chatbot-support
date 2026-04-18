import os
import re
import pickle
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# ---------- App Initialization ----------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email configuration (for guardian alerts)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ============================================================
# LOAD ML MODEL
# ============================================================
model = pickle.load(open("model/risk_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
print("[OK] ML risk detection model loaded.")

# ---------- Database Models ----------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    guardian_email = db.Column(db.String(120), nullable=False)
    guardian_phone = db.Column(db.String(20), nullable=True, default='')

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), default='New Chat')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('sessions', lazy=True))

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=True)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('chats', lazy=True))
    session = db.relationship('ChatSession', backref=db.backref('messages', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ============================================================
# TEXT CLEANING & ML RISK PREDICTION
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z\' ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------------------------------
# Keyword Safety Layer — runs BEFORE ML model.
# Any match here forces HIGH risk immediately.
# This catches obvious crisis phrases the ML might miss.
# -------------------------------------------------------
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

LOW_RISK_KEYWORDS = [
    "happy", "great", "good", "amazing", "love",
    "wonderful", "fantastic", "blessed", "grateful",
    "excited", "joyful", "cheerful", "proud",
    "feeling good", "doing well", "doing great",
]

def predict_risk(text):
    """
    Risk prediction pipeline:
    1. Keyword safety layer (highest priority — catches explicit crisis phrases)
    2. ML model (handles everything else)
    """
    cleaned = clean_text(text)

    for phrase in HIGH_RISK_KEYWORDS:
        if phrase in cleaned:
            return "HIGH"

    for phrase in MEDIUM_RISK_KEYWORDS:
        if phrase in cleaned:
            return "MEDIUM"

    for word in LOW_RISK_KEYWORDS:
        if word in cleaned:
            return "LOW"

    vec = vectorizer.transform([cleaned])
    pred_enc = model.predict(vec)[0]
    return label_encoder.inverse_transform([pred_enc])[0]


# ============================================================
# CONVERSATIONAL RESPONSE ENGINE
# ============================================================

EXERCISES = [
    "🧘 Try this — sit comfy, breathe in 4 counts, hold 4, out for 6. Do it 5 times!",
    "🚶 How about a quick 10-min walk? Fresh air honestly hits different.",
    "💪 Stretch it out — reach up, touch your toes, roll those shoulders. Feels amazing.",
    "🌬️ Box breathing is magic: in 4s, hold 4s, out 4s, hold 4s. Try 4 rounds!",
    "🤸 Do 5 jumping jacks right now, then 5 deep breaths. Quick reset!",
    "🏃 Put on your fav song and just dance for 3 min. No rules, just move!",
    "🌿 Grounding time: name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 taste.",
    "🦋 Butterfly hug — cross arms on chest, tap shoulders one at a time. So calming.",
    "🧎 Try child's pose for a minute — kneel, sit back, arms forward. Just breathe.",
    "🧘‍♀️ Tense your toes 5 sec, release. Then calves, thighs, hands, shoulders, face. Ahh.",
]

NEGATION_WORDS = ["not", "no", "don't", "dont", "can't", "cant", "never",
                  "isn't", "isnt", "wasn't", "wasnt", "won't", "wont", "hardly", "barely"]
INTENSITY_WORDS = ["so", "very", "really", "extremely", "incredibly", "absolutely",
                   "totally", "completely", "super", "too"]

FEELING_MAP = {
    "stress":  ["stress", "overwhelm", "pressure", "too much", "can't handle", "burnt out",
                "burnout", "hectic", "overload", "suffocating", "drowning in work",
                "swamped", "deadline", "breaking point"],
    "anxiety": ["anxi", "panic", "nervous", "worry", "scared", "fear", "restless",
                "uneasy", "on edge", "can't relax", "racing thoughts", "overthink",
                "what if", "dread", "tense"],
    "sad":     ["sad", "crying", "tears", "depress", "unhappy", "miserable", "heartbroken",
                "grief", "mourning", "lost someone", "miss them", "hurting inside",
                "pain", "ache", "sorrow", "gloomy", "blue"],
    "lonely":  ["lonely", "alone", "nobody", "no friends", "isolated", "no one",
                "left out", "abandoned", "invisible", "forgotten", "disconnected",
                "don't belong", "unwanted"],
    "sleep":   ["sleep", "insomnia", "can't sleep", "tired", "exhausted", "fatigue",
                "restless night", "nightmares", "tossing and turning", "waking up",
                "no energy", "drained"],
    "angry":   ["angry", "furious", "mad", "frustrated", "annoyed", "irritated",
                "rage", "pissed", "hate", "sick of", "fed up", "done with", "can't stand"],
    "happy":   ["happy", "great", "amazing", "wonderful", "fantastic", "grateful",
                "excited", "blessed", "joyful", "proud", "cheerful", "thrilled",
                "pumped", "awesome", "brilliant"],
}

EMOTIONAL_PATTERNS = {
    "sad":     ["i feel like crying", "nothing makes me happy", "everything feels grey",
                "what's the point", "i don't care anymore", "nothing matters",
                "feel empty", "feel hollow", "feel numb", "lost interest", "don't enjoy"],
    "stress":  ["i have so much to do", "everything is piling up", "i can't keep up",
                "my head is spinning", "i need a break", "it's all too much"],
    "anxiety": ["my heart is racing", "i can't breathe", "i feel like something bad",
                "stomach is churning", "i keep thinking", "can't stop worrying"],
    "lonely":  ["i have no one to talk to", "no one understands me",
                "i feel so disconnected", "wish someone was here", "i miss having someone"],
    "sleep":   ["i was up all night", "i haven't slept", "my eyes are heavy",
                "i can barely stay awake", "i just want to sleep forever"],
    "angry":   ["i want to scream", "this makes me want to", "i can't believe they",
                "how could they", "it's not fair"],
    "happy":   ["best day ever", "i can't stop smiling", "everything feels right",
                "life is good", "i feel alive", "on top of the world"],
}

def _detect_mood(text):
    """Detect the emotional mood of a message."""
    if not text:
        return "general"
    t = text.lower().strip()
    has_negation = any(neg in t.split() for neg in NEGATION_WORDS)

    for mood, patterns in EMOTIONAL_PATTERNS.items():
        for p in patterns:
            if p in t:
                if mood == "happy" and has_negation:
                    return "sad"
                return mood

    scores = {}
    for mood, words in FEELING_MAP.items():
        score = 0
        for w in words:
            if w in t:
                score += 1
                for iw in INTENSITY_WORDS:
                    if f"{iw} {w}" in t or iw in t:
                        score += 0.5
                        break
        if score > 0:
            scores[mood] = score

    if scores:
        best_mood = max(scores, key=scores.get)
        if best_mood == "happy" and has_negation:
            return "sad"
        if best_mood in ("stress", "sad", "angry", "anxiety") and has_negation:
            return "happy"
        return best_mood

    return "general"

# --- Building blocks for responses ---
HIGH_OPENERS = [
    "Hey… I'm really glad you said something. That took guts. 💛",
    "I hear you. And I need you to know — you matter. Seriously.",
    "I'm not going to pretend I know exactly how you feel, but I'm here. 🤍",
    "Thank you for telling me this. Please don't go through this alone.",
    "I care about what happens to you. And right now, I'm worried. 💙",
    "Hey. I see you. This sounds really heavy, and I'm not going anywhere.",
]
HIGH_ACTIONS = [
    "Please call 988 or text HOME to 741741 — someone's there 24/7, just for this.",
    "There are people trained for exactly this — reach out: 988, or text HOME to 741741. 📞",
    "Can you do one thing for me? Call 988 or text HOME to 741741. They'll listen.",
    "Please talk to someone real — 988 or text HOME to 741741. You deserve that support.",
]
HIGH_CLOSERS = [
    "You're not alone in this. I promise. 🌟",
    "One step at a time. I'm right here with you.",
    "This feeling won't last forever, even if it feels like it right now. 💛",
    "You've already been brave enough to say it. That's the hardest part.",
]

# -------------------------------------------------------
# Context bridge phrases — used when the user's current message
# is on the same emotional topic as their PREVIOUS message.
# This makes the bot feel like it's following the conversation,
# not starting fresh every time.
# -------------------------------------------------------
CONTEXT_BRIDGES = {
    "stress": [
        "Still dealing with all that? Let's keep talking — ",
        "Sounds like the stress hasn't let up. ",
        "I hear you, this has been going on a while — ",
    ],
    "anxiety": [
        "Still feeling that anxious energy? ",
        "Sounds like your mind's still racing. ",
        "That anxiety is really persisting, huh? ",
    ],
    "sad": [
        "You're still going through a tough time — ",
        "I know this hasn't been easy for you. ",
        "Still feeling heavy? I'm right here. ",
    ],
    "lonely": [
        "Still feeling disconnected? That's really hard. ",
        "Loneliness like this doesn't just go away overnight — ",
        "You're still feeling that alone-ness. I hear you. ",
    ],
    "sleep": [
        "Sleep still not coming? That's exhausting. ",
        "Another rough night? I'm sorry. ",
        "Sounds like rest is still hard to find. ",
    ],
    "angry": [
        "Still fired up about it? Totally valid. ",
        "That frustration is clearly still there. ",
        "Sounds like this is still bothering you a lot. ",
    ],
}

STRESS_LINES = [
    "Ugh, stress is the worst. It literally sits in your body, you know?",
    "When everything feels like too much, it usually means you need a pause, not more effort.",
    "Sounds like you've got a lot on your plate right now. That's exhausting.",
    "I get it — when you're stressed, even small things feel huge.",
]
ANXIETY_LINES = [
    "Anxiety can make your brain feel like it's on a hamster wheel, huh?",
    "That racing feeling is so uncomfortable. Your body's just trying to protect you, but it's on overdrive.",
    "When anxiety hits, it can feel like the walls are closing in. But they're not. You're safe.",
    "I know that nervous energy is hard to shake. Let's try to slow things down a bit.",
]
SAD_LINES = [
    "It's okay to feel sad. You don't have to smile through everything. 💙",
    "Some days just feel heavy, and that's alright. You don't need a reason.",
    "I wish I could give you a hug through the screen right now. 🤗",
    "Sadness is just your heart telling you something matters. That's not weakness.",
]
LONELY_LINES = [
    "Loneliness can hurt just as much as anything physical. I see you. 💛",
    "Even when it feels like no one gets it, someone always does. Including me right now.",
    "Being alone and being lonely are different things. And what you're feeling is real.",
    "Connection is a basic human need — there's nothing wrong with wanting it.",
]
SLEEP_LINES = [
    "Not sleeping properly messes with everything, doesn't it? Your body needs rest so badly.",
    "When your brain won't shut off at night… that's the worst kind of tired.",
    "Sleep issues can make the whole world feel harder. You're not being dramatic.",
    "A tired mind makes everything feel 10x worse. Let's see if we can help with that.",
]
ANGRY_LINES = [
    "Anger usually means something important got crossed. Your feelings are valid.",
    "It's okay to be mad. Seriously. You don't have to be calm all the time.",
    "Sometimes you just need to let it out. No judgment here at all. 🔥",
    "Frustration is exhausting. But hey, at least you're feeling something — that matters.",
]
HAPPY_LINES = [
    "Okay wait, I love this energy!! 😊✨",
    "Yesss, that's what I like to hear! You deserve this!",
    "This genuinely made me smile. Tell me everything! 🥰",
    "Look at you, glowing! What happened? I want details! 🌟",
    "That's amazing!! Keep riding this wave, you've earned it! 🎉",
]
GENERAL_LINES = [
    "Hey, thanks for sharing that with me! How are you really feeling though? 😊",
    "I'm here! Tell me more — what's going on in your world today?",
    "Hmm, interesting! Is there something specific on your mind? I'm all ears. 🌱",
    "I appreciate you reaching out! What's the vibe today — good, meh, or rough?",
]

SUGGESTIONS = {
    "stress": [
        "Maybe step away from the screen for 5 min? Even just staring out a window helps.",
        "Have you tried writing down what's stressing you? Getting it out of your head onto paper is oddly freeing.",
        "A warm cup of tea and 5 minutes of silence can reset more than you'd think. ☕",
    ],
    "anxiety": [
        "Try putting your hand on your chest and feeling your heartbeat. It reminds you — you're here, you're safe.",
        "Sometimes splashing cold water on your face snaps you right out of a spiral. Weird but it works!",
        "Focus on what's in front of you right now. Not tomorrow, not later. Just this moment.",
    ],
    "sad": [
        "Could you do one tiny nice thing for yourself today? Even a favorite snack counts. 🍫",
        "Sometimes just going outside for 5 minutes and feeling the air helps more than you'd expect.",
        "If you can, talk to someone you trust. Even just texting a friend 'hey' can help.",
    ],
    "lonely": [
        "Could you reach out to one person today? Even a quick text. You might be surprised how they respond. 💛",
        "Sometimes just being around people helps — a café, a park, anywhere with life around you.",
        "Online communities can be amazing. There are people who get it, I promise.",
    ],
    "sleep": [
        "Try putting your phone down 30 min before bed and just... doing nothing. Boring, but it works!",
        "A warm shower before bed can trick your body into sleep mode. Worth a shot! 🚿",
        "Write down whatever's keeping you up. Once it's on paper, your brain can let go a little.",
    ],
    "angry": [
        "Physical movement is the best anger release — go for a fast walk, do push-ups, anything.",
        "Try writing down exactly what made you mad. Sometimes seeing it on paper takes the edge off.",
        "Walk away from whatever triggered it for 10 min. Give yourself permission to cool down.",
    ],
    "happy": [
        "Write down what's making you happy right now — when tough days come, you'll want to remember this! 📝",
        "Share this feeling with someone! Call a friend, text a family member. Spread the joy! 💛",
        "Use this energy to try something new today — you'll associate the good feeling with it forever!",
    ],
    "general": [
        "If you're up for it, tell me more about how you're feeling. I'm genuinely curious!",
        "Sometimes just checking in with yourself is a big step. How would you rate today, 1 to 10?",
        "Is there anything you've been wanting to talk about? No topic is too small. 🌻",
    ],
}

# Track recently used responses per user (avoids repetition)
_recent = {}

def _pick(pool, user_id, key):
    """Pick a random item from pool, avoiding recent repeats for this user."""
    k = f"{user_id}_{key}"
    used = _recent.get(k, [])
    avail = [i for i in range(len(pool)) if i not in used]
    if not avail:
        used = []
        avail = list(range(len(pool)))
    idx = random.choice(avail)
    used.append(idx)
    _recent[k] = used[-max(len(pool) // 2, 2):]
    return pool[idx]

def generate_response(user_message, risk_level, user_id, prev_message=None):
    """
    Build a dynamic, empathetic reply.

    Parameters:
        user_message  - what the user just typed
        risk_level    - HIGH / MEDIUM / LOW from predict_risk()
        user_id       - used to track variety and avoid repetition
        prev_message  - the user's PREVIOUS message in this session (context)
    """
    mood = _detect_mood(user_message)
    prev_mood = _detect_mood(prev_message) if prev_message else None

    # HIGH risk always gets crisis response first
    if risk_level == "HIGH":
        opener   = _pick(HIGH_OPENERS, user_id, "ho")
        action   = _pick(HIGH_ACTIONS, user_id, "ha")
        closer   = _pick(HIGH_CLOSERS, user_id, "hc")
        exercise = _pick(EXERCISES, user_id, "ex")
        return {"text": f"{opener}\n{action}\n{closer}", "exercise": exercise}

    # Handle casual short messages (greetings, thanks, etc.)
    casual = _casual_reply(user_message.lower().strip())
    if casual:
        return {"text": casual, "exercise": None}

    # Build topic-aware acknowledgment
    ack = _build_acknowledgment(user_message, mood)

    # If this message is on the SAME topic as the previous one,
    # prefix with a context bridge to make it feel like a real ongoing chat
    if prev_mood and prev_mood == mood and mood in CONTEXT_BRIDGES:
        bridge = random.choice(CONTEXT_BRIDGES[mood])
        ack = bridge + ack[0].lower() + ack[1:]

    # Emotional messages → acknowledgment + practical suggestion + optional exercise
    if risk_level == "MEDIUM" or mood in ("stress", "anxiety", "sad", "lonely", "sleep", "angry"):
        sug      = _pick(SUGGESTIONS.get(mood, SUGGESTIONS["general"]), user_id, f"s_{mood}")
        exercise = _pick(EXERCISES, user_id, "ex") if mood != "angry" else None
        return {"text": f"{ack}\n{sug}", "exercise": exercise}

    # Happy / general → keep it short
    return {"text": ack, "exercise": None}


def _casual_reply(msg):
    """Handle greetings, yes/no, thanks, bye — like a real conversation."""
    words = msg.split()
    clean = msg.strip().rstrip("!?.").strip()

    greetings = ["hi", "hii", "hiii", "hey", "hello", "heya", "heyy", "yo",
                 "sup", "wassup", "whatsup", "hola", "howdy", "namaste"]
    if clean in greetings or (len(words) <= 2 and words[0] in greetings):
        return random.choice([
            "Heyyy! 😊 How's it going?",
            "Hello! Good to see you here 💛 What's up?",
            "Hey hey! How are you doing today?",
            "Hii! 👋 How's your day been so far?",
            "Heya! 😊 What's on your mind today?",
            "Yo! What's good? Tell me about your day!",
        ])

    how_are_you = ["how are you", "how r u", "how r you", "how are u",
                   "how you doing", "how's it going", "hows it going",
                   "what's up", "whats up", "what up"]
    if any(h in msg for h in how_are_you):
        return random.choice([
            "I'm doing great, thanks for asking! 😊 But this is about YOU — how are you really feeling?",
            "Aww, I'm good! But more importantly, how are YOU doing? 💛",
            "I'm always here and ready! How about you — what's going on?",
        ])

    yes_words = ["yes", "yeah", "yep", "yup", "ya", "yea", "sure", "definitely",
                 "absolutely", "of course", "mhm", "mmhm", "right"]
    if clean in yes_words:
        return random.choice([
            "Okay, tell me more! I'm all ears 😊",
            "Got it! So what's going on with that?",
            "Alright, I'm listening — go ahead! 💛",
            "Okay! Take your time and share whatever feels right.",
        ])

    no_words = ["no", "nah", "nope", "not really", "no thanks", "naa", "naah"]
    if clean in no_words:
        return random.choice([
            "That's totally fine! Is there something else you'd like to talk about? 😊",
            "No worries at all! What would you like to chat about instead?",
            "Okay, no pressure! I'm here whenever you want to talk. 💛",
            "All good! What's actually on your mind today?",
        ])

    thanks = ["thanks", "thank you", "thx", "ty", "thank u", "thankyou", "tysm"]
    if any(t in clean for t in thanks):
        return random.choice([
            "Anytime! That's what I'm here for 😊💛",
            "You're welcome! I'm always here if you need me 🌻",
            "Of course! Don't hesitate to come back anytime 💛",
            "No need to thank me! I'm glad I could help 😊",
        ])

    bye_words = ["bye", "goodbye", "good night", "goodnight", "gn", "see you",
                 "gotta go", "ttyl", "talk later", "cya", "night"]
    if any(b in clean for b in bye_words):
        return random.choice([
            "Take care of yourself! Come back anytime you need to talk 💛",
            "Bye! Remember, I'm always here for you 😊🌻",
            "Goodnight! Rest well — you deserve it 🌙",
            "See you soon! Take it easy, okay? 💛",
        ])

    ok_words = ["ok", "okay", "k", "kk", "alright", "fine", "hmm", "hm", "ohh", "oh", "ooh", "ohk"]
    if clean in ok_words:
        return random.choice([
            "Everything okay? You seem quiet today. What's on your mind? 😊",
            "I'm here if you want to share anything! No pressure though 💛",
            "Wanna tell me about your day? I'm curious!",
            "Something on your mind? I've got all the time in the world for you 🌱",
        ])

    laugh = ["haha", "lol", "lmao", "hehe", "rofl", "😂", "🤣", "😄"]
    if any(l in clean for l in laugh):
        return random.choice([
            "Haha glad that made you smile! 😄 What else is going on?",
            "Love to hear you laughing! 😊 How are you feeling overall?",
            "That's the energy I like! 😂 What's making you laugh?",
        ])

    fine_phrases = ["i'm fine", "im fine", "i am fine", "i'm okay", "im okay",
                    "i am okay", "i'm good", "im good", "i am good", "i'm alright", "im alright"]
    if any(f in msg for f in fine_phrases):
        return random.choice([
            "Are you really fine, or just saying that? 😊 No judgment either way!",
            "Glad to hear it! Anything fun happening today?",
            "That's good! But if anything changes, I'm right here 💛",
            "Okay! But real talk — how are you *actually* doing? 😊",
        ])

    idk = ["i don't know", "idk", "i dont know", "not sure", "dunno", "no idea"]
    if any(d in msg for d in idk):
        return random.choice([
            "That's okay! Sometimes we don't have the words. Did anything happen today that stuck with you?",
            "No worries, we can figure it out together. How has your day been so far?",
            "Totally valid. On a scale of 1-10, how's your mood right now?",
        ])

    return None


def _build_acknowledgment(msg, mood):
    """Build a reply that feels like it actually heard what the user said."""
    about_work         = any(w in msg for w in ["work", "job", "boss", "office", "deadline", "meeting", "coworker", "colleague"])
    about_school       = any(w in msg for w in ["school", "exam", "study", "college", "class", "homework", "grades", "teacher", "professor"])
    about_family       = any(w in msg for w in ["family", "mom", "dad", "parent", "brother", "sister", "mother", "father"])
    about_friend       = any(w in msg for w in ["friend", "bestie", "buddy", "pal", "mate"])
    about_relationship = any(w in msg for w in ["boyfriend", "girlfriend", "partner", "ex", "breakup", "relationship", "dating", "crush"])
    about_health       = any(w in msg for w in ["sick", "ill", "hospital", "doctor", "pain", "headache", "health"])
    about_money        = any(w in msg for w in ["money", "broke", "debt", "rent", "bills", "financial", "salary"])
    about_future       = any(w in msg for w in ["future", "career", "life", "purpose", "direction", "goal", "plan"])

    if about_work:
        replies = [
            "Work stuff can be so draining. What's going on — is it the workload or the people?",
            "Ugh, work stress is real. Are you getting any time for yourself outside of it?",
            "That sounds rough. Work shouldn't take over your whole life. What happened?",
            "I hear you on the work thing. Is it a specific situation or just... everything piling up?",
        ]
    elif about_school:
        replies = [
            "School pressure is no joke. Are exams coming up or is it something else?",
            "I totally get it — academic stress hits different. What's weighing on you?",
            "Been there. School can feel like it's your whole world sometimes. What's going on?",
            "That sounds stressful. Are you putting too much pressure on yourself? Be honest with me.",
        ]
    elif about_family:
        replies = [
            "Family stuff is complicated, isn't it? They can push buttons nobody else can.",
            "I get it. Family can be the best and hardest thing at the same time. What happened?",
            "That sounds like a lot to deal with. Family issues hit close to home — literally.",
            "I hear you. Sometimes the people closest to us cause the most stress. Wanna talk about it?",
        ]
    elif about_relationship:
        replies = [
            "Relationship stuff can really mess with your head. What's going on?",
            "That sounds tough. Love stuff is never simple, is it? Tell me more.",
            "I'm sorry you're going through that. Relationship pain is a special kind of hard.",
            "Hearts are complicated things. What happened — if you want to share?",
        ]
    elif about_friend:
        replies = [
            "Friend drama? That's always tough because you care about them. What's up?",
            "Friendships matter so much. What's going on with you and them?",
            "I get it — when things are off with a friend, it affects everything. Tell me more?",
        ]
    elif about_money:
        replies = [
            "Money stress is the worst because it touches everything. What's going on?",
            "Financial stuff can be so overwhelming. Are you dealing with something specific?",
            "I hear you. Money worries keep you up at night. Let's talk through it.",
        ]
    elif about_health:
        replies = [
            "I'm sorry to hear that. Health stuff is scary. Are you taking care of yourself?",
            "That sounds really tough. Your health has to come first. How are you holding up?",
            "I hope you're okay. Health worries can make everything feel heavier.",
        ]
    elif about_future:
        replies = [
            "The future can feel overwhelming when you don't have all the answers yet. That's normal though.",
            "Not knowing what's next is scary, but honestly? Nobody has it figured out. What's on your mind?",
            "I get that feeling of uncertainty. What specifically is worrying you about the future?",
        ]
    elif mood == "stress":
        replies = [
            "That sounds really overwhelming. What's the biggest thing stressing you right now?",
            "I can feel the stress in your words. Have you been able to take any breaks lately?",
            "Ugh, that's a lot. When did it start feeling this heavy?",
        ]
    elif mood == "anxiety":
        replies = [
            "Anxiety is the worst — your mind just won't stop, right? What's it focused on?",
            "I hear you. That anxious feeling is so exhausting. Is it about something specific?",
            "That racing feeling in your chest... I know it. What triggered it this time?",
        ]
    elif mood == "sad":
        replies = [
            "I'm sorry you're feeling this way. Do you know what's making you sad, or is it just... everything?",
            "It's okay to feel sad. You don't have to explain it. But I'm here if you want to. 💙",
            "That sucks. Honestly. Some days just feel heavy. How long have you been feeling like this?",
        ]
    elif mood == "lonely":
        replies = [
            "Loneliness is such a real ache. I'm here with you right now, at least. 💛",
            "I hear you. Feeling disconnected from people is painful. When did it start?",
            "You're not alone right now — I'm here. What would make you feel more connected?",
        ]
    elif mood == "sleep":
        replies = [
            "Not sleeping is torture. Is your mind racing at night, or is it something else?",
            "I feel you — when you can't sleep, everything gets harder. How long has this been going on?",
            "Sleep issues mess with everything. What's keeping you up — thoughts, stress, or just can't switch off?",
        ]
    elif mood == "angry":
        replies = [
            "You sound frustrated, and honestly? That's valid. What set it off?",
            "I can tell you're fired up. Want to vent? I'm all ears, no judgment. 🔥",
            "Anger means something crossed a line. What happened?",
        ]
    elif mood == "happy":
        replies = [
            "Wait, I love this! What's making you feel so good? Tell me everything! 😊",
            "Yesss! This energy is contagious! What happened? I want the details! 🌟",
            "Okay you're literally glowing right now. Spill — what's the good news? 🎉",
            "This makes me SO happy to hear! You deserve every bit of this. What brought it on?",
        ]
    else:
        if len(msg) < 15:
            replies = [
                "Tell me more! I want to understand what you're going through. 😊",
                "I'm listening! Can you share a bit more about how you're feeling?",
                "Hey! What's behind that? I'm curious to hear more from you. 🌱",
            ]
        elif "?" in msg:
            replies = [
                "Great question! Let me think about that with you. What made you ask?",
                "Hmm, I love that you're thinking about this. What's your gut telling you?",
                "That's a really thoughtful thing to wonder about. What do you think? 💭",
            ]
        else:
            replies = [
                "I hear you. Thanks for sharing that with me. What else is on your mind?",
                "That's interesting — tell me more? I feel like there's more to this. 😊",
                "I appreciate you opening up! What would make today better for you?",
                "Got it! How does talking about it make you feel? Sometimes just saying it helps. 🌻",
            ]

    return random.choice(replies)

# ============================================================
# GUARDIAN EMAIL ALERT
# ============================================================
# ============================================================
# GUARDIAN EMAIL ALERT
# ============================================================
def send_guardian_alert(username, guardian_email, risk_message):
    """
    Sends an urgent alert email to the guardian in a background thread.
    Credentials are hardcoded — do NOT change them.
    """
    import smtplib
    import threading
    from email.mime.text import MIMEText

    GMAIL       = "mindcarealert1@gmail.com"   # ← fixed sender, never change
    APP_PASSWORD = "mbpbssuaydclptwn"          # ← Gmail App Password, never change

    def _send():
        try:
            body = (
                f"Dear Guardian,\n\n"
                f"🚨 CRITICAL SAFETY ALERT — Immediate Action Required\n\n"
                f"MindCare AI has detected that your ward, {username}, "
                f"may be experiencing a mental health crisis and could be in danger.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚠️  Detected Message:\n"
                f"  \"{risk_message}\"\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Please reach out to {username} IMMEDIATELY. "
                f"Your presence and support can save a life.\n\n"
                f"If you believe they are in immediate danger, "
                f"please contact emergency services right away.\n\n"
                f"📞 Crisis Helplines:\n"
                f"  • 988 Suicide & Crisis Lifeline (call or text 988)\n"
                f"  • Crisis Text Line: text HOME to 741741\n"
                f"  • iCall India: 9152987821\n\n"
                f"Do NOT ignore this alert.\n\n"
                f"— MindCare AI Safety System\n"
                f"This is an automated alert. Please act immediately."
            )

            msg = MIMEText(body)
            msg["Subject"] = "🚨 URGENT — MindCare Safety Alert"
            msg["From"]    = f"MindCare AI <{GMAIL}>"
            msg["To"]      = guardian_email

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(GMAIL, APP_PASSWORD)
                smtp.send_message(msg)

            print(f"[ALERT] Guardian email sent to {guardian_email} for user '{username}'")

        except Exception as e:
            print(f"[ALERT FAILED] Could not send guardian email: {e}")

    threading.Thread(target=_send, daemon=True).start()

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return render_template('landing.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username       = request.form.get('username')
        password       = request.form.get('password')
        confirm        = request.form.get('confirm_password')
        guardian_email = request.form.get('guardian_email')
        guardian_phone = request.form.get('guardian_phone', '')

        if password != confirm:
            flash('Passwords do not match', 'danger')
            return render_template('signup.html')

        if User.query.filter_by(username=username).first():
            flash('Username already taken', 'danger')
            return render_template('signup.html')

        # werkzeug.security hashes the password — plain text is never stored
        hashed_pw = generate_password_hash(password)
        new_user = User(
            username=username,
            password_hash=hashed_pw,
            guardian_email=guardian_email,
            guardian_phone=guardian_phone
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if not user:
            flash('User does not exist', 'danger')
        elif not check_password_hash(user.password_hash, password):
            flash('Incorrect password', 'danger')
        else:
            login_user(user)
            return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        flash('If an account exists with that information, reset instructions have been sent.', 'success')
    return render_template('forgot_password.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/chat')
@app.route('/chat/<int:session_id>')
@login_required
def chat(session_id=None):
    if session_id is None:
        active = ChatSession.query.filter_by(user_id=current_user.id)\
            .order_by(ChatSession.created_at.desc()).first()
        if active is None:
            active = ChatSession(user_id=current_user.id, title='New Chat')
            db.session.add(active)
            db.session.commit()
        session_id = active.id
    return render_template('index.html',
        username=current_user.username,
        session_id=session_id,
        guardian_email=current_user.guardian_email,
        guardian_phone=current_user.guardian_phone)

@app.route('/new_session', methods=['POST'])
@login_required
def new_session():
    s = ChatSession(user_id=current_user.id, title='New Chat')
    db.session.add(s)
    db.session.commit()
    return jsonify({'session_id': s.id})

@app.route('/sessions')
@login_required
def get_sessions():
    sessions = ChatSession.query.filter_by(user_id=current_user.id)\
        .order_by(ChatSession.created_at.desc()).all()
    result = []
    for s in sessions:
        msg_count = ChatHistory.query.filter_by(session_id=s.id).count()
        first_msg = ChatHistory.query.filter_by(session_id=s.id)\
            .order_by(ChatHistory.timestamp).first()
        title = s.title
        if first_msg and title == 'New Chat':
            title = first_msg.message[:40] + ('...' if len(first_msg.message) > 40 else '')
        result.append({
            'id': s.id,
            'title': title,
            'msg_count': msg_count,
            'created_at': s.created_at.strftime('%b %d, %Y')
        })
    return jsonify(result)

@app.route('/delete_session/<int:session_id>', methods=['DELETE'])
@login_required
def delete_session(session_id):
    s = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if s:
        ChatHistory.query.filter_by(session_id=session_id).delete()
        db.session.delete(s)
        db.session.commit()
    return jsonify({'ok': True})

@app.route('/dashboard_data')
@login_required
def dashboard_data():
    total    = ChatHistory.query.filter_by(user_id=current_user.id).count()
    high     = ChatHistory.query.filter_by(user_id=current_user.id, risk_level='HIGH').count()
    medium   = ChatHistory.query.filter_by(user_id=current_user.id, risk_level='MEDIUM').count()
    low      = ChatHistory.query.filter_by(user_id=current_user.id, risk_level='LOW').count()
    sessions = ChatSession.query.filter_by(user_id=current_user.id).count()
    return jsonify({
        'total_messages': total,
        'high_risk': high,
        'medium_risk': medium,
        'low_risk': low,
        'total_sessions': sessions,
        'username': current_user.username,
        'guardian_email': current_user.guardian_email
    })

@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    sid = data.get('session_id')

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    # Ensure session exists
    if sid:
        sess = ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()
        if not sess:
            sid = None
    if not sid:
        sess = ChatSession(user_id=current_user.id, title='New Chat')
        db.session.add(sess)
        db.session.commit()
        sid = sess.id

    # -------------------------------------------------------
    # Fetch previous message for conversational context.
    # Passing this to generate_response lets the bot detect
    # if the user is continuing the same emotional topic
    # and respond with a context bridge phrase accordingly.
    # -------------------------------------------------------
    prev_entry = ChatHistory.query.filter_by(
        user_id=current_user.id,
        session_id=sid
    ).order_by(ChatHistory.timestamp.desc()).first()
    prev_message = prev_entry.message if prev_entry else None

    # Keyword safety layer runs first, then ML model
    risk = predict_risk(user_message)

    # Generate context-aware, empathetic response
    response_data = generate_response(user_message, risk, current_user.id, prev_message)

    full_reply = response_data["text"]
    if response_data.get("exercise"):
        full_reply += "\n\n" + response_data["exercise"]

    # Save chat with timestamp
    chat_entry = ChatHistory(
        user_id=current_user.id,
        session_id=sid,
        message=user_message,
        response=full_reply,
        risk_level=risk,
        timestamp=datetime.utcnow()
    )
    db.session.add(chat_entry)
    db.session.commit()

    # Guardian alert fires for every HIGH risk message — no exceptions
    if risk == "HIGH":
        print(f"[DEBUG] HIGH risk detected — sending alert to {current_user.guardian_email}")
        send_guardian_alert(
        current_user.username,
        current_user.guardian_email,
        user_message
         )

    return jsonify({
        'response': response_data["text"],
        'exercise': response_data.get("exercise"),
        'risk': risk,
        'session_id': sid,
        'timestamp': chat_entry.timestamp.strftime('%H:%M')
    })

@app.route('/history')
@app.route('/history/<int:session_id>')
@login_required
def history(session_id=None):
    if session_id:
        chats = ChatHistory.query.filter_by(user_id=current_user.id, session_id=session_id)\
            .order_by(ChatHistory.timestamp).all()
    else:
        chats = ChatHistory.query.filter_by(user_id=current_user.id)\
            .order_by(ChatHistory.timestamp).all()
    return jsonify([{
        'message': c.message,
        'response': c.response,
        'risk_level': c.risk_level,
        'timestamp': c.timestamp.strftime('%H:%M')
    } for c in chats])

# ---------- Run ----------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
