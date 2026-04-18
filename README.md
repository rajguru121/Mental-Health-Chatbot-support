🧠 Mental Health Chatbot with Sentiment & Risk Detection

An AI-powered mental health chatbot that interacts with users in real time, analyzes emotional state using Natural Language Processing (NLP), and detects potential stress, anxiety, or high-risk situations.

The chatbot provides context-aware responses and can alert a guardian/parent via email when high-risk mental health indicators are detected.

📌 Project Overview

Many individuals hesitate to seek help for mental health concerns due to fear, stigma, or lack of awareness. This project aims to provide early emotional support using AI by:

Understanding user emotions from text input
Detecting negative sentiment and risk level
Providing supportive responses
Alerting guardians in high-risk situations
✨ Features
💬 Real-Time AI Chatbot
Interactive conversational interface
Generates responses dynamically (not predefined)
Context-aware replies
😊 Sentiment Analysis

Classifies emotional tone of user input:

Positive
Neutral
Negative
⚠️ Risk Detection System

Detects mental health risk levels:

Low Risk
Medium Risk
High Risk
📧 Guardian Alert System
Sends email notification when high-risk messages are detected
Helps ensure timely support
🔐 Secure Authentication System
User Signup and Login
Strong password validation
Show/Hide password option
Forgot password feature
Prevent duplicate usernames
📊 Data Preprocessing
Text cleaning
Stopword removal
Tokenization
Feature extraction
Converts raw dataset into meaningful input for ML model
🌐 Web Interface
Login Page
Signup Page
Chat Interface
Sentiment Output Display

⚙️ Tech Stack
Programming Language
Python
Machine Learning / NLP
Scikit-learn
NLTK / TextBlob
Pandas
NumPy
Backend Framework
Flask
Frontend
HTML
CSS
JavaScript
Database
CSV / SQLite
Email Service
SMTP (Gmail App Password)

🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/mental-health-chatbot.git
cd mental-health-chatbot
2️⃣ Install Required Libraries
pip install -r requirements.txt
3️⃣ Run the Application
python app.py

Open in browser:

http://127.0.0.1:5000
📊 Dataset Information

The dataset contains mental health related text samples used for training the sentiment and risk detection model.

Example format:

Text	Sentiment	Risk Level
I feel very sad today	Negative	Medium
I am feeling great	Positive	Low
I want to end my life	Negative	High
🔐 Password Requirements

User password must contain:

Minimum 8 characters
At least 1 uppercase letter
At least 1 number
At least 1 special symbol

Example:

Strong@123
⚠️ Risk Detection Workflow
User enters message in chatbot
NLP model analyzes text sentiment
Risk level is predicted
If risk level is HIGH:
Alert email is sent to guardian
📧 Email Configuration

To enable email alerts:

Enable 2-step verification in Gmail
Generate App Password
Add credentials in email configuration file

Example:

EMAIL = "your_email@gmail.com"
PASSWORD = "gmail_app_password"
🎯 Future Enhancements
Voice-based chatbot
Mobile app version
Multi-language support
Therapist recommendation system
Emotion detection using speech
Dashboard for monitoring risk levels
