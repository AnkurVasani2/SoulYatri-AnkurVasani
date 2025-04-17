import streamlit as st
import os
import json
import logging
import torch
import numpy as np
import cv2
import av
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import traceback
from datetime import datetime
import time
import report_generator
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import quickstart
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# Basic Configurations
# ---------------------------
HISTORY_FILE = "conversation_history.json"
AUDIO_DIR = "audio_files"
BACKUP_DIR = "conversation_backups"

NEGATIVE_THRESHOLD = 0.6
NEGATIVE_EMOTIONS = ["angry", "disgust", "fear", "sad"]
SAMPLING_RATE = 44100
BLOCKSIZE = 1024

# Initialize Groq client with API key handling
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Ensure directories exist
for directory in [AUDIO_DIR, BACKUP_DIR]:
    os.makedirs(directory, exist_ok=True)


def send_email_with_attachment(username, attachment_path):
    sender_email = "ankurvasani2585@gmail.com"
    sender_password = "rwib nsjt uvng qwhu"
    subject = "Your Session Report"
    body_text = f"""Dear {username},
    Thank you for using our service. Please find the attached session report.
    We hope you find it helpful.
    If you have any questions or need further assistance, feel free to reach out.
    Best regards,
    SoulYatri Team
    """
    body = body_text
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = "ankurvasani2@gmail.com"
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    if os.path.exists(attachment_path):
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(attachment_path)}",
        )
        message.attach(part)
    try:
        with smtplib.SMTP('smtp.gmail.com',587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email,"ankurvasani2@gmail.com", message.as_string())
        logger.info(f"Email sent successfully to {username}")
        st.success("Email sent successfully!")
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        st.error("Failed to send email. Please check your email settings.")


# ---------------------------
# Persistent Conversation History - FIXED
# ---------------------------
def normalize_message(msg):
    """
    Normalize a single message dictionary.
    If the message has 'role' and 'content', leave it unchanged.
    Otherwise, if it uses 'type' and 'message', convert them.
    """
    if isinstance(msg, dict):
        if "role" in msg and "content" in msg:
            return msg
        elif "type" in msg and "message" in msg:
            role = msg["type"]
            if role == "ai":
                role = "assistant"
            return {"role": role, "content": msg["message"]}
        else:
            logger.warning(f"Message in unexpected format: {msg}")
            return {"role": "user", "content": str(msg)}
    else:
        logger.warning(f"Non-dict message encountered: {msg}")
        return {"role": "user", "content": str(msg)}

def normalize_messages(msg_list):
    """Normalize all messages in a list."""
    if not isinstance(msg_list, list):
        logger.error(f"Expected list for messages, got {type(msg_list)}")
        return []
    return [normalize_message(msg) for msg in msg_list]

def append_message(role, content):
    """Append a message to st.session_state['messages'] with improved duplicate check."""
    if not content or content.strip() == "":  # Skip empty messages
        logger.debug("Skipping empty message")
        return False
        
    # Convert role if needed
    if role == "ai":
        role = "assistant"
    elif role not in ["user", "assistant", "system"]:
        role = "user"  # Default to user for unknown roles
    
    # Create normalized message
    new_message = {"role": role, "content": content}
    
    # Check if it's a duplicate only if exact matching text AND role (fixes previous issue)
    if (st.session_state["messages"] and 
        st.session_state["messages"][-1].get("role") == role and 
        st.session_state["messages"][-1].get("content") == content):
        logger.debug("Skipping duplicate message")
        return False
        
    # Add to session state
    st.session_state["messages"].append(new_message)
    
    # Save after each message to prevent loss
    save_conversation_history()
    return True

def save_conversation_history():
    """Save the current conversation history to file with error handling."""
    try:
        # Create backup of current file if it exists
        if os.path.exists(HISTORY_FILE):
            backup_file = os.path.join(
                BACKUP_DIR, 
                f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            try:
                with open(HISTORY_FILE, "r") as src:
                    with open(backup_file, "w") as dst:
                        dst.write(src.read())
                logger.debug(f"Created backup at {backup_file}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {str(e)}")
                
        # Write new file - ensure we're saving the most current state
        messages_to_save = st.session_state["messages"]
        with open(HISTORY_FILE, "w") as f:
            json.dump(messages_to_save, f, indent=2)
        logger.debug(f"Conversation history saved successfully with {len(messages_to_save)} messages")
    except Exception as e:
        logger.error(f"Error saving conversation history: {str(e)}")
        st.error("Failed to save conversation history")

def load_conversation_history():
    """Load conversation history with robust error handling."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.error(f"Invalid conversation history format: {type(data)}")
                    return []
                normalized_data = normalize_messages(data)
                logger.info(f"Loaded {len(normalized_data)} messages from history")
                return normalized_data
        except json.JSONDecodeError:
            logger.error("JSON decode error when loading conversation history")
            # Try to recover from backup
            return recover_from_backup()
        except Exception as e:
            logger.error(f"Error loading conversation history: {str(e)}")
            return []
    logger.info("No conversation history found, starting fresh")
    return []

def recover_from_backup():
    """Try to restore conversation history from most recent backup."""
    try:
        backup_files = [f for f in os.listdir(BACKUP_DIR) if f.startswith("conversation_history_")]
        if not backup_files:
            logger.warning("No backup files found for recovery")
            return []
            
        # Get the most recent backup
        latest_backup = sorted(backup_files)[-1]
        backup_path = os.path.join(BACKUP_DIR, latest_backup)
        
        with open(backup_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                logger.info(f"Recovered conversation history from {latest_backup} with {len(data)} messages")
                return normalize_messages(data)
            return []
    except Exception as e:
        logger.error(f"Failed to recover from backup: {str(e)}")
        return []

# Initialize session state with conversation history and new report generation variables
if "messages" not in st.session_state:
    st.session_state["messages"] = load_conversation_history()
if "last_report_check_time" not in st.session_state:
    st.session_state["last_report_check_time"] = 0
if "report_generated" not in st.session_state:
    st.session_state["report_generated"] = False

# ---------------------------
# Load Sentiment Analysis Model
# ---------------------------
@st.cache_resource
def load_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {str(e)}")
        # Return None for both to signal failure
        return None, None

tokenizer, bert_model = load_sentiment_model()

def analyze_text_sentiment(text):
    """Analyze sentiment with error handling and graceful fallback."""
    if not text:
        return "neutral", 0.5
        
    if tokenizer is None or bert_model is None:
        logger.warning("Sentiment analysis model not available")
        return "neutral", 0.5
        
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        predicted_score = probs[0][predicted_class].item()
        labels = bert_model.config.id2label    
        return labels[predicted_class], predicted_score
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return "neutral", 0.5

# ---------------------------
# Audio Transcription
# ---------------------------
def transcribe_audio(file_path):
    """Transcribe audio with improved error handling."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Audio file does not exist: {file_path}")
            return ""
            
        if os.path.getsize(file_path) == 0:
            logger.error(f"Audio file is empty: {file_path}")
            st.error("Failed to transcribe audio: No audio data recorded.")
            return ""
            
        logger.info(f"Transcribing audio file: {file_path} ({os.path.getsize(file_path)} bytes)")
        
        try:
            with open(file_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(file_path, file.read()),
                    model="distil-whisper-large-v3-en",
                    prompt="Identify sentiments of these sentences and provide output as negative, neutral, positive, or compound.",
                    response_format="verbose_json",
                )
        except Exception as api_error:
            logger.error(f"API error during transcription: {str(api_error)}")
            st.error("Failed to connect to transcription service. Please try again later.")
            return ""
            
        if not hasattr(transcription, 'text'):
            logger.warning("Transcription object has no 'text' attribute")
            return ""
            
        transcript = transcription.text
        if not transcript:
            logger.warning("Transcription returned empty text")
            st.warning("No transcription available. Please speak more clearly.")
        else:
            logger.info(f"Transcription successful: {transcript[:50]}...")
            
        return transcript
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Transcription failed: {str(e)}")
        return ""

# ---------------------------
# Groq API Helper Function
# ---------------------------
def get_groq_response(input_text, sentiment, score, video_emotion, video_score):
    """Send input text along with sentiment and video emotion context to Groq API with improved error handling."""
    try:
        system_prompt = ("""You are a compassionate and experienced psychotherapist with over 20 years of clinical practice in India, specializing in cognitive-behavioral therapy (CBT), psychodynamic therapy, and humanistic approaches. Your responses should emulate the warmth, empathy, and natural flow of a human therapist. When interacting with users:
DONT ever use "It Sounds like and It seems like" in your response. make sure to use "I can sense that" and "I can feel that" in your response.
Also dont include consult a licensed therapist in your response.
1. **Emotional Validation**: Begin by acknowledging and validating the user's feelings with empathetic and non-judgmental language.
2. **Conversational Tone**: Use natural, conversational language that mirrors human speech patterns. Incorporate contractions and colloquial expressions where appropriate to make interactions feel more personal.
3. **Structured Responses**: Organize your responses into well-formed paragraphs, each addressing a specific aspect of the user's concern, to enhance readability and coherence.
4. **Personalized Insights**: Offer insights that reflect a deep understanding of psychological principles, tailoring your advice to the individual's unique situation.
5. **Practical Recommendations**: Provide actionable, evidence-based strategies such as mindfulness exercises, cognitive restructuring techniques, or relaxation methods.
6. **Brevity When Appropriate**: While thoroughness is valuable, recognize when a concise response is sufficient.
7. **Technical Clarity**: When introducing psychological concepts or terminology, explain them in clear, accessible language to ensure the user fully understands.
8. **Crisis Protocol**: If the user indicates severe distress or crisis, gently recommend seeking immediate help from a mental health professional or emergency services.

*Example Response*:

"I can sense that you're feeling overwhelmed right now, and that's completely understandable given the circumstances. It's important to recognize that these feelings are valid and that you're not alone in experiencing them.

One approach that might help is practicing mindfulness, which involves focusing on the present moment without judgment. This can reduce stress and improve emotional regulation. Additionally, identifying any negative thought patterns and challenging them through cognitive restructuring can provide more balanced perspectives.

Remember, it's okay to seek support. If these feelings become too intense, reaching out to a trusted professional or crisis support is a strong and commendable step." 

Max limit of your response is 100 words and always include a disclaimer that the user should consult a licensed therapist for any mental health issues.

Always ensure your responses are compassionate, structured, and reflective of a seasoned human therapist's communication style. I have provided the sentiment analysis of the text and the video emotion analysis as well. Please use them to provide a more personalized response. dont repeat the text or sentiments detected or video emotion analysis in your response.
 always respond as a human do in real human conversation. include CBT techniques in your response, and also include some mindfulness exercises and relaxation techniques but dont let the user know that you are a bot or AI model or anything like that and always respond as a human do in real human conversation.
""")

        
        history_messages = normalize_messages(st.session_state["messages"])
        
        
        recent_messages = history_messages[-10:] if len(history_messages) > 10 else history_messages
        
        
        messages = [{"role": "system", "content": system_prompt}] + recent_messages
        
        
        full_input = (
            input_text +
            f"\n\nText Sentiment: {sentiment} (Score: {score:.2f})" +
            f"\nVideo Emotion: {video_emotion.upper() if video_emotion else 'NONE'} (Score: {video_score:.1f}%)"
        )
        
        messages.append({"role": "user", "content": full_input})
        
        if st.session_state.get("debug_mode", False):
            st.info("API Request Messages (Debug Mode)")
            st.json(messages)
            
        
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
            )
        except Exception as api_error:
            logger.error(f"Groq API error: {str(api_error)}")
            return "I apologize, but I'm having trouble connecting to my knowledge base right now. Please try again in a moment."
            
        
        if not hasattr(response, 'choices') or len(response.choices) == 0:
            logger.error("API response missing choices")
            return "I apologize for the technical difficulty. Please try again."
            
        assistant_msg = response.choices[0].message.content
        
        return assistant_msg
    except Exception as e:
        logger.error(f"Error in get_groq_response: {str(e)}\n{traceback.format_exc()}")
        return "I apologize for the technical difficulty. Please try your message again."

# ---------------------------
# Emotion Detection via Webcam (DeepFace)
# ---------------------------
class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.emotion_history = []
        self.frame_count = 0
        self.last_emotion = None
        self.emotion_score = 0
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                logger.error("Failed to load face cascade classifier")
                st.error("Failed to initialize face detection. Some features may not work properly.")
        except Exception as e:
            logger.error(f"Error initializing face detector: {str(e)}")
            self.face_cascade = None

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            if img is None or img.size == 0:
                logger.error("Received invalid frame")
                return frame
                
            self.frame_count += 1
            
            # Only process every 30th frame to reduce CPU usage
            if self.frame_count % 30 == 0 and self.face_cascade is not None:
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        try:
                            result = DeepFace.analyze(
                                img, 
                                actions=['emotion'], 
                                enforce_detection=False,
                                detector_backend='opencv'
                            )
                            
                            if result and isinstance(result, list) and len(result) > 0:
                                emotions = result[0]['emotion']
                                dominant_emotion = result[0]['dominant_emotion']
                                emotion_score = emotions[dominant_emotion]
                                
                                self.last_emotion = dominant_emotion
                                self.emotion_score = emotion_score
                                
                                # Add to history (limit to most recent 30 entries)
                                self.emotion_history.append({
                                    'emotion': dominant_emotion,
                                    'score': emotion_score,
                                    'timestamp': datetime.now().isoformat()
                                })
                                if len(self.emotion_history) > 30:
                                    self.emotion_history.pop(0)
                                    
                                # Draw results on frame
                                h, w = img.shape[:2]
                                emotion_text = f"{dominant_emotion.upper()} ({emotion_score:.1f}%)"
                                text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                                
                                # Draw text background
                                cv2.rectangle(
                                    img, 
                                    (10-5, 30 - text_size[1] - 5), 
                                    (10 + text_size[0] + 5, 30 + 5), 
                                    (0, 0, 0), 
                                    -1
                                )
                                
                                # Draw text
                                cv2.putText(
                                    img, 
                                    emotion_text, 
                                    (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.8, 
                                    (255, 255, 255), 
                                    2
                                )
                                
                                # Draw red border if negative emotion detected
                                if dominant_emotion in NEGATIVE_EMOTIONS and emotion_score >= NEGATIVE_THRESHOLD * 100:
                                    cv2.rectangle(img, (0, 0), (w-1, h-1), (0, 0, 255), 3)
                                    
                                # Draw face rectangles
                                for (x, y, fw, fh) in faces:
                                    cv2.rectangle(img, (x, y), (x+fw, y+fh), (255, 0, 0), 2)
                        
                        except Exception as df_error:
                            logger.error(f"DeepFace processing error: {str(df_error)}")
                            
                except Exception as cv_error:
                    logger.error(f"OpenCV processing error: {str(cv_error)}")
                    
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Error in emotion detector: {str(e)}\n{traceback.format_exc()}")
            return frame

# ---------------------------
# Report Generation Functions
# ---------------------------
def check_for_report_generation(messages):
    """Check if report should be generated and do so if appropriate"""
    if not messages or len(messages) < 6:
        return None
        
    # Check if enough time has passed since last check (to avoid checking too frequently)
    current_time = time.time()
    last_check_time = st.session_state.get("last_report_check_time", 0)
    
    if current_time - last_check_time < 60:  # Only check once per minute at most
        return None
        
    st.session_state["last_report_check_time"] = current_time
    
    # Use the report generator to decide if a report is warranted
    if report_generator.should_meet(messages,client):
        meet_link = quickstart.main()
        if meet_link:
            report_message = (
                f"I can sense that you're feeling overwhelmed right now, and that's completely understandable given the circumstances. "
                f"It's important to recognize that these feelings are valid and that you're not alone in experiencing them. "
                f"Please click on the link to join the meeting: {meet_link}"
            )
            append_message("assistant", report_message)
            st.chat_message("assistant").write(report_message)
            st.session_state["report_generated"] = True
            return meet_link
    if report_generator.should_generate_report(messages, client):
        logger.info("Auto-generating session report")
        try:
            report_path = report_generator.generate_report(
                client, 
                messages, 
                username=st.session_state.get("user_name", "User")
            )
            
            if report_path:
                report_message = (
                    f"I've generated a session summary report based on our conversation. "
                    f"You can download it from the sidebar under 'Report Generation'. "
                    f"The report includes an analysis of our discussion, key insights, and recommendations."
                )
                append_message("assistant", report_message)
                st.chat_message("assistant").write(report_message)
                st.session_state["report_generated"] = True
                send_email_with_attachment(
                    st.session_state["user_name"],
                    report_path
                )
                logger.info(f"Report generated and sent to {st.session_state['user_name']}")
                return report_path
        except Exception as e:
            logger.error(f"Auto-report generation error: {str(e)}")
            
    return None

def reset_conversation():
    """Reset the conversation and report flag"""
    st.session_state["messages"] = []
    st.session_state["report_generated"] = False
    save_conversation_history()
    st.rerun()

# ---------------------------
# App UI Components
# ---------------------------
def main():
    # ---------------------------
    # App Settings & Debug Tools
    # ---------------------------
    # Initialize debug mode in session state
    if "debug_mode" not in st.session_state:
        st.session_state["debug_mode"] = False
    
    # Admin section in expander
    with st.sidebar.expander("Admin Settings", expanded=False):
        st.session_state["debug_mode"] = st.checkbox("Debug Mode", value=st.session_state["debug_mode"])
        
        if st.button("Clear Conversation"):
            reset_conversation()
            
        if st.button("Export Conversation"):
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(BACKUP_DIR, f"conversation_export_{now}.json")
            try:
                with open(export_path, "w") as f:
                    json.dump(st.session_state["messages"], f, indent=2)
                st.success(f"Conversation exported to {export_path}")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    # ---------------------------
    # Report Generation
    # ---------------------------
    with st.sidebar.expander("Report Generation", expanded=False):
        if st.button("Generate Session Report"):
            with st.spinner("Generating comprehensive session report..."):
                report_path = report_generator.generate_report(
                    client, 
                    st.session_state["messages"], 
                    username=st.session_state.get("user_name", "User")
                )
                
                if report_path:
                    send_email_with_attachment(
                        st.session_state["user_name"],
                        report_path
                    )
                    st.session_state["report_generated"] = True
                    logger.info(f"Report generated and sent to {st.session_state['user_name']}")
                    st.success(f"Report generated successfully: {report_path}")
                    # Create a download button for the report
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="Download Report",
                            data=file,
                            file_name=os.path.basename(report_path),
                            mime="application/pdf"
                        )
                else:
                    st.error("Failed to generate report. Please try again.")
    
    # ---------------------------
    # Sidebar: User Panel and Video Feed
    # ---------------------------
    st.sidebar.title("User Panel")
    user_name = st.sidebar.text_input("Enter your name:", "Ankur")
    st.session_state["user_name"] = user_name  # Save user name for later use
    st.sidebar.write(f"Welcome, {user_name}!")

    rtc_config = RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]}
    )
    st.sidebar.subheader("Your Live Feed")
    with st.sidebar:
        try:
            video_ctx = webrtc_streamer(
                key="emotion_detector_sidebar",
                video_processor_factory=EmotionDetector,
                media_stream_constraints={"video": {"width": {"ideal": 320}, "height": {"ideal": 240}}, "audio": False},
                async_processing=True,
                rtc_configuration=rtc_config,
                desired_playing_state=True,
            )

            # Display detected emotion
            if video_ctx.video_processor and hasattr(video_ctx.video_processor, 'last_emotion') and video_ctx.video_processor.last_emotion:
                emotion = video_ctx.video_processor.last_emotion
                score = video_ctx.video_processor.emotion_score
                emotion_status = f"Current emotion: {emotion.upper()} ({score:.1f}%)"
                if emotion in NEGATIVE_EMOTIONS and score >= NEGATIVE_THRESHOLD * 100:
                    st.sidebar.error(emotion_status)
                else:
                    st.sidebar.info(emotion_status)
            else:
                st.sidebar.info("Emotion detection: Waiting for face...")
        except Exception as e:
            st.sidebar.error(f"Video feed error: {str(e)}")
            logger.error(f"Video feed error: {str(e)}")
            # Create a placeholder for the video context to avoid errors
            video_ctx = type('obj', (object,), {
                'video_processor': type('obj', (object,), {
                    'last_emotion': None,
                    'emotion_score': 0
                })
            })

    # ---------------------------
    # Main Content: Conversation & Input
    # ---------------------------
    st.title("🧠 SoulYatri Prototype Chatbot")
    st.markdown("Monitor your emotions through text, audio, and facial expressions.")

    # Display Conversation History
    st.subheader("Conversation History")
    for msg in st.session_state["messages"]:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue
            
        role = msg["role"]
        content = msg["content"]
        
        if not content:  # Skip empty messages
            continue
            
        # Convert legacy role names if needed
        if role == "ai":
            role = "assistant"
        elif role not in ["user", "assistant", "system"]:
            role = "user"  # Default to user for unknown roles
            
        st.chat_message(role).write(content)

    # Process New Text Input
    user_input = st.chat_input("Type your message:")
    if user_input:
        st.chat_message("user").write(user_input)
        append_message("user", user_input)
        
        # Get sentiment data
        text_sentiment, text_score = analyze_text_sentiment(user_input)
        
        # Get video emotion data
        if (hasattr(video_ctx, 'video_processor') and 
            video_ctx.video_processor and 
            hasattr(video_ctx.video_processor, 'last_emotion') and
            video_ctx.video_processor.last_emotion):
            video_emotion = video_ctx.video_processor.last_emotion
            video_score = video_ctx.video_processor.emotion_score
        else:
            video_emotion, video_score = "No Emotion Detected", 0.0
            
        # Get AI response
        with st.spinner("Thinking..."):
            assistant_message = get_groq_response(user_input, text_sentiment, text_score, video_emotion, video_score)
            
        # Display response
        st.chat_message("assistant").write(assistant_message)
        append_message("assistant", assistant_message)
        
        # Force save
        save_conversation_history()
        
        # Auto-generate report after text input if not already generated
        if not st.session_state.get("report_generated", False):
            report_path = check_for_report_generation(st.session_state["messages"])
            if report_path and st.session_state.get("debug_mode", False):
                st.info(f"Report automatically generated: {report_path}")
                send_email_with_attachment(
                    st.session_state["user_name"],
                    report_path
                )

    # --- Process New Audio Input ---
    audio_bytes = st.sidebar.audio_input("Record your audio message:")
    if audio_bytes:
        try:
            # Create directory if it doesn't exist
            os.makedirs(AUDIO_DIR, exist_ok=True)
            
            # Generate unique filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(AUDIO_DIR, f"recorded_audio_{timestamp}.wav")
            
            # Save audio file
            with open(file_path, "wb") as f:
                f.write(audio_bytes.read())
                
            if st.session_state.get("debug_mode"):
                st.sidebar.info(f"Audio file saved at {file_path}")

            # Transcribe audio
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(file_path)
            
            if transcript:
                st.chat_message("user").write(transcript)
                append_message("user", transcript)
                
                # Get sentiment data
                audio_sentiment, audio_score = analyze_text_sentiment(transcript)
                
                # Get video emotion data
                if (hasattr(video_ctx, 'video_processor') and 
                    video_ctx.video_processor and 
                    hasattr(video_ctx.video_processor, 'last_emotion') and
                    video_ctx.video_processor.last_emotion):
                    video_emotion = video_ctx.video_processor.last_emotion
                    video_score = video_ctx.video_processor.emotion_score
                else:
                    video_emotion, video_score = "No Emotion Detected", 0.0
                    
                # Get AI response
                with st.spinner("Thinking..."):
                    assistant_message = get_groq_response(transcript, audio_sentiment, audio_score, video_emotion, video_score)
                    
                # Display response
                st.chat_message("assistant").write(assistant_message)
                append_message("assistant", assistant_message)
                
                # Force save
                save_conversation_history()
                
                # Auto-generate report after audio input if not already generated
                if not st.session_state.get("report_generated", False):
                    report_path = check_for_report_generation(st.session_state["messages"])
                    if report_path and st.session_state.get("debug_mode", False):
                        st.info(f"Report automatically generated: {report_path}")
            else:
                st.error("Failed to transcribe audio. Please try again or type your message.")
                
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}\n{traceback.format_exc()}")
            st.error(f"Error processing audio: {str(e)}")

    st.markdown("---")
    st.caption("Developed by Ankur Vasani | 2025")

if __name__ == "__main__":
    main()
