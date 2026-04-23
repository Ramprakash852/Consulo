import streamlit as st
import ollama
import random
from datetime import datetime
import pandas as pd
from hashlib import pbkdf2_hmac
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt
import os
import time
import json
from streamlit.web.server.websocket_headers import _get_websocket_headers

# Check for optional voice packages
VOICE_AVAILABLE = False
try:
    import pyttsx3
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    pass

# Set page configuration
st.set_page_config(
    page_title="CONSULO: Mental Health Support",
    page_icon="🧘‍♀️",
    layout="wide"
)

# Emotion mapping from detection model to mood tracker
CHAT_FILE = "chat_data.json"

EMOTION_TO_MOOD = {
    "happy": "Good",
    "neutral": "Neutral",
    "angry": "Low",
    "sad": "Low",
    "disgust": "Low",
    "fear": "Very Low",
    "surprise": "Neutral",
    "contempt": "Low",
    "no_detection": None  # No update if no face detected
}


def load_chat():
    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as file:
            messages = json.load(file)
            return messages if isinstance(messages, list) else []
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []


def save_chat(messages):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def should_escalate(user_input):
    if st.session_state.mood_data.get('current_mood') == 'Very Low':
        return True

    if not user_input:
        return False

    keywords = ['suicide', 'hopeless', 'depressed']
    text = user_input.lower()
    return any(keyword in text for keyword in keywords)

# Initialize ALL session state variables properly
def init_session_state():
    # Mood tracking
    if 'mood_track' not in st.session_state:
        st.session_state['mood_track'] = []
    
    if 'mood_data' not in st.session_state:
        st.session_state.mood_data = {
            'current_mood': 'Neutral',
            'history': [],
            'mood_values': {'Very Low': 1, 'Low': 2, 'Neutral': 3, 'Good': 4, 'Excellent': 5}
        }
    
    # Journal entries
    if 'journal_entries' not in st.session_state:
        st.session_state.journal_entries = []
    
    # Voice settings
    if 'audio_playing' not in st.session_state:
        st.session_state.audio_playing = False

    if 'agent_mode' not in st.session_state:
        st.session_state.agent_mode = "AI"

    if 'role' not in st.session_state:
        st.session_state['role'] = 'USER'
    
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False and VOICE_AVAILABLE
    
    if 'conversation_speed' not in st.session_state:
        st.session_state.conversation_speed = "normal"
    
    # Reflection count - Fix for the error
    if 'reflection_count' not in st.session_state:
        st.session_state['reflection_count'] = 0
    
    # Journal UI state
    if 'show_journal' not in st.session_state:
        st.session_state['show_journal'] = False
    
    # Voice input
    if 'spoken_input' not in st.session_state:
        st.session_state['spoken_input'] = None
    
    # Response timing
    if 'last_response_time' not in st.session_state:
        st.session_state['last_response_time'] = 0
        
    # Voice listening mode
    if 'voice_listening_mode' not in st.session_state:
        st.session_state['voice_listening_mode'] = False
        
    # Last response for auto-speech
    if 'last_response_spoken' not in st.session_state:
        st.session_state['last_response_spoken'] = None
        
    # Auto emotion detection
    if 'auto_emotion_detection' not in st.session_state:
        st.session_state['auto_emotion_detection'] = False
        
    # Last detected emotion
    if 'last_detected_emotion' not in st.session_state:
        st.session_state['last_detected_emotion'] = None
        
    # Last emotion update time
    if 'last_emotion_update' not in st.session_state:
        st.session_state['last_emotion_update'] = 0


def role_selector():
    st.sidebar.header("👤 Role")

    selected_role = st.sidebar.selectbox(
        "Select role",
        options=['USER', 'THERAPIST'],
        index=['USER', 'THERAPIST'].index(st.session_state['role'])
    )

    st.session_state['role'] = selected_role


# Text-to-speech function
def text_to_speech(text):
    if not st.session_state.voice_enabled or not VOICE_AVAILABLE:
        return
    
    try:
        # Use pyttsx3 for offline TTS
        engine = pyttsx3.init()
        
        # Adjust voice properties based on selected speed
        speed_settings = {"slow": 150, "normal": 180, "fast": 220}
        engine.setProperty('rate', speed_settings[st.session_state.conversation_speed])
        
        # Get available voices and set a more natural voice if available
        voices = engine.getProperty('voices')
        # Try to find a female voice for the therapist
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        
        # Convert to audio
        temp_file = 'temp_speech.mp3'
        engine.save_to_file(text, temp_file)
        engine.runAndWait()
        
        # Play audio
        with open(temp_file, 'rb') as f:
            audio_bytes = f.read()
        
        st.audio(audio_bytes, format='audio/mp3', start_time=0, autoplay=True)
        
        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass
            
    except Exception as e:
        st.error(f"TTS Error: {e}")


# Speech-to-text function - improved for better UX
def speech_to_text():
    if not VOICE_AVAILABLE:
        st.warning("Voice recognition packages not installed.")
        return None
        
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=5)
            st.info("Processing speech...")
            
        try:
            text = r.recognize_google(audio)
            st.success(f"Heard: '{text}'")
            return text
        except sr.UnknownValueError:
            st.warning("Sorry, I couldn't understand what you said.")
            return None
        except sr.RequestError:
            st.error("Speech recognition service is unavailable.")
            return None
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return None


# Endpoint for receiving emotions from the detection model
def handle_emotion_webhook():
    # This runs for every request to check if it's the emotion webhook
    try:
        # Check if this is our emotion endpoint
        request_path = st.runtime.get_instance()._server_request._request_path
        print(request_path)
        if request_path == "/emotion":
            # Parse the incoming data
            request_data = st.runtime.get_instance()._server_request.get_json()
            
            # Extract the emotion
            detected_emotion = request_data.get("emotion", "").lower()
            
            # Store the detected emotion in session state
            st.session_state['last_detected_emotion'] = detected_emotion
            
            # Update the mood if auto-detection is enabled
            if st.session_state['auto_emotion_detection'] and detected_emotion:
                mapped_mood = EMOTION_TO_MOOD.get(detected_emotion)
                if mapped_mood and mapped_mood != st.session_state.mood_data['current_mood']:
                    # Only update if the mood has changed
                    st.session_state.mood_data['current_mood'] = mapped_mood
                    
                    # Add to mood history
                    st.session_state.mood_data['history'].append({
                        'mood': mapped_mood,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Update reflection count
                    st.session_state['reflection_count'] += 1
                    st.session_state['last_emotion_update'] = time.time()
                    
                    # Force a rerun to update the UI
                    st.rerun()
                    
            return {"status": "success"}
    except Exception as e:
        print(f"Error in emotion webhook: {e}")
    
    return None

# In your Streamlit app
def check_emotion_updates():
    try:
        if os.path.exists("emotion_data.json"):
            with open("emotion_data.json", "r") as f:
                data = json.load(f)
                
            # Check if this is a new update (within last 10 seconds)
            if time.time() - data["timestamp"] < 10:  
                detected_emotion = data["emotion"].lower()
                print(detected_emotion)
                st.session_state['last_detected_emotion'] = detected_emotion
                
                # Update the mood if auto-detection is enabled
                if st.session_state['auto_emotion_detection'] and detected_emotion:
                    mapped_mood = EMOTION_TO_MOOD.get(detected_emotion)
                    if mapped_mood and mapped_mood != st.session_state.mood_data['current_mood']:
                        # Only update if the mood has changed
                        st.session_state.mood_data['current_mood'] = mapped_mood
                        
                        # Add to mood history
                        st.session_state.mood_data['history'].append({
                            'mood': mapped_mood,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Update reflection count
                        st.session_state['reflection_count'] += 1
                        st.session_state['last_emotion_update'] = time.time()
                        
    except Exception as e:
        print(f"Error checking emotion updates: {e}")

# Mood tracking function with simplified UI and auto-detection toggle
def track_mood():
    st.sidebar.header("🌈 Mood Tracker")
    
    # Add auto-detection toggle
    auto_detect = st.sidebar.toggle(
        "Auto-detect emotions", 
        value=st.session_state['auto_emotion_detection'],
        help="Enable to automatically update your mood based on facial expressions"
    )
    
    if auto_detect != st.session_state['auto_emotion_detection']:
        st.session_state['auto_emotion_detection'] = auto_detect
        
    # Show last detected emotion if auto-detection is enabled
    if st.session_state['auto_emotion_detection'] and st.session_state['last_detected_emotion']:
        emotion_emoji = {
            "happy": "😄", 
            "neutral": "😐", 
            "angry": "😠",
            "sad": "😢",
            "disgust": "🤢",
            "fear": "😨",
            "surprise": "😲",
            "contempt": "😒",
            "no_detection": "❓"
        }
        emoji = emotion_emoji.get(st.session_state['last_detected_emotion'], "❓")
        st.sidebar.info(f"Detected emotion: {emoji} {st.session_state['last_detected_emotion'].capitalize()}")
        
        # Show when the last update happened
        if st.session_state['last_emotion_update'] > 0:
            time_diff = time.time() - st.session_state['last_emotion_update']
            if time_diff < 60:
                st.sidebar.caption(f"Last updated {int(time_diff)} seconds ago")
            else:
                st.sidebar.caption(f"Last updated {int(time_diff/60)} minutes ago")
    
    # Mood selector with emojis for better UX
    mood_options = {
        'Very Low': '😞', 
        'Low': '😔', 
        'Neutral': '😐', 
        'Good': '😊', 
        'Excellent': '😁'
    }
    
    mood_display = [f"{mood} {emoji}" for mood, emoji in mood_options.items()]
    current_mood_index = list(mood_options.keys()).index(st.session_state.mood_data['current_mood'])
    
    # Disable manual selection if auto-detection is enabled
    new_mood_selection = st.sidebar.select_slider(
        "How are you feeling right now?",
        options=mood_display,
        value=mood_display[current_mood_index],
        disabled=st.session_state['auto_emotion_detection']
    )
    
    # Only process manual mood updates if auto-detection is disabled
    if not st.session_state['auto_emotion_detection']:
        # Extract the mood without emoji
        new_mood = ' '.join(new_mood_selection.split(' ')[:-1])
        
        # If mood changes, update the reflection count
        if new_mood != st.session_state.mood_data['current_mood']:
            st.session_state.mood_data['current_mood'] = new_mood
            st.session_state['reflection_count'] += 1
    
            # Log mood change with timestamp
            st.session_state.mood_data['history'].append({
                'mood': new_mood,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # Mood history visualization - only show if there's data
    if st.session_state.mood_data['history']:
        if st.sidebar.checkbox("Show Mood Timeline", value=False):
            mood_df = pd.DataFrame(st.session_state.mood_data['history'])
            mood_df['numeric_mood'] = mood_df['mood'].map(st.session_state.mood_data['mood_values'])
            st.sidebar.line_chart(mood_df.set_index('timestamp')['numeric_mood'])


# Optimized response generation for faster replies
def generate_response(user_input):
    # Start the timer
    start_time = time.time()

    agent_mode = st.session_state.get('agent_mode', 'AI')
    agent_persona = "AI assistant" if agent_mode == "AI" else "therapist"
    
    # Include detected emotion in context if available
    emotion_context = ""
    if st.session_state['auto_emotion_detection'] and st.session_state['last_detected_emotion']:
        emotion_context = f"Detected facial emotion: {st.session_state['last_detected_emotion']}"
    
    # Simplify the prompt for faster responses
    mood_context = f"User mood: {st.session_state.mood_data['current_mood']}"
    print(mood_context)

    system_prompt = {
        "role": "system",
        "content": f"""You are a concise, compassionate mental health {agent_persona} named COUNSULO. Use these guidelines:
        1. Current Mood: {st.session_state.mood_data['current_mood']}
        2. Always validate feelings first
        3. Suggest mood-appropriate coping strategies
        4. Be warm but direct
        5. Maintain supportive tone and never say anything that make user uncomfortable and sad
        6. Keep responses under 5-6 sentences for faster dialogue
        7. mood_context: {mood_context}
        8. {emotion_context}"""
    }

    chat_history = load_chat()

    # Use only the last 3 messages for context to improve speed
    messages = [system_prompt]
    messages.extend(chat_history[-3:])
    messages.append({"role": "user", "content": user_input})

    try:
        with st.spinner("Thinking..."):
            # Call the LLM with a timeout
            response = ollama.chat(
                model="qwen2.5:7b", 
                messages=messages,
                options={"temperature": 0.7, "top_p": 0.9}
            )
            ai_response = response['message']['content']

            # Update shared conversation history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": ai_response})
            save_chat(chat_history)
            
            # Calculate response time
            response_time = time.time() - start_time
            st.session_state['last_response_time'] = response_time
            
            # Flag that this is a new response to speak
            st.session_state['last_response_spoken'] = ai_response

            return ai_response
    except Exception as e:
        return f"I'm having trouble right now. Could you try again in a moment? Error: {str(e)}"


# Simplified personalized therapy section
def personalized_therapy():
    st.sidebar.header("🧠 Personalized Therapy")
    
    # Use tabs for better organization
    issue_tab, tips_tab = st.sidebar.tabs(["Your Concerns", "Get Help"])
    
    with issue_tab:
        issues = st.multiselect(
            "What are you struggling with today?", 
            ['Anxiety', 'Depression', 'Stress', 'Self-esteem', 'Sleep', 'Loneliness']
        )
    
    with tips_tab:
        if issues:
            therapy_tips = {
                'Anxiety': "🧘 Try 4-7-8 breathing: inhale for 4s, hold for 7s, exhale for 8s.",
                'Depression': "🚶 Even a 5-minute walk outside can help lift your mood.",
                'Stress': "✋ Try the 5-5-5 method: name 5 things you see, hear, and feel.",
                'Self-esteem': "📝 Write down 3 things you like about yourself.",
                'Sleep': "🌙 No screens 1 hour before bed. Try reading instead.",
                'Loneliness': "📱 Send a message to someone you haven't talked to lately."
            }
            
            for issue in issues:
                st.info(therapy_tips[issue])
        else:
            st.write("Select concerns to get personalized tips")


# Progress dashboard with cleaner visualization
def progress_dashboard():
    st.header("📈 Your Progress")
    
    # Calculate metrics
    mood_values = {'Very Low': 1, 'Low': 2, 'Neutral': 3, 'Good': 4, 'Excellent': 5}

    # Helper function to parse timestamps
    def parse_timestamp(ts):
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M")

    # Parse dates with error handling
    mood_dates = []
    mood_scores = []
    for entry in st.session_state.mood_data.get('history', []):
        try:
            dt = parse_timestamp(entry['timestamp'])
            mood_dates.append(dt.date())
            mood_scores.append(mood_values[entry['mood']])
        except (KeyError, ValueError):
            continue

    unique_dates = sorted(list(set(mood_dates))) if mood_dates else []
    
    # Streak calculation
    current_streak = 0
    if unique_dates:
        current_date = datetime.now().date()
        last_date = unique_dates[-1]
        
        if last_date == current_date:
            current_streak = 1
            for i in range(len(unique_dates)-2, -1, -1):
                if (unique_dates[i+1] - unique_dates[i]).days == 1:
                    current_streak += 1
                else:
                    break

    # Simple metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mood Streak", f"{current_streak} days")
    
    with col2:
        if len(unique_dates) > 0:
            today_entries = [
                e for e in st.session_state.mood_data['history']
                if parse_timestamp(e['timestamp']).date() == unique_dates[-1]
            ]
            
            if today_entries:
                total_mood = sum(mood_values[entry['mood']] for entry in today_entries)
                avg_mood = total_mood / len(today_entries)
                st.metric("Today's Mood", f"{avg_mood:.1f}/5")
            else:
                st.metric("Today's Mood", "N/A")
    
    with col3:
        st.metric("Check-ins", f"{st.session_state['reflection_count']}")
        
    # Display response time
    if 'last_response_time' in st.session_state and st.session_state['last_response_time'] > 0:
        st.caption(f"Last response time: {st.session_state['last_response_time']:.2f} seconds")

    # Only show report option if there's enough data
    if len(st.session_state.mood_data['history']) > 3:
        if st.button("📄 Get PDF Report", use_container_width=True):
            generate_pdf_report()


# Generate PDF report
def generate_pdf_report():
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add content
        pdf.cell(200, 10, txt="Your Mental Wellness Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated on {datetime.now().strftime('%Y-%m-%d')}", ln=2, align='C')
        
        # Mood chart
        mood_values = {'Very Low': 1, 'Low': 2, 'Neutral': 3, 'Good': 4, 'Excellent': 5}
        mood_data = [mood_values[entry['mood']] for entry in st.session_state.mood_data['history']]
        
        plt.figure(figsize=(10, 4))
        plt.plot(mood_data)
        plt.title('Mood Progression')
        plt.savefig('mood_chart.png')
        pdf.image('mood_chart.png', w=180)
        
        # Save and offer download
        pdf_output = "wellness_report.pdf"
        pdf.output(pdf_output)
        
        with open(pdf_output, "rb") as f:
            pdf_bytes = f.read()
            
        st.download_button(
            label="Download Report",
            data=pdf_bytes,
            file_name="wellness_report.pdf",
            mime="application/pdf"
        )
        
    except Exception as e:
        st.error(f"Error generating report: {e}")


# Voice interaction settings - Improved with better UX
def voice_settings():
    st.sidebar.header("🎙️ Voice Interaction")
    
    if not VOICE_AVAILABLE:
        st.sidebar.warning("Voice packages not installed. Run: pip install pyttsx3 SpeechRecognition pyaudio")
        st.sidebar.info("For conda: conda install -c anaconda pyaudio")
        return
        
    voice_enabled = st.sidebar.toggle("Enable Voice", value=st.session_state.voice_enabled)
    if voice_enabled != st.session_state.voice_enabled:
        st.session_state.voice_enabled = voice_enabled
    
    if st.session_state.voice_enabled:
        speed_options = {"slow": "Slow", "normal": "Normal", "fast": "Fast"}
        selected_speed = st.sidebar.radio(
            "Voice Speed", 
            options=list(speed_options.keys()),
            format_func=lambda x: speed_options[x],
            horizontal=True,
            index=list(speed_options.keys()).index(st.session_state.conversation_speed)
        )
        
        if selected_speed != st.session_state.conversation_speed:
            st.session_state.conversation_speed = selected_speed
        
        # Voice input toggle
        voice_listening = st.sidebar.toggle("Voice Input Mode", value=st.session_state.voice_listening_mode)
        if voice_listening != st.session_state.voice_listening_mode:
            st.session_state.voice_listening_mode = voice_listening
            
        if st.session_state.voice_listening_mode:
            st.sidebar.info("Voice input mode ON - use the microphone button below the chat to speak")
        
        # Keep the manual button for backup
        if st.sidebar.button("🎤 Speak Now", use_container_width=True):
            with st.spinner("Listening..."):
                spoken_text = speech_to_text()
                if spoken_text and spoken_text != "Sorry, I couldn't understand what you said.":
                    st.session_state.spoken_input = spoken_text
                    st.rerun()


# Crisis support with simplified UI
def crisis_support():
    st.sidebar.header("🆘 Need Urgent Help?")
    
    if st.sidebar.button("🚨 I Need Immediate Help", use_container_width=True):
        st.sidebar.error("""
        **Call now:**
        - National Emergency: 112
        - Mental Health Helpline: 1800-599-0019
        """)
        
        # Show hospitals map
        hospital_locations = pd.DataFrame({
            'lat': [28.6139, 28.6280, 28.5805],
            'lon': [77.2090, 77.3649, 77.3316],
            'name': ['AIIMS Hospital', 'Safdarjung Hospital', 'RML Hospital']
        })
        st.sidebar.map(hospital_locations,
            latitude='lat',
            longitude='lon',
            size=20,
            color='#FF0000')


# Journaling feature
def journaling():
    st.sidebar.header("📔 Therapy Journal")
    
    if st.sidebar.button("Write in Journal", use_container_width=True):
        st.session_state['show_journal'] = True
        st.rerun()
        
    if st.session_state.get('show_journal', False):
        with st.form("journal_entry"):
            st.subheader("Today's Journal Entry")
            entry = st.text_area("What's on your mind today?", height=150)
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Save Entry")
            with col2:
                cancel = st.form_submit_button("Close Journal")
                
            if submit and entry:
                # Save journal entry
                st.session_state.journal_entries.append({
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'entry': entry,
                    'mood': st.session_state.mood_data['current_mood']
                })
                st.success("Journal entry saved!")
                st.session_state['show_journal'] = False
                st.rerun()
                
            if cancel:
                st.session_state['show_journal'] = False
                st.rerun()


# Main app with improved UI
def main():
    # Initialize session state first thing
    init_session_state()
    
    # Check for emotion webhook data
    # handle_emotion_webhook()
    check_emotion_updates()

    # Top banner
    st.title("🌿 CONSULO: Your Personal AI Therapist")
    
    # Show the progress dashboard in a collapsible section
    with st.expander("📊 Your Progress Dashboard", expanded=False):
        progress_dashboard()

    # Sidebar features
    role_selector()
    track_mood()
    voice_settings()
    personalized_therapy()
    journaling()
    crisis_support()

    # Chat interface
    st.subheader("💬 Talk with CONSULO")
    if st.session_state.get('agent_mode', 'AI') == 'AI':
        st.info("🤖 AI is responding")
    else:
        st.info("👨‍⚕️ Therapist connected")
    
    # Modern chat container
    chat_container = st.container()
    with chat_container:
        # Create a scrollable area for chat history
        chat_area = st.container(height=400, border=True)
        with chat_area:
            chat_history = load_chat()
            recent_messages = chat_history[-10:]

            # Show welcome message if no conversation history
            if not recent_messages:
                welcome_msg = "Hello! I'm CONSULO, your AI therapist. How are you feeling today?"
                st.chat_message("assistant").write(welcome_msg)
                
                # Auto-speak the welcome message if voice is enabled
                if st.session_state.voice_enabled and VOICE_AVAILABLE:
                    text_to_speech(welcome_msg)
                    st.session_state['last_response_spoken'] = None
            else:
                # Display chat history (limited to last 10 messages for performance)
                for msg in recent_messages:
                    role = msg.get("role", "assistant")
                    content = msg.get("content", "")
                    if role == "system":
                        role = "assistant"

                    with st.chat_message(role):
                        st.write(content)

    # User input area (below the chat)
    user_message = None
    if st.session_state.get('role', 'USER') != 'THERAPIST':
        user_message = st.chat_input("Share your thoughts or ask for support...")
    
    # Voice input button directly in the chat
    if st.session_state.get('role', 'USER') != 'THERAPIST' and st.session_state.voice_enabled and st.session_state.voice_listening_mode:
        voice_col1, voice_col2 = st.columns([1, 9])
        with voice_col1:
            if st.button("🎤", help="Click to speak"):
                with st.spinner("Listening..."):
                    spoken_text = speech_to_text()
                    if spoken_text:
                        st.session_state.spoken_input = spoken_text
                        st.rerun()
    
    # Check if there's voice input
    if st.session_state.get('role', 'USER') != 'THERAPIST' and 'spoken_input' in st.session_state and st.session_state['spoken_input']:
        user_message = st.session_state['spoken_input']
        st.session_state['spoken_input'] = None
    
    # Auto-speak the last response if it hasn't been spoken yet
    if st.session_state.voice_enabled and st.session_state['last_response_spoken']:
        text_to_speech(st.session_state['last_response_spoken'])
        st.session_state['last_response_spoken'] = None
    
    if user_message:
        # Display user message
        with st.chat_message("user"):
            st.write(user_message)

        if st.session_state.get('agent_mode', 'AI') == 'AI' and should_escalate(user_message):
            chat_history = load_chat()
            chat_history.append({"role": "user", "content": user_message})
            st.session_state['agent_mode'] = 'THERAPIST'
            chat_history.append({"role": "assistant", "content": "👨‍⚕️ Connecting to therapist..."})
            save_chat(chat_history)
        elif st.session_state.get('agent_mode', 'AI') == 'THERAPIST':
            chat_history = load_chat()
            chat_history.append({"role": "user", "content": user_message})
            save_chat(chat_history)
            st.info("Therapist mode is active. Waiting for therapist response.")
        else:
            # Generate AI response
            with st.chat_message("assistant"):
                ai_response = generate_response(user_message)
                st.write(ai_response)
                
                # Automatically speak the response if voice is enabled
                if st.session_state.voice_enabled and VOICE_AVAILABLE:
                    text_to_speech(ai_response)
                    st.session_state['last_response_spoken'] = None

    if st.session_state.get('agent_mode', 'AI') == 'THERAPIST':
        st.caption("↻ Waiting for therapist response...")
        time.sleep(2)
        st.rerun()

    # Compact emergency support section
    st.markdown("---")
    with st.container():
        emergency_col1, emergency_col2, emergency_col3 = st.columns(3)
        with emergency_col1:
            st.markdown("<small>🏥 Emergency: 112</small>", unsafe_allow_html=True)
        with emergency_col2:
            st.markdown("<small>📞 NIMHANS: 080-2678500</small>", unsafe_allow_html=True)
        with emergency_col3:
            st.markdown("<small>💬 Manodarpan: 1800-599-0019</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()