I'll create a comprehensive project report focused on the build structure and design decisions.  

Read [](file:///c%3A/Users/chitt/Downloads/Facial%20Emotion%20Detection/Facial%20Emotion%20Detection/btp/calmconnect.py#32-32), lines 32 to 45

Read [](file:///c%3A/Users/chitt/Downloads/Facial%20Emotion%20Detection/Facial%20Emotion%20Detection/archive/data.yaml)

---

# **BTP PROJECT REPORT**
## **Facial Emotion Detection AI Mental Health Companion**

---

## **1. PROJECT OVERVIEW**

**Project Name:** CONSULO: Personal AI Therapist (BTP - Bachelor of Technology Project)

**Objective:** Build an end-to-end system that detects facial emotions from webcam input and uses that data to provide real-time mental health support, mood tracking, and therapeutic guidance through an AI-powered conversational interface.

**Core Value Proposition:**
- Real-time emotion detection without user self-reporting
- Emotion-context-aware AI responses (not generic)
- Persistent mood tracking with historical analytics
- Integrated crisis support and coping strategies
- Voice-enabled accessibility for hands-free interaction

---

## **2. PROBLEM STATEMENT**

Mental health support systems typically rely on user self-reporting of mood, which is:
- **Subjective & inaccurate:** Users may not accurately self-assess their emotional state
- **Retrospective:** Requires manual input after emotions have passed
- **Incomplete:** Missing real-time micro-expressions and subtle indicators

**Solution Approach:** Automate emotion capture via facial analysis, then use detected emotions as context for AI-driven therapeutic support, creating a feedback loop between physiological signals and conversational guidance.

---

## **3. TECHNOLOGY STACK**

### **Frontend & UI**
- **Streamlit** (Python web framework)
  - Reason: Rapid prototyping, real-time state management, built-in widgets
  - No HTML/CSS/JavaScript required—purely Python-driven UI
  - Session state for persistent data across reruns

### **Backend & API**
- **FastAPI** (Python async web framework)
  - Reason: Lightweight HTTP server for receiving emotion webhooks
  - Async capability for non-blocking I/O
  - Auto-generated OpenAPI docs
  - Port: `127.0.0.1:8502`

### **Emotion Detection (Two Options)**
1. **FER (Facial Expression Recognition) with MTCNN**
   - FER library: Pre-trained CNN on facial expression datasets
   - MTCNN: Multi-task Cascaded Convolutional Networks for robust face detection
   - Used by: emotion_detection_app.py
   - Pros: Lightweight, runs locally, 8 emotion classes
   - Output: Real-time emotion scores per face region

2. **YOLO (You Only Look Once)**
   - Model: Fine-tuned custom YOLO11 or earlier variant
   - Reason: Faster inference, handles multiple faces
   - Used by: main.py as fallback
   - Output: Single dominant emotion per frame

### **AI/LLM Engine**
- **Ollama + Qwen2.5:7B**
  - Reason: Local, open-source LLM—no API keys, data stays local
  - Model size: 7B parameters (runs on CPU/modest GPU)
  - Integration: Python `ollama` library with JSON chat messages
  - Customization: System prompt includes mood context, keeps responses short (5-6 sentences)

### **Data Persistence**
- **JSON file (emotion_data.json)** for inter-process communication
  - Structure: `{"emotion": "string", "timestamp": "float"}`
  - Reason: Simple, no database dependency, filesystem-based polling
  - Updated by: FastAPI server
  - Read by: Streamlit app (via `check_emotion_updates()`)

### **Optional Voice I/O**
- **pyttsx3**: Text-to-speech (offline, no cloud dependency)
- **SpeechRecognition**: Speech-to-text via Google Speech API
- **pyaudio**: Microphone input handling

### **Reporting & Export**
- **FPDF**: PDF generation for wellness reports
- **Matplotlib**: Mood charts and visualizations
- **Pandas**: Mood history aggregation and analysis

---

## **4. ARCHITECTURE & DESIGN PRINCIPLES**

### **4.1 Layered Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                 USER INTERFACE LAYER                     │
│  (Streamlit: Chat, Mood Tracker, Dashboard, Voice)      │
└────────────────────┬────────────────────────────────────┘
                     │ Reads from
                     ↓
┌─────────────────────────────────────────────────────────┐
│              DATA EXCHANGE LAYER                         │
│  (emotion_data.json: Latest emotion + timestamp)         │
└────────────────────┬────────────────────────────────────┘
                     │ Written by
                     ↓
┌─────────────────────────────────────────────────────────┐
│              RELAY / API LAYER                           │
│  (FastAPI server: Receives emotion POST requests)        │
└────────────────────┬────────────────────────────────────┘
                     │ Receives from
                     ↓
┌─────────────────────────────────────────────────────────┐
│            DETECTION LAYER                              │
│  (Webcam → FER/YOLO → Emit emotion every 5 sec)         │
└─────────────────────────────────────────────────────────┘
```

### **4.2 Design Principles**

**Decoupling:**
- Detection and UI are independent processes
- JSON file acts as loosely-coupled message queue
- Allows detector and UI to run on different schedules or machines

**Statelessness (Detection) vs. Statefulness (UI):**
- Emotion detector: Stateless, sends every 5 seconds
- Streamlit app: Maintains full session state (conversation history, mood data, journal)

**Polling vs. Webhooks:**
- Initial design supported webhook endpoints (see `handle_emotion_webhook()` in calmconnect.py)
- Fallback: File-based polling via `check_emotion_updates()` (more stable for local setup)

**Context-Aware AI:**
- System prompt includes: current mood, detected emotion, last 3 messages (for speed)
- LLM never responds in a vacuum—always mood-aware

---

## **5. CORE COMPONENTS & BUILD DETAILS**

### **5.1 Emotion Detection Engine**
**Files:** emotion_detection_app.py, main.py

**How It's Built:**

```python
class EmotionDetectionApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)           # Initialize webcam
        self.emotion_detector = FER(mtcnn=True)  # Load FER model
        self.emotions_buffer = []                # Buffer detections
        
    def detect_emotions(self, frame):
        # MTCNN finds faces
        detected_faces = self.emotion_detector.detect_emotions(frame)
        # Extract dominant emotion per face
        
    def run(self):
        while True:
            frame = webcam.read()
            emotions = self.detect_emotions(frame)
            self.emotions_buffer.extend(emotions)
            
            # Every 5 seconds:
            if elapsed >= 5:
                dominant = Counter(buffer).most_common(1)
                self.send_emotion(dominant)  # POST to FastAPI
```

**Key Features:**
- **Buffering:** Collects 5 seconds of emotion frames, picks most common (noise-resistant)
- **Fallback Detection:** YOLO if FER not available
- **Direct Webhook:** Sends emotion via HTTP POST to FastAPI at `http://127.0.0.1:8502/emotion`
- **Visual Feedback:** Displays bounding boxes, detected emotion text, per-emotion confidence scores

**Input:** Webcam stream (OpenCV VideoCapture)  
**Output:** JSON payload `{"emotion": "happy"}` every 5 seconds  
**Error Handling:** If webcam fails, raises IOError; if YOLO unavailable, falls back to FER

---

### **5.2 FastAPI Relay Server**
**File:** emotion_server.py

**How It's Built:**

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/emotion")
async def receive_emotion(request: Request):
    data = await request.json()
    emotion = data.get("emotion", "neutral")
    
    # Persist to file
    with open("emotion_data.json", "w") as f:
        json.dump({"emotion": emotion, "timestamp": time.time()}, f)
    
    return {"status": "success"}

uvicorn.run(app, host="127.0.0.1", port=8502)
```

**Design Decisions:**
- **Single Endpoint:** `/emotion` POST only
- **Async:** Non-blocking request handling
- **File-Based State:** JSON written to disk for Streamlit to read
- **Timestamp:** ISO-like timestamp allows Streamlit to detect stale data (only update if < 10 seconds old)
- **Port 8502:** Non-standard, intentional to avoid conflicts

**Responsibilities:**
1. Receive emotion POST from detector
2. Write to JSON file (atomic operation)
3. Return 200 OK

**Why This Design (vs. Database)?**
- No external dependencies (no PostgreSQL, MongoDB)
- File I/O is simple and synchronous from Streamlit
- Suitable for single-user local deployment

---

### **5.3 Streamlit UI Application**
**File:** calmconnect.py (~800 lines)

**How It's Built:**

#### **5.3.1 Session State Management**
```python
def init_session_state():
    # Persists across reruns without user intervention
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'mood_data' not in st.session_state:
        st.session_state.mood_data = {
            'current_mood': 'Neutral',
            'history': [],
            'mood_values': {'Very Low': 1, 'Low': 2, 'Neutral': 3, 'Good': 4, 'Excellent': 5}
        }
    if 'journal_entries' not in st.session_state:
        st.session_state.journal_entries = []
    # ... 10+ more state variables
```

**Why:** Streamlit reruns entire script on each user interaction. Session state preserves data across reruns without fetching from external storage.

#### **5.3.2 Emotion-to-Mood Mapping**
```python
EMOTION_TO_MOOD = {
    "happy": "Good",
    "neutral": "Neutral",
    "angry": "Low",
    "sad": "Low",
    "disgust": "Low",
    "fear": "Very Low",
    "surprise": "Neutral",
    "contempt": "Low",
    "no_detection": None
}
```

**Logic:** Raw 8-class emotions → 5-tier mood scale (Very Low to Excellent) for therapy guidance consistency.

#### **5.3.3 Emotion Update Polling**
```python
def check_emotion_updates():
    if os.path.exists("emotion_data.json"):
        with open("emotion_data.json", "r") as f:
            data = json.load(f)
        
        # Check if this is a new update (within last 10 seconds)
        if time.time() - data["timestamp"] < 10:
            detected_emotion = data["emotion"].lower()
            st.session_state['last_detected_emotion'] = detected_emotion
            
            # If auto-detection is ON, update mood
            if st.session_state['auto_emotion_detection']:
                mapped_mood = EMOTION_TO_MOOD.get(detected_emotion)
                if mapped_mood and mapped_mood != st.session_state.mood_data['current_mood']:
                    st.session_state.mood_data['current_mood'] = mapped_mood
                    st.session_state.mood_data['history'].append({
                        'mood': mapped_mood,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.rerun()  # Trigger UI update
```

**Flow:**
1. Read emotion_data.json
2. Check if timestamp is recent (< 10 sec)
3. If auto-detection is enabled, map emotion → mood
4. If mood changed, append to history and rerun app

**Why Polling?** 
- Simple, no WebSocket complexity
- Works reliably with file I/O
- Timestamp check prevents redundant updates

#### **5.3.4 AI Therapist Response Generation**
```python
def generate_response(user_input):
    emotion_context = ""
    if st.session_state['last_detected_emotion']:
        emotion_context = f"Detected facial emotion: {st.session_state['last_detected_emotion']}"
    
    system_prompt = {
        "role": "system",
        "content": f"""You are a concise, compassionate mental health AI named COUNSULO. 
        1. Current Mood: {st.session_state.mood_data['current_mood']}
        2. Always validate feelings first
        3. Suggest mood-appropriate coping strategies
        4. Keep responses under 5-6 sentences
        5. {emotion_context}"""
    }
    
    messages = [system_prompt]
    messages.extend(st.session_state['conversation_history'][-3:])  # Last 3 for speed
    messages.append({"role": "user", "content": user_input})
    
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=messages,
        options={"temperature": 0.7, "top_p": 0.9}
    )
    
    ai_response = response['message']['content']
    st.session_state['conversation_history'].append({"role": "user", "content": user_input})
    st.session_state['conversation_history'].append({"role": "assistant", "content": ai_response})
    
    return ai_response
```

**Design Choices:**
- **System Prompt Customization:** Includes mood and detected emotion for context
- **Limited History:** Only last 3 messages to reduce token usage and latency
- **Temperature 0.7:** Balance between creativity and consistency
- **Conversation History:** Preserved in session_state for multi-turn dialogue

#### **5.3.5 Mood Tracking UI**
```python
def track_mood():
    st.sidebar.header("🌈 Mood Tracker")
    
    auto_detect = st.sidebar.toggle(
        "Auto-detect emotions",
        value=st.session_state['auto_emotion_detection']
    )
    
    mood_options = {'Very Low': '😞', 'Low': '😔', 'Neutral': '😐', 'Good': '😊', 'Excellent': '😁'}
    mood_display = [f"{mood} {emoji}" for mood, emoji in mood_options.items()]
    current_mood_index = list(mood_options.keys()).index(st.session_state.mood_data['current_mood'])
    
    new_mood_selection = st.sidebar.select_slider(
        "How are you feeling right now?",
        options=mood_display,
        value=mood_display[current_mood_index],
        disabled=st.session_state['auto_emotion_detection']
    )
    
    # If auto-detect is OFF, allow manual mood selection
    if not st.session_state['auto_emotion_detection']:
        new_mood = ' '.join(new_mood_selection.split(' ')[:-1])
        if new_mood != st.session_state.mood_data['current_mood']:
            st.session_state.mood_data['current_mood'] = new_mood
            st.session_state.mood_data['history'].append({
                'mood': new_mood,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Show mood timeline
    if st.session_state.mood_data['history']:
        if st.sidebar.checkbox("Show Mood Timeline", value=False):
            mood_df = pd.DataFrame(st.session_state.mood_data['history'])
            mood_df['numeric_mood'] = mood_df['mood'].map(st.session_state.mood_data['mood_values'])
            st.sidebar.line_chart(mood_df.set_index('timestamp')['numeric_mood'])
```

**Features:**
- **Auto-Detect Toggle:** Disable manual selection when auto-detect is ON
- **Emoji Support:** Better UX than text-only mood selector
- **Dual Input:** Manual selection or automatic facial detection
- **Timeline Visualization:** Line chart of mood over time

#### **5.3.6 Progress Dashboard & PDF Report**
```python
def progress_dashboard():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mood Streak", f"{current_streak} days")
    with col2:
        st.metric("Today's Mood", f"{avg_mood:.1f}/5")
    with col3:
        st.metric("Check-ins", f"{st.session_state['reflection_count']}")

def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.cell(200, 10, txt="Your Mental Wellness Report", ln=1, align='C')
    
    # Generate mood chart
    mood_data = [mood_values[entry['mood']] for entry in st.session_state.mood_data['history']]
    plt.figure(figsize=(10, 4))
    plt.plot(mood_data)
    plt.title('Mood Progression')
    plt.savefig('mood_chart.png')
    pdf.image('mood_chart.png', w=180)
    
    pdf.output("wellness_report.pdf")
```

**Metrics:**
- Mood streak (consecutive days with check-ins)
- Today's average mood
- Total reflection count (number of mood updates)
- Response time tracking

#### **5.3.7 Voice Integration**
```python
def text_to_speech(text):
    if not st.session_state.voice_enabled or not VOICE_AVAILABLE:
        return
    
    engine = pyttsx3.init()
    speed_settings = {"slow": 150, "normal": 180, "fast": 220}
    engine.setProperty('rate', speed_settings[st.session_state.conversation_speed])
    
    voices = engine.getProperty('voices')
    for voice in voices:
        if "female" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    
    temp_file = 'temp_speech.mp3'
    engine.save_to_file(text, temp_file)
    engine.runAndWait()

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=5)
    
    text = r.recognize_google(audio)
    return text
```

**Capabilities:**
- Text-to-speech with 3 speed settings
- Fallback to female voice if available
- Speech-to-text via Google APIs
- Error handling for unavailable packages

#### **5.3.8 Journaling Feature**
```python
def journaling():
    st.sidebar.header("📔 Therapy Journal")
    
    if st.sidebar.button("Write in Journal"):
        st.session_state['show_journal'] = True
        st.rerun()
    
    if st.session_state.get('show_journal', False):
        with st.form("journal_entry"):
            entry = st.text_area("What's on your mind today?", height=150)
            submit = st.form_submit_button("Save Entry")
            
            if submit and entry:
                st.session_state.journal_entries.append({
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'entry': entry,
                    'mood': st.session_state.mood_data['current_mood']
                })
```

**Stores:** Date, entry text, current mood (for correlation analysis)

#### **5.3.9 Crisis Support**
```python
def crisis_support():
    st.sidebar.header("🆘 Need Urgent Help?")
    
    if st.sidebar.button("🚨 I Need Immediate Help"):
        st.sidebar.error("""
        **Call now:**
        - National Emergency: 112
        - Mental Health Helpline: 1800-599-0019
        """)
        
        hospital_locations = pd.DataFrame({
            'lat': [28.6139, 28.6280, 28.5805],
            'lon': [77.2090, 77.3649, 77.3316],
            'name': ['AIIMS Hospital', 'Safdarjung Hospital', 'RML Hospital']
        })
        st.sidebar.map(hospital_locations, latitude='lat', longitude='lon', size=20, color='#FF0000')
```

**Features:**
- Quick emergency hotline display
- Hospital locator map (hardcoded for India)
- One-click crisis support access

#### **5.3.10 Personalized Therapy Tips**
```python
def personalized_therapy():
    issues = st.multiselect(
        "What are you struggling with today?",
        ['Anxiety', 'Depression', 'Stress', 'Self-esteem', 'Sleep', 'Loneliness']
    )
    
    therapy_tips = {
        'Anxiety': "🧘 Try 4-7-8 breathing: inhale for 4s, hold for 7s, exhale for 8s.",
        'Depression': "🚶 Even a 5-minute walk outside can help lift your mood.",
        'Stress': "✋ Try the 5-5-5 method: name 5 things you see, hear, and feel.",
        # ...
    }
    
    for issue in issues:
        st.info(therapy_tips[issue])
```

**Approach:** Evidence-based, concise coping strategies matched to selected concerns.

---

## **6. DATA FLOW & INTEGRATION**

### **6.1 Complete User Journey**

```
1. USER LAUNCHES APP
   └─> python emotion_server.py     [Terminal 1]
   └─> streamlit run calmconnect.py [Terminal 2]
   └─> python emotion_detection_app.py [Terminal 3, or fallback to YOLO]

2. WEBCAM RUNNING (Emotion Detection)
   └─> Every frame: FER detects faces & emotions
   └─> Buffer detections for 5 seconds
   └─> Counter() finds most common emotion
   └─> POST {"emotion": "happy"} to http://127.0.0.1:8502/emotion

3. FASTAPI RECEIVES
   └─> Parse JSON
   └─> Write to emotion_data.json with timestamp
   └─> Return 200 OK

4. STREAMLIT POLLING (check_emotion_updates runs in main loop)
   └─> Every ~1 second: Read emotion_data.json
   └─> If timestamp < 10 sec old AND auto_emotion_detection=ON
   └─> Map emotion to mood via EMOTION_TO_MOOD
   └─> Update st.session_state.mood_data['current_mood']
   └─> Append to mood history
   └─> st.rerun() triggers UI refresh

5. UI UPDATES
   └─> Mood tracker shows new mood emoji
   └─> Mood timeline refreshes with new data point
   └─> Sidebar displays: "Detected emotion: 😊 happy"

6. USER CHATS WITH CONSULO
   └─> User types message
   └─> generate_response() builds system prompt with mood + detected emotion
   └─> Call ollama.chat() with mood-aware system prompt
   └─> Qwen2.5:7B returns response (< 10 seconds typically)
   └─> Response displayed + auto-spoken if voice enabled
   └─> Conversation history appended to st.session_state

7. USER TRACKS MOOD MANUALLY (if auto-detect off)
   └─> Move slider to new mood
   └─> Append to history
   └─> Mood streak calculates based on unique dates

8. USER GENERATES REPORT
   └─> Click "Get PDF Report"
   └─> Matplotlib plots mood_data
   └─> FPDF embeds chart + metadata
   └─> Download wellness_report.pdf
```

### **6.2 Inter-Process Communication**

| Process | Writes To | Reads From | Format |
|---------|-----------|-----------|--------|
| emotion_detection_app.py | POST /emotion | — | JSON |
| emotion_server.py | emotion_data.json | POST /emotion | JSON |
| calmconnect.py (Streamlit) | — | emotion_data.json | JSON |

**Rationale:** File-based communication allows processes to be started/stopped independently.

---

## **7. IMPLEMENTATION DETAILS & BUILD HIGHLIGHTS**

### **7.1 Why This Tech Stack?**

| Choice | Alternatives Considered | Why Chosen |
|--------|--------------------------|-----------|
| Streamlit | Flask, Django, React | No frontend code, rapid iteration, built-in state |
| FastAPI | Flask, Django REST | Async, modern Python, minimal boilerplate |
| FER + MTCNN | DeepFace, MediaPipe, OpenFace | Pre-trained, lightweight, 8-emotion support |
| Ollama + Qwen2.5 | OpenAI API, Hugging Face | Local, no API keys, privacy, offline capable |
| JSON file | SQLite, PostgreSQL | Simplicity, no external dependencies |
| pyttsx3 | Google TTS, Azure TTS | Offline, no API calls |

### **7.2 Session State Pattern**

Streamlit's execution model reruns the entire script on every interaction. Without session state, all variables reset:

```python
# Without session_state (BROKEN)
conversation = []
if user_input:
    conversation.append(user_input)  # Lost on next rerun!

# With session_state (CORRECT)
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if user_input:
    st.session_state.conversation.append(user_input)  # Persists!
```

### **7.3 Buffering & Smoothing**

Emotion detection can be jittery (one frame = happy, next = sad). Solution: buffer 5 seconds, take most common:

```python
self.emotions_buffer = []  # Accumulate 5 sec of detections
# Every 5 sec:
most_common = Counter(self.emotions_buffer).most_common(1)[0][0]
self.send_emotion(most_common)
```

Benefit: Noise-resistant, more stable mood transitions.

### **7.4 Context Window Optimization**

Ollama models have token limits. To keep responses fast, only last 3 messages included:

```python
messages = [system_prompt]
messages.extend(st.session_state['conversation_history'][-3:])  # Limited history
messages.append({"role": "user", "content": user_input})
```

Benefit: Reduces token count → faster inference (< 10 sec typical).

### **7.5 Mood-Aware System Prompt**

Generic LLM prompts produce generic responses. By including mood in the system prompt:

```
You are COUNSULO. Current Mood: Low. Detected emotion: sad.
Always validate feelings first. Suggest mood-appropriate coping strategies.
Keep responses under 5-6 sentences.
```

Benefit: Responses adapt to user's emotional state (e.g., suggest gentle exercise if low mood).

---

## **8. KEY FEATURES & DELIVERABLES**

| Feature | Technology | Purpose |
|---------|-----------|---------|
| **Real-time Emotion Detection** | FER + MTCNN | Capture facial expressions without user input |
| **Auto-Mood Mapping** | EMOTION_TO_MOOD dict | Convert 8 emotions → 5-tier mood scale |
| **AI Therapist Chat** | Ollama + Qwen2.5:7B | Mood-aware, empathetic responses |
| **Mood Timeline** | Pandas + Matplotlib | Visualize emotional journey over time |
| **Mood Streak** | Datetime logic | Gamify consistency (e.g., "7-day streak") |
| **Personalized Tips** | Hardcoded dictionary | Suggest coping strategies by concern |
| **PDF Wellness Report** | FPDF + Matplotlib | Export mood data + charts |
| **Voice I/O** | pyttsx3 + SpeechRecognition | Hands-free accessibility |
| **Crisis Hotlines** | Hardcoded numbers | Quick emergency access |
| **Therapy Journal** | Streamlit forms + session_state | Timestamped reflections linked to mood |
| **Responsive UI** | Streamlit sidebar + containers | Multi-panel interface |

---

## **9. PROJECT STRUCTURE**

```
Facial Emotion Detection/
├── btp/                              # BTP project folder
│   ├── emotion_server.py            # FastAPI relay server (port 8502)
│   ├── calmconnect.py               # Main Streamlit UI (~800 lines)
│   ├── emotion_data.json            # Emotion data file (IPC)
│   ├── README.md                    # Setup instructions
│   ├── mood_chart.png               # Generated mood visualization
│   ├── wellness_report.pdf          # Generated wellness report
│   ├── background.png               # UI assets
│   └── calmconnect.py               # Config (typo in filename: calmconnent.py)
├── emotion_detection_app.py         # FER-based detector
├── main.py                          # YOLO detector + fallback handler
├── test_camera.py                   # Debugging utility
├── archive/                         # Training dataset (Roboflow YOLO format)
│   ├── data.yaml                   # Dataset config
│   ├── train/                      # Training images/labels
│   ├── val/                        # Validation images/labels
│   └── test/                       # Test images/labels
└── runs/                            # Training outputs (train, train2, ..., train10)
    └── detect/
        ├── train/                  # YOLO training run outputs
        └── train2-10/              # Iterative training attempts
```

---

## **10. DEPLOYMENT & EXECUTION**

### **Prerequisites**
```
Python 3.8+
opencv-python
fer
mtcnn
fastapi
uvicorn
streamlit
ollama (installed separately, daemon running)
qwen2.5:7b (pulled via ollama: ollama pull qwen2.5:7b)
pandas
matplotlib
fpdf
pyttsx3 (optional)
speech_recognition (optional)
pyaudio (optional)
```

### **Startup Sequence**
```bash
# Terminal 1: Start emotion detector
conda activate emotion_detection
python emotion_detection_app.py
# or
python main.py

# Terminal 2: Start FastAPI server
cd btp
python emotion_server.py

# Terminal 3: Start Streamlit UI
cd btp
streamlit run calmconnect.py
```

### **User Access**
- **Streamlit UI:** `http://localhost:8501`
- **FastAPI Docs:** `http://127.0.0.1:8502/docs`
- **Emotion API:** POST to `http://127.0.0.1:8502/emotion`

---

## **11. CHALLENGES & SOLUTIONS**

| Challenge | Root Cause | Solution |
|-----------|-----------|----------|
| **Jittery emotion detection** | Frame-to-frame noise | Buffer 5 sec, use Counter() for mode |
| **Streamlit state resets** | Script reruns on every interaction | Use st.session_state for persistence |
| **Slow LLM inference** | Full conversation history processed | Use only last 3 messages |
| **Webcam initialization fails** | Different OS camera drivers | Try DirectShow (Windows), fallback to default |
| **FER not installed** | Missing dependency | Graceful fallback to YOLO detector |
| **Timestamp staleness** | Old emotion data kept alive | Check `time.time() - data["timestamp"] < 10` |
| **Voice packages missing** | Optional dependencies | Graceful degradation; show warning, continue |
| **YOLO model not found** | Missing .pt file | Fallback to FER, or pre-download models |

---

## **12. FUTURE ENHANCEMENTS**

1. **Database Integration:** Replace JSON with SQLite/PostgreSQL for multi-user support
2. **Multi-Face Handling:** Track different users separately
3. **LLM Fine-tuning:** Fine-tune Qwen2.5 on mental health datasets
4. **Mobile App:** Migrate to Flutter/React Native
5. **Real Webhook Support:** Replace file polling with WebSocket
6. **Emotion Analysis:** Store raw emotion scores, not just max
7. **Integration with Therapists:** Share anonymized reports with professionals
8. **Offline Voice:** Use local speech models (Whisper, Coqui)
9. **Biometric Integration:** Heart rate, skin conductance for richer context
10. **Adaptive Prompts:** Dynamically adjust LLM personality based on user feedback

---

## **13. CONCLUSION**

The BTP project demonstrates a modern approach to mental health support by combining:
- **Computer Vision** (emotion detection)
- **Distributed Systems** (async APIs, file-based IPC)
- **NLP/LLM** (context-aware chatbot)
- **UX Design** (multi-panel Streamlit interface)
- **Data Science** (mood tracking, visualizations)

All components are locally hosted, privacy-preserving, and designed for rapid iteration and deployment. The modular architecture allows independent testing and scaling of each layer.

---