import streamlit as st
import pandas as pd
import google.generativeai as genai
from gtts import gTTS
import io
from streamlit_mic_recorder import speech_to_text

def text_to_speech(text):
    # Convert text to audio bytes
    audio_bytes = io.BytesIO()
    tts = gTTS(text=text, lang='en')
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# ==========================================
# 1. CONFIGURATION
# ==========================================
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
SHEET_ID = st.secrets["SHEET_ID"]
GID = st.secrets["GID"]

SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. DATA MANAGEMENT (Google Sheet Connection)
# ==========================================
@st.cache_data(ttl=60)
def load_cases():
    try:
        df = pd.read_csv(SHEET_URL, on_bad_lines='skip')
        
        rename_map = {
            "Case Name": "Case_Name",
            "Hidden Diagnosis": "Hidden_Diagnosis",
            "Parent Persona": "Parent_Persona",
            "Parent_Personae": "Parent_Persona",
            "Chief Complaint": "Chief_Complaint",
            "HPI Timeline": "HPI_Timeline",
            "Symptom Visuals": "Symptom_Visuals",
            "Symptom Behavior": "Symptom_Behavior",
            "Medical History": "Medical_History",
            "Medications": "Medications",
            "Jargon Triggers": "Jargon_Triggers",
            "Lab Results": "Lab_Results",
            "Imaging Results": "Imaging_Results",
            "Correct Mgmt": "Correct_Mgmt",
            "Critical Pitfalls": "Critical_Pitfalls",
            "Educational Pearl": "Educational_Pearl"
        }
        
        df = df.rename(columns=rename_map)
        df.columns = df.columns.str.strip()

        if 'Case_Name' in df.columns:
            return df.dropna(subset=['Case_Name'])
        else:
            st.error("Error: Could not find 'Case_Name' even after cleaning. Check Sheet headers.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. THE BRAIN (System Prompt)
# ==========================================
def build_system_prompt(case):
    return f"""
    ### SYSTEM ROLE
    You are an Advanced Medical Education Simulator used to train pediatric residents.
    You must dynamically switch between three distinct roles based on the user's intent.
    
    ### ROLE 1: THE PARENT (The Default State)
    **Trigger:** When the user asks about symptoms, history, feelings, or environment for the patient.
    **Tone:** {case['Parent_Persona']}
    **Constraints:**
    1. You know ONLY what you observe.
    2. You do NOT understand medical jargon triggers: [{case['Jargon_Triggers']}]. Act confused.
    3. You never reveal the diagnosis.
    
    **Data:**
    * **CC:** "{case['Chief_Complaint']}"
    * **HPI:** {case['HPI_Timeline']}
    * **Visuals:** {case['Symptom_Visuals']}
    * **Behavior:** {case['Symptom_Behavior']}
    * **Hx:** {case['Medical_History']}
    * **Meds:** {case['Medications']}

    ### ROLE 2: THE PROCTOR / EMR
    **Trigger:** Commands like "Order Labs", "Get X-Ray", "Check Vitals". When user states they would like to perform an action like exam or test, provide data, labs or imaging pertinent to that request.
    **Action:** State "EMR Record:" and provide objective data.
    * **Labs:** State "Recent Labs:" and provide {case['Lab_Results']}
    * **Imaging:** State "Imaging:" and provide {case['Imaging_Results']}

    ### ROLE 3: THE GRADER
    **Trigger:** Diagnosis or Admission statements.
    **Action:** Break character. Provide teaching summary in a narrative form. If a pitfall is encountered, emphasize it and include proper management. Separate the pearl as a distinct discussion item.
    1. **Truth:** {case['Hidden_Diagnosis']}
    2. **Gold Standard:** {case['Correct_Mgmt']}
    3. **Pitfalls:** {case['Critical_Pitfalls']}
    4. **Pearl:** {case['Educational_Pearl']}
    
    ### IMMEDIATE INSTRUCTION
    Start by acting as the PARENT. State your Chief Complaint naturally. If non-contributory or irrelevant information is asked for during the session, provide details but be brief. 
    """

# ==========================================
# 4. THE USER INTERFACE (Streamlit)
# ==========================================
st.set_page_config(page_title="PedsSim Bot", page_icon="🩺")

st.title("🩺 Pediatric History Trainer")
st.warning("⚠️ **DISCLAIMER: Do not enter actual patient information. For simulation only.**")

df = load_cases()

if not df.empty:
    # --- Sidebar ---
    with st.sidebar:
        st.header("Case Files")
        case_names = df['Case_Name'].tolist()
        selected_case_name = st.selectbox("Choose a Patient:", case_names)
        
        if st.button("Start / Reset Case", type="primary"):
            st.session_state.messages = []
            st.session_state.current_case_data = df[df['Case_Name'] == selected_case_name].iloc[0]
            st.session_state.chat_started = True
            st.rerun()
            
        st.divider()
        
        # --- AUDIO SETTINGS ---
        st.header("Audio Settings")
        enable_audio = st.toggle("Enable Text-to-Speech Response", value=False)
        
        st.header("🎤 Voice Input")
        voice_text = speech_to_text(
            language='en',
            start_prompt="Click to Speak",
            stop_prompt="Stop Recording",
            just_once=True,
            key='STT'
        )

    # --- Chat Logic ---
    if "chat_started" not in st.session_state:
        st.info("👈 Please select a case and click 'Start Case' in the sidebar.")
    
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display History
        for message in st.session_state.messages:
            role = "user" if message["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(message["content"])

        # Input Handling
        chat_input_text = st.chat_input("Interview the parent...")

        prompt = None
        if chat_input_text:
            prompt = chat_input_text
        elif 'voice_text' in locals() and voice_text:
            prompt = voice_text

        if prompt:
            # 1. Display User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Build Brain & History
            sys_prompt = build_system_prompt(st.session_state.current_case_data)
            gemini_history = [{"role": "user", "parts": [sys_prompt]}]
            
            for msg in st.session_state.messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})

            # 3. Generate Response
            try:
                response = model.generate_content(gemini_history)
                ai_reply = response.text
                
                st.session_state.messages.append({"role": "model", "content": ai_reply})
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)

                # --- AUDIO GENERATION (Fixed Logic) ---
                if enable_audio:
                    try:
                        audio_data = text_to_speech(ai_reply)
                        st.audio(audio_data, format='audio/mp3')
                    except Exception as e:
                        st.error(f"Audio Generation Error: {e}")

            except Exception as e:
                st.error(f"AI Generation Error: {e}")

else:
    st.warning("Waiting for data... Please check your Google Sheet connection.")

