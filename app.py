import streamlit as st
import pandas as pd
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION
# ==========================================
# We grab these from the "Secrets" tab in Streamlit Cloud (NOT hardcoded!)
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
        
        # --- THE CLEANING MAP ---
        # Keeps your app robust against Google Form naming changes
        rename_map = {
            "Case Name": "Case_Name",
            "Hidden Diagnosis": "Hidden_Diagnosis",
            "Parent Persona": "Parent_Persona",
            "Parent_Personae": "Parent_Persona", # Catches that specific typo
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
        
        # Apply the renaming and strip whitespace
        df = df.rename(columns=rename_map)
        df.columns = df.columns.str.strip()

        # Final Verification
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
    """
    This function takes a single row of data (a case) and constructs the 
    instruction manual for Gemini.
    """
    return f"""
    ### SYSTEM ROLE
    You are an Advanced Medical Education Simulator used to train pediatric residents.
    You must dynamically switch between three distinct roles based on the user's intent.

    ---
    
    ### ROLE 1: THE PARENT (The Default State)
    **Trigger:** When the user asks about symptoms, history, feelings, or observes the patient.
    **Tone:** {case['Parent_Persona']}
    **Knowledge Constraints:**
    1. You know ONLY what you observe.
    2. You do NOT understand medical jargon. If the user says words like: [{case['Jargon_Triggers']}], you must act confused or ask "What does that mean?"
    3. You never reveal the diagnosis.
    
    **Your Script Data:**
    * **Chief Complaint:** "{case['Chief_Complaint']}"
    * **HPI (The Story):** {case['HPI_Timeline']}
    * **Visual Symptoms:** {case['Symptom_Visuals']}
    * **Behavioral Symptoms:** {case['Symptom_Behavior']}
    * **Medical Hx:** {case['Medical_History']}
    * **Meds:** {case['Medications']}

    ---

    ### ROLE 2: THE PROCTOR / EMR (The Data Provider)
    **Trigger:** When the user commands to "Order Labs", "Get X-Ray", "Check Vitals", or "Start IV".
    **Action:** Immediately break character. State "EMR DATA:" and provide the objective data below. Do not add parent commentary.
    
    **Clinical Data:**
    * **Lab Results:** {case['Lab_Results']}
    * **Imaging/Workup:** {case['Imaging_Results']}

    ---

    ### ROLE 3: THE GRADER (The Feedback Loop)
    **Trigger:** When the user states a Final Diagnosis or Disposition (e.g., "I am admitting for...", "The diagnosis is...").
    **Action:** Break character completely. Provide a teaching summary.
    
    **Grading Rubric:**
    1. **Compare** their diagnosis to the Truth: {case['Hidden_Diagnosis']}
    2. **Compare** their plan to the Gold Standard: {case['Correct_Mgmt']}
    3. **Reveal Pitfalls:** {case['Critical_Pitfalls']}
    4. **Teaching Pearl:** {case['Educational_Pearl']}

    ---
    
    ### IMMEDIATE INSTRUCTION
    The student has just entered the room. Start by acting as the PARENT. 
    State your Chief Complaint naturally.
    """

# ==========================================
# 4. THE USER INTERFACE (Streamlit)
# ==========================================
st.set_page_config(page_title="PedsSim Bot", page_icon="🩺")

st.title("🩺 Pediatric History Trainer")

# --- DISCLAIMER ADDED HERE ---
st.warning("⚠️ **DISCLAIMER: Do not enter actual patient information. For simulation only.**")
# -----------------------------

st.markdown("Select a case below to begin the simulation.")

# Load Data
df = load_cases()

if not df.empty:
    # --- Sidebar for Case Selection ---
    with st.sidebar:
        st.header("Case Files")
        case_names = df['Case_Name'].tolist()
        selected_case_name = st.selectbox("Choose a Patient:", case_names)
        
        # Button to Start/Restart
        if st.button("Start / Reset Case", type="primary"):
            st.session_state.messages = []
            st.session_state.current_case_data = df[df['Case_Name'] == selected_case_name].iloc[0]
            st.session_state.chat_started = True
            st.rerun()

    # --- Chat Logic ---
    if "chat_started" not in st.session_state:
        st.info("👈 Please select a case and click 'Start Case' in the sidebar.")
    
    else:
        # Initialize chat history if needed
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages
        for message in st.session_state.messages:
            role = "user" if message["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Interview the parent..."):
            
            # 1. Display User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Build the System Prompt (The Brain)
            # We rebuild this every turn to ensure the AI always remembers its rules
            sys_prompt = build_system_prompt(st.session_state.current_case_data)
            
            # 3. Construct the full conversation for Gemini
            # We put the System Prompt *first* in the history we send to the model
            gemini_history = [{"role": "user", "parts": [sys_prompt]}]
            
            # Append the rest of the conversation
            for msg in st.session_state.messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})

            # 4. Generate & Display AI Response
            try:
                response = model.generate_content(gemini_history)
                ai_reply = response.text
                
                st.session_state.messages.append({"role": "model", "content": ai_reply})
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)
            except Exception as e:
                st.error(f"AI Error: {e}")

else:
    st.warning("Waiting for data... Please check your Google Sheet connection.")