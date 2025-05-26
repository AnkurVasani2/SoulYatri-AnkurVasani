# Soulyatri - Ankur Vasani

1. Data Collection & Analysis 📥🔍  
   - It gathers data from text, voice, and facial expressions.  
   - The system transcribes voice inputs and analyzes the tone, while computer vision assesses facial emotions like joy, sadness, anger, and surprise.  
   - All the captured emotions, along with the transcriptions, are fed into the Llama model to generate an in-depth understanding of the conversation.

2. Empathetic Voice-Based Chatbot 🎤💬  
   - The prototype features an empathetic voice-based chatbot that interacts in both Hindi and English.  
   - This makes the tool more accessible and comforting, allowing users to communicate in the language they’re most comfortable with.

3. Severe Situation Detection & Emergency Response 🚨
   - It automatically detects critical situations by analyzing the emotional cues and conversation context.  
   - If a severe condition is identified, it can connect the patient with local support bodies and mental health professionals.
   - The system can even generate real-time Google Meet links to facilitate immediate online consultations with a doctor, and it sends these links along with the patient’s situation details to both the doctor and in the chat.

4. Comprehensive Reporting 📄📊 
   - After each session, a detailed report is generated that includes:  
     - A comprehensive summary of the conversation.  
     - Graphs and charts showing mood progression and emotion distribution.  
     - Insights from text, voice, and face emotion analysis.
   - This report is automatically emailed to both the concerned authorities and the patient, ensuring everyone stays informed.

For main app:
```
streamlit run myapp.py
```

For conversational app:
```
python conversation.py
```
