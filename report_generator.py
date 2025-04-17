import os
from datetime import datetime
import json
import logging
import base64
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.widgets.markers import makeMarker
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Ensure reports directory exists
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def create_conversation_summary(client, conversation_history, username="User"):
    """
    Generate a conversation summary using the LLM (Llama model).
    
    Args:
        client: The Groq client for API calls
        conversation_history: List of conversation messages
        username: The username for context
        
    Returns:
        dict: JSON summary data from the LLM
    """
    try:
        # Prepare system prompt for summary generation
        system_prompt = f"""
        You are an analysis expert tasked with evaluating a conversation between a user named {username} and SoulYatri, 
        a mental health chatbot. Please analyze the conversation and provide a comprehensive summary with the following details:
        
        1. Overall conversation summary (300-500 words)
        2. Mental state assessment of the user on a scale from -100 (extremely negative) to +100 (extremely positive)
        3. Key emotional patterns identified
        4. Primary concerns or issues discussed
        5. Notable progress or insights gained
        6. Recommendations for follow-up
        
        Format your response as valid JSON with the following structure:
        ```json
        {{
            "conversation_summary": "string",
            "mental_state_score": int,
            "mental_state_description": "string",
            "emotional_patterns": ["string", "string", ...],
            "primary_concerns": ["string", "string", ...],
            "progress_insights": ["string", "string", ...],
            "recommendations": ["string", "string", ...],
            "session_date": "string" (current date),
            "mood_timeline": [
                {{
                    "message_index": int,
                    "sentiment_value": int,
                    "notable_point": "string"
                }},
                ...
            ],
            "emotion_distribution": {{
                "joy": float,
                "sadness": float,
                "anger": float,
                "fear": float,
                "surprise": float,
                "neutral": float
            }}
        }}
        ```
        
        Ensure your analysis is compassionate, balanced, and therapeutic in nature.
        The output should be valid JSON only, with no additional text or formatting.
        """

        # Extract just the content for analysis to avoid token limits
        simplified_history = []
        for msg in conversation_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                simplified_history.append({
                    "role": msg["role"],
                    "content": msg["content"][:500]  # Truncate long messages
                })
        
        # Limit to most recent/relevant messages if too long
        if len(simplified_history) > 20:
            simplified_history = simplified_history[-20:]
            
        # Create API request
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the conversation to analyze:\n\n{json.dumps(simplified_history)}"}
                ],
                temperature=0.2,  # Lower for more consistent structured output
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse response
            summary_text = response.choices[0].message.content
            summary_data = json.loads(summary_text)
            
            # Validate required fields
            required_fields = ["conversation_summary", "mental_state_score", "mental_state_description"]
            for field in required_fields:
                if field not in summary_data:
                    summary_data[field] = "Not available"
                    
            # Add current date if missing
            if "session_date" not in summary_data:
                summary_data["session_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                
            return summary_data
            
        except Exception as api_error:
            logger.error(f"API error during summary generation: {str(api_error)}")
            # Return basic structure with error message
            return {
                "conversation_summary": "Error generating summary",
                "mental_state_score": 0,
                "mental_state_description": f"Analysis error: {str(api_error)}",
                "emotional_patterns": ["Error in analysis"],
                "primary_concerns": ["Error in analysis"],
                "progress_insights": ["Error in analysis"],
                "recommendations": ["Please try generating the report again"],
                "session_date": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
    except Exception as e:
        logger.error(f"Error in create_conversation_summary: {str(e)}")
        return {
            "conversation_summary": "Error generating summary",
            "mental_state_score": 0,
            "mental_state_description": f"System error: {str(e)}",
            "session_date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

def create_mood_chart(mood_data):
    """Create a matplotlib chart showing mood/sentiment progression as BytesIO object"""
    try:
        # Extract data
        indices = [item.get("message_index", i) for i, item in enumerate(mood_data)]
        values = [item.get("sentiment_value", 0) for item in mood_data]
        
        if not indices or not values:
            # Create empty placeholder chart if no data
            plt.figure(figsize=(7, 3))
            plt.text(0.5, 0.5, "No mood data available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.ylim(-100, 100)
            plt.xlabel("Conversation Progress")
            plt.ylabel("Sentiment")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            img_data = BytesIO()
            plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            img_data.seek(0)
            return img_data
        
        # Create the chart
        plt.figure(figsize=(7, 3))
        plt.plot(indices, values, marker='o', linestyle='-', color='#3366cc', linewidth=2)
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='#999999', linestyle='-', alpha=0.3)
        
        # Fill areas
        plt.fill_between(indices, values, 0, where=(np.array(values) >= 0), 
                        color='#99ccff', alpha=0.3)
        plt.fill_between(indices, values, 0, where=(np.array(values) < 0), 
                        color='#ffcccc', alpha=0.3)
        
        # Customize the chart
        plt.ylim(-100, 100)
        plt.xlabel("Conversation Progress")
        plt.ylabel("Sentiment")
        plt.title("Mood Progression Through Conversation")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save to BytesIO
        img_data = BytesIO()
        plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        img_data.seek(0)
        return img_data
    except Exception as e:
        logger.error(f"Error creating mood chart: {str(e)}")
        # Return a blank image
        plt.figure(figsize=(7, 3))
        plt.text(0.5, 0.5, f"Chart generation error: {str(e)}", 
                horizontalalignment='center', verticalalignment='center')
        img_data = BytesIO()
        plt.savefig(img_data, format='png', dpi=100)
        plt.close()
        img_data.seek(0)
        return img_data

def create_emotion_pie_chart(emotion_data):
    """Create a pie chart showing emotion distribution as BytesIO object"""
    try:
        # Filter out empty values
        emotion_dict = {k: v for k, v in emotion_data.items() if v > 0}
        
        if not emotion_dict:
            # Create empty placeholder chart if no data
            plt.figure(figsize=(5, 4))
            plt.text(0.5, 0.5, "No emotion data available", 
                    horizontalalignment='center', verticalalignment='center')
            img_data = BytesIO()
            plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            img_data.seek(0)
            return img_data
        
        # Color mapping for emotions
        color_map = {
            'joy': '#66cc66',      # Green
            'sadness': '#6699cc',  # Blue
            'anger': '#cc6666',    # Red
            'fear': '#cc99cc',     # Purple
            'surprise': '#cccc66', # Yellow
            'neutral': '#cccccc',  # Gray
            'disgust': '#cc9966',  # Brown
            'happy': '#66cc66',    # Same as joy
            'sad': '#6699cc'       # Same as sadness
        }
        
        # Use consistent colors for known emotions, generate for unknown
        colors = [color_map.get(emotion, f'#{hash(emotion) % 0xffffff:06x}') 
                for emotion in emotion_dict.keys()]
        
        # Create the pie chart
        plt.figure(figsize=(5, 4))
        wedges, texts, autotexts = plt.pie(
            emotion_dict.values(),
            labels=None,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors
        )
        
        # Customize text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
        
        # Add legend
        plt.legend(
            wedges,
            emotion_dict.keys(),
            title="Emotions",
            loc="center left",
            bbox_to_anchor=(0.9, 0, 0.5, 1)
        )
        
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        plt.title("Emotion Distribution")
        
        # Save to BytesIO
        img_data = BytesIO()
        plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        img_data.seek(0)
        return img_data
    except Exception as e:
        logger.error(f"Error creating emotion pie chart: {str(e)}")
        # Return a blank image
        plt.figure(figsize=(5, 4))
        plt.text(0.5, 0.5, f"Chart generation error: {str(e)}", 
                horizontalalignment='center', verticalalignment='center')
        img_data = BytesIO()
        plt.savefig(img_data, format='png', dpi=100)
        plt.close()
        img_data.seek(0)
        return img_data

def generate_report_pdf(summary_data, username, conversation_history):
    """
    Generate a PDF report based on conversation summary data
    
    Args:
        summary_data: Dictionary of summary information from the LLM
        username: Name of the user
        conversation_history: List of conversation messages
    
    Returns:
        str: Path to the generated PDF file
    """
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_username = ''.join(c for c in username if c.isalnum() or c in (' ', '_')).strip()
        if not sanitized_username:
            sanitized_username = "Anonymous"
            
        filename = f"{sanitized_username}_{timestamp}.pdf"
        file_path = os.path.join(REPORTS_DIR, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Prepare styles
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading1"]
        heading2_style = styles["Heading2"]
        normal_style = styles["Normal"]
        
        # Create custom styles
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2c3e50')
        )
        
        subheader_style = ParagraphStyle(
            'SubHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#34495e')
        )
        
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            spaceBefore=6,
            spaceAfter=6
        )
        
        score_style = ParagraphStyle(
            'Score',
            parent=styles['Normal'],
            fontSize=12,
            alignment=1,  # Center alignment
            textColor=colors.HexColor('#3498db')
        )
        
        # Start building the document
        elements = []
        
        # Add header with logo placeholder
        elements.append(Paragraph("SoulYatri Therapy", title_style))
        elements.append(Paragraph("Session Analysis Report", header_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add session info
        session_date = summary_data.get("session_date", datetime.now().strftime("%Y-%m-%d %H:%M"))
        elements.append(Paragraph(f"Client: {username}", subheader_style))
        elements.append(Paragraph(f"Session Date: {session_date}", body_style))
        elements.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add mental state score (with color coding)
        score = summary_data.get("mental_state_score", 0)
        if isinstance(score, str):
            try:
                score = int(score)
            except:
                score = 0
                
        # Determine color based on score
        if score > 50:
            score_color = colors.HexColor('#27ae60')  # Green for very positive
        elif score > 0:
            score_color = colors.HexColor('#2ecc71')  # Light green for positive
        elif score == 0:
            score_color = colors.HexColor('#7f8c8d')  # Gray for neutral
        elif score > -50:
            score_color = colors.HexColor('#e67e22')  # Orange for negative
        else:
            score_color = colors.HexColor('#e74c3c')  # Red for very negative
            
        score_style.textColor = score_color
        elements.append(Paragraph("Mental State Assessment", heading_style))
        elements.append(Paragraph(f"Score: {score} / 100", score_style))
        elements.append(Paragraph(summary_data.get("mental_state_description", "No description available"), body_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add mood chart if data available
        if "mood_timeline" in summary_data and summary_data["mood_timeline"]:
            elements.append(Paragraph("Mood Progression", heading_style))
            mood_chart = create_mood_chart(summary_data["mood_timeline"])
            if mood_chart:
                img = Image(mood_chart, width=6*inch, height=2.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.1*inch))
                
            # Add notable points from mood timeline if available
            notable_points = [item.get("notable_point") for item in summary_data["mood_timeline"] 
                             if item.get("notable_point")]
            if notable_points:
                elements.append(Paragraph("Notable Moments:", subheader_style))
                for point in notable_points[:5]:  # Limit to 5 points
                    if point and len(point) > 3:  # Skip empty or very short points
                        elements.append(Paragraph(f"• {point}", body_style))
            elements.append(Spacer(1, 0.25*inch))
        
        # Add emotion distribution chart if available
        if "emotion_distribution" in summary_data and any(summary_data["emotion_distribution"].values()):
            elements.append(Paragraph("Emotion Distribution", heading_style))
            emotion_chart = create_emotion_pie_chart(summary_data["emotion_distribution"])
            if emotion_chart:
                img = Image(emotion_chart, width=5*inch, height=3*inch)
                elements.append(img)
            elements.append(Spacer(1, 0.25*inch))
        
        # Add conversation summary
        elements.append(Paragraph("Conversation Summary", heading_style))
        summary_text = summary_data.get("conversation_summary", "No summary available")
        # Split into paragraphs for better formatting
        for paragraph in summary_text.split('\n'):
            if paragraph.strip():
                elements.append(Paragraph(paragraph, body_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add emotional patterns
        if "emotional_patterns" in summary_data and summary_data["emotional_patterns"]:
            elements.append(Paragraph("Emotional Patterns", heading_style))
            for pattern in summary_data["emotional_patterns"]:
                elements.append(Paragraph(f"• {pattern}", body_style))
            elements.append(Spacer(1, 0.25*inch))
        
        # Add primary concerns
        if "primary_concerns" in summary_data and summary_data["primary_concerns"]:
            elements.append(Paragraph("Primary Concerns", heading_style))
            for concern in summary_data["primary_concerns"]:
                elements.append(Paragraph(f"• {concern}", body_style))
            elements.append(Spacer(1, 0.25*inch))
        
        # Add progress and insights
        if "progress_insights" in summary_data and summary_data["progress_insights"]:
            elements.append(Paragraph("Progress & Insights", heading_style))
            for insight in summary_data["progress_insights"]:
                elements.append(Paragraph(f"• {insight}", body_style))
            elements.append(Spacer(1, 0.25*inch))
        
        # Add recommendations
        if "recommendations" in summary_data and summary_data["recommendations"]:
            elements.append(Paragraph("Recommendations", heading_style))
            for rec in summary_data["recommendations"]:
                elements.append(Paragraph(f"• {rec}", body_style))
            elements.append(Spacer(1, 0.25*inch))
        
        # Add conversation excerpt (last few exchanges)
        if conversation_history:
            elements.append(Paragraph("Recent Exchange Highlights", heading_style))
            # Get the last 3-5 message pairs (user-assistant)
            recent_msgs = conversation_history[-min(10, len(conversation_history)):]
            for i, msg in enumerate(recent_msgs):
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    continue
                
                role = msg["role"]
                content = msg["content"]
                
                if not content:
                    continue
                    
                # Format based on role
                if role == "user":
                    elements.append(Paragraph(f"Client: {content[:200]}{'...' if len(content) > 200 else ''}", 
                                             ParagraphStyle("user", parent=body_style, 
                                                          textColor=colors.HexColor('#2980b9'))))
                elif role == "assistant":
                    elements.append(Paragraph(f"Therapist: {content[:200]}{'...' if len(content) > 200 else ''}", 
                                             ParagraphStyle("assistant", parent=body_style, 
                                                          textColor=colors.HexColor('#16a085'))))
                # Add separator between message pairs
                if role == "assistant" and i < len(recent_msgs) - 1:
                    elements.append(Spacer(1, 0.1*inch))
        
        # Add disclaimer
        elements.append(Spacer(1, 0.5*inch))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Italic'],
            fontSize=8,
            textColor=colors.gray
        )
        disclaimer_text = (
            "DISCLAIMER: This report was generated by an AI system and should not replace professional "
            "mental health evaluation. If you're experiencing serious mental health concerns, please "
            "consult with a licensed therapist or counselor."
        )
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build PDF
        doc.build(elements)
        logger.info(f"Report generated successfully: {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return None

def should_meet(messages, client):
    """
    Determine if a meeting should be scheduled based on conversation signals.

    Args:
        messages: The chat history to analyze.
        client: Groq client for API calls.

    Returns:
        bool: True if a meeting should be scheduled, otherwise False.
    """
    # Only consider scheduling if there are sufficient exchanges
    if len(messages) < 6:
        return False

    # Extract the last few messages
    recent_msgs = messages[-6:]
    meeting_signals = [
        "schedule a meeting", "set up a meeting", "let's meet", "can we schedule a call",
        "meeting request", "let's have a meeting", "book a meeting", "arrange a meeting", 
        "follow-up", "appointment"
    ]

    # Check for meeting signals in the last user message
    for msg in reversed(recent_msgs):
        if msg.get("role") == "user":
            user_text = msg.get("content", "").lower()
            if any(signal in user_text for signal in meeting_signals):
                logger.info("Detected meeting scheduling signals in user message")
                return True
            break

    # If conversation is long enough, ask the LLM to determine if a meeting is needed
    if len(messages) >= 10:
        try:
            # Get the last 5 messages for analysis
            last_exchanges = messages[-5:]
            simplified_exchanges = []
            for msg in last_exchanges:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    simplified_exchanges.append({
                        "role": msg["role"],
                        "content": msg["content"][:200]  # Truncate long messages
                    })

            system_prompt = """
            Your task is to determine if a therapy conversation appears to indicate that scheduling a follow-up meeting 
            with a professional would be appropriate. Look for signals like:
            
            1. The user expressing a desire for a follow-up meeting.
            2. The user requesting additional support or indicating a need to continue discussions.
            3. Any language suggesting an appointment or meeting should be scheduled.
            
            Reply with ONLY 'yes' or 'no', with no additional explanation.
            """

            response = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Based on these recent exchanges, should we schedule a meeting with a professional? Reply with ONLY 'yes' or 'no'.\n\n{json.dumps(simplified_exchanges)}"}
                ],
                temperature=0.1,
                max_tokens=5
            )

            # Extract answer
            answer = response.choices[0].message.content.strip().lower()
            logger.info(f"LLM meeting decision: {answer}")

            return answer == "yes"

        except Exception as e:
            logger.error(f"Error in meeting decision: {str(e)}")
            # Default to false on errors
            return False

    return False
    

def should_generate_report(conversation_history, client):
    """
    Determine if a report should be generated based on conversation signals
    
    Args:
        conversation_history: The chat history to analyze
        client: Groq client to use for analysis
        
    Returns:
        bool: True if report should be generated
    """
    # Only consider generating if there are sufficient exchanges
    if len(conversation_history) < 6:
        return False
    
    # Extract the last few messages
    recent_msgs = conversation_history[-6:]
    closing_signals = ["goodbye", "thank you", "thanks for your help", "that's all", "talk later", 
                      "end session", "end the session", "see you", "disconnect", "until next time"]
    
    # Check for closing signals in the last user message
    for msg in reversed(recent_msgs):
        if msg.get("role") == "user":
            user_text = msg.get("content", "").lower()
            if any(signal in user_text for signal in closing_signals):
                logger.info("Detected closing signals in user message")
                return True
            break
    
    # If conversation is long enough, ask the LLM to determine if it's a good time for a report
    if len(conversation_history) >= 10:
        try:
            # Get the last 5 messages for analysis
            last_exchanges = conversation_history[-5:]
            
            # Prepare prompt to ask if conversation appears to be ending
            system_prompt = """
            Your task is to determine if a therapy conversation appears to be concluding and would be 
            appropriate for generating a session summary report. Look for signals like:
            
            1. The user expressing satisfaction with the advice received
            2. The user thanking the therapist and indicating they're done
            3. Natural conclusion of therapeutic goals
            4. Statements about implementing advice or "taking it from here"
            5. Expressions suggesting the user is ready to end the session
            
            Reply with ONLY 'yes' or 'no', with no additional explanation.
            """
            
            simplified_exchanges = []
            for msg in last_exchanges:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    simplified_exchanges.append({
                        "role": msg["role"],
                        "content": msg["content"][:200]  # Truncate long messages
                    })
            
            response = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Based on these recent exchanges, is the conversation concluding? Reply with ONLY 'yes' or 'no'.\n\n{json.dumps(simplified_exchanges)}"}
                ],
                temperature=0.1,
                max_tokens=5
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip().lower()
            logger.info(f"LLM report generation decision: {answer}")
            
            return answer == "yes"
            
        except Exception as e:
            logger.error(f"Error in report generation decision: {str(e)}")
            # Default to false on errors
            return False
    
    return False

def generate_report(client, conversation_history, username="User"):
    """
    Main function to generate a conversation report
    
    Args:
        client: The Groq client for API calls
        conversation_history: List of conversation messages
        username: The username for the report
        
    Returns:
        str: Path to generated PDF or None on failure
    """
    try:
        # Get conversation summary from LLM
        summary_data = create_conversation_summary(client, conversation_history, username)
        
        # Generate PDF report
        pdf_path = generate_report_pdf(summary_data, username, conversation_history)
        
        return pdf_path
    
    except Exception as e:
        logger.error(f"Error in generate_report: {str(e)}")
        return None
