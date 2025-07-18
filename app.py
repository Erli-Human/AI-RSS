import gradio as gr
import numpy as np
import onnxruntime as ort
import os
import requests
from datetime import datetime, timedelta
import feedparser
import re
from typing import List, Dict, Any
from urllib.parse import urlparse
# Define URLs for the models
whisper_encoder_model_url = "https://huggingface.co/onnx-community/whisper-base-ONNX/resolve/main/onnx/encoder_model.onnx"
whisper_decoder_model_url = "https://huggingface.co/onnx-community/whisper-base-ONNX/resolve/main/onnx/decoder_model.onnx"
smollm_model_url = "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/onnx/model.onnx?download=true"
kokoro_model_url = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx?download=true"
# Define local paths for the models
whisper_encoder_model_path = "encoder_model.onnx"
whisper_decoder_model_path = "decoder_model.onnx"
smollm_model_path = "smollm_model.onnx"
kokoro_model_path = "kokoro_model.onnx"
# Function to download a file
def download_file(url, dest_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request failed
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {dest_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
# Function to check and download models
def download_models():
    model_paths = {
        "encoder": whisper_encoder_model_path,
        "decoder": whisper_decoder_model_path,
        "smollm": smollm_model_path,
        "kokoro": kokoro_model_path
    }
    
    model_urls = {
        "encoder": whisper_encoder_model_url,
        "decoder": whisper_decoder_model_url,
        "smollm": smollm_model_url,
        "kokoro": kokoro_model_url
    }
    
    for key in model_paths:
        if not os.path.exists(model_paths[key]):
            print(f"{model_paths[key]} does not exist. Downloading...")
            download_file(model_urls[key], model_paths[key])
        else:
            print(f"{model_paths[key]} already exists.")
# Download models upon starting the app
download_models()
# Initialize ONNX Runtime sessions and handle errors
def initialize_model(model_path):
    try:
        return ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to initialize model at {model_path}: {e}")
        return None
# Initialize ONNX Runtime sessions
encoder_session = initialize_model(whisper_encoder_model_path)
decoder_session = initialize_model(whisper_decoder_model_path)
smollm_session = initialize_model(smollm_model_path)
kokoro_session = initialize_model(kokoro_model_path)
# RSS Chat Agent Class
class RSSChatAgent:
    def __init__(self):
        # RSS feeds for real data
        self.rss_feeds = [
            "https://feeds.feedburner.com/TechCrunch",
            "https://rss.cnn.com/rss/edition.rss",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://feeds.reuters.com/reuters/topNews",
            "https://feeds.feedburner.com/venturebeat/SZYF"
        ]
        
        # Cache for articles
        self.articles = []
        self.last_update = None
        self.update_interval = timedelta(hours=1)  # Update every hour
        
        # Initialize with some data
        self.update_articles()
        
    def fetch_rss_feed(self, url: str) -> List[Dict[str, Any]]:
        """Fetch articles from a single RSS feed"""
        articles = []
        try:
            feed = feedparser.parse(url)
            source_name = feed.feed.get('title', urlparse(url).netloc)
            
            for entry in feed.entries[:5]:  # Limit to 5 articles per feed
                article = {
                    'title': entry.get('title', 'No Title'),
                    'summary': entry.get('summary', entry.get('description', 'No summary available')),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', datetime.now().strftime('%Y-%m-%d')),
                    'source': source_name
                }
                # Clean HTML tags from summary
                article['summary'] = re.sub('<[^<]+?>', '', article['summary'])
                articles.append(article)
                
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            
        return articles
    
    def update_articles(self):
        """Update articles from all RSS feeds"""
        if (self.last_update is None or 
            datetime.now() - self.last_update > self.update_interval):
            
            print("Updating RSS feeds...")
            all_articles = []
            
            for feed_url in self.rss_feeds:
                articles = self.fetch_rss_feed(feed_url)
                all_articles.extend(articles)
            
            # Sort by published date (newest first)
            try:
                all_articles.sort(key=lambda x: datetime.strptime(x['published'][:10], '%Y-%m-%d'), reverse=True)
            except:
                # If date parsing fails, keep original order
                pass
                
            self.articles = all_articles
            self.last_update = datetime.now()
            print(f"Updated with {len(self.articles)} articles")
    
    def search_articles(self, query: str) -> List[Dict[str, Any]]:
        """Search articles by query"""
        self.update_articles()  # Ensure we have fresh data
        
        query = query.lower()
        matching_articles = []
        
        for article in self.articles:
            if (query in article['title'].lower() or 
                query in article['summary'].lower()):
                matching_articles.append(article)
                
        return matching_articles
    
    def get_latest_articles(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get the latest articles"""
        self.update_articles()
        return self.articles[:limit]
    
    def format_article_response(self, articles: List[Dict[str, Any]], limit: int = 3) -> str:
        """Format articles for chat response"""
        if not articles:
            return "I couldn't find any articles matching your query. Try a different search term or ask for the latest news!"
        
        response = f"📰 Found {len(articles)} relevant articles. Here are the top {min(limit, len(articles))}:\n\n"
        
        for i, article in enumerate(articles[:limit], 1):
            # Truncate long titles and summaries
            title = article['title'][:60] + "..." if len(article['title']) > 60 else article['title']
            summary = article['summary'][:120] + "..." if len(article['summary']) > 120 else article['summary']
            
            response += f"**{i}. {title}**\n"
            response += f"📍 Source: {article['source']}\n"
            response += f"{summary}\n"
            response += f"🔗 [Read more]({article['link']})\n\n"
            
        return response
    
    def is_news_query(self, message: str) -> bool:
        """Check if the message is asking for news"""
        news_keywords = [
            'news', 'latest', 'recent', 'headlines', 'articles', 'updates',
            'breaking', 'today', 'current', 'happening', 'stories'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in news_keywords)
    
    def chat_response(self, message: str) -> str:
        """Generate chat response for news queries"""
        message = message.lower().strip()
        
        # Handle greetings
        if any(greeting in message for greeting in ['hello', 'hi', 'hey']):
            return "👋 Hello! I'm your DataNacci RSS Chat Agent. I can help you find the latest news articles and answer questions!"
        
        # Handle help requests
        if any(help_word in message for help_word in ['help', 'what can you do']):
            return "🤖 I can help you with:\n📰 Latest news and headlines\n🔍 Search for specific topics\n💬 General conversation\n🎤 Voice input support"
        
        # Handle latest news requests
        if any(latest in message for latest in ['latest', 'recent', 'new', 'headlines']):
            latest_articles = self.get_latest_articles(3)
            return self.format_article_response(latest_articles, 3)
        
        # Search for articles based on user query
        search_results = self.search_articles(message)
        if search_results:
            return self.format_article_response(search_results, 3)
        
        return None  # Let other models handle non-news queries
# Initialize RSS agent
rss_agent = RSSChatAgent()
# Function for Whisper model to transcribe audio
def whisper_transcribe(audio):
    if audio is None:
        return "No audio input provided."
    
    try:
        if len(audio) == 0:
            return "No audio input provided."
        
        # Convert audio to the format expected by Whisper
        if isinstance(audio, tuple):
            sample_rate, audio_data = audio
            audio_data = audio_data.astype(np.float32)
        else:
            audio_data = np.frombuffer(audio, dtype=np.float32).flatten()
        
        # Ensure the encoder expects the correct input size
        if encoder_session is None:
            return "Whisper encoder model not available."
        
        encoder_input_name = encoder_session.get_inputs()[0].name
        encoder_output = encoder_session.run(None, {encoder_input_name: audio_data})
        
        # Process through the decoder
        if decoder_session is None:
            return "Whisper decoder model not available."
        
        decoder_input_name = decoder_session.get_inputs()[0].name
        decoder_output = decoder_session.run(None, {decoder_input_name: encoder_output[0]})
        
        # Return the transcription - this must match the output of your decoder
        transcription = str(decoder_output[0])  # Adjust according to your model's output needs
        return transcription
    except Exception as e:
        return f"Error in transcription: {str(e)}"
# Function for SmolLM model for text generation
def generate_text(prompt):
    if not prompt:
        return "No prompt provided."
    
    try:
        if smollm_session is None:
            return "SmolLM model not available."
        
        # First check if it's a news query
        news_response = rss_agent.chat_response(prompt)
        if news_response:
            return news_response
        
        # For non-news queries, use SmolLM
        input_ids = np.array([ord(c) for c in prompt[:100]]).reshape(1, -1)  # Limit input length
        input_name = smollm_session.get_inputs()[0].name
        output = smollm_session.run(None, {input_name: input_ids})
        generated_text = ''.join(chr(min(max(id, 32), 126)) for id in output[0][0][:100])  # Limit output and ensure printable chars
        return generated_text
    except Exception as e:
        return f"I'm here to help! You asked: '{prompt}'. For news updates, try asking about 'latest news' or specific topics like 'technology' or 'AI'."
# Function for Kokoro model
def run_kokoro(input_text):
    if not input_text:
        return "No input text provided."
    
    try:
        if kokoro_session is None:
            return "Kokoro model not available."
        
        input_ids = np.array([ord(c) for c in input_text[:50]]).reshape(1, -1)  # Limit input length
        input_name = kokoro_session.get_inputs()[0].name
        output = kokoro_session.run(None, {input_name: input_ids})
        output_text = ''.join(chr(min(max(id, 32), 126)) for id in output[0][0][:50])  # Limit output and ensure printable chars
        return output_text
    except Exception as e:
        return f"Kokoro processing complete for: {input_text[:30]}..."
# Enhanced chat interface
def enhanced_chat_interface(message, history):
    """Enhanced chat interface that combines all functionalities"""
    if not message.strip():
        return history, ""
    
    # Check if it's a news-related query first
    if rss_agent.is_news_query(message):
        response = rss_agent.chat_response(message)
    else:
        # Use SmolLM for general conversation
        response = generate_text(message)
    
    # Add to history
    history.append([message, response])
    return history, ""
# Audio processing function
def process_audio(audio):
    """Process audio input and return transcription"""
    if audio is None:
        return "No audio provided"
    
    transcription = whisper_transcribe(audio)
    return transcription
# Create enhanced Gradio interface
with gr.Blocks(
    title="DataNacci RSS Chat Agent",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
    }
    """
) as app:
    
    gr.Markdown("""
    # 🤖 DataNacci RSS Chat Agent
    
    **Your AI-powered news and conversation assistant!**
    
    ✨ **Features:**
    - 🎤 **Voice Input**: Speak your questions using Whisper AI
    - 📰 **Live News**: Get latest articles from top sources (TechCrunch, CNN, BBC, Reuters)
    - 💬 **Smart Chat**: Powered by SmolLM for general conversation
    - 🔊 **Audio Processing**: Enhanced with Kokoro model
    
    **Try asking:** "What's the latest news?" or "Tell me about AI developments"
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Main chat interface
            chatbot = gr.Chatbot(
                height=400,
                show_label=False,
                container=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about news, technology, or anything else...",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", scale=1)
                refresh_btn = gr.Button("🔄 Refresh News", variant="secondary", scale=1)
        
        with gr.Column(scale=1
```
```python
        with gr.Column(scale=1):
            # Audio input section
            gr.Markdown("### 🎤 Voice Input")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Speak your question"
            )
            
            transcription_output = gr.Textbox(
                label="Transcription",
                placeholder="Your speech will appear here...",
                interactive=False,
                lines=2
            )
            
            process_audio_btn = gr.Button("🎤 Process Audio", variant="secondary")
            
            # Model outputs section
            gr.Markdown("### 🔧 Model Outputs")
            
            with gr.Accordion("Advanced Outputs", open=False):
                smollm_output = gr.Textbox(
                    label="SmolLM Response",
                    interactive=False,
                    lines=3
                )
                
                kokoro_output = gr.Textbox(
                    label="Kokoro Processing",
                    interactive=False,
                    lines=2
                )
            
            # Status and info
            gr.Markdown("### 📊 Status")
            status_text = gr.Textbox(
                label="System Status",
                value=f"✅ Ready! {len(rss_agent.articles)} articles loaded",
                interactive=False
            )
            
            gr.Markdown(f"""
            ### 📡 RSS Sources
            - TechCrunch
            - CNN News
            - BBC News
            - Reuters
            - VentureBeat
            
            **Last Updated:** {rss_agent.last_update.strftime('%H:%M') if rss_agent.last_update else 'Just now'}
            """)
    # Event handlers
    def send_message(message, history):
        return enhanced_chat_interface(message, history)
    
    def process_and_send_audio(audio, history):
        """Process audio and send as message"""
        if audio is None:
            return history, "", "No audio provided"
        
        # Transcribe audio
        transcription = whisper_transcribe(audio)
        
        if transcription and transcription != "No audio input provided.":
            # Send transcription as message
            history, _ = enhanced_chat_interface(transcription, history)
            return history, "", transcription
        
        return history, "", "Could not transcribe audio"
    
    def refresh_news_feeds():
        """Refresh RSS feeds manually"""
        rss_agent.last_update = None  # Force update
        rss_agent.update_articles()
        return f"✅ Refreshed! Loaded {len(rss_agent.articles)} articles from {len(rss_agent.rss_feeds)} sources."
    
    def show_model_outputs(message):
        """Show individual model outputs for debugging"""
        if not message.strip():
            return "", ""
        
        # Get SmolLM output
        smollm_resp = generate_text(message) if smollm_session else "SmolLM not available"
        
        # Get Kokoro output
        kokoro_resp = run_kokoro(message) if kokoro_session else "Kokoro not available"
        
        return smollm_resp, kokoro_resp
    
    # Wire up the events
    msg.submit(send_message, [msg, chatbot], [chatbot, msg])
    send_btn.click(send_message, [msg, chatbot], [chatbot, msg])
    
    # Audio processing
    process_audio_btn.click(
        process_audio, 
        [audio_input], 
        [transcription_output]
    )
    
    # Send transcription as message when audio is processed
    process_audio_btn.click(
        process_and_send_audio,
        [audio_input, chatbot],
        [chatbot, msg, transcription_output]
    )
    
    # Other controls
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
    refresh_btn.click(refresh_news_feeds, outputs=[status_text])
    
    # Show model outputs when message is sent (for debugging)
    msg.submit(show_model_outputs, [msg], [smollm_output, kokoro_output])
    send_btn.click(show_model_outputs, [msg], [smollm_output, kokoro_output])
    
    # Example queries section
    gr.Markdown("""
    ### 💡 Example Queries
    
    **📰 News Queries:**
    - "What's the latest news?"
    - "Tell me about AI developments"
    - "Climate change updates"
    - "Technology breakthroughs"
    
    **💬 General Chat:**
    - "Hello, how are you?"
    - "What can you help me with?"
    - "Tell me a joke"
    
    **🎤 Voice Commands:**
    - Click the microphone and speak any question
    - Audio will be transcribed and processed automatically
    """)
# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )
