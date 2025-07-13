import gradio as gr
from feedparser import parseRSSFeed
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import dateutil.parser
import ollama
import aiohttp
import typing
import urllib3
import certifi
import charset_normalizer
import idna
import schedule

# Define your RSS feeds here
RSS_FEEDS = {
    "Tech": "https://www.theverge.com/rss/index.xml",
    "AI": "https://ai.googleblog.com/rss",
    "Blockchain": "https://blockonomi.com/feed/",
}

def format_category_feeds_html(category_name):
    """Formats the RSS feeds for a given category into an HTML string."""
    if category_name not in RSS_FEEDS:
        return "<p>Category not found.</p>"

    try:
        feed = parseRSSFeed(RSS_FEEDS[category_name])
        html = "<h2>" + category_name + "</h2><ul>"
        for entry in feed.entries[:5]:  # Limit to 5 entries for brevity
            title = entry.title
            link = entry.link
            summary = entry.summary
            published = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else "Date Unavailable"

            html += f'<li><a href="{link}" target="_blank">{title}</a> - {published}<br>{summary}</li>'
        html += "</ul>"
        return html
    except Exception as e:
        return f"<p>Error fetching feeds for {category_name}: {e}</p>"

def chat_with_feeds(chatbot, msg, category_name, ollama_model):
    """Chats with the selected RSS feed using Ollama."""
    try:
        feed = parseRSSFeed(RSS_FEEDS[category_name])
        context = ""
        for entry in feed.entries[:5]:  # Use only 5 entries for context
            context += f"Title: {entry.title}\nSummary: {entry.summary}\n\n"

        if not context:
            return chatbot + [[None, "No articles found in this category."]], ""

        try:
            response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": f"Summarize the following news articles and answer my question:\n{context}\nQuestion: {msg}"}])
            answer = response['message']['content']
            chatbot += [[msg, answer]]
        except Exception as e:
            chatbot += [[msg, f"Error communicating with Ollama: {e} Please ensure Ollama is running (`ollama serve`)."]]

        return chatbot, ""  # Return the updated chatbot and clear the input box

    except Exception as e:
        chatbot += [[msg, f"Error fetching or processing feeds: {e}"]]
        return chatbot, ""


def update_ollama_status():
    """Updates the Ollama status display in the Gradio interface."""
    try:
        models = ollama.list()
        num_models = len(models.get('models', []))
        status_html = f"<p style='color: green;'>✅ Datanacci.blockchain is Running!</p><p>Available Models: {num_models}</p>"
    except Exception as e:
        status_html = f"<p style='color: red;'>❌ Datanacci Server Not Reachable. Error: {e}</p><p>Please ensure Ollama is installed and running (`ollama serve`).</p>"

    # Find the HTML element in the Gradio interface and update its content
    for component in app.get_components():
        if isinstance(component, gr.HTML) and "Datanacci Server Status" in component.label:
            component.value = status_html
            break


def create_enhanced_rss_viewer():
    with gr.Blocks() as app:
        gr.Markdown("# Datanacci RSS Chat Agent")

        with gr.TabItem("Chat with Feeds"):
            chat_category_select = gr.Dropdown(choices=list(RSS_FEEDS.keys()), label="Select Category", value=list(RSS_FEEDS.keys())[0] if list(RSS_FEEDS.keys()) else None)

            with gr.Row():
                chatbot = gr.Chatbot(label="RSS Chat")  # Output component!
                msg = gr.Textbox(label="Your Question", placeholder="e.g., What are the latest AI advancements?") # Input Component!

            clear = gr.Button("Clear Chat")

            ollama_model_select = gr.Dropdown(choices=[m['name'] for m in ollama.list()['models']], label="Select Ollama Model", value=list(ollama.list()['models'])[0]['name'] if list(ollama.list()['models']) else "mistral")

            msg.submit(chat_with_feeds, [chatbot, msg, chat_category_select, gr.State(ollama_model_select)], [chatbot, msg]) # Pass ollama model as state
            clear.click(lambda: None, None, chatbot, queue=False)


        with gr.TabItem("Settings"):
            status_html = "<p>Loading status...</p>"  # Initial status message
            status_display = gr.HTML(label="Datanacci Server Status", value=status_html)

        schedule.every(10).seconds.do(update_ollama_status)


    return app  # Return the interface object!



if __name__ == "__main__":
    app = create_enhanced_rss_viewer()
    app.launch()
