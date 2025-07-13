import gradio as gr
from feedparser import parse
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import schedule
import ollama  # Import Ollama here
from typing import List
import lxml
import urllib3
import certifi
import charset_normalizer
import idna

# Dummy data for RSS feeds (replace with your actual feed URLs)
RSS_FEEDS = {
    "AI News": [
        "https://www.technologyreview.com/feed/",
        "https://feeds.bbci.co.uk/news/rss/technology",
    ],
    "Machine Learning": ["https://machinelearningmastery.com/feed/"],
}

OLLAMA_MODELS = ollama.list().get('models', [])  # Get available models from Ollama


def get_ollama_models():
    """Fetches the list of available Ollama models."""
    try:
        models = ollama.list()
        model_names = [model['name'] for model in models.get('models', [])]
        return model_names
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []


def fetch_and_parse_feed(url):
    """Fetches and parses an RSS feed."""
    try:
        response = requests.get(url, timeout=10)  # Add a timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        feed = parse(response.text)
        return feed
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_article_details(entry):
    """Extracts relevant details from an RSS entry."""
    title = entry.get("title", "No Title")
    link = entry.get("link", "")
    summary = entry.get("summary", "")  # Use summary if available
    published = entry.get("published")

    if published:
        try:
            from dateutil import parser
            published_date = parser.parse(published).strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            published_date = "Date Unavailable"
    else:
        published_date = "Date Unavailable"

    return title, link, summary, published_date


def format_category_feeds_html(category_name):
    """Formats the RSS feeds for a given category into HTML."""
    if category_name not in RSS_FEEDS:
        return "<p>Category not found.</p>"

    articles_html = ""
    for feed_url in RSS_FEEDS[category_name]:
        feed = fetch_and_parse_feed(feed_url)
        if feed:
            for entry in feed.entries:
                title, link, summary, published_date = extract_article_details(entry)
                articles_html += f"""
                    <div class="article">
                        <h3><a href="{link}" target="_blank">{title}</a></h3>
                        <p>{summary}</p>
                        <small>Published: {published_date}</small>
                    </div>
                """

    return articles_html


def chat_with_feeds(chatbot, msg, category_name, model):
    """Chats with the RSS feeds using Ollama."""
    if not category_name or not RSS_FEEDS.get(category_name):
        chatbot.append((msg, "Please select a valid category."))
        return chatbot

    # Concatenate article titles and summaries for context
    context = ""
    for feed_url in RSS_FEEDS[category_name]:
        feed = fetch_and_parse_feed(feed_url)
        if feed:
            for entry in feed.entries:
                title, link, summary, _ = extract_article_details(entry)
                context += f"{title}: {summary}\n"

    # Use Ollama to generate a response
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": f"Answer the following question based on this context:\n{context}\n\nQuestion: {msg}"}])
        answer = response['message']['content']
        chatbot.append((msg, answer))
    except Exception as e:
        chatbot.append((msg, f"Error chatting with Ollama: {e}"))

    return chatbot


def create_enhanced_rss_viewer():
    """Creates the Gradio app."""
    app = gr.Interface(
        fn=None,  # No initial function; tabs handle logic
        inputs=[],
        outputs=[],
        title="Datanacci - HelixEncoder",
        description="A Gradio interface for viewing and chatting with RSS feeds.",
    )

    with app:
        with gr.Blocks():
            with gr.Tabs():
                # Dynamically create a tab for each category
                for category_name in RSS_FEEDS.keys():
                    with gr.TabItem(category_name):
                        gr.Markdown(f"### Recent Articles in {category_name}")

                        articles_html_output = gr.HTML(
                            value=format_category_feeds_html(category_name),  # Initial content
                            elem_id=f"articles_display_{category_name}"  # Unique ID for each HTML component
                        )

                        refresh_btn = gr.Button("🔄 Refresh Feeds", variant="primary")
                        refresh_btn.click(
                            fn=format_category_feeds_html,
                            inputs=[gr.State(category_name)],  # Pass the category name as a state
                            outputs=articles_html_output
                        )

                # New "Chat with RSS Feeds" Tab
                with gr.TabItem("💬 Chat with RSS Feeds"):
                    gr.Markdown("### Ask questions about the loaded RSS feeds!")
                    gr.Markdown(
                        "First, switch to any category tab to load its articles. Then you can chat here about the articles from the *currently loaded* category."
                    )

                    with gr.Row():
                        chat_category_select = gr.Dropdown(
                            choices=list(RSS_FEEDS.keys()),
                            label="Select Category for Chat (Articles from this category will be used as context)",
                            interactive=True,
                            value=list(RSS_FEEDS.keys())[0] if RSS_FEEDS else None,
                            scale=1
                        )
                        ollama_model_dropdown = gr.Dropdown(
                            choices=OLLAMA_MODELS,
                            label="Select HelixEncoder Model",
                            interactive=True,
                            value=OLLAMA_MODELS[0] if OLLAMA_MODELS else None,
                            scale=1
                        )

                    chatbot = gr.Chatbot(label="RSS Chat")
                    msg = gr.Textbox(label="Your Question", placeholder="e.g., What are the latest AI advancements?", container=False)
                    clear = gr.Button("Clear Chat")

                    msg.submit(
                        chat_with_feeds,
                        [chatbot, msg, chat_category_select, ollama_model_dropdown],
                        [chatbot, msg]
                    )
                    clear.click(lambda: None, None, chatbot, queue=False)  # Clears the chatbot

                # Settings Tab
                with gr.TabItem("⚙️ Datanacci Settings"):
                    gr.Markdown("### HelixEncoder Settings")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Feed Sources")
                            feed_count = sum(len(feeds) for feeds in RSS_FEEDS.values())
                            gr.Markdown(f"**Total Categories:** {len(RSS_FEEDS)}")
                            gr.Markdown(f"**Total Feeds:** {feed_count}")

                            for category, feeds in RSS_FEEDS.items():
                                gr.Markdown(f"**{category}:** {len(feeds)} feeds")

                        with gr.Column():
                            gr.Markdown("#### System Info")
                            gr.Markdown(f"**Last Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            gr.Markdown("**Status:** Datanacci Running")
                            gr.Markdown("**Version:** 14.11.1")
                            gr.Markdown("---")
                            gr.Markdown("#### Datanacci.HelixEncoder Status")
                            ollama_status_display = gr.HTML(label="Datanacci Server Status")

                    # Function to check Ollama status
                    def check_ollama_status():
                        try:
                            models = ollama.list()
                            num_models = len(models.get('models', []))
                            return f"<p style='color: green;'>✅ Datanacci.blockchain is Running!</p><p>Available Models: {num_models}</p>"
                        except Exception as e:
                            return f"<p style='color: red;'>❌ Datanacci Server Not Reachable. Error: {e}</p><p>Please ensure Ollama is installed and running (`ollama serve`).</p>"

                    # Update status every 5 seconds
                    schedule.every(5).seconds.do(update_ollama_status)

    return app


import schedule
import time

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


if __name__ == "__main__":
    app = create_enhanced_rss_viewer()
    app.launch()

