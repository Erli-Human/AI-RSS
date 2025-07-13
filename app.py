import gradio as gr
from news_fetcher import get_news_from_urls
from news_store import load_news, save_news
import ollama
import time

# Define RSS feed URLs (you can make this configurable in the UI)
RSS_URLS = [
    "https://www.bbc.com/news/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml"
]


def refresh_news():
    """Refreshes news articles from RSS feeds and saves them."""
    try:
        articles = get_news_from_urls(RSS_URLS)
        save_news(articles)
        return "News refreshed successfully!"
    except Exception as e:
        return f"Error refreshing news: {e}"

def chat_with_ollama(message, history):
    """Chats with the Ollama model using the loaded news articles."""
    try:
        news = load_news()  # Load the stored news articles
        context = "\n".join([f"Title: {a['title']}\nSummary: {a['summary'][:200]}" for a in news]) # Limit summary length

        prompt = f"{context}\n\nUser: {message}"

        try:
            response = ollama.chat(model='datanacci-rss-model', messages=[{"role": "user", "content": prompt}])
            bot_message = response['message']['content']
            history.append((message, bot_message))
            return history
        except Exception as e:
            return history + [(message, f"Error communicating with Ollama model: {e}")]

    except Exception as e:
        return history + [(message, f"Error loading news or processing chat request: {e}")]


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your message")
    refresh_button = gr.Button("Refresh News")
    clear_button = gr.ClearButton([msg, chatbot])

    refresh_button.click(refresh_news)
    msg.submit(chat_with_ollama, [msg, chatbot], chatbot)


if __name__ == '__main__':
    demo.launch()
