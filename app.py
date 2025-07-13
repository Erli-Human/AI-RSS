import gradio as gr
from news_fetcher import get_news_from_urls, save_news, load_news
import ollama
import schedule
import time

# Schedule to fetch news every 60 minutes (adjust as needed)
schedule.every(60).minutes.do(lambda: update_news())

def update_news():
    """Fetches and saves the latest news articles."""
    urls = [
        "https://www.reuters.com/technology/",
        "https://www.bbc.com/news/technology",
        "https://techcrunch.com/"
    ]
    try:
        news = get_news_from_urls(urls)
        save_news(news)
        print("News updated successfully!")
    except Exception as e:
        print(f"Error updating news: {e}")

def chat_with_ollama(message, history):
    """Chats with the Ollama model using the loaded news articles."""
    try:
        news = load_news()  # Load the stored news articles
        context = "\n".join([f"Title: {a['title']}\nSummary: {a['summary'][:200]}" for a in news]) # Limit summary length

        prompt = f"{context}\n\nUser: {message}"

        try:
            response = ollama.chat(model='datanacci-rss-model', messages=[{"role": "user", "content": prompt}])  # No API key needed
            bot_message = response['message']['content']
            history.append((message, bot_message))
            return history
        except Exception as e:
            return history + [(message, f"Error communicating with Ollama model: {e}")]

    except Exception as e:
        return history + [(message, f"Error loading news or processing chat request: {e}")]


if __name__ == "__main__":
    # Initial news fetch on startup
    update_news()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Enter your message")
        clear = gr.Button("Clear")

        def respond(message, chat_history):
            bot_message = chat_with_ollama(message, chat_history)
            return bot_message

        msg.submit(respond, [msg, chatbot], chatbot)
        clear.click(lambda: None, inputs=msg, outputs=msg)

    demo.launch()
