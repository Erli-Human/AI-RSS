import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import gradio as gr
import feedparser
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import onnxruntime as ort

# --- Constants ---
CONFIG_PATH = "rss_config.json"
HISTORY_PATH = "article_history.json"
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "OpenAI Blog": "https://openai.com/blog/rss.xml",
        "Hugging Face Blog": "https://huggingface.co/blog/feed.xml"
    },
    "🚨 Breaking News": {
        "Reuters Top News": "http://feeds.reuters.com/reuters/topNews",
        "Associated Press": "https://apnews.com/hub/ap-top-news/rss"
    },
    "🌍 World News": {
        "Reuters World News": "http://feeds.reuters.com/Reuters/worldNews",
        "BBC World News": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Global News": "https://globalnews.ca/feed/"
    },
    "💻 Technology": {
        "TechCrunch": "https://techcrunch.com/feed/",
        "Wired": "https://www.wired.com/feed/rss"
    },
    "⚽ Sports": {
        "ESPN": "https://www.espn.com/espn/rss/news",
        "Olympic News": "https://olympics.com/en/rss/"
    },
    "💼 Business": {
        "Financial Times": "https://www.ft.com/rss/home",
        "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss"
    }
}

# --- ONNX Model: Switched to a public GPT-2 model that doesn't require login ---
GPT2_MODEL_URL = "https://huggingface.co/onnx-community/gpt2-medium-onnx/resolve/main/gpt2-medium-10.onnx?download=true"
GPT2_MODEL_PATH = "gpt2_medium.onnx"

# --- Model Download and Initialization ---

def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists.")
        return
    print(f"{dest_path} not found. Downloading...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {dest_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)

def initialize_onnx_session(model_path: str) -> ort.InferenceSession:
    try:
        return ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to initialize ONNX session for {model_path}: {e}")
        return None

download_file(GPT2_MODEL_URL, GPT2_MODEL_PATH)
gpt2_session = initialize_onnx_session(GPT2_MODEL_PATH)

# --- Data Models & JSON Persistence ---
@dataclass
class Article:
    title: str; link: str; published: str; summary: str; feed_name: str
    author: str = ""; fetched_at: str = datetime.utcnow().isoformat()

def load_json(path: str, default: list = []) -> list:
    if not os.path.exists(path) or os.path.getsize(path) == 0: return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure data is a list
            return data if isinstance(data, list) else default
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {path}. Returning default.")
        return default

def save_json(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def initialize_config() -> List[Dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    # FIX: Check if entry is a dict before accessing keys to prevent crashes
    existing_urls = {feed['url'] for feed in config if isinstance(feed, dict)}
    
    new_feeds_added = False
    for cat, feeds in RSS_FEEDS.items():
        for name, url in feeds.items():
            if url not in existing_urls:
                config.append({
                    "category": cat, "feed_name": name, "url": url,
                    "created": datetime.utcnow().isoformat(), "key": f"{cat}_{name}"
                })
                new_feeds_added = True

    # Ensure config only contains dictionaries before saving
    clean_config = [feed for feed in config if isinstance(feed, dict)]
    if new_feeds_added or len(clean_config) != len(config):
        save_json(CONFIG_PATH, clean_config)
        
    return clean_config

# --- Core RSS Logic ---
def fetch_single_feed(url: str, feed_name: str, timeout: int = 10) -> List[Article]:
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=timeout)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        if feed.bozo: print(f"Warning: Feed {feed_name} malformed. {feed.bozo_exception}")
        return [Article(title=e.get("title","No Title"), link=e.get("link",""),
                        published=e.get("published","Unknown"), summary=e.get("summary","")[:300]+"...",
                        author=e.get("author",""), feed_name=feed_name) for e in feed.entries]
    except Exception as ex:
        print(f"Error fetching {feed_name}: {ex}")
        return []

def update_article_history() -> str:
    config = load_json(CONFIG_PATH)
    history = load_json(HISTORY_PATH)
    existing_links = {a['link'] for a in history if isinstance(a, dict)}
    new_articles_found = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_feed = {executor.submit(fetch_single_feed, f['url'], f['feed_name']): f for f in config if isinstance(f, dict)}
        for future in as_completed(future_to_feed):
            for article in future.result():
                if article.link not in existing_links:
                    history.append(asdict(article))
                    existing_links.add(article.link)
                    new_articles_found += 1
    
    if new_articles_found > 0:
        history.sort(key=lambda x: x.get('published', ''), reverse=True)
        save_json(HISTORY_PATH, history)
        return f"✅ Found {new_articles_found} new articles. Total: {len(history)}."
    return f"ℹ️ No new articles found. Total: {len(history)}."

# --- ONNX Text Generation ---
def generate_text_from_onnx(prompt: str) -> str:
    if not gpt2_session: return "GPT-2 ONNX model not initialized."
    if not prompt: return "No prompt."
    
    # Placeholder for a real tokenizer
    input_ids = np.array([ord(c) for c in prompt if ord(c) < 50257], dtype=np.int64).reshape(1, -1) # Using GPT-2 vocab size
    
    try:
        input_name = gpt2_session.get_inputs()[0].name
        output = gpt2_session.run(None, {input_name: input_ids})
        # Placeholder for real decoding
        return ''.join(chr(id) for id in output[0][0] if id < 256)
    except Exception as e: return f"Error during ONNX inference: {e}"

# --- Gradio UI ---
def create_app():
    def chat_with_history(history: List[Dict[str, str]], query: str) -> Tuple[List[Dict[str, str]], None]:
        if not query.strip(): return history, None
        context = load_json(HISTORY_PATH)
        system_prompt = f"CONTEXT: {json.dumps(context, indent=2)}"
        full_prompt = system_prompt + "\n\n"
        for h in history: full_prompt += f"{h['role']}: {h['content']}\n"
        full_prompt += f"user: {query}\nassistant:"
        
        resp = generate_text_from_onnx(full_prompt)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": resp})
        return history, None

    with gr.Blocks(title="Datanacci RSS", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📰 Datanacci RSS App with Local ONNX")

        with gr.Tabs():
            with gr.TabItem("📖 Article History"):
                fetch_button = gr.Button("Fetch All RSS Feeds", variant="primary")
                fetch_status = gr.Markdown("Click to update article history.")
                history_df = gr.Dataframe(value=pd.DataFrame(load_json(HISTORY_PATH)), interactive=False, height=600, wrap=True)
                
                def refresh_ui():
                    status = update_article_history()
                    df = pd.DataFrame(load_json(HISTORY_PATH))
                    return status, df
                fetch_button.click(fn=refresh_ui, outputs=[fetch_status, history_df])

            with gr.TabItem("💬 Chat with History (RAG)"):
                gr.Markdown("## Chat using Local GPT-2 ONNX Model")
                gr.Markdown("⚠️ **Warning**: Model tokenization is a placeholder. A proper tokenizer is required for valid results.")
                chatbot = gr.Chatbot(type="messages", value=[], height=600)
                ask_textbox = gr.Textbox(placeholder="Enter your question...", label="Your Question")
                clear_button = gr.Button("Clear Chat")
                ask_textbox.submit(chat_with_history, [chatbot, ask_textbox], [chatbot, ask_textbox])
                clear_button.click(lambda: [], None, chatbot)
            
            with gr.TabItem("🛠️ Configurations"):
                gr.Markdown("## RSS Feed Configuration (`rss_config.json`)")
                config_df = gr.Dataframe(value=pd.DataFrame(initialize_config()), interactive=True, wrap=True)
                def save_config_df(data: pd.DataFrame):
                    save_json(CONFIG_PATH, data.to_dict('records'))
                    gr.Info("Configuration saved!")
                config_df.change(save_config_df, config_df, None)

    return app

if __name__ == "__main__":
    initialize_config()
    app = create_app()
    app.launch()
