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

--- Constants ---
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

--- ONNX Model URLs and Paths ---
GPT2_MODEL_URL = "https://huggingface.co/onnx/models/gpt2/resolve/main/model.onnx?download=true"
GPT2_MODEL_PATH = "gpt2_model.onnx"

def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        return
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"Downloaded {dest_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)

def initialize_onnx_session(model_path: str):
    if not os.path.exists(model_path):
        return None
    try:
        return ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to initialize ONNX session for {model_path}: {e}")
        return None

Download and initialize
download_file(GPT2_MODEL_URL, GPT2_MODEL_PATH)
gpt2_session = initialize_onnx_session(GPT2_MODEL_PATH)

--- Data Models & Persistence ---
@dataclass
class Article:
    title: str; link: str; published: str; summary: str; feed_name: str
    author: str = ""; fetched_at: str = datetime.utcnow().isoformat()

def load_json(path: str, default: list = []) -> list:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        return data if isinstance(data, list) else default
    except json.JSONDecodeError:
        return default

def save_json(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def initialize_config() -> List[Dict[str, Any]]:
    cfg = load_json(CONFIG_PATH)
    existing_urls = {f["url"] for f in cfg if isinstance(f, dict)}
    changed = False
    for cat, feeds in RSS_FEEDS.items():
        for name, url in feeds.items():
            if url not in existing_urls:
                cfg.append({
                    "category": cat,
                    "feed_name": name,
                    "url": url,
                    "created": datetime.utcnow().isoformat(),
                    "key": f"{cat}_{name}"
                })
                changed = True
    cfg = [f for f in cfg if isinstance(f, dict)]
    if changed:
        save_json(CONFIG_PATH, cfg)
    return cfg

--- RSS Logic ---
def fetch_single_feed(url: str, feed_name: str, timeout: int = 10) -> List[Article]:
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=timeout)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        return [
            Article(
                title=e.get("title","No Title"),
                link=e.get("link",""),
                published=e.get("published","Unknown"),
                summary=e.get("summary","")[:300]+"...",
                author=e.get("author",""),
                feed_name=feed_name
            )
            for e in feed.entries
        ]
    except:
        return []

def update_article_history() -> str:
    cfg = load_json(CONFIG_PATH)
    history = load_json(HISTORY_PATH)
    links = {a["link"] for a in history if isinstance(a, dict)}
    new_count = 0
    with ThreadPoolExecutor(max_workers=8) as exe:
        fut2feed = {exe.submit(fetch_single_feed, f["url"], f["feed_name"]): f for f in cfg if isinstance(f, dict)}
        for fut in as_completed(fut2feed):
            for art in fut.result():
                if art.link not in links:
                    history.append(asdict(art))
                    links.add(art.link)
                    new_count += 1
    if new_count:
        history.sort(key=lambda x: x.get("published",""), reverse=True)
        save_json(HISTORY_PATH, history)
        return f"✅ Found {new_count} new articles. Total: {len(history)}."
    return f"ℹ️ No new articles. Total: {len(history)}."

--- ONNX Text Generation ---
def generate_text(prompt: str) -> str:
    if not gpt2_session or not prompt:
        return "Model not ready or empty prompt."
    # Placeholder tokenizer: not for production
    ids = np.array([ord(c) for c in prompt if ord(c)<50257], dtype=np.int64).reshape(1,-1)
    try:
        name = gpt2_session.get_inputs()[0].name
        out = gpt2_session.run(None, {name: ids})
        return "".join(chr(i) for i in out[0][0] if i<256)
    except Exception as e:
        return f"Error: {e}"

--- Gradio UI ---
def create_app():
    def chat(history: List[Dict[str,str]], query: str) -> Tuple[List[Dict[str,str]], None]:
        if not query.strip():
            return history, None
        ctx = load_json(HISTORY_PATH)
        sys = {"role":"system","content":f"CONTEXT:{json.dumps(ctx)[:1000]}..."}
        full = sys["content"] + "\n\n"
        for h in history:
            full += f"{h['role']}: {h['content']}\n"
        full += f"user: {query}\nassistant:"
        resp = generate_text(full)
        history.append({"role":"user","content":query})
        history.append({"role":"assistant","content":resp})
        return history, None

    with gr.Blocks() as app:
        gr.Markdown("# Datanacci RSS with ONNX GPT2")
        with gr.Tabs():
            with gr.TabItem("History"):
                btn = gr.Button("Fetch RSS")
                status = gr.Markdown()
                df = gr.Dataframe(value=pd.DataFrame(load_json(HISTORY_PATH)), interactive=False)
                def refresh():
                    s = update_article_history()
                    return s, pd.DataFrame(load_json(HISTORY_PATH))
                btn.click(refresh, outputs=[status, df])
            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(type="messages", value=[])
                txt = gr.Textbox(placeholder="Ask anything...")
                clr = gr.Button("Clear")
                txt.submit(chat, [chatbot, txt], [chatbot, txt])
                clr.click(lambda: [], None, chatbot)
    return app

if name=="main":
    initialize_config()
    create_app().launch()
