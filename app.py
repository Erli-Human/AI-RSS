import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import gradio as gr
import feedparser
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import onnxruntime as ort

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

GPT2_MODEL_PATH = "gpt2_model.onnx"
GPT2_SESSION = None
if os.path.exists(GPT2_MODEL_PATH):
    try:
        GPT2_SESSION = ort.InferenceSession(GPT2_MODEL_PATH)
    except:
        GPT2_SESSION = None

@dataclass
class Article:
    title: str
    link: str
    published: str
    summary: str
    feed_name: str
    author: str = ""
    fetched_at: str = datetime.utcnow().isoformat()

def load_json(path: str, default=[]):
    if not os.path.exists(path) or os.path.getsize(path)==0:
        return default
    try:
        data = json.load(open(path,"r",encoding="utf-8"))
        return data if isinstance(data,list) else default
    except:
        return default

def save_json(path: str, data):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2)

def init_config():
    cfg = load_json(CONFIG_PATH)
    urls = {f["url"] for f in cfg if isinstance(f,dict)}
    updated = False
    for cat, feeds in RSS_FEEDS.items():
        for name, url in feeds.items():
            if url not in urls:
                cfg.append({
                    "category": cat,
                    "feed_name": name,
                    "url": url,
                    "created": datetime.utcnow().isoformat(),
                    "key": f"{cat}_{name}"
                })
                updated = True
    cfg = [f for f in cfg if isinstance(f,dict)]
    if updated:
        save_json(CONFIG_PATH, cfg)
    return cfg

def fetch_feed(url, name):
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        return [
            Article(
                title=e.get("title","No title"),
                link=e.get("link",""),
                published=e.get("published","Unknown"),
                summary=e.get("summary","")[:300]+"...",
                author=e.get("author",""),
                feed_name=name
            ) for e in feed.entries
        ]
    except:
        return []

def update_history():
    cfg = load_json(CONFIG_PATH)
    history = load_json(HISTORY_PATH)
    links = {a["link"] for a in history if isinstance(a,dict)}
    new = 0
    with ThreadPoolExecutor(max_workers=8) as exe:
        fut2 = {exe.submit(fetch_feed,f["url"],f["feed_name"]):f for f in cfg if isinstance(f,dict)}
        for fut in as_completed(fut2):
            for art in fut.result():
                if art.link not in links:
                    history.append(asdict(art))
                    links.add(art.link)
                    new += 1
    if new:
        history.sort(key=lambda x: x.get("published",""), reverse=True)
        save_json(HISTORY_PATH, history)
        return f"✅ {new} new articles. Total {len(history)}."
    return f"ℹ️ No new articles. Total {len(history)}."

def generate_text(prompt: str) -> str:
    if not GPT2_SESSION or not prompt:
        return "Model unavailable or empty prompt."
    ids = np.array([ord(c) for c in prompt if ord(c)<50257], dtype=np.int64).reshape(1,-1)
    try:
        inp = GPT2_SESSION.get_inputs()[0].name
        out = GPT2_SESSION.run(None, {inp: ids})
        return "".join(chr(i) for i in out[0][0] if i<256)
    except Exception as e:
        return f"Error: {e}"

def create_app():
    def chat(history: List[Dict[str,str]], query: str) -> Tuple[List[Dict[str,str]],None]:
        if not query.strip(): return history, None
        ctx = load_json(HISTORY_PATH)
        sys = {"role":"system","content":f"CONTEXT:{json.dumps(ctx)[:1000]}..."}
        full = sys["content"]+"\n"
        for h in history:
            full += f"{h['role']}: {h['content']}\n"
        full += f"user: {query}\nassistant:"
        r = generate_text(full)
        history.append({"role":"user","content":query})
        history.append({"role":"assistant","content":r})
        return history, None

    with gr.Blocks() as app:
        gr.Markdown("# Datanacci RSS with ONNX GPT2")
        with gr.Tabs():
            with gr.TabItem("History"):
                btn = gr.Button("Fetch RSS")
                status = gr.Markdown()
                df = gr.Dataframe(value=pd.DataFrame(load_json(HISTORY_PATH)), interactive=False)
                def ref():
                    s = update_history()
                    return s, pd.DataFrame(load_json(HISTORY_PATH))
                btn.click(ref, outputs=[status, df])
            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(type="messages", value=[])
                txt = gr.Textbox(placeholder="Ask...")
                clr = gr.Button("Clear")
                txt.submit(chat, [chatbot, txt], [chatbot, txt])
                clr.click(lambda: [], None, chatbot)
            with gr.TabItem("Config"):
                cfg_df = gr.Dataframe(value=pd.DataFrame(init_config()), interactive=True)
                def save_cfg(df):
                    save_json(CONFIG_PATH, df.to_dict("records"))
                cfg_df.change(save_cfg, cfg_df, None)
    return app

if name=="main":
    init_config()
    create_app().launch()
