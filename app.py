import os
import json
from datetime import datetime, timedelta
import csv
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO, BytesIO
import base64
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import gradio as gr
import schedule
import time
import smtplib
from email.mime.text import MIMEText
import feedparser
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import ollama

# ——————————————————————————
# CONFIGURATION PERSISTENCE
# ——————————————————————————
CONFIG_PATH = "rss_config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH) or os.path.getsize(CONFIG_PATH) == 0:
        # seed with 3 months backdate
        seed_date = datetime.utcnow() - timedelta(days=90)
        cfg = []
        for cat, feeds in RSS_FEEDS.items():
            for name, url in feeds.items():
                cfg.append({
                    "category": cat,
                    "feed_name": name,
                    "url": url,
                    "created": seed_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "key": f"{cat}_{name}"
                })
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return cfg
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

# Load or seed on start
RSS_CONFIG = load_config()

# ——————————————————————————
# DATA MODELS
# ——————————————————————————
@dataclass
class Article:
    title: str
    link: str
    published: str
    summary: str
    author: str = ""
    feed_name: str = ""

@dataclass
class FeedData:
    status: str
    articles: List[Article]
    last_updated: str
    error: str = ""

# (RSS_FEEDS unchanged from your existing code…)

GLOBAL_ARTICLE_CACHE: Dict[str, List[Article]] = []

# ——————————————————————————
# RSS FETCHERS
# ——————————————————————————
def fetch_rss_feed(url: str, feed_name: str, timeout: int = 10) -> FeedData:
    # … your existing implementation …

def fetch_category_feeds_parallel(category: str, max_workers: int = 5) -> Dict[str, FeedData]:
    # … your existing implementation …

# ——————————————————————————
# OLLAMA UTILS
# ——————————————————————————
def get_ollama_models() -> List[str]:
    try:
        info = ollama.list()
        return [m["name"] for m in info.get("models", [])]
    except:
        return []

def generate_ollama_response(model: str, messages: List[Dict[str, str]]) -> str:
    try:
        resp = ollama.chat(model=model, messages=messages)
        return resp["message"]["content"]
    except Exception as e:
        return json.dumps({"error": str(e)})

# ——————————————————————————
# GRADIO APP
# ——————————————————————————
def create_enhanced_rss_viewer():
    # helper to parse dates
    def get_pub_date(a): 
        try:
            d = feedparser._parse_date_rfc822(a.published) or feedparser._parse_date_iso8601(a.published)
            return datetime(*d[:6]) if d else datetime.min
        except:
            return datetime.min

    def format_category_feeds_html(cat, n=3):
        feeds_data = fetch_category_feeds_parallel(cat)
        GLOBAL_ARTICLE_CACHE[:] = []  # flatten list
        for fd in feeds_data.values():
            if fd.status=="success":
                GLOBAL_ARTICLE_CACHE.extend(fd.articles)
        # … build your HTML exactly as before …

    def chat_with_feeds(history: List[List[str]], query: str) -> Tuple[List[List[str]], str]:
        if not query.strip(): return history, "Enter a question."
        # context = the config dict by key
        ctx = {e["key"]: e for e in RSS_CONFIG}
        sys = {
            "role":"system",
            "content":(
                "You are datanacci-rss-model. "
                "Using RSS_CONFIG metadata, reply in JSON with keys: category, feed_name, url, created, key, "
                "technical_metadata (load_time, run_date, source_date), and any article previews. "
                f"CONFIG: {ctx}"
            )
        }
        msgs = [sys]
        for h, r in history:
            msgs += [{"role":"user","content":h},{"role":"assistant","content":r}]
        msgs.append({"role":"user","content":query})
        res = generate_ollama_response("datanacci-rss-model", msgs)
        history.append([query, res])
        return history, ""

    def check_ollama_status():
        try:
            m = ollama.list().get("models",[])
            return f"<p style='color:green;'>✅ Ollama running, {len(m)} models</p>"
        except Exception as e:
            return f"<p style='color:red;'>❌ Ollama error: {e}</p>"

    with gr.Blocks(title="Datanacci RSS") as app:
        gr.Markdown("# 📰 Datanacci RSS App")
        with gr.Tabs():
            # feed tabs (unchanged)…

            # CHAT TAB
            with gr.TabItem("💬 Chat with RSS"):
                gr.Markdown("**Model fixed to datanacci-rss-model; JSON output.**")
                chatbot = gr.Chatbot()
                ask = gr.Textbox(placeholder="Ask about feeds...")
                clr = gr.Button("Clear")
                ask.submit(chat_with_feeds, [chatbot, ask], [chatbot, ask])
                clr.click(lambda: None, None, chatbot)

            # SETTINGS TAB
            with gr.TabItem("⚙️ Settings"):
                # counts
                tot_cat = len(RSS_FEEDS)
                tot_feed = sum(len(v) for v in RSS_FEEDS.values())
                gr.Markdown(f"**Categories:** {tot_cat} • **Feeds:** {tot_feed}")
                # per-feed reachability
                items = []
                for e in RSS_CONFIG:
                    ok = "✅" if requests.get(e["url"],timeout=3).ok else "❌"
                    items.append(f"{ok} {e['feed_name']}")
                gr.Markdown("<br>".join(items))
                gr.HTML(check_ollama_status())
                # model availability
                mods = get_ollama_models()
                okm = "✅" if "datanacci-rss-model" in mods else "❌"
                gr.Markdown(f"**datanacci-rss-model** {okm}")

            # CONFIGURATION TAB
            with gr.TabItem("🛠️ Configurations"):
                df = pd.DataFrame(RSS_CONFIG)
                table = gr.Dataframe(value=df, interactive=False)
                cat_in = gr.Textbox(label="Category")
                name_in = gr.Textbox(label="Feed Name")
                url_in = gr.Textbox(label="Feed URL")
                add = gr.Button("Add Feed")

                def _add(cat, nm, u):
                    ent = {
                        "category":cat,
                        "feed_name":nm,
                        "url":u,
                        "created":datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "key":f"{cat}_{nm}"
                    }
                    RSS_CONFIG.append(ent)
                    save_config(RSS_CONFIG)
                    return pd.DataFrame(RSS_CONFIG)
                add.click(_add, [cat_in, name_in, url_in], table)

                upload = gr.File(file_types=[".json"])
                def _upl(f):
                    cfg = json.load(open(f.name, encoding="utf-8"))
                    save_config(cfg)
                    return pd.DataFrame(cfg)
                upload.upload(_upl, upload, table)

                down = gr.Button("Download Config")
                down.click(lambda: CONFIG_PATH, None, None)

        return app

if __name__ == "__main__":
    create_enhanced_rss_viewer().launch()
