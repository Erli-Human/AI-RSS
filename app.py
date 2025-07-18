import os
import json
from datetime import datetime, timedelta
import csv
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO, BytesIO
import base64
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
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

RSS Feed Sources
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "Science Daily - AI": "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        "Science Daily - Technology": "https://www.sciencedaily.com/rss/top/technology.xml",
        "OpenAI Blog": "https://openai.com/blog/rss.xml",
        "DeepMind Blog": "https://deepmind.com/blog/feed/basic/",
        "Microsoft AI Blog": "https://blogs.microsoft.com/ai/feed/",
        "Machine Learning Mastery": "https://machinelearningmastery.com/feed/",
        "MarkTechPost": "https://www.marktechpost.com/feed/",
        "Berkeley AI Research": "https://bair.berkeley.edu/blog/feed.xml",
        "Distill": "https://distill.pub/rss.xml",
        "AI News": "https://www.artificialintelligence-news.com/feed/",
        "MIT Technology Review": "https://www.technologyreview.com/feed/",
        "IEEE Spectrum": "https://spectrum.ieee.org/rss/fulltext"
    },
    # Add other categories here...
}

Configuration Persistence
CONFIG_PATH = "rss_config.json"

def load_config() -> List[Dict[str, Any]]:
    if not os.path.exists(CONFIG_PATH) or os.path.getsize(CONFIG_PATH) == 0:
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

def save_config(cfg: List[Dict[str, Any]]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

RSS_CONFIG = load_config()

Data Models
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

GLOBAL_ARTICLE_CACHE: List[Article] = []

RSS Fetchers
def fetch_rss_feed(url: str, feed_name: str, timeout: int = 10) -> FeedData:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        if feed.bozo and feed.bozo_exception:
            return FeedData("error", [], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            error=str(feed.bozo_exception))
        arts = []
        for e in feed.entries:
            arts.append(Article(
                title=e.get("title","No title"),
                link=e.get("link",""),
                published=e.get("published","Unknown"),
                summary=e.get("summary","No summary")[:200] + "...",
                author=e.get("author",""),
                feed_name=feed_name
            ))
        return FeedData("success", arts, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as ex:
        return FeedData("error", [], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        error=str(ex))

def fetch_category_feeds_parallel(category: str, max_workers: int = 5) -> Dict[str, FeedData]:
    if category not in RSS_FEEDS:
        return {}
    results: Dict[str, FeedData] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        fut2name = {
            exe.submit(fetch_rss_feed, url, name): name
            for name, url in RSS_FEEDS[category].items()
        }
        for fut in as_completed(fut2name):
            nm = fut2name[fut]
            try:
                results[nm] = fut.result()
            except Exception as ex:
                results[nm] = FeedData("error", [], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                       error=str(ex))
    return results

Ollama Utilities
def get_ollama_models() -> List[str]:
    try:
        info = ollama.list()
        return [m["name"] for m in info.get("models",[])]
    except:
        return []

def generate_ollama_response(model: str, messages: List[Dict[str, str]]) -> str:
    try:
        resp = ollama.chat(model=model, messages=messages)
        return resp["message"]["content"]
    except Exception as ex:
        return json.dumps({"error": str(ex)})

Gradio App
def create_enhanced_rss_viewer():
    def get_pub_date(a: Article):
        try:
            d = feedparser._parse_date_rfc822(a.published) or feedparser._parse_date_iso8601(a.published)
            return datetime(*d[:6]) if d else datetime.min
        except:
            return datetime.min

    def format_category_feeds_html(cat: str, n: int = 3) -> str:
        fd = fetch_category_feeds_parallel(cat)
        GLOBAL_ARTICLE_CACHE.clear()
        for v in fd.values():
            if v.status == "success":
                GLOBAL_ARTICLE_CACHE.extend(v.articles)
        return f"<p>Loaded {len(fd)} feeds.</p>"

    def chat_with_feeds(history: List[List[str]], query: str) -> Tuple[List[List[str]], str]:
        if not query.strip():
            return history, "Enter a question."
        ctx = {e["key"]: e for e in RSS_CONFIG if isinstance(e, dict)}
        system = {
            "role":"system",
            "content":(
                "You are datanacci-rss-model. "
                "Using RSS_CONFIG metadata, reply in JSON with keys: "
                "category, feed_name, url, created, key, "
                "technical_metadata (load_time, run_date, source_date), "
                "and any article previews. "
                f"CONFIG: {ctx}"
            )
        }
        msgs = [system]
        for h,a in history:
            msgs += [{"role":"user","content":h},{"role":"assistant","content":a}]
        msgs.append({"role":"user","content":query})
        resp = generate_ollama_response("datanacci-rss-model", msgs)
        history.append([query, resp])
        return history, ""

    def check_ollama_status() -> str:
        try:
            m = ollama.list().get("models",[])
            return f"<p style='color:green;'>✅ Ollama up, {len(m)} models</p>"
        except Exception as ex:
            return f"<p style='color:red;'>❌ Ollama error: {ex}</p>"

    with gr.Blocks(title="Datanacci RSS") as app:
        gr.Markdown("# 📰 Datanacci RSS App")

        with gr.Tabs():
            # Category tabs
            for cat in RSS_FEEDS:
                with gr.TabItem(cat):
                    gr.Markdown(f"### {cat}")
                    html = gr.HTML(format_category_feeds_html(cat))
                    btn = gr.Button("Refresh")
                    btn.click(fn=format_category_feeds_html, inputs=[gr.State(cat)], outputs=[html])

            # Chat with RSS tab
            with gr.TabItem("💬 Chat with RSS"):
                gr.Markdown("Model fixed to datanacci-rss-model; JSON output.")
                chatbot = gr.Chatbot(type="messages")
                ask = gr.Textbox(placeholder="Ask about feeds...")
                clr = gr.Button("Clear")
                ask.submit(chat_with_feeds, [chatbot, ask], [chatbot, ask])
                clr.click(lambda: None, None, chatbot)

            # Settings tab
            with gr.TabItem("⚙️ Settings"):
                tot_cat = len(RSS_FEEDS)
                tot_feed = sum(len(v) for v in RSS_FEEDS.values())
                gr.Markdown(f"Categories: {tot_cat} • Feeds: {tot_feed}")
                items = []
                for entry in RSS_CONFIG:
                    if isinstance(entry, dict):
                        url  = entry.get("url","")
                        name = entry.get("feed_name","")
                        ok   = "✅" if url and requests.get(url,timeout=3).ok else "❌"
                        items.append(f"{ok} {name}")
                gr.Markdown("<br>".join(items))
                gr.HTML(check_ollama_status())
                mods = get_ollama_models()
                okm = "✅" if "datanacci-rss-model" in mods else "❌"
                gr.Markdown(f"datanacci-rss-model {okm}")

            # Configurations tab
            with gr.TabItem("🛠️ Configurations"):
                df = pd.DataFrame(RSS_CONFIG if isinstance(RSS_CONFIG, list) else [])
                table = gr.Dataframe(value=df, interactive=False)

                cat_in  = gr.Textbox(label="Category")
                name_in = gr.Textbox(label="Feed Name")
                url_in  = gr.Textbox(label="Feed URL")
                add_btn = gr.Button("Add Feed")

                def _add(cat, nm, u):
                    ent = {
                        "category": cat,
                        "feed_name": nm,
                        "url": u,
                        "created": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "key": f"{cat}_{nm}"
                    }
                    RSS_CONFIG.append(ent)
                    save_config(RSS_CONFIG)
                    return pd.DataFrame(RSS_CONFIG)

                add_btn.click(_add, [cat_in, name_in, url_in], table)

                upload = gr.File(file_types=[".json"])
                def _upload(f):
                    cfg = json.load(open(f.name, encoding="utf-8"))
                    save_config(cfg)
                    return pd.DataFrame(cfg)
                upload.upload(_upload, upload, table)

                down = gr.Button("Download Config")
                down.click(lambda: CONFIG_PATH, None, None)

        return app

if name == "main":
    create_enhanced_rss_viewer().launch()
