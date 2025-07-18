import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import gradio as gr
import feedparser
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

RSS Feed Sources
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "OpenAI Blog": "https://openai.com/blog/rss.xml"
    }
}

Config persistence
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

Dummy local ONNX-based/model runner or mock function
def generate_local_response(messages: List[Dict[str, str]]) -> str:
    prompt = messages[-1]["content"]
    return f"[llama3-onnx-dummy]: Echo: {prompt}"

def create_enhanced_rss_viewer():
    def format_category_feeds_html(cat: str, n: int = 3) -> str:
        fd = fetch_category_feeds_parallel(cat)
        GLOBAL_ARTICLE_CACHE.clear()
        for v in fd.values():
            if v.status == "success":
                GLOBAL_ARTICLE_CACHE.extend(v.articles)
        return f"<p>Loaded {len(fd)} feeds.</p>"

    def chat_with_feeds(history: List[Dict[str, str]], query: str) -> Tuple[List[Dict[str, str]], None]:
        if not query.strip():
            return history, None
        ctx = {e["key"]: e for e in RSS_CONFIG if isinstance(e, dict)}
        system = {
            "role":"system",
            "content":("You are llama3-onnx-dummy. "
                       "Use RSS_CONFIG metadata for context. "
                       f"CONFIG: {ctx}")
        }
        msgs = [system]
        for h in history:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": query})
        resp = generate_local_response(msgs)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": resp})
        return history, None

    with gr.Blocks(title="Datanacci RSS") as app:
        gr.Markdown("# 📰 Datanacci RSS App")
        with gr.Tabs():
            for cat in RSS_FEEDS:
                with gr.TabItem(cat):
                    gr.Markdown(f"### {cat}")
                    html = gr.HTML(format_category_feeds_html(cat))
                    btn = gr.Button("Refresh")
                    btn.click(fn=format_category_feeds_html, inputs=[gr.State(cat)], outputs=[html])
            with gr.TabItem("💬 Chat with RSS"):
                gr.Markdown("Model fixed to browser ONNX/dummy; JSON output.")
                chatbot = gr.Chatbot(type="messages", value=[])
                ask = gr.Textbox(placeholder="Ask about feeds...")
                clr = gr.Button("Clear")
                ask.submit(chat_with_feeds, [chatbot, ask], [chatbot, ask])
                clr.click(lambda: [], None, chatbot)
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
