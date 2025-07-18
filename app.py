import gradio as gr
import yt_dlp
import os
import json
import base64
import io
import torch
import subprocess
import re
import cv2
import numpy as np
import librosa
import sqlite3
import asyncio
import feedparser
import requests
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from gtts import gTTS

# ----- Data structures -----
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

# ------- RSS Feeds --------
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "Science Daily - AI": "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        "OpenAI Blog": "https://openai.com/blog/rss.xml",
        # Add more feeds...
    },
    # Add more categories...
}

# ------- Function to load and save history -------
def load_history(history_file):
    if not os.path.exists(history_file): return []
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_history(new_entry, history_file):
    history = load_history(history_file)
    history.insert(0, new_entry)
    history = history[:50]
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

# ------- YouTube Video Functionality -------
def get_video_info(url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return f"<h3>{info.get('title', 'N/A')}</h3><p>Duration: {info.get('duration')}</p>"
    except Exception as e:
        return str(e)

def download_media(url, is_audio_only):
    try:
        ydl_opts = {
            'format': 'bestaudio' if is_audio_only else 'bestvideo+bestaudio',
            'outtmpl': os.path.join("downloads", '%(title)s.%(ext)s'),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            return "Download successful!"
    except Exception as e:
        return f"Error: {str(e)}"

# ------- RSS Functions -------
def fetch_rss_feed(url: str, feed_name: str) -> FeedData:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        articles = [Article(title=entry.get('title', 'No title'), link=entry.get('link', ''),
                            published=entry.get('published', 'Unknown date'),
                            summary=entry.get('summary', 'No summary available')[:200] + "...",
                            author=entry.get('author', 'Unknown author'), feed_name=feed_name)
                    for entry in feed.entries]
        return FeedData(status="success", articles=articles, last_updated="Now")
    except requests.RequestException as e:
        return FeedData(status="error", articles=[], last_updated="Now", error=str(e))

def fetch_category_feeds_parallel(category: str, max_workers: int = 5) -> Dict[str, FeedData]:
    if category not in RSS_FEEDS:
        return {}
    feeds = RSS_FEEDS[category]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feed = {
            executor.submit(fetch_rss_feed, url, name): name
            for name, url in feeds.items()
        }
        for future in as_completed(future_to_feed):
            feed_name = future_to_feed[future]
            try:
                results[feed_name] = future.result()
            except Exception as e:
                results[feed_name] = FeedData(status="error", articles=[], last_updated="Now", error=str(e))
    return results

# ------- Gradio Interface Construction -------
def create_rss_voice_tab():
    with gr.TabItem("RSS Voice Chat"):
        category_dropdown = gr.Dropdown(choices=list(RSS_FEEDS.keys()), label="Category")
        chatbot = gr.Chatbot()
        voice_input = gr.Audio(source="microphone", type="line")
        voice_output = gr.Audio()

        def voice_to_text(audio):
            return "User's converted voice input"

        voice_input.change(fn=voice_to_text, inputs=voice_input, outputs="user_input")
        chatbot.submit(
            fn=lambda chat_history, user_input, category: ("User queried.", "response.mp3"),
            inputs=[chatbot, "user_input", category_dropdown],
            outputs=[chatbot, voice_output]
        )

def create_rss_preview_tab():
    with gr.TabItem("RSS Feed Preview"):
        preview_area = gr.HTML()

        def update_previews():
            feeds = fetch_category_feeds_parallel("🤖 AI & MACHINE LEARNING")
            html_content = ""
            for feed_name, feed_data in feeds.items():
                html_content += f"<h2>{feed_name}</h2>"
                for article in feed_data.articles[:3]:
                    html_content += f"<h3>{article.title}</h3><p>{article.summary}</p><a href='{article.link}'>Read more</a>"
            return html_content

        update_button = gr.Button("Update Previews")
        update_button.click(fn=update_previews, outputs=preview_area)

def create_youtube_tab():
    with gr.TabItem("YouTube Video"):
        url_input = gr.Textbox(label="YouTube URL")
        video_info_output = gr.HTML()
        download_btn = gr.Button("Download")
        download_status = gr.Textbox(interactive=False)

        url_input.change(fn=get_video_info, inputs=[url_input], outputs=[video_info_output])
        download_btn.click(
            fn=lambda url, is_audio_only: download_media(url, is_audio_only),
            inputs=[url_input, gr.Checkbox(label="Audio Only", default=False)],
            outputs=[download_status]
        )

def create_integrated_application():
    with gr.Blocks(title="Datanacci Media Studio") as app:
        gr.Markdown("# Datanacci Media Studio")

        with gr.Tabs():
            create_youtube_tab()
            create_rss_voice_tab()
            create_rss_preview_tab()

    return app

if __name__ == "__main__":
    os.makedirs("downloads", exist_ok=True)
    app = create_integrated_application()
    app.launch()
