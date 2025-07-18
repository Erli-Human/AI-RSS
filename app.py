import gradio as gr
import yt_dlp
import os
import json
import base64
import io
import torch
import time
import subprocess
import whisper
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
from PIL import Image
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from email.mime.text import MIMEText
import smtplib

# Data structures for RSS feeds
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

# RSS Feed Sources
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "OpenAI Blog": "https://openai.com/blog/rss.xml",
        # Add other feeds here
    },
    # Continue with other categories...
}

# Constants
DOWNLOAD_HISTORY_FILE = "download_history.json"
MAX_HISTORY_ITEMS = 50
OUTPUT_FOLDER = "downloads"
GLOBAL_ARTICLE_CACHE: Dict[str, List[Article]] = {}

# Function to load history
def load_history(history_file):
    if not os.path.exists(history_file): return []
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

# Function to save history
def save_history(new_entry, history_file):
    history = load_history(history_file)
    history.insert(0, new_entry)
    history = history[:MAX_HISTORY_ITEMS]
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

# RSS feed fetching function
def fetch_rss_feed(url: str, feed_name: str) -> FeedData:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        if feed.bozo and feed.bozo_exception:
            return FeedData(status="error", articles=[], last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            error=f"Feed parsing error: {feed.bozo_exception}")
        articles = [Article(
            title=entry.get('title', 'No title'),
            link=entry.get('link', ''),
            published=entry.get('published', 'Unknown date'),
            summary=entry.get('summary', 'No summary available')[:200] + "...",
            author=entry.get('author', 'Unknown author'),
            feed_name=feed_name
        ) for entry in feed.entries]
        return FeedData(status="success", articles=articles, last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except requests.exceptions.RequestException as e:
        return FeedData(status="error", articles=[], last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        error=f"Network error: {str(e)}")
    except Exception as e:
        return FeedData(status="error", articles=[], last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        error=f"Unexpected error: {str(e)}")

# Full File Implementation below...
# We will start adding more functions as needed from both apps

def create_integrated_application():
    with gr.Blocks(title="Datanacci Media Studio") as app:
        gr.Markdown("# Datanacci Media Studio")

        with gr.Tabs():
            # Tab for Video Analysis
            with gr.TabItem("Video Analysis"):
                gr.Markdown("## Video Processing and Analysis")
                
                video_input = gr.Video(label="Upload Video")
                analysis_btn = gr.Button("Analyze Video")
                analysis_output = gr.HTML()

                analysis_btn.click(
                    fn=analyze_video,  # This should connect to the actual analysis function from app.py's video analysis section
                    inputs=[video_input],
                    outputs=[analysis_output]
                )

            # Tab for RSS Feed Viewer
            with gr.TabItem("RSS Feed Viewer"):
                gr.Markdown("## RSS Feed Monitoring")

                category_dropdown = gr.Dropdown(choices=list(RSS_FEEDS.keys()), label="Select Category")
                feed_display = gr.HTML()
                chat_input = gr.Textbox(label="Query Feeds")
                chat_output = gr.Chatbot()

                category_dropdown.change(
                    fn=lambda category: format_category_feeds_html(category),
                    inputs=[category_dropdown],
                    outputs=[feed_display]
                )
                chat_input.submit(
                    fn=lambda chat_history, user_input, category: chat_with_feeds(chat_history, user_input, category),
                    inputs=[chat_output, chat_input, category_dropdown],
                    outputs=[chat_output, chat_input]
                )

            # Settings and additional configurations etc.
            
            with gr.TabItem("Settings"):
                gr.Markdown("### Application Settings")
                # Add settings related components here

    return app

# To run the Gradio app
if __name__ == "__main__":
    app = create_integrated_application()
    app.launch()
