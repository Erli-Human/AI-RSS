import asyncio
import datetime
import json
import logging
import os
import re
import subprocess
import threading
import time
import typing
from html import escape

import aiohttp
import feedparser
import gradio as gr
from bs4 import BeautifulSoup
from lxml import html as lh
from requests import get
from urllib3.util import retry


RSS_FEEDS = {}  # Global dictionary to store RSS feeds
GLOBAL_ARTICLE_CACHE = {}  # Global cache for parsed articles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def fetch_rss_feeds():
    global RSS_FEEDS, GLOBAL_ARTICLE_CACHE
    try:
        with open("rss_config.json", "r") as f:
            RSS_FEEDS = json.load(f)
    except FileNotFoundError:
        logging.warning("rss_config.json not found. Using default feeds.")
        RSS_FEEDS = {
            "Technology": [
                "https://www.theverge.com/rss/index.xml",
                "https://techcrunch.com/feed/"
            ],
            "Science": ["https://www.sciencemag.org/rss/news"],
            "World News": [
                "http://feeds.bbci.co.uk/news/rss.xml",
                "https://www.reuters.com/feed/worldNews"
            ]
        }
    except json.JSONDecodeError:
        logging.error("Invalid JSON in rss_config.json. Using default feeds.")
        RSS_FEEDS = {
            "Technology": [
                "https://www.theverge.com/rss/index.xml",
                "https://techcrunch.com/feed/"
            ],
            "Science": ["https://www.sciencemag.org/rss/news"],
            "World News": [
                "http://feeds.bbci.co.uk/news/rss.xml",
                "https://www.reuters.com/feed/worldNews"
            ]
        }

    for category, feeds in RSS_FEEDS.items():
        GLOBAL_ARTICLE_CACHE[category] = []
        for feed_url in feeds:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(feed_url) as response:
                        if response.status == 200:
                            feed = await response.text()
                            parsed_feed = feedparser.parse(feed)
                            articles = []
                            for entry in parsed_feed.entries:
                                try:
                                    article = {
                                        "title": entry.title,
                                        "link": entry.link,
                                        "summary": entry.summary if hasattr(entry, 'summary') else "",
                                        "published": datetime.datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z") if hasattr(entry, 'published') else None
                                    }
                                    articles.append(article)
                                except (ValueError, TypeError) as e:
                                    logging.error(f"Error parsing date for entry in {feed_url}: {e}")

                            GLOBAL_ARTICLE_CACHE[category].extend(articles)
                        else:
                            logging.error(f"Failed to fetch feed {feed_url}: Status code {response.status}")
            except aiohttp.ClientError as e:
                logging.error(f"Error fetching feed {feed_url}: {e}")
            except Exception as e:
                logging.exception(f"An unexpected error occurred while processing {feed_url}: {e}")


def format_category_feeds_html(category):
    if category not in GLOBAL_ARTICLE_CACHE:
        return "<p>Loading feeds...</p>"

    articles = GLOBAL_ARTICLE_CACHE[category]
    html_output = "<h2>" + category + "</h2><ul>"
    for article in articles[:10]:  # Limit to 10 articles for display
        title = escape(article["title"])
        link = article["link"]
        summary = escape(article["summary"])
        published_date = article["published"].strftime("%Y-%m-%d %H:%M") if article["published"] else "Date not available"

        html_output += f'<li><a href="{link}" target="_blank">{title}</a><br>{summary}<br>Published: {published_date}</li>'
    html_output += "</ul>"
    return html_output


def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [line.split()[0] for line in result.stdout.splitlines() if line]  # Extract model names
        return models
    except FileNotFoundError:
        return ["mistral"]  # Default model if Ollama is not found
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting Ollama models: {e}")
        return ["mistral"]


def chat_with_ollama(category, query, model):
    if category not in GLOBAL_ARTICLE_CACHE:
        return "Loading feeds first..."

    articles = GLOBAL_ARTICLE_CACHE[category]
    context = "\n".join([f"{a.title}: {a.summary}" for a in articles[:20]])  # Limit context to 20 articles

    prompt = f"You are an AI assistant summarizing news articles.\nHere is the context:\n{context}\n\nUser Query: {query}\nAI Response:"
    try:
        import subprocess

        process = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
        )
        response = process.stdout
        return response
    except FileNotFoundError:
        return "Ollama not found. Please install Ollama."
    except subprocess.CalledProcessError as e:
        return f"Error running Ollama: {e}"


def ollama_status():
    try:
        import subprocess

        subprocess.run(["ollama", "pull", "mistral"], check=True, capture_output=True)  # Check if Ollama is accessible
        return "Ollama is running."
    except FileNotFoundError:
        return "Ollama not found. Please install it."
    except subprocess.CalledProcessError as e:
        return f"Ollama is not running or there was an error: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# RSS Feed Reader with Ollama")

    category_select = gr.Dropdown(choices=list(RSS_FEEDS.keys()), label="Select Category", value="Technology")
    refresh_button = gr.Button("Refresh Feeds")
    query_input = gr.Textbox(label="Enter your query:")
    model_select = gr.Dropdown(choices=get_ollama_models(), label="Select Ollama Model", value="mistral")  # Default model

    chat_button = gr.Button("Chat with Ollama")
    output_text = gr.Textbox(label="Ollama Response")

    status_display = gr.Textbox(label="Ollama Status:", interactive=False)

    refresh_button.click(fetch_rss_feeds, inputs=[], outputs=[])  # Refresh feeds on button click
    chat_button.click(chat_with_ollama, inputs=[category_select, query_input, model_select], outputs=output_text)
    category_select.change(format_category_feeds_html, inputs=[category_select], outputs=[])  # Update on category change

    demo.load(fetch_rss_feeds, inputs=[], outputs=[]) # Load feeds initially
    demo.load(ollama_status, inputs=[], outputs=status_display)


if __name__ == "__main__":
    asyncio.run(fetch_rss_feeds())  # Start fetching feeds when the app starts
    demo.launch()
