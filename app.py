import csv
import json
from datetime import datetime, timedelta
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

# Import the ollama client library
import ollama

# Data structures
@dataclass
class Article:
    title: str
    link: str
    published: str
    summary: str
    author: str = ""
    feed_name: str = "" # Add feed_name to Article dataclass

@dataclass
class FeedData:
    status: str
    articles: List[Article]
    last_updated: str
    error: str = ""

# RSS Feed Sources
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "Science Daily - AI":    
        "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        "Science Daily - Technology":   
        "https://www.sciencedaily.com/rss/top/technology.xml",
        "OpenAI Blog": "https://openai.com/blog/rss.xml",
        "DeepMind Blog": "https://deepmind.com/blog/feed/basic/",
        "Microsoft AI Blog": "https://blogs.microsoft.com/ai/feed/",
        "Machine Learning Mastery": "https://machinelearningmastery.com/feed/",
        "MarkTechPost": "https://www.marktechpost.com/feed/",
        "Berkeley AI Research": "https://bair.berkeley.edu/blog/feed.xml",
        "Distill": "https://distill.pub/rss.xml",
        "AI News": "https://www.artificialintelligence-news.com/feed/",
        "VentureBeat AI": "https://venturebeat.com/ai/feed/",
        "MIT Technology Review": "https://www.technologyreview.com/feed/",
        "IEEE Spectrum": "https://spectrum.ieee.org/rss/fulltext"
    },
    
    "💰 FINANCE & BUSINESS": {
        "Investing.com": "https://www.investing.com/rss/news.rss",
        "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
        "Fortune": "https://fortune.com/feed",
        "Forbes Business": "https://www.forbes.com/business/feed/",
        "Economic Times": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
        "Financial Samurai": "https://www.financialsamurai.com/feed/",
        "NerdWallet": "https://www.nerdwallet.com/blog/feed/",
        "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss"
    },
    
    "🔬 SCIENCE & PHYSICS": {
        "Phys.org": "https://phys.org/rss-feed/",
        "Nature": "https://www.nature.com/nature.rss",
        "Physical Review Letters": "https://feeds.aps.org/rss/recent/prl.xml",
        "New Scientist": "https://www.newscientist.com/feed/home/",
        "Physics World": "https://physicsworld.com/feed/",
        "Space.com": "https://www.space.com/feeds/all",
        "NASA Breaking News": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
        "Sky & Telescope": "https://www.skyandtelescope.com/feed/",
        "Science Daily": "https://www.sciencedaily.com/rss/all.xml"
    },
    
    "💻 TECHNOLOGY": {
        "TechCrunch": "https://techcrunch.com/feed/",
        "The Verge": "https://www.theverge.com/rss/index.xml",
        "Ars Technica": "https://arstechnica.com/feed/",
        "Wired": "https://www.wired.com/feed/rss",
        "Gizmodo": "https://gizmodo.com/rss",
        "Engadget": "https://www.engadget.com/rss.xml",
        "Hacker News": "https://news.ycombinator.com/rss",
        "Slashdot": "https://slashdot.org/slashdot.rss",
        "Reddit Technology": "https://www.reddit.com/r/technology/.rss",
        "The Next Web": "https://thenextweb.com/feed/",
        "ZDNet": "https://www.zdnet.com/news/rss.xml",
        "TechRadar": "https://www.techradar.com/rss"
    },
    
    "📰 GENERAL NEWS": {
        "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
        "CNN": "http://rss.cnn.com/rss/edition.rss",
        "New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "The Guardian": "https://www.theguardian.com/world/rss",
        "Washington Post": "https://feeds.washingtonpost.com/rss/world",
        "Google News": "https://news.google.com/rss",
        "NPR": "https://feeds.npr.org/1001/rss.xml",
        "CBS News": "https://www.cbsnews.com/latest/rss/main"
    },
    
    "🏈 SPORTS": {
        "ESPN": "https://www.espn.com/espn/rss/news",
        "Fox Sports": "https://api.foxsports.com/v1/rss?partnerKey=zBaFxRyGKCfxBagJG9b8pqLyndmvo7UU",
        "The Athletic": "https://theathletic.com/rss/",
        "Yahoo Sports": "https://sports.yahoo.com/rss/",
        "CBS Sports": "https://www.cbssports.com/rss/headlines"
    },
    
    "🎬 ENTERTAINMENT": {
        "Variety": "https://variety.com/feed/",
        "The Hollywood Reporter": "https://www.hollywoodreporter.com/feed/",
        "Rolling Stone": "https://www.rollingstone.com/feed/",
        "Billboard": "https://www.billboard.com/feed/",
        "IGN": "https://feeds.ign.com/ign/all",
        "GameSpot": "https://www.gamespot.com/feeds/mashup/",
        "Polygon": "https://www.polygon.com/rss/index.xml"
    },
    
    "🏥 HEALTH & MEDICINE": {
        "Mayo Clinic": "https://newsnetwork.mayoclinic.org/feed/",
        "CDC": "https://tools.cdc.gov/api/v2/resources/media/132608.rss"
    },
    
    "🔗 BLOCKCHAIN & CRYPTO": {
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "Decrypt": "https://decrypt.co/feed",
        "The Block": "https://www.theblockcrypto.com/rss.xml",
        "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/"
    },
    
    "📊 DATA SCIENCE": {
        "KDnuggets": "https://www.kdnuggets.com/feed",
        "Analytics Vidhya": "https://www.analyticsvidhya.com/feed/",
        "Towards Data Science": "https://towardsdatascience.com/feed"
    },
    
    "🌍 WORLD NEWS": {
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "Deutsche Welle": "https://rss.dw.com/rdf/rss-en-all",
        "RT": "https://www.rt.com/rss/",
        "Times of India": "https://timesofindia.indiatim\
es.com/rssfeedstopstories.cms"
    },
    
    "🍔 FOOD & COOKING": {
        "Bon Appétit": "https://www.bonappetit.com/feed/rss",
        "Serious Eats": "https://feeds.feedburner.com/seriouseats/recipes"
    },
    
    "🎨 DESIGN & CREATIVITY": {
        "Behance": "https://feeds.feedburner.com/behance/vorr",
        "Dribbble": "https://dribbble.com/shots/popular.rss",
        "Creative Bloq": "https://www.creativebloq.com/feed",
        "Smashing Magazine": "https://www.smashingmagazine.com/feed/"
    },
    
    "🌱 ENVIRONMENT & SUSTAINABILITY": {
        "Green Tech Media": "https://www.greentechmedia.com/rss/all"
    }
}

# Global cache for fetched articles to enable chat functionality
GLOBAL_ARTICLE_CACHE: Dict[str, List[Article]] = {}
# Global variable to store available Ollama models
OLLAMA_MODELS: List[str] = []

# --- RSS Core Functions (No Changes Here) ---
def fetch_rss_feed(url: str, feed_name: str, timeout: int = 10) -> FeedData:
    """Fetch and parse a single RSS feed."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        
        if feed.bozo and feed.bozo_exception:
            return FeedData(
                status="error",
                articles=[],
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                error=f"Feed parsing error: {feed.bozo_exception}"
            )
        
        articles = []
        for entry in feed.entries:
            article = Article(
                title=entry.get('title', 'No title'),
                link=entry.get('link', ''),
                published=entry.get('published', 'Unknown date'),
                summary=entry.get('summary', 'No summary available')[:200] + "...",
                author=entry.get('author', 'Unknown author'),
                feed_name=feed_name
            )
            articles.append(article)
        
        return FeedData(
            status="success",
            articles=articles,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except requests.exceptions.RequestException as e:
        return FeedData(
            status="error",
            articles=[],
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error=f"Network error: {str(e)}"
        )
    except Exception as e:
        return FeedData(
            status="error",
            articles=[],
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error=f"Unexpected error: {str(e)}"
        )

def fetch_category_feeds_parallel(category: str, max_workers: int = 5) -> Dict[str, FeedData]:
    """Fetch all feeds in a category using parallel processing."""
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
                results[feed_name] = FeedData(
                    status="error",
                    articles=[],
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    error=f"Processing error: {str(e)}"
                )
    
    return results

# --- Ollama Integration Functions (No Changes Here) ---
def get_ollama_models() -> List[str]:
    """Fetches a list of available Ollama models."""
    try:
        models_info = ollama.list()
        return [model['name'] for model in models_info['models']]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return ["Error: Could not fetch models. Is Ollama server running?"]

def generate_ollama_response(model: str, messages: List[Dict[str, str]]) -> str:
    """Generates a response from Ollama based on the provided messages."""
    try:
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    except Exception as e:
        print(f"Error calling Ollama model '{model}': {e}")
        return f"Error: Could not get a response from Ollama model '{model}'. Please ensure Ollama server is running and the model is downloaded. Error details: {e}"

# --- Main Application Logic ---
def create_enhanced_rss_viewer():
    """Create the main RSS viewer application."""
    
    def get_published_date(article):
        """Helper to safely parse article published date for sorting."""
        try:
            parsed_date = feedparser._parse_date_rfc822(article.published)
            if parsed_date is None:
                parsed_date = feedparser._parse_date_iso8601(article.published)
            
            if parsed_date is not None:
                return datetime(*parsed_date[:6])  
            else:
                return datetime.min
        except Exception:
            return datetime.min

    def format_category_feeds_html(category: str, num_articles_per_feed: int = 3) -> str:
        """
        Fetches and formats articles for a specific category as HTML,
        with each feed in its own scrolling box and articles in a 3-column layout.
        """
        feeds_data = fetch_category_feeds_parallel(category)
        
        # Cache ALL articles (across all feeds in the category) for the chat tab
        all_articles_for_cache = []
        for feed_data in feeds_data.values():
            if feed_data.status == 'success':
                all_articles_for_cache.extend(feed_data.articles)
        GLOBAL_ARTICLE_CACHE[category] = all_articles_for_cache

        if not feeds_data:
            return "<p>No feeds found for this category.</p>"
        
        html_content = f"""
        <style>
            /* Main container for all feeds in a category */
            .category-feeds-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px; /* Gap between individual feed boxes */
                height: 100%; /* Occupy full height of the parent tab */
                overflow-y: auto; /* Enable scrolling for the entire category view if needed */
                padding: 10px;
                box-sizing: border-box;
            }}

            /* Individual feed box */
            .feed-box {{
                flex: 1 1 calc(50% - 30px); /* Two columns, adjust for gap */
                min-width: 350px; /* Minimum width before wrapping */
                max-width: calc(50% - 30px); /* Max width to ensure two columns */
                height: 400px; /* Fixed height for consistent scrolling box */
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 15px;
                background-color: #f9f9f9;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column; /* Stack header and article grid */
                overflow: hidden; /* Hide overflow to make inner content scrollable */
                box-sizing: border-box;
            }}

            .feed-box h3 {{
                margin-top: 0;
                margin-bottom: 10px;
                color: #333;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }}

            .feed-articles-scroll {{
                flex-grow: 1; /* Allows this part to take remaining height */
                overflow-y: auto; /* Make articles scrollable within the box */
                padding-right: 10px; /* Add some padding for scrollbar */
            }}

            .article-grid {{
                display: flex;
                flex-wrap: wrap; /* Allows articles to wrap to the next line */
                gap: 15px; /* Space between articles */
                justify-content: flex-start; /* Align items to the start */
            }}
            .article-item {{
                flex: 1 1 calc(33.333% - 10px); /* Adjust to allow for 3 columns with gap */
                box-sizing: border-box; /* Include padding and border in the element's total width and height */
                border: 1px solid #eee;
                padding: 10px;
                border-radius: 5px;
                background-color: #fff;
                min-width: 180px; /* Smaller min-width for 3 cols within a smaller box */
                max-width: 100%; /* Ensure it doesn't exceed its container */
                margin-bottom: 10px; /* Space below each article */
            }}
            .article-item h4 {{
                margin-top: 0;
                margin-bottom: 5px;
                font-size: 1.0em; /* Slightly smaller font for compactness */
                line-height: 1.3;
            }}
            .article-item p {{
                margin-bottom: 5px;
                font-size: 0.85em; /* Smaller font for meta info */
                color: #666;
            }}
            .article-item a {{
                color: #2196F3;
                text-decoration: none;
            }}
            .article-item a:hover {{
                text-decoration: underline;
            }}

            /* Responsive adjustments */
            @media (max-width: 1200px) {{
                .feed-box {{
                    flex: 1 1 calc(100% - 20px); /* Single column layout on smaller screens */
                    max-width: 100%;
                }}
                .article-item {{
                    flex: 1 1 calc(50% - 10px); /* 2 columns within feed box on medium screens */
                }}
            }}
            @media (max-width: 768px) {{
                .article-item {{
                    flex: 1 1 100%; /* Single column within feed box on very small screens */
                }}
            }}
        </style>
        <div class='category-feeds-container'>
        """
        
        for feed_name, feed_data in feeds_data.items():
            status_icon = "✅" if feed_data.status == 'success' else "❌"
            error_message = f"<p style='color: red;'><strong>Error:</strong> {feed_data.error}</p>" if feed_data.error else ""
            
            html_content += f"""
            <div class='feed-box'>
                <h3>{status_icon} {feed_name}</h3>
                {error_message}
                <div class='feed-articles-scroll'>
            """
            
            if feed_data.status == 'success' and feed_data.articles:
                articles = sorted(feed_data.articles, key=get_published_date, reverse=True)
                displayed_articles = articles[:num_articles_per_feed]
                
                html_content += "<div class='article-grid'>"
                for article in displayed_articles:
                    html_content += f"""
                    <div class='article-item'>
                        <h4><a href='{article.link}' target='_blank'>{article.title}</a></h4>
                        <p>📅 {article.published} | ✍️ {article.author}</p>
                        <p>{article.summary}</p>
                    </div>
                    """
                html_content += "</div>" # Close article-grid
            else:
                html_content += "<p>No articles available or feed error.</p>"
                
            html_content += "</div>" # Close feed-articles-scroll
            html_content += "</div>" # Close feed-box
            
        html_content += "</div>" # Close category-feeds-container
        return html_content

    # Function for the "Chat with RSS feeds" tab (now part of the main layout)
    def chat_with_feeds(
        chat_history: List[List[str]],
        user_input: str,
        chat_category: str,
        ollama_model_name: str
    ) -> Tuple[List[List[str]], str]:
        """
        Processes user input and generates a response using Ollama,
        leveraging cached RSS articles as context.
        """
        if not user_input.strip():
            return chat_history, "Please enter a query."

        if not chat_category or chat_category not in GLOBAL_ARTICLE_CACHE:
            return chat_history, "Please select a category and load its feeds first in the Feed Viewer tabs."
        
        if not ollama_model_name or ollama_model_name.startswith("Error:"):
            return chat_history, "Please select a valid Ollama model. Ensure Ollama server is running."

        articles_to_search = GLOBAL_ARTICLE_CACHE[chat_category]

        # Construct context from articles
        context_articles_str = ""
        if articles_to_search:
            context_articles_str = "Here is a list of recent articles from the selected RSS category. Please use this information to answer the user's questions:\n\n"
            # Limit context to top 20 articles to avoid exceeding context window
            for i, article in enumerate(articles_to_search[:20]):  
                context_articles_str += (
                    f"Article {i+1}:\n"
                    f"  Feed: {article.feed_name}\n"
                    f"  Title: {article.title}\n"
                    f"  Link: {article.link}\n"
                    f"  Published: {article.published}\n"
                    f"  Summary: {article.summary}\n\n"
                )
            context_articles_str += "\nIf the user asks about general knowledge not related to these articles, answer generally but prioritize the provided article context when relevant. Keep your responses concise.\n"
        else:
            context_articles_str = "No articles are available for the selected category. Please answer the user's questions based on general knowledge. Keep your responses concise.\n"

        # Prepare messages for Ollama API
        messages = [
            {"role": "system", "content": context_articles_str}
        ]

        # Add past conversation to messages
        for human_msg, ai_msg in chat_history:
            messages.append({"role": "user", "content": human_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})

        try:
            # Call Ollama for response
            ai_response = generate_ollama_response(ollama_model_name, messages)
        except Exception as e:
            ai_response = f"An error occurred while communicating with Ollama: {e}"
        
        chat_history.append([user_input, ai_response])
        return chat_history, "" # Clear user input box

    # Initial population of Ollama models
    # This needs to be done once when the app starts
    global OLLAMA_MODELS
    OLLAMA_MODELS = get_ollama_models()
    
    # Set preferred model, ensure it's in the list, otherwise fallback to first available
    preferred_ollama_model = 'gemma3n:e4b' # <--- THIS IS THE MODIFIED LINE
    if preferred_ollama_model in OLLAMA_MODELS:
        default_ollama_model = preferred_ollama_model
    elif OLLAMA_MODELS:
        default_ollama_model = OLLAMA_MODELS[0]
    else:
        default_ollama_model = "No models found. Run `ollama run <model_name>`"


    # Create Gradio interface
    with gr.Blocks(title="Advanced RSS Feed Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📰 Advanced RSS Feed Viewer")
        gr.Markdown("Monitor and view RSS feeds from various sources with integrated local Ollama LLM chat.")
        
        # This hidden state will store the currently active category name from the tabs
        active_category = gr.State(list(RSS_FEEDS.keys())[0] if RSS_FEEDS else "")

        with gr.Tabs() as main_tabs: # Encapsulate all main content in tabs
            with gr.TabItem("Feed Viewer 📺"): # Renamed for clarity and added emoji
                with gr.Row(): # Main layout row
                    with gr.Column(scale=2): # Left column for Feed Viewer Tabs
                        gr.Markdown("## Feed Viewer")
                        gr.Markdown("Select a category to view its recent articles. Each feed is in a scrolling box.")
                        
                        # Tabs for categories
                        with gr.Tabs() as category_tabs:
                            for category_name in RSS_FEEDS.keys():
                                with gr.TabItem(category_name, id=f"tab_{category_name}"):
                                    gr.Markdown(f"### Recent Articles in {category_name}")
                                    
                                    # Refresh button moved to the top of each tab
                                    refresh_btn = gr.Button("🔄 Refresh Feeds", variant="primary")
                                    articles_html_output = gr.HTML(
                                        value=format_category_feeds_html(category_name), # Initial content
                                        elem_id=f"articles_display_{category_name}" # Unique ID for each HTML component
                                    )
                                    
                                    refresh_btn.click(
                                        fn=format_category_feeds_html,
                                        inputs=[gr.State(category_name)], # Pass the category name as a state
                                        outputs=articles_html_output
                                    )
                            
                            # When a tab is selected, update the active_category state and also trigger a refresh for chat context
                            category_tabs.select(
                                fn=lambda category_id: category_id.replace("tab_", ""), # Extract category name from tab ID
                                inputs=category_tabs,
                                outputs=active_category
                            ).success( # After the category ID is set, refresh the content of that specific tab
                                fn=lambda category_name: format_category_feeds_html(category_name),
                                inputs=active_category,
                                outputs=articles_html_output # This needs to point to the *currently active* HTML output. This is tricky with dynamic tabs.
                                                # A simpler approach for the active_category for chat is to rely on user selection in the chat dropdown.
                            )
                
            with gr.TabItem("Chat with RSS 💬"): # New tab for chat
                with gr.Column(scale=1): # Right column for Chat with RSS
                    gr.Markdown("## 💬 Chat with RSS Feeds")
                    gr.Markdown("Ask questions about the articles from the selected category.")
                    
                    with gr.Row():
                        # Category for chat: Default to the first category, but allow user override.
                        # We rely on the user to select the correct category for chat context,
                        # as dynamically linking the current tab to the dropdown value in real-time
                        # across separate components can be complex in Gradio.
                        chat_category_select = gr.Dropdown(
                            choices=list(RSS_FEEDS.keys()),
                            label="Select Category for Chat Context",
                            interactive=True,
                            value=list(RSS_FEEDS.keys())[0] if RSS_FEEDS else None,
                            scale=1
                        )
                        
                        ollama_model_dropdown = gr.Dropdown(
                            choices=OLLAMA_MODELS,
                            label="Select Ollama Model",
                            interactive=True,
                            value=default_ollama_model, # Pre-populate with the chosen default model
                            scale=1
                        )
                        refresh_models_btn = gr.Button("Refresh Models", scale=0)
                    
                    chatbot = gr.Chatbot(label="RSS Chat", height=400) # Give chatbot a fixed height for better layout
                    msg = gr.Textbox(label="Your Question", placeholder="e.g., Summarize the latest news on AI?", container=False)
                    with gr.Row(): # Buttons for chat input
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                        clear_chat_btn = gr.Button("Clear Chat", scale=0)

                    # Event listeners for chat
                    msg.submit(
                        chat_with_feeds,
                        [chatbot, msg, chat_category_select, ollama_model_dropdown],
                        [chatbot, msg]
                    )
                    submit_btn.click(
                        chat_with_feeds,
                        [chatbot, msg, chat_category_select, ollama_model_dropdown],
                        [chatbot, msg]
                    )
                    clear_chat_btn.click(lambda: None, None, chatbot, queue=False) # Clears the chatbot
                    
                    refresh_models_btn.click(
                        fn=get_ollama_models,
                        outputs=ollama_model_dropdown
                    )
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown("### Application Settings")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Feed Sources")
                        feed_count = sum(len(feeds) for feeds in RSS_FEEDS.values())
                        gr.Markdown(f"**Total Categories:** {len(RSS_FEEDS)}")
                        gr.Markdown(f"**Total Feeds:** {feed_count}")
                        
                        for category, feeds in RSS_FEEDS.items():
                            gr.Markdown(f"**{category}:** {len(feeds)} feeds")
                    
                    with gr.Column():
                        gr.Markdown("#### System Info")
                        gr.Markdown(f"**Last Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        gr.Markdown("**Status:** Running")
                        gr.Markdown("**Version:** 1.0.0")
                        gr.Markdown("---")
                        gr.Markdown("#### Ollama Status")
                        ollama_status_display = gr.HTML(label="Ollama Server Status")

                # Function to check Ollama status
                def check_ollama_status():
                    try:
                        models = ollama.list()
                        num_models = len(models.get('models', []))
                        return f"<p style='color: green;'>✅ Ollama Server is Running!</p><p>Available Models: {num_models}</p>"
                    except Exception as e:
                        return f"<p style='color: red;'>❌ Ollama Server Not Reachable. Error: {e}</p><p>Please ensure Ollama is installed and running (`ollama serve`).</p>"

                app.load(
                    fn=check_ollama_status,
                    outputs=ollama_status_display
                )

    return app

# To run the Gradio app:
if __name__ == "__main__":
    app = create_enhanced_rss_viewer()
    app.launch()
