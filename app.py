import csv
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO, BytesIO
import base64
from typing import Dict, Any, List
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

# RSS Feed Sources - UPDATED WITH YOUR NEW LIST
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
        "The Guardian": "https://www.guardian.com/world/rss",
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
        "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"
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
# This is a simple in-memory cache. For a real app, consider persistent storage.
# Key: category name, Value: List of Article objects
GLOBAL_ARTICLE_CACHE: Dict[str, List[Article]] = {}


# Core RSS functionality
def fetch_rss_feed(url: str, feed_name: str, timeout: int = 10) -> FeedData:
    """Fetch and parse a single RSS feed."""
    try:
        # Set user agent to avoid blocking
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
                feed_name=feed_name # Store the feed name with the article
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
            executor.submit(fetch_rss_feed, url, name): name  # Pass feed_name
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

# Main application
def create_enhanced_rss_viewer():
    """Create the main RSS viewer application."""
    
    # Modified to accept a dictionary of feed_name -> List[Article]
    def format_articles_html(feeds_with_articles: Dict[str, List[Article]], num_articles_per_feed: int = 3) -> str:
        """Format articles from multiple feeds as HTML, showing a preview per feed."""
        if not feeds_with_articles:
            return "<p>No articles found for the selected category.</p>"
        
        html = "<div style='max-height: 600px; overflow-y: auto;'>"
        
        for feed_name, articles in feeds_with_articles.items():
            html += f"<div style='border: 1px solid #ccc; margin: 20px 0; padding: 15px; border-radius: 8px; background-color: #f9f9f9;'>"
            html += f"<h3 style='color: #333;'>Feed: {feed_name}</h3>"
            
            if not articles:
                html += "<p>No articles available for this feed.</p>"
            else:
                # Re-introducing sorting with a robust date parsing
                # This helps ensure the 'top N' articles are truly the latest.
                def get_published_date(article):
                    try:
                        # Attempt to parse common RSS date formats using feedparser's internal methods
                        parsed_date = feedparser._parse_date_rfc822(article.published)
                        if parsed_date is None:
                            parsed_date = feedparser._parse_date_iso8601(article.published)
                        
                        if parsed_date is not None:
                            # Convert time.struct_time to datetime object for comparison
                            return datetime(*parsed_date[:6]) 
                        else:
                            # Fallback for unparseable dates: treat as very old
                            return datetime.min
                    except Exception:
                        # Catch any other parsing errors and treat as very old
                        return datetime.min
                
                # Sort articles by published date (most recent first)
                articles.sort(key=get_published_date, reverse=True)
                
                displayed_articles = articles[:num_articles_per_feed]

                for article in displayed_articles:
                    html += f"""
                    <div style='border: 1px solid #eee; margin: 10px 0; padding: 10px; border-radius: 5px; background-color: #fff;'>
                        <h4><a href='{article.link}' target='_blank' style='color: #2196F3; text-decoration: none;'>{article.title}</a></h4>
                        <p style='color: #666; font-size: 0.9em;'>📅 {article.published} | ✍️ {article.author}</p>
                        <p>{article.summary}</p>
                    </div>
                    """
            html += "</div>"
        html += "</div>"
        return html
    
    def load_category_feeds(category):
        """Load feeds for a specific category and cache articles."""
        if not category or category not in RSS_FEEDS:
            return "Please select a category.", ""
        
        try:
            feeds_data = fetch_category_feeds_parallel(category)
            
            # Create status summary
            total_feeds = len(feeds_data)
            working_feeds = sum(1 for feed in feeds_data.values() if feed.status == 'success')
            
            status_html = f"""
            <div style='background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
                <h3>📊 Category: {category}</h3>
                <p><strong>Status:</strong> {working_feeds}/{total_feeds} feeds working ({(working_feeds/total_feeds)*100:.1f}%)</p>
                <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """
            
            # Create feed details and prepare articles for display and caching
            feeds_html = ""
            all_articles_for_cache = [] # All articles for the chat cache
            articles_for_display = {} # Dictionary: feed_name -> List[Article] for display
            
            for feed_name, feed_data in feeds_data.items():
                if feed_data.status == 'success':
                    status_icon = "✅"
                    article_count = len(feed_data.articles)
                    all_articles_for_cache.extend(feed_data.articles)
                    articles_for_display[feed_name] = feed_data.articles
                else:
                    status_icon = "❌"
                    article_count = 0
                
                feeds_html += f"""
                <div style='border-left: 4px solid {"#4CAF50" if feed_data.status == "success" else "#f44336"}; padding-left: 15px; margin: 10px 0;'>
                    <h4>{status_icon} {feed_name}</h4>
                    <p><strong>Articles:</strong> {article_count} | <strong>Updated:</strong> {feed_data.last_updated}</p>
                    {f"<p style='color: red;'><strong>Error:</strong> {feed_data.error}</p>" if feed_data.error else ""}
                </div>
                """
            
            # Cache ALL articles (across all feeds in the category) for the chat tab
            # Sorting for the cache isn't strictly necessary here as chat will filter/search
            GLOBAL_ARTICLE_CACHE[category] = all_articles_for_cache
            
            # Format articles for display, showing previews per feed
            articles_html = format_articles_html(articles_for_display, num_articles_per_feed=3) 
            
            return status_html + feeds_html, articles_html
            
        except Exception as e:
            error_msg = f"Error loading feeds: {str(e)}"
            return error_msg, ""
    
    def refresh_feeds(category):
        """Refresh feeds for the selected category."""
        return load_category_feeds(category)

    # Function for the "Chat with RSS feeds" tab
    def chat_with_feeds(chat_history: List[List[str]], user_input: str, chat_category: str):
        if not user_input.strip():
            return chat_history, "Please enter a query."

        if not chat_category or chat_category not in GLOBAL_ARTICLE_CACHE:
            return chat_history, "Please select a category and load its feeds first in the Feed Viewer tab."
        
        articles_to_search = GLOBAL_ARTICLE_CACHE[chat_category]
        
        # --- Simple Keyword-based "Chat" (replace with actual LLM integration for real chat) ---
        response_articles = []
        user_input_lower = user_input.lower()
        
        for article in articles_to_search:
            # Check title, summary, and feed_name for keywords
            if user_input_lower in article.title.lower() or \
               user_input_lower in article.summary.lower() or \
               user_input_lower in article.feed_name.lower():
                response_articles.append(article)
                if len(response_articles) >= 5: # Limit responses to top 5 matches
                    break
        
        if response_articles:
            chat_response = "Here are some relevant articles:\n\n"
            for article in response_articles:
                chat_response += f"**Feed:** {article.feed_name}\n" # Include feed name in chat response
                chat_response += f"**Title:** [{article.title}]({article.link})\n"
                chat_response += f"**Summary:** {article.summary}\n\n"
        else:
            chat_response = "Sorry, I couldn't find any articles matching your query in the current category. Try a different keyword or load another category."
        # --- End of Simple Keyword-based "Chat" ---

        chat_history.append([user_input, chat_response])
        return chat_history, "" # Clear user input box


    # Create Gradio interface
    with gr.Blocks(title="Datanacci Advanced RSS Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📰 Datanacci Advanced RSS Viewer")
        gr.Markdown("Monitor and view RSS feeds from various sources with a basic chat functionality.")
        
        with gr.Tabs():
            # Main Feed Viewer Tab
            with gr.TabItem("📖 Feed Viewer"):
                with gr.Row():
                    category_dropdown = gr.Dropdown(
                        choices=list(RSS_FEEDS.keys()),
                        label="Select Category",
                        value=list(RSS_FEEDS.keys())[0] if RSS_FEEDS else None # Handle empty RSS_FEEDS
                    )
                    refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        feed_status = gr.HTML(label="Feed Status")
                    with gr.Column(scale=2):
                        articles_display = gr.HTML(label="Recent Articles Previews (Per Feed)")
                
                # Load initial data
                if RSS_FEEDS: # Only add change listener if there are feeds to load
                    category_dropdown.change(
                        fn=load_category_feeds,
                        inputs=[category_dropdown],
                        outputs=[feed_status, articles_display]
                    )
                    
                    refresh_btn.click(
                        fn=refresh_feeds,
                        inputs=[category_dropdown],
                        outputs=[feed_status, articles_display]
                    )
            
            # New "Chat with RSS Feeds" Tab
            with gr.TabItem("💬 Chat with RSS Feeds"):
                gr.Markdown("### Ask questions about the loaded RSS feeds!")
                gr.Markdown("First, go to the 'Feed Viewer' tab and select a category to load its articles. Then you can chat here.")
                
                chat_category_select = gr.Dropdown(
                    choices=list(RSS_FEEDS.keys()),
                    label="Select Category for Chat",
                    interactive=True,
                    value=list(RSS_FEEDS.keys())[0] if RSS_FEEDS else None
                )

                chatbot = gr.Chatbot(label="RSS Chat")
                msg = gr.Textbox(label="Your Question", placeholder="e.g., What are the latest AI advancements?")
                clear = gr.Button("Clear Chat")

                msg.submit(chat_with_feeds, [chatbot, msg, chat_category_select], [chatbot, msg])
                clear.click(lambda: None, None, chatbot, queue=False) # Clears the chatbot
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown("### Application Settings")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Feed Sources")
                        feed_count = sum(len(feeds) for feeds in RSS_FEEDS.values())
                        gr.Markdown(f"**Total Feeds:** {feed_count}")
                        
                        for category, feeds in RSS_FEEDS.items():
                            gr.Markdown(f"**{category}:** {len(feeds)} feeds")
                    
                    with gr.Column():
                        gr.Markdown("#### System Info")
                        gr.Markdown(f"**Last Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        gr.Markdown("**Status:** Running")
                        gr.Markdown("**Version:** 1.0.0")
        
        # Load initial data on startup
        if RSS_FEEDS: # Only load if there are feeds
            app.load(
                fn=lambda: load_category_feeds(list(RSS_FEEDS.keys())[0]),
                outputs=[feed_status, articles_display]
            )
    
    return app

# Monitoring Script (kept as is, but it's not directly part of the Gradio app's tabs)
def create_monitoring_script():
    """Create a separate monitoring script for continuous feed checking."""
    
    # Removed emojis from the string literal to avoid encoding issues
    monitoring_script = '''#!/usr/bin/env python3
"""
RSS Feed Monitoring Script
Runs continuous monitoring of RSS feeds and generates reports.
"""

import time
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import schedule

# IMPORTANT: These functions (fetch_rss_feed, fetch_category_feeds_parallel, RSS_FEEDS)
# must be available in the context where this script runs.
# For a standalone monitoring script, you would typically copy these functions
# or import them from a shared utility file.
# For this example, we assume they are globally accessible or copied for simplicity.

# Re-define or import necessary components for the standalone script context
import requests
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

# Re-define data structures for the monitoring script's self-containment
@dataclass
class Article:
    title: str
    link: str
    published: str
    summary: str
    author: str = ""
    feed_name: str = "" # Added for consistency if needed by monitoring

@dataclass
class FeedData:
    status: str
    articles: List[Article]
    last_updated: str
    error: str = ""

# RSS Feed Sources - Matches the main app's RSS_FEEDS
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "Science Daily - AI":   
        "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        "Science Daily - Technology":  
        "https://www.sciencedaily.com/rss/top/technology.xml",
        "Sam Altman Blog": "https://blog.samaltman.com/",
        "O'Reilly Radar": "https://feeds.feedburner.com/oreilly-radar",
        "Google AI Blog": "https://ai.googleblog.com/feeds/posts/default",
        "OpenAI Blog": "https://openai.com/blog/rss.xml",
        "DeepMind Blog": "https://deepmind.com/blog/feed/basic/",
        "Microsoft AI Blog": "https://blogs.microsoft.com/ai/feed/",
        "Machine Learning Mastery": "https://machinelearningmastery.com/feed/",
        "MarkTechPost": "https://www.marktechpost.com/feed/",
        "Berkeley AI Research": "https://bair.berkeley.edu/blog/feed.xml",
        "Distill": "https://distill.pub/rss.xml",
        "Unite.AI": "https://www.unite.ai/feed/",
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
        "Money Under 30": "https://www.moneyunder30.com/feed",
        "Wall Street Journal": "https://www.wsj.com/xml/rss/3_7085.xml",
        "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss"
    },
    
    "🔬 SCIENCE & PHYSICS": {
        "Phys.org": "https://phys.org/rss-feed/",
        "Nature": "https://www.nature.com/nature.rss",
        "Physical Review Letters": "https://feeds.aps.org/rss/recent/prl.xml",
        "Scientific American": "https://rss.sciam.com/ScientificAmerican-Global",
        "New Scientist": "https://www.newscientist.com/feed/home/",
        "Physics World": "https://physicsworld.com/feed/",
        "Symmetry Magazine": "https://www.symmetrymagazine.org/rss/all-articles.xml",
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
        "Reuters": "https://www.reuters.com/rssFeed/topNews",
        "Associated Press": "https://feeds.apnews.com/ApNews/apf-topnews",
        "NPR": "https://feeds.npr.org/1001/rss.xml",
        "CBS News": "https://www.cbsnews.com/latest/rss/main"
    },
    
    "🏈 SPORTS": {
        "ESPN": "https://www.espn.com/espn/rss/news",
        "Fox Sports": "https://api.foxsports.com/v1/rss?partnerKey=zBaFxRyGKCfxBagJG9b8pqLyndmvo7UU",
        "Sports Illustrated": "https://www.si.com/rss/si_topstories.rss",
        "Bleacher Report": "https://bleacherreport.com/articles/feed",
        "The Athletic": "https://theathletic.com/rss/",
        "Yahoo Sports": "https://sports.yahoo.com/rss/",
        "CBS Sports": "https://www.cbssports.com/rss/headlines",
        "NFL": "https://www.nfl.com/feeds/rss/news",
        "NBA": "https://www.nba.com/rss/nba_rss.xml"
    },
    
    "🎬 ENTERTAINMENT": {
        "Entertainment Weekly": "https://ew.com/feed/",
        "Variety": "https://variety.com/feed/",
        "The Hollywood Reporter": "https://www.hollywoodreporter.com/feed/",
        "Rolling Stone": "https://www.rollingstone.com/feed/",
        "Billboard": "https://www.billboard.com/feed/"
    }
} # This was the missing closing brace for the RSS_FEEDS dict inside the string!
''' # This was the missing closing triple quote for the string!
    return monitoring_script

# To run the Gradio app:
if __name__ == "__main__":
    app = create_enhanced_rss_viewer()
    app.launch()
