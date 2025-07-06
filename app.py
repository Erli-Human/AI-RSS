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
        "Sam Altman Blog": "https://blog.samaltman.com/", # Corrected entry
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
        "Billboard": "https://www.billboard.com/feed/",
        "IGN": "https://feeds.ign.com/ign/all",
        "GameSpot": "https://www.gamespot.com/feeds/mashup/",
        "Polygon": "https://www.polygon.com/rss/index.xml"
    },
    
    "🏥 HEALTH & MEDICINE": {
        "WebMD": "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
        "Mayo Clinic": "https://newsnetwork.mayoclinic.org/feed/",
        "Harvard Health": "https://www.health.harvard.edu/blog/feed",
        "Medical News Today": "https://www.medicalnewstoday.com/rss",
        "Healthline": "https://www.healthline.com/rss",
        "CDC": "https://tools.cdc.gov/api/v2/resources/media/132608.rss"
    },
    
    "🔗 BLOCKCHAIN & CRYPTO": {
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "Decrypt": "https://decrypt.co/feed",
        "The Block": "https://www.theblockcrypto.com/rss.xml",
        "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
        "Crypto News": "https://www.crypto-news.net/feed/"
    },
    
    "📊 DATA SCIENCE": {
        "KDnuggets": "https://www.kdnuggets.com/feed",
        "Analytics Vidhya": "https://www.analyticsvidhya.com/feed/",
        "Towards Data Science": "https://towardsdatascience.com/feed",
        "Data Science Central": "https://www.datasciencecentral.com/profiles/blog/feed"
    },
    
    "🌍 WORLD NEWS": {
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "Deutsche Welle": "https://rss.dw.com/rdf/rss-en-all",
        "France24": "https://www.france24.com/en/rss",
        "RT": "https://www.rt.com/rss/",
        "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"
    },
    
    "🍔 FOOD & COOKING": {
        "Food Network": "https://www.foodnetwork.com/feeds/all-latest-recipes.xml",
        "Bon Appétit": "https://www.bonappetit.com/feed/rss",
        "Serious Eats": "https://feeds.feedburner.com/seriouseats/recipes",
        "Epicurious": "https://www.epicurious.com/services/rss/recipes/latest"
    },
    
    "🎨 DESIGN & CREATIVITY": {
        "Behance": "https://feeds.feedburner.com/behance/vorr",
        "Dribbble": "https://dribbble.com/shots/popular.rss",
        "Creative Bloq": "https://www.creativebloq.com/feed",
        "Smashing Magazine": "https://www.smashingmagazine.com/feed/"
    },
    
    "🌱 ENVIRONMENT & SUSTAINABILITY": {
        "TreeHugger": "https://www.treehugger.com/feeds/rss/",
        "Environmental News Network": "https://www.enn.com/rss/",
        "Climate Central": "https://www.climatecentral.org/rss/news.xml",
        "Green Tech Media": "https://www.greentechmedia.com/rss/all"
    }
}

# Core RSS functionality
def fetch_rss_feed(url: str, timeout: int = 10) -> FeedData:
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
        for entry in feed.entries[:10]:  # Limit to 10 most recent articles
            article = Article(
                title=entry.get('title', 'No title'),
                link=entry.get('link', ''),
                published=entry.get('published', 'Unknown date'),
                summary=entry.get('summary', 'No summary available')[:200] + "...",
                author=entry.get('author', 'Unknown author')
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
            executor.submit(fetch_rss_feed, url): name  
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

class FeedAnalytics:
    """Analytics and export functionality for RSS feeds."""
    
    def __init__(self):
        self.feed_history = []
    
    def log_feed_check(self, category: str, feed_name: str, status: str, article_count: int):
        """Log feed check results for analytics."""
        self.feed_history.append({
            'timestamp': datetime.now(),
            'category': category,
            'feed_name': feed_name,
            'status': status,
            'article_count': article_count
        })
    
    def export_to_csv(self, results: Dict[str, Any]) -> str:
        """Export feed results to CSV format."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Category', 'Feed Name', 'Status', 'Articles', 'Last Updated', 'Error'])
        
        # Write data
        for category, feeds in results.items():
            for feed_name, feed_data in feeds.items():
                writer.writerow([
                    category,
                    feed_name,
                    feed_data.status,
                    len(feed_data.articles) if feed_data.status == 'success' else 0,
                    feed_data.last_updated,
                    feed_data.error if feed_data.status == 'error' else ''
                ])
        
        return output.getvalue()
    
    def export_to_json(self, results: Dict[str, Any]) -> str:
        """Export feed results to JSON format."""
        export_data = {}
        
        for category, feeds in results.items():
            export_data[category] = {}
            for feed_name, feed_data in feeds.items():
                export_data[category][feed_name] = {
                    'status': feed_data.status,
                    'articles': [asdict(article) for article in feed_data.articles],
                    'last_updated': feed_data.last_updated,
                    'error': feed_data.error if feed_data.status == 'error' else None
                }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def create_status_chart(self, results: Dict[str, Any]) -> str:
        """Create a status chart as base64 encoded image."""
        categories = []
        working_counts = []
        total_counts = []
        
        for category, feeds in results.items():
            # Remove emojis for chart labels (if any are still present from the category names)
            clean_category = category.replace('🤖 ', '').replace('💻 ', '').replace('🔬 ', '').replace('📰 ', '') \
                                     .replace('💰 ', '').replace('🏈 ', '').replace('🎬 ', '').replace('🏥 ', '') \
                                     .replace('🔗 ', '').replace('📊 ', '').replace('🌍 ', '').replace('🍔 ', '') \
                                     .replace('🎨 ', '').replace('🌱 ', '')
            categories.append(clean_category[:15])
            working = sum(1 for feed in feeds.values() if feed.status == 'success')
            total = len(feeds)
            working_counts.append(working)
            total_counts.append(total)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(categories))
        
        ax.bar([i - 0.2 for i in x], working_counts, 0.4, label='Working', color='#00b894')
        ax.bar([i + 0.2 for i in x], total_counts, 0.4, label='Total', color='#ddd')
        
        ax.set_xlabel('Categories')
        ax.set_ylabel('Number of Feeds')
        ax.set_title('RSS Feed Status by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"

# Main application
def create_enhanced_rss_viewer():
    """Create the main RSS viewer application."""
    
    analytics = FeedAnalytics()
    
    def format_articles_html(articles: List[Article]) -> str:
        """Format articles as HTML."""
        if not articles:
            return "<p>No articles found.</p>"
        
        html = "<div style='max-height: 600px; overflow-y: auto;'>"
        for article in articles:
            html += f"""
            <div style='border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;'>
                <h3><a href='{article.link}' target='_blank' style='color: #2196F3; text-decoration: none;'>{article.title}</a></h3>
                <p style='color: #666; font-size: 0.9em;'>📅 {article.published} | ✍️ {article.author}</p>
            </div>
            """
        html += "</div>"
        return html
    
    def load_category_feeds(category):
        """Load feeds for a specific category."""
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
            
            # Create feed details
            feeds_html = ""
            all_articles = []
            
            for feed_name, feed_data in feeds_data.items():
                if feed_data.status == 'success':
                    status_icon = "✅"
                    article_count = len(feed_data.articles)
                    all_articles.extend(feed_data.articles)
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
            
            # Sort articles by date (most recent first)
            try:
                # Filter out articles with 'Unknown date' or non-parseable dates for sorting
                parseable_articles = [
                    a for a in all_articles 
                    if a.published != 'Unknown date' and 'published_parsed' in feedparser.parse(a.link).entries[0]
                ]
                # Sort based on parsed time
                parseable_articles.sort(key=lambda x: feedparser.parse(x.link).entries[0].published_parsed, reverse=True)
                # Re-add unparseable articles at the end if necessary, or just use parseable ones
                all_articles = parseable_articles 
            except Exception as e:
                # Fallback if sorting based on feedparser's internal parsed_published fails
                print(f"Warning: Could not sort articles by date. Error: {e}")
                pass  
            
            articles_html = format_articles_html(all_articles[:20])  # Show top 20 articles
            
            return status_html + feeds_html, articles_html
            
        except Exception as e:
            error_msg = f"Error loading feeds: {str(e)}"
            return error_msg, ""
    
    def refresh_feeds(category):
        """Refresh feeds for the selected category."""
        return load_category_feeds(category)
    
    # Create Gradio interface
    with gr.Blocks(title="Advanced RSS Feed Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📰 Advanced RSS Feed Viewer")
        gr.Markdown("Monitor and view RSS feeds from various sources with analytics and export capabilities.")
        
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
                        articles_display = gr.HTML(label="Recent Articles")
                
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
            
            # Export & Analytics Tab
            with gr.TabItem("📤 Export & Analytics"):
                gr.Markdown("### Export feed data and view analytics")
                
                with gr.Row():
                    export_category = gr.Dropdown(
                        choices=["All Categories"] + list(RSS_FEEDS.keys()),
                        label="Select Category to Export",
                        value="All Categories"
                    )
                    export_format = gr.Radio(
                        choices=["CSV", "JSON"],
                        label="Export Format",
                        value="CSV"
                    )
                
                export_btn = gr.Button("📊 Generate Export", variant="primary")
                
                with gr.Row():
                    export_output = gr.File(label="Download Export")
                    chart_output = gr.Image(label="Status Chart")
                
                def generate_export(category, format_type):
                    # Fetch data
                    if category == "All Categories":
                        all_results = {}
                        for cat in RSS_FEEDS.keys():
                            all_results[cat] = fetch_category_feeds_parallel(cat)
                    else:
                        all_results = {category: fetch_category_feeds_parallel(category)}
                    
                    # Generate export
                    if format_type == "CSV":
                        content = analytics.export_to_csv(all_results)
                        filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    else:
                        content = analytics.export_to_json(all_results)
                        filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    # Save file with UTF-8 encoding
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Generate chart
                    chart_image = analytics.create_status_chart(all_results)
                    
                    return filename, chart_image
                
                export_btn.click(
                    fn=generate_export,
                    inputs=[export_category, export_format],
                    outputs=[export_output, chart_output]
                )
            
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

# Monitoring Script
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
        "Billboard": "https://www.billboard.com/feed/",
        "IGN": "https://feeds.ign.com/ign/all",
        "GameSpot": "https://www.gamespot.com/feeds/mashup/",
        "Polygon": "https://www.polygon.com/rss/index.xml"
    },
    
    "🏥 HEALTH & MEDICINE": {
        "WebMD": "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
        "Mayo Clinic": "https://newsnetwork.mayoclinic.org/feed/",
        "Harvard Health": "https://www.health.harvard.edu/blog/feed",
        "Medical News Today": "https://www.medicalnewstoday.com/rss",
        "Healthline": "https://www.healthline.com/rss",
        "CDC": "https://tools.cdc.gov/api/v2/resources/media/132608.rss"
    },
    
    "🔗 BLOCKCHAIN & CRYPTO": {
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "Decrypt": "https://decrypt.co/feed",
        "The Block": "https://www.theblockcrypto.com/rss.xml",
        "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
        "Crypto News": "https://www.crypto-news.net/feed/"
    },
    
    "📊 DATA SCIENCE": {
        "KDnuggets": "https://www.kdnuggets.com/feed",
        "Analytics Vidhya": "https://www.analyticsvidhya.com/feed/",
        "Towards Data Science": "https://towardsdatascience.com/feed",
        "Data Science Central": "https://www.datasciencecentral.com/profiles/blog/feed"
    },
    
    "🌍 WORLD NEWS": {
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "Deutsche Welle": "https://rss.dw.com/rdf/rss-en-all",
        "France24": "https://www.france24.com/en/rss",
        "RT": "https://www.rt.com/rss/",
        "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"
    },
    
    "🍔 FOOD & COOKING": {
        "Food Network": "https://www.foodnetwork.com/feeds/all-latest-recipes.xml",
        "Bon Appétit": "https://www.bonappetit.com/feed/rss",
        "Serious Eats": "https://feeds.feedburner.com/seriouseats/recipes",
        "Epicurious": "https://www.epicurious.com/services/rss/recipes/latest"
    },
    
    "🎨 DESIGN & CREATIVITY": {
        "Behance": "https://feeds.feedburner.com/behance/vorr",
        "Dribbble": "https://dribbble.com/shots/popular.rss",
        "Creative Bloq": "https://www.creativebloq.com/feed",
        "Smashing Magazine": "https://www.smashingmagazine.com/feed/"
    },
    
    "🌱 ENVIRONMENT & SUSTAINABILITY": {
        "TreeHugger": "https://www.treehugger.com/feeds/rss/",
        "Environmental News Network": "https://www.enn.com/rss/",
        "Climate Central": "https://www.climatecentral.org/rss/news.xml",
        "Green Tech Media": "https://www.greentechmedia.com/rss/all"
    }
}


# Core RSS functionality
def fetch_rss_feed(url: str, timeout: int = 10) -> FeedData:
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
        for entry in feed.entries[:10]:  # Limit to 10 most recent articles
            article = Article(
                title=entry.get('title', 'No title'),
                link=entry.get('link', ''),
                published=entry.get('published', 'Unknown date'),
                summary=entry.get('summary', 'No summary available')[:200] + "...",
                author=entry.get('author', 'Unknown author')
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
            executor.submit(fetch_rss_feed, url): name  
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

class FeedAnalytics:
    """Analytics and export functionality for RSS feeds."""
    
    def __init__(self):
        self.feed_history = []
    
    def log_feed_check(self, category: str, feed_name: str, status: str, article_count: int):
        """Log feed check results for analytics."""
        self.feed_history.append({
            'timestamp': datetime.now(),
            'category': category,
            'feed_name': feed_name,
            'status': status,
            'article_count': article_count
        })
    
    def export_to_csv(self, results: Dict[str, Any]) -> str:
        """Export feed results to CSV format."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Category', 'Feed Name', 'Status', 'Articles', 'Last Updated', 'Error'])
        
        # Write data
        for category, feeds in results.items():
            for feed_name, feed_data in feeds.items():
                writer.writerow([
                    category,
                    feed_name,
                    feed_data.status,
                    len(feed_data.articles) if feed_data.status == 'success' else 0,
                    feed_data.last_updated,
                    feed_data.error if feed_data.status == 'error' else ''
                ])
        
        return output.getvalue()
    
    def export_to_json(self, results: Dict[str, Any]) -> str:
        """Export feed results to JSON format."""
        export_data = {}
        
        for category, feeds in results.items():
            export_data[category] = {}
            for feed_name, feed_data in feeds.items():
                export_data[category][feed_name] = {
                    'status': feed_data.status,
                    'articles': [asdict(article) for article in feed_data.articles],
                    'last_updated': feed_data.last_updated,
                    'error': feed_data.error if feed_data.status == 'error' else None
                }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def create_status_chart(self, results: Dict[str, Any]) -> str:
        """Create a status chart as base64 encoded image."""
        categories = []
        working_counts = []
        total_counts = []
        
        for category, feeds in results.items():
            # Remove emojis for chart labels (if any are still present from the category names)
            clean_category = category.replace('🤖 ', '').replace('💻 ', '').replace('🔬 ', '').replace('📰 ', '') \
                                     .replace('💰 ', '').replace('🏈 ', '').replace('🎬 ', '').replace('🏥 ', '') \
                                     .replace('🔗 ', '').replace('📊 ', '').replace('🌍 ', '').replace('🍔 ', '') \
                                     .replace('🎨 ', '').replace('🌱 ', '')
            categories.append(clean_category[:15])
            working = sum(1 for feed in feeds.values() if feed.status == 'success')
            total = len(feeds)
            working_counts.append(working)
            total_counts.append(total)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(categories))
        
        ax.bar([i - 0.2 for i in x], working_counts, 0.4, label='Working', color='#00b894')
        ax.bar([i + 0.2 for i in x], total_counts, 0.4, label='Total', color='#ddd')
        
        ax.set_xlabel('Categories')
        ax.set_ylabel('Number of Feeds')
        ax.set_title('RSS Feed Status by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"

# Main application
def create_enhanced_rss_viewer():
    """Create the main RSS viewer application."""
    
    analytics = FeedAnalytics()
    
    def format_articles_html(articles: List[Article]) -> str:
        """Format articles as HTML."""
        if not articles:
            return "<p>No articles found.</p>"
        
        html = "<div style='max-height: 600px; overflow-y: auto;'>"
        for article in articles:
            html += f"""
            <div style='border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;'>
                <h3><a href='{article.link}' target='_blank' style='color: #2196F3; text-decoration: none;'>{article.title}</a></h3>
                <p style='color: #666; font-size: 0.9em;'>📅 {article.published} | ✍️ {article.author}</p>
                <p>{article.summary}</p>
            </div>
            """
        html += "</div>"
        return html
    
    def load_category_feeds(category):
        """Load feeds for a specific category."""
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
            
            # Create feed details
            feeds_html = ""
            all_articles = []
            
            for feed_name, feed_data in feeds_data.items():
                if feed_data.status == 'success':
                    status_icon = "✅"
                    article_count = len(feed_data.articles)
                    all_articles.extend(feed_data.articles)
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
            
            # Sort articles by date (most recent first)
            try:
                # To sort, we need to convert the 'published' string to a datetime object.
                # This needs careful handling as date formats can vary in RSS feeds.
                # We'll try common formats or fall back to feedparser's parsed_published
                # if available and reliable.
                
                # A more robust way to parse diverse dates would involve dateutil.parser
                # from dateutil.parser import parse
                # sorted_articles = sorted(all_articles, key=lambda x: parse(x.published, fuzzy=True), reverse=True)
                
                # For simplicity and to avoid over-complicating without the full feedparser entry,
                # we'll stick to a basic sort by string or assume feedparser handles it.
                # If feedparser.parse(url).entries[0].published_parsed is reliable, use that.
                
                # Let's use the 'published' string directly for sorting for now,
                # assuming it's somewhat consistent, or that `feedparser` provides a parsable string.
                # A more advanced solution would involve parsing the date string into datetime objects.
                all_articles.sort(key=lambda x: x.published, reverse=True)
            except Exception as e:
                # If sorting fails (e.g., inconsistent date formats), just print a warning
                # and proceed with unsorted articles.
                print(f"Warning: Could not sort articles by date. Error: {e}")
                pass  
            
            articles_html = format_articles_html(all_articles[:20])  # Show top 20 articles
            
            return status_html + feeds_html, articles_html
            
        except Exception as e:
            error_msg = f"Error loading feeds: {str(e)}"
            return error_msg, ""
    
    def refresh_feeds(category):
        """Refresh feeds for the selected category."""
        return load_category_feeds(category)
    
    # Create Gradio interface
    with gr.Blocks(title="Advanced RSS Feed Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📰 Advanced RSS Feed Viewer")
        gr.Markdown("Monitor and view RSS feeds from various sources with analytics and export capabilities.")
        
        with gr.Tabs():
            # Main Feed Viewer Tab
            with gr.TabItem("📖 Feed Viewer"):
                with gr.Row():
                    category_dropdown = gr.Dropdown(
                        choices=list(RSS_FEEDS.keys()),
                        label="Select Category",
                        # Ensure a default value if RSS_FEEDS is not empty
                        value=list(RSS_FEEDS.keys())[0] if RSS_FEEDS else None 
                    )
                    refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        feed_status = gr.HTML(label="Feed Status")
                    with gr.Column(scale=2):
                        articles_display = gr.HTML(label="Recent Articles")
                
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
            
            # Export & Analytics Tab
            with gr.TabItem("📤 Export & Analytics"):
                gr.Markdown("### Export feed data and view analytics")
                
                with gr.Row():
                    export_category = gr.Dropdown(
                        choices=["All Categories"] + list(RSS_FEEDS.keys()),
                        label="Select Category to Export",
                        value="All Categories"
                    )
                    export_format = gr.Radio(
                        choices=["CSV", "JSON"],
                        label="Export Format",
                        value="CSV"
                    )
                
                export_btn = gr.Button("📊 Generate Export", variant="primary")
                
                with gr.Row():
                    export_output = gr.File(label="Download Export")
                    chart_output = gr.Image(label="Status Chart")
                
                def generate_export(category, format_type):
                    # Fetch data
                    if category == "All Categories":
                        all_results = {}
                        for cat in RSS_FEEDS.keys():
                            all_results[cat] = fetch_category_feeds_parallel(cat)
                    else:
                        all_results = {category: fetch_category_feeds_parallel(category)}
                    
                    # Generate export
                    if format_type == "CSV":
                        content = analytics.export_to_csv(all_results)
                        filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    else:
                        content = analytics.export_to_json(all_results)
                        filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    # Save file with UTF-8 encoding
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Generate chart
                    chart_image = analytics.create_status_chart(all_results)
                    
                    return filename, chart_image
                
                export_btn.click(
                    fn=generate_export,
                    inputs=[export_category, export_format],
                    outputs=[export_output, chart_output]
                )
            
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

# Monitoring Script
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
        "Billboard": "https://www.billboard.com/feed/",
        "IGN": "https://feeds.ign.com/ign/all",
        "GameSpot": "https://www.gamespot.com/feeds/mashup/",
        "Polygon": "https://www.polygon.com/rss/index.xml"
    },
    
    "🏥 HEALTH & MEDICINE": {
        "WebMD": "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
        "Mayo Clinic": "https://newsnetwork.mayoclinic.org/feed/",
        "Harvard Health": "https://www.health.harvard.edu/blog/feed",
        "Medical News Today": "https://www.medicalnewstoday.com/rss",
        "Healthline": "https://www.healthline.com/rss",
        "CDC": "https://tools.cdc.gov/api/v2/resources/media/132608.rss"
    },
    
    "🔗 BLOCKCHAIN & CRYPTO": {
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "Decrypt": "https://decrypt.co/feed",
        "The Block": "https://www.theblockcrypto.com/rss.xml",
        "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
        "Crypto News": "https://www.crypto-news.net/feed/"
    },
    
    "📊 DATA SCIENCE": {
        "KDnuggets": "https://www.kdnuggets.com/feed",
        "Analytics Vidhya": "https://www.analyticsvidhya.com/feed/",
        "Towards Data Science": "https://towardsdatascience.com/feed",
        "Data Science Central": "https://www.datasciencecentral.com/profiles/blog/feed"
    },
    
    "🌍 WORLD NEWS": {
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "Deutsche Welle": "https://rss.dw.com/rdf/rss-en-all",
        "France24": "https://www.france24.com/en/rss",
        "RT": "https://www.rt.com/rss/",
        "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"
    },
    
    "🍔 FOOD & COOKING": {
        "Food Network": "https://www.foodnetwork.com/feeds/all-latest-recipes.xml",
        "Bon Appétit": "https://www.bonappetit.com/feed/rss",
        "Serious Eats": "https://feeds.feedburner.com/seriouseats/recipes",
        "Epicurious": "https://www.epicurious.com/services/rss/recipes/latest"
    },
    
    "🎨 DESIGN & CREATIVITY": {
        "Behance": "https://feeds.feedburner.com/behance/vorr",
        "Dribbble": "https://dribbble.com/shots/popular.rss",
        "Creative Bloq": "https://www.creativebloq.com/feed",
        "Smashing Magazine": "https://www.smashingmagazine.com/feed/"
    },
    
    "🌱 ENVIRONMENT & SUSTAINABILITY": {
        "TreeHugger": "https://www.treehugger.com/feeds/rss/",
        "Environmental News Network": "https://www.enn.com/rss/",
        "Climate Central": "https://www.climatecentral.org/rss/news.xml",
        "Green Tech Media": "https://www.greentechmedia.com/rss/all"
    }
}


# Core RSS functionality
def fetch_rss_feed(url: str, timeout: int = 10) -> FeedData:
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
        for entry in feed.entries[:10]:  # Limit to 10 most recent articles
            article = Article(
                title=entry.get('title', 'No title'),
                link=entry.get('link', ''),
                published=entry.get('published', 'Unknown date'),
                summary=entry.get('summary', 'No summary available')[:200] + "...",
                author=entry.get('author', 'Unknown author')
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
            executor.submit(fetch_rss_feed, url): name  
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

class FeedAnalytics:
    """Analytics and export functionality for RSS feeds."""
    
    def __init__(self):
        self.feed_history = []
    
    def log_feed_check(self, category: str, feed_name: str, status: str, article_count: int):
        """Log feed check results for analytics."""
        self.feed_history.append({
            'timestamp': datetime.now(),
            'category': category,
            'feed_name': feed_name,
            'status': status,
            'article_count': article_count
        })
    
    def export_to_csv(self, results: Dict[str, Any]) -> str:
        """Export feed results to CSV format."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Category', 'Feed Name', 'Status', 'Articles', 'Last Updated', 'Error'])
        
        # Write data
        for category, feeds in results.items():
            for feed_name, feed_data in feeds.items():
                writer.writerow([
                    category,
                    feed_name,
                    feed_data.status,
                    len(feed_data.articles) if feed_data.status == 'success' else 0,
                    feed_data.last_updated,
                    feed_data.error if feed_data.status == 'error' else ''
                ])
        
        return output.getvalue()
    
    def export_to_json(self, results: Dict[str, Any]) -> str:
        """Export feed results to JSON format."""
        export_data = {}
        
        for category, feeds in results.items():
            export_data[category] = {}
            for feed_name, feed_data in feeds.items():
                export_data[category][feed_name] = {
                    'status': feed_data.status,
                    'articles': [asdict(article) for article in feed_data.articles],
                    'last_updated': feed_data.last_updated,
                    'error': feed_data.error if feed_data.status == 'error' else None
                }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def create_status_chart(self, results: Dict[str, Any]) -> str:
        """Create a status chart as base64 encoded image."""
        categories = []
        working_counts = []
        total_counts = []
        
        for category, feeds in results.items():
            # Remove emojis for chart labels (if any are still present from the category names)
            clean_category = category.replace('🤖 ', '').replace('💻 ', '').replace('🔬 ', '').replace('📰 ', '') \
                                     .replace('💰 ', '').replace('🏈 ', '').replace('🎬 ', '').replace('🏥 ', '') \
                                     .replace('🔗 ', '').replace('📊 ', '').replace('🌍 ', '').replace('🍔 ', '') \
                                     .replace('🎨 ', '').replace('🌱 ', '')
            categories.append(clean_category[:15])
            working = sum(1 for feed in feeds.values() if feed.status == 'success')
            total = len(feeds)
            working_counts.append(working)
            total_counts.append(total)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(categories))
        
        ax.bar([i - 0.2 for i in x], working_counts, 0.4, label='Working', color='#00b894')
        ax.bar([i + 0.2 for i in x], total_counts, 0.4, label='Total', color='#ddd')
        
        ax.set_xlabel('Categories')
        ax.set_ylabel('Number of Feeds')
        ax.set_title('RSS Feed Status by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"

# Main application
def create_enhanced_rss_viewer():
    """Create the main RSS viewer application."""
    
    analytics = FeedAnalytics()
    
    def format_articles_html(articles: List[Article]) -> str:
        """Format articles as HTML."""
        if not articles:
            return "<p>No articles found.</p>"
        
        html = "<div style='max-height: 600px; overflow-y: auto;'>"
        for article in articles:
            html += f"""
            <div style='border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;'>
                <h3><a href='{article.link}' target='_blank' style='color: #2196F3; text-decoration: none;'>{article.title}</a></h3>
                <p style='color: #666; font-size: 0.9em;'>📅 {article.published} | ✍️ {article.author}</p>
                <p>{article.summary}</p>
            </div>
            """
        html += "</div>"
        return html
    
    def load_category_feeds(category):
        """Load feeds for a specific category."""
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
            
            feeds_html = ""
            all_articles = []
            
            for feed_name, feed_data in feeds_data.items():
                if feed_data.status == 'success':
                    status_icon = "✅"
                    article_count = len(feed_data.articles)
                    all_articles.extend(feed_data.articles)
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
            
            # Sort articles by date (most recent first)
            try:
                # Use a more robust date parsing for sorting
                from dateutil.parser import parse
                all_articles.sort(key=lambda x: parse(x.published, fuzzy=True) if x.published != 'Unknown date' else datetime.min, reverse=True)
            except ImportError:
                print("Warning: 'python-dateutil' not installed. Article sorting by date may be less accurate.")
                # Fallback to string sort if dateutil is not available
                all_articles.sort(key=lambda x: x.published, reverse=True)
            except Exception as e:
                print(f"Warning: Could not sort articles by date. Error: {e}")
                pass  # Keep original order if sorting fails
            
            articles_html = format_articles_html(all_articles[:20])  # Show top 20 articles
            
            return status_html + feeds_html, articles_html
            
        except Exception as e:
            error_msg = f"Error loading feeds: {str(e)}"
            return error_msg, ""
    
    def refresh_feeds(category):
        """Refresh feeds for the selected category."""
        return load_category_feeds(category)
    
    # Create Gradio interface
    with gr.Blocks(title="Advanced RSS Feed Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📰 Advanced RSS Feed Viewer")
        gr.Markdown("Monitor and view RSS feeds from various sources with analytics and export capabilities.")
        
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
                        articles_display = gr.HTML(label="Recent Articles")
                
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
            
            # Export & Analytics Tab
            with gr.TabItem("📤 Export & Analytics"):
                gr.Markdown("### Export feed data and view analytics")
                
                with gr.Row():
                    export_category = gr.Dropdown(
                        choices=["All Categories"] + list(RSS_FEEDS.keys()),
                        label="Select Category to Export",
                        value="All Categories"
                    )
                    export_format = gr.Radio(
                        choices=["CSV", "JSON"],
                        label="Export Format",
                        value="CSV"
                    )
                
                export_btn = gr.Button("📊 Generate Export", variant="primary")
                
                with gr.Row():
                    export_output = gr.File(label="Download Export")
                    chart_output = gr.Image(label="Status Chart")
                
                def generate_export(category, format_type):
                    # Fetch data
                    if category == "All Categories":
                        all_results = {}
                        for cat in RSS_FEEDS.keys():
                            all_results[cat] = fetch_category_feeds_parallel(cat)
                    else:
                        all_results = {category: fetch_category_feeds_parallel(category)}
                    
                    # Generate export
                    if format_type == "CSV":
                        content = analytics.export_to_csv(all_results)
                        filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    else:
                        content = analytics.export_to_json(all_results)
                        filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    # Save file with UTF-8 encoding
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Generate chart
                    chart_image = analytics.create_status_chart(all_results)
                    
                    return filename, chart_image
                
                export_btn.click(
                    fn=generate_export,
                    inputs=[export_category, export_format],
                    outputs=[export_output, chart_output]
                )
            
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

# Monitoring Script
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
        "Billboard": "https://www.billboard.com/feed/",
        "IGN": "https://feeds.ign.com/ign/all",
        "GameSpot": "https://www.gamespot.com/feeds/mashup/",
        "Polygon": "https://www.polygon.com/rss/index.xml"
    },
    
    "🏥 HEALTH & MEDICINE": {
        "WebMD": "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
        "Mayo Clinic": "https://newsnetwork.mayoclinic.org/feed/",
        "Harvard Health": "https://www.health.harvard.edu/blog/feed",
        "Medical News Today": "https://www.medicalnewstoday.com/rss",
        "Healthline": "https://www.healthline.com/rss",
        "CDC": "https://tools.cdc.gov/api/v2/resources/media/132608.rss"
    },
    
    "🔗 BLOCKCHAIN & CRYPTO": {
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "Decrypt": "https://decrypt.co/feed",
        "The Block": "https://www.theblockcrypto.com/rss.xml",
        "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
        "Crypto News": "https://www.crypto-news.net/feed/"
    },
    
    "📊 DATA SCIENCE": {
        "KDnuggets": "https://www.kdnuggets.com/feed",
        "Analytics Vidhya": "https://www.analyticsvidhya.com/feed/",
        "Towards Data Science": "https://towardsdatascience.com/feed",
        "Data Science Central": "https://www.datasciencecentral.com/profiles/blog/feed"
    },
    
    "🌍 WORLD NEWS": {
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "Deutsche Welle": "https://rss.dw.com/rdf/rss-en-all",
        "France24": "https://www.france24.com/en/rss",
        "RT": "https://www.rt.com/rss/",
        "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"
    },
    
    "🍔 FOOD & COOKING": {
        "Food Network": "https://www.foodnetwork.com/feeds/all-latest-recipes.xml",
        "Bon Appétit": "https://www.bonappetit.com/feed/rss",
        "Serious Eats": "https://feeds.feedburner.com/seriouseats/recipes",
        "Epicurious": "https://www.epicurious.com/services/rss/recipes/latest"
    },
    
    "🎨 DESIGN & CREATIVITY": {
        "Behance": "https://feeds.feedburner.com/behance/vorr",
        "Dribbble": "https://dribbble.com/shots/popular.rss",
        "Creative Bloq": "https://www.creativebloq.com/feed",
        "Smashing Magazine": "https://www.smashingmagazine.com/feed/"
    },
    
    "🌱 ENVIRONMENT & SUSTAINABILITY": {
        "TreeHugger": "https://www.treehugger.com/feeds/rss/",
        "Environmental News Network": "https://www.enn.com/rss/",
        "Climate Central": "https://www.climatecentral.org/rss/news.xml",
        "Green Tech Media": "https://www.greentechmedia.com/rss/all"
    }
}


# Core RSS functionality
def fetch_rss_feed(url: str, timeout: int = 10) -> FeedData:
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
        for entry in feed.entries[:10]:  # Limit to 10 most recent articles
            article = Article(
                title=entry.get('title', 'No title'),
                link=entry.get('link', ''),
                published=entry.get('published', 'Unknown date'),
                summary=entry.get('summary', 'No summary available')[:200] + "...",
                author=entry.get('author', 'Unknown author')
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
            executor.submit(fetch_rss_feed, url): name  
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

class FeedAnalytics:
    """Analytics and export functionality for RSS feeds."""
    
    def __init__(self):
        self.feed_history = []
    
    def log_feed_check(self, category: str, feed_name: str, status: str, article_count: int):
        """Log feed check results for analytics."""
        self.feed_history.append({
            'timestamp': datetime.now(),
            'category': category,
            'feed_name': feed_name,
            'status': status,
            'article_count': article_count
        })
    
    def export_to_csv(self, results: Dict[str, Any]) -> str:
        """Export feed results to CSV format."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Category', 'Feed Name', 'Status', 'Articles', 'Last Updated', 'Error'])
        
        # Write data
        for category, feeds in results.items():
            for feed_name, feed_data in feeds.items():
                writer.writerow([
                    category,
                    feed_name,
                    feed_data.status,
                    len(feed_data.articles) if feed_data.status == 'success' else 0,
                    feed_data.last_updated,
                    feed_data.error if feed_data.status == 'error' else ''
                ])
        
        return output.getvalue()
    
    def export_to_json(self, results: Dict[str, Any]) -> str:
        """Export feed results to JSON format."""
        export_data = {}
        
        for category, feeds in results.items():
            export_data[category] = {}
            for feed_name, feed_data in feeds.items():
                export_data[category][feed_name] = {
                    'status': feed_data.status,
                    'articles': [asdict(article) for article in feed_data.articles],
                    'last_updated': feed_data.last_updated,
                    'error': feed_data.error if feed_data.status == 'error' else None
                }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def create_status_chart(self, results: Dict[str, Any]) -> str:
        """Create a status chart as base64 encoded image."""
        categories = []
        working_counts = []
        total_counts = []
        
        for category, feeds in results.items():
            # Remove emojis for chart labels (if any are still present from the category names)
            clean_category = category.replace('🤖 ', '').replace('💻 ', '').replace('🔬 ', '').replace('📰 ', '') \
                                     .replace('💰 ', '').replace('🏈 ', '').replace('🎬 ', '').replace('🏥 ', '') \
                                     .replace('🔗 ', '').replace('📊 ', '').replace('🌍 ', '').replace('🍔 ', '') \
                                     .replace('🎨 ', '').replace('🌱 ', '')
            categories.append(clean_category[:15])
            working = sum(1 for feed in feeds.values() if feed.status == 'success')
            total = len(feeds)
            working_counts.append(working)
            total_counts.append(total)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(categories))
        
        ax.bar([i - 0.2 for i in x], working_counts, 0.4, label='Working', color='#00b894')
        ax.bar([i + 0.2 for i in x], total_counts, 0.4, label='Total', color='#ddd')
        
        ax.set_xlabel('Categories')
        ax.set_ylabel('Number of Feeds')
        ax.set_title('RSS Feed Status by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"

# Main application
def create_enhanced_rss_viewer():
    """Create the main RSS viewer application."""
    
    analytics = FeedAnalytics()
    
    def format_articles_html(articles: List[Article]) -> str:
        """Format articles as HTML."""
        if not articles:
            return "<p>No articles found.</p>"
        
        html = "<div style='max-height: 600px; overflow-y: auto;'>"
        for article in articles:
            html += f"""
            <div style='border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;'>
                <h3><a href='{article.link}' target='_blank' style='color: #2196F3; text-decoration: none;'>{article.title}</a></h3>
                <p style='color: #666; font-size: 0.9em;'>📅 {article.published} | ✍️ {article.author}</p>
                <p>{article.summary}</p>
            </div>
            """
        html += "</div>"
        return html
    
    def load_category_feeds(category):
        """Load feeds for a specific category."""
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
            
            feeds_html = ""
            all_articles = []
            
            for feed_name, feed_data in feeds_data.items():
                if feed_data.status == 'success':
                    status_icon = "✅"
                    article_count = len(feed_data.articles)
                    all_articles.extend(feed_data.articles)
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
            
            # Sort articles by date (most recent first)
            try:
                # Use a more robust date parsing for sorting
                from dateutil.parser import parse
                # Parse date, if it's 'Unknown date', assign datetime.min to push it to the end
                all_articles.sort(key=lambda x: parse(x.published, fuzzy=True) if x.published != 'Unknown date' else datetime.min, reverse=True)
            except ImportError:
                print("Warning: 'python-dateutil' not installed. Article sorting by date may be less accurate. Install with 'pip install python-dateutil'.")
                # Fallback to string sort if dateutil is not available or parsing fails
                all_articles.sort(key=lambda x: x.published, reverse=True)
            except Exception as e:
                print(f"Warning: Could not sort articles by date. Error: {e}")
                pass  # Keep original order if sorting fails
            
            articles_html = format_articles_html(all_articles[:20])  # Show top 20 articles
            
            return status_html + feeds_html, articles_html
            
        except Exception as e:
            error_msg = f"Error loading feeds: {str(e)}"
            return error_msg, ""
    
    def refresh_feeds(category):
        """Refresh feeds for the selected category."""
        return load_category_feeds(category)
    
    # Create Gradio interface
    with gr.Blocks(title="Advanced RSS Feed Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📰 Advanced RSS Feed Viewer")
        gr.Markdown("Monitor and view RSS feeds from various sources with analytics and export capabilities.")
        
        with gr.Tabs():
            # Main Feed Viewer Tab
            with gr.TabItem("📖 Feed Viewer"):
                with gr.Row():
                    category_dropdown = gr.Dropdown(
                        choices=list(RSS_FEEDS.keys()),
                        label="Select Category",
                        value=list(RSS_FEEDS.keys())[0] if RSS_FEEDS else None 
                    )
                    refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        feed_status = gr.HTML(label="Feed Status")
                    with gr.Column(scale=2):
                        articles_display = gr.HTML(label="Recent Articles")
                
                # Load initial data
                if RSS_FEEDS: 
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
            
            # Export & Analytics Tab
            with gr.TabItem("📤 Export & Analytics"):
                gr.Markdown("### Export feed data and view analytics")
                
                with gr.Row():
                    export_category = gr.Dropdown(
                        choices=["All Categories"] + list(RSS_FEEDS.keys()),
                        label="Select Category to Export",
                        value="All Categories"
                    )
                    export_format = gr.Radio(
                        choices=["CSV", "JSON"],
                        label="Export Format",
                        value="CSV"
                    )
                
                export_btn = gr.Button("📊 Generate Export", variant="primary")
                
                with gr.Row():
                    export_output = gr.File(label="Download Export")
                    chart_output = gr.Image(label="Status Chart")
                
                def generate_export(category, format_type):
                    # Fetch data
                    if category == "All Categories":
                        all_results = {}
                        for cat in RSS_FEEDS.keys():
                            all_results[cat] = fetch_category_feeds_parallel(cat)
                    else:
                        all_results = {category: fetch_category_feeds_parallel(category)}
                    
                    # Generate export
                    if format_type == "CSV":
                        content = analytics.export_to_csv(all_results)
                        filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    else:
                        content = analytics.export_to_json(all_results)
                        filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    # Save file with UTF-8 encoding
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Generate chart
                    chart_image = analytics.create_status_chart(all_results)
                    
                    return filename, chart_image
                
                export_btn.click(
                    fn=generate_export,
                    inputs=[export_category, export_format],
                    outputs=[export_output, chart_output]
                )
            
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
        if RSS_FEEDS: 
            app.load(
                fn=lambda: load_category_feeds(list(RSS_FEEDS.keys())[0]),
                outputs=[feed_status, articles_display]
            )
    
    return app

# Monitoring Script
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
        "Billboard": "https://www.billboard.com/feed/",
        "IGN": "https://feeds.ign.com/ign/all",
        "GameSpot": "https://www.gamespot.com/feeds/mashup/",
        "Polygon": "https://www.polygon.com/rss/index.xml"
    },
    
    "🏥 HEALTH & MEDICINE": {
        "WebMD": "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
        "Mayo Clinic": "https://newsnetwork.mayoclinic.org/feed/",
        "Harvard Health": "https://www.health.harvard.edu/blog/feed",
        "Medical News Today": "https://www.medicalnewstoday.com/rss",
        "Healthline": "https://www.healthline.com/rss",
        "CDC": "https://tools.cdc.gov/api/v2/resources/media/132608.rss"
    },
    
    "🔗 BLOCKCHAIN & CRYPTO": {
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "Decrypt": "https://decrypt.co/feed",
        "The Block": "https://www.theblockcrypto.com/rss.xml",
        "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
        "Crypto News": "https://www.crypto-news.net/feed/"
    },
    
    "📊 DATA SCIENCE": {
        "KDnuggets": "https://www.kdnuggets.com/feed",
        "Analytics Vidhya": "https://www.analyticsvidhya.com/feed/",
        "Towards Data Science": "https://towardsdatascience.com/feed",
        "Data Science Central": "https://www.datasciencecentral.com/profiles/blog/feed"
    },
    
    "🌍 WORLD NEWS": {
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "Deutsche Welle": "https://rss.dw.com/rdf/rss-en-all",
        "France24": "https://www.france24.com/en/rss",
        "RT": "https://www.rt.com/rss/",
        "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"
    },
    
    "🍔 FOOD & COOKING": {
        "Food Network": "https://www.foodnetwork.com/feeds/all-latest-recipes.xml",
        "Bon Appétit": "https://www.bonappetit.com/feed/rss",
        "Serious Eats": "https://feeds.feedburner.com/seriouseats/recipes",
        "Epicurious": "https://www.epicurious.com/services/rss/recipes/latest"
    },
    
    "🎨 DESIGN & CREATIVITY": {
        "Behance": "https://feeds.feedburner.com/behance/vorr",
        "Dribbble": "https://dribbble.com/shots/popular.rss",
        "Creative Bloq": "https://www.creativebloq.com/feed",
        "Smashing Magazine": "https://www.smashingmagazine.com/feed/"
    },
    
    "🌱 ENVIRONMENT & SUSTAINABILITY": {
        "TreeHugger": "https://www.treehugger.com/feeds/rss/",
        "Environmental News Network": "https://www.enn.com/rss/",
        "Climate Central": "https://www.climatecentral.org/rss/news.xml",
        "Green Tech Media": "https://www.greentechmedia.com/rss/all"
    }
}


# Docker Configuration
def create_docker_files():
    """Create Docker configuration for easy deployment."""
    
    dockerfile = '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
'''
    
    docker_compose = '''version: '3.8'

services:
  rss-viewer:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
    restart: unless-stopped
'''
    
    requirements_txt = '''gradio>=4.0.0
feedparser>=6.0.10
requests>=2.31.0
python-dateutil>=2.8.2
matplotlib>=3.8.0
pandas>=2.1.0
schedule>=1.2.0
'''
    
    # Write Docker files with UTF-8 encoding
    with open("Dockerfile", "w", encoding='utf-8') as f:
        f.write(dockerfile)
    
    with open("docker-compose.yml", "w", encoding='utf-8') as f:
        f.write(docker_compose)
    
    with open("requirements.txt", "w", encoding='utf-8') as f:
        f.write(requirements_txt)
    
    print("🐳 Created Docker configuration files")
    print("📋 Created requirements.txt")

if __name__ == "__main__":
    try:
        # Create additional files
        create_monitoring_script()
        create_docker_files()
        
        # Launch main application
        print("🚀 Starting Advanced RSS Feed Viewer...")
        app = create_enhanced_rss_viewer()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=False
        )
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Please check that all required packages are installed:")
        print("pip install gradio feedparser requests matplotlib pandas schedule python-dateutil")
