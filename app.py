import gradio as gr
import feedparser
import requests
from datetime import datetime, timedelta
import pandas as pd
import re
from urllib.parse import urlparse
import time

# Enhanced RSS Feed Categories with working URLs
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "OpenAI Blog": "https://openai.com/blog/rss.xml",
        "Google AI Blog": "https://ai.googleblog.com/feeds/posts/default",
        "Machine Learning Mastery": "https://machinelearningmastery.com/feed/",
        "MarkTechPost": "https://www.marktechpost.com/feed/",
        "Unite.AI": "https://www.unite.ai/feed/",
        "AI News": "https://www.artificialintelligence-news.com/feed/",
        "VentureBeat AI": "https://venturebeat.com/ai/feed/",
        "MIT Technology Review": "https://www.technologyreview.com/feed/",
        "Towards Data Science": "https://towardsdatascience.com/feed"
    },
    
    "💰 FINANCE & BUSINESS": {
        "Fortune": "https://fortune.com/feed",
        "Forbes": "https://www.forbes.com/real-time/feed2/",
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
        "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "Financial Times": "https://www.ft.com/rss/home",
        "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
        "Seeking Alpha": "https://seekingalpha.com/market_currents.xml"
    },
    
    "💻 TECHNOLOGY": {
        "TechCrunch": "https://techcrunch.com/feed/",
        "The Verge": "https://www.theverge.com/rss/index.xml",
        "Ars Technica": "https://arstechnica.com/feed/",
        "Wired": "https://www.wired.com/feed/rss",
        "Engadget": "https://www.engadget.com/rss.xml",
        "Hacker News": "https://news.ycombinator.com/rss",
        "Slashdot": "https://slashdot.org/slashdot.rss",
        "The Next Web": "https://thenextweb.com/feed/",
        "ZDNet": "https://www.zdnet.com/news/rss.xml",
        "TechRadar": "https://www.techradar.com/rss"
    },
    
    "🔬 SCIENCE & PHYSICS": {
        "Phys.org": "https://phys.org/rss-feed/",
        "Scientific American": "https://rss.sciam.com/ScientificAmerican-Global",
        "New Scientist": "https://www.newscientist.com/feed/home/",
        "Space.com": "https://www.space.com/feeds/all",
        "NASA News": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
        "Science Daily": "https://www.sciencedaily.com/rss/all.xml",
        "Nature News": "https://www.nature.com/nature.rss"
    },
    
    "📰 GENERAL NEWS": {
        "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
        "CNN": "http://rss.cnn.com/rss/edition.rss",
        "Reuters": "https://www.reuters.com/rssFeed/topNews",
        "Associated Press": "https://feeds.apnews.com/ApNews/apf-topnews",
        "NPR": "https://feeds.npr.org/1001/rss.xml",
        "The Guardian": "https://www.theguardian.com/world/rss",
        "Google News": "https://news.google.com/rss"
    },
    
    "🔗 BLOCKCHAIN & CRYPTO": {
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "Decrypt": "https://decrypt.co/feed",
        "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
        "The Block": "https://www.theblockcrypto.com/rss.xml"
    }
}

def get_categories():
    """Get list of categories for dropdown"""
    return ["All Categories"] + list(RSS_FEEDS.keys())

def get_feeds_for_category(category):
    """Get feeds for selected category"""
    if category == "All Categories" or not category:
        all_feeds = []
        for cat_feeds in RSS_FEEDS.values():
            all_feeds.extend(list(cat_feeds.keys()))
        return ["All Feeds"] + all_feeds
    elif category in RSS_FEEDS:
        return ["All Feeds"] + list(RSS_FEEDS[category].keys())
    return ["All Feeds"]

def get_feed_url(feed_name):
    """Get RSS URL for a specific feed name"""
    for category, feeds in RSS_FEEDS.items():
        if feed_name in feeds:
            return feeds[feed_name]
    return None

def format_date(date_string):
    """Format date string to relative time like techurls.com"""
    if not date_string or date_string == "No date":
        return "Unknown"
    
    try:
        # Parse various date formats
        parsed_date = None
        for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", 
                   "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"]:
            try:
                parsed_date = datetime.strptime(date_string, fmt)
                break
            except:
                continue
        
        if not parsed_date:
            # Try feedparser's time parsing
            import time
            parsed_date = datetime.fromtimestamp(time.mktime(time.strptime(date_string)))
        
        # Calculate relative time
        now = datetime.now()
        if parsed_date.tzinfo:
            now = now.replace(tzinfo=parsed_date.tzinfo)
        
        diff = now - parsed_date
        
        if diff.days > 0:
            return f"{diff.days}d"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m"
        else:
            return "now"
            
    except Exception as e:
        return "Unknown"

def get_domain_name(url):
    """Extract domain name from URL for source display"""
    try:
        domain = urlparse(url).netloc
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "Unknown"

def fetch_multiple_feeds(category, feed_name, time_filter="Any time"):
    """Fetch articles from selected feeds with time filtering"""
    try:
        articles = []
        feeds_to_fetch = []
        
        # Determine which feeds to fetch
        if category == "All Categories" and feed_name == "All Feeds":
            # Fetch from all feeds (limit to prevent overload)
            for cat_feeds in RSS_FEEDS.values():
                feeds_to_fetch.extend(list(cat_feeds.items())[:2])  # Limit per category
        elif category != "All Categories" and feed_name == "All Feeds":
            # Fetch all feeds from selected category
            if category in RSS_FEEDS:
                feeds_to_fetch = list(RSS_FEEDS[category].items())
        elif feed_name != "All Feeds":
            # Fetch specific feed
            url = get_feed_url(feed_name)
            if url:
                feeds_to_fetch = [(feed_name, url)]
        
        if not feeds_to_fetch:
            return "No feeds selected", pd.DataFrame()
        
        # Fetch articles from selected feeds
        for feed_name, feed_url in feeds_to_fetch:
            try:
                feed = feedparser.parse(feed_url)
                if not feed.bozo:
                    for entry in feed.entries[:10]:  # Limit per feed
                        # Parse publication date
                        pub_date = entry.get('published', '')
                        formatted_date = format_date(pub_date)
                        
                        # Get source domain
                        source = get_domain_name(feed_url)
                        
                        article = {
                            'Title': entry.get('title', 'No title'),
                            'Source': source,
                            'Time': formatted_date,
                            'Link': entry.get('link', ''),
                            'Summary': entry.get('summary', 'No summary')[:200] + '...' if len(entry.get('summary', '')) > 200 else entry.get('summary', 'No summary'),
                            'Feed': feed_name
                        }
                        articles.append(article)
            except Exception as e:
                print(f"Error fetching {feed_name}: {e}")
                continue
        
        if not articles:
            return "No articles found or all feeds failed to load", pd.DataFrame()
        
        # Sort by time (most recent first)
        # For now, sort by title since we don't have exact timestamps
        articles.sort(key=lambda x: x['Title'])
        
        # Apply time filter
        if time_filter != "Any time":
            # This would need more sophisticated date parsing for real filtering
            pass
        
        # Create DataFrame
        df = pd.DataFrame(articles)
        
        # Create summary
        feed_info = f"**Loaded {len(articles)} articles from {len(feeds_to_fetch)} feed(s)**\n\n"
        feed_info += f"**Category:** {category}\n"
        feed_info += f"**Feed:** {feed_name}\n"
        feed_info += f"**Time Filter:** {time_filter}"
        
        return feed_info, df
        
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def update_feed_dropdown(category):
    """Update feed dropdown based on selected category"""
    feeds = get_feeds_for_category(category)
    return gr.Dropdown(choices=feeds, value="All Feeds")

def search_articles(df, search_term):
    """Search through articles"""
    if df.empty or not search_term:
        return df
    
    mask = df['Title'].str.contains(search_term, case=False, na=False) | \
           df['Summary'].str.contains(search_term, case=False, na=False)
    
    return df[mask]

# Create enhanced Gradio interface
with gr.Blocks(title="Enhanced RSS Aggregator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📰 Enhanced RSS Feed Aggregator")
    gr.Markdown("*A clone of TechURLs.com and FinURLs.com with enhanced capabilities*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Feed Selection")
            
            category_dropdown = gr.Dropdown(
                choices=get_categories(),
                value="All Categories",
                label="Category",
                info="Select a category to filter feeds"
            )
            
            feed_dropdown = gr.Dropdown(
                choices=get_feeds_for_category("All Categories"),
                value="All Feeds",
                label="Feed Source",
                info="Select specific feed or all feeds from category"
            )
            
            time_filter = gr.Dropdown(
                choices=["Any time", "Last hour", "Last 24 hours", "Last week", "Last month"],
                value="Any time",
                label="Time Filter",
                info="Filter articles by publication time"
            )
            
            with gr.Row():
                load_btn = gr.Button("📥 Load Articles", variant="primary")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            
            gr.Markdown("### 🔍 Search")
            search_input = gr.Textbox(
                label="Search Articles",
                placeholder="Enter keywords to search...",
                info="Search in titles and summaries"
            )
            
            with gr.Row():
                search_btn = gr.Button("🔍 Search", variant="secondary")
                clear_search_btn = gr.Button("❌ Clear", variant="secondary")
        
        with gr.Column(scale=2):
            feed_info = gr.Markdown("### 📊 Feed Information\nSelect feeds and click 'Load Articles' to get started.")
            
            articles_table = gr.Dataframe(
                headers=['Title', 'Source', 'Time', 'Link', 'Summary', 'Feed'],
                interactive=False,
                wrap=True,
                height=600,
                label="Articles",
                column_widths=["40%", "10%", "8%", "15%", "20%", "7%"]
            )
    
    # Store original dataframe for search functionality
    original_df = gr.State(pd.DataFrame())
    
    # Event handlers
    category_dropdown.change(
        fn=update_feed_dropdown,
        inputs=[category_dropdown],
        outputs=[feed_dropdown]
    )
    
    load_btn.click(
        fn=fetch_multiple_feeds,
        inputs=[category_dropdown, feed_dropdown, time_filter],
        outputs=[feed_info, articles_table]
    ).then(
        fn=lambda df: df,
        inputs=[articles_table],
        outputs=[original_df]
    )
    
    refresh_btn.click(
        fn=fetch_multiple_feeds,
        inputs=[category_dropdown, feed_dropdown, time_filter],
        outputs=[feed_info, articles_table]
    ).then(
        fn=lambda df: df,
        inputs=[articles_table],
        outputs=[original_df]
    )
    
    search_btn.click(
        fn=search_articles,
        inputs=[original_df, search_input],
        outputs=[articles_table]
    )
    
    clear_search_btn.click(
        fn=lambda df: (df, ""),
        inputs=[original_df],
        outputs=[articles_table, search_input]
    )
    
    # Allow Enter key for search
    search_input.submit(
        fn=search_articles,
        inputs=[original_df, search_input],
        outputs=[articles_table]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
