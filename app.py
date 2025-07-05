
import gradio as gr
import feedparser
import requests
from datetime import datetime
import pandas as pd

# Comprehensive RSS Feed Categories
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "Science Daily - AI": "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        "Science Daily - Technology": "https://www.sciencedaily.com/rss/top/technology.xml",
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

def get_all_feeds_list():
    """Get a flat list of all feeds for dropdown"""
    feeds_list = []
    for category, feeds in RSS_FEEDS.items():
        for name, url in feeds.items():
            feeds_list.append(f"{category} - {name}")
    return feeds_list

def get_feed_url_by_name(feed_name):
    """Get RSS URL by feed name"""
    for category, feeds in RSS_FEEDS.items():
        for name, url in feeds.items():
            if f"{category} - {name}" == feed_name:
                return url
    return None

def fetch_rss_feed(url):
    """Fetch and parse RSS feed from given URL"""
    try:
        # Parse the RSS feed
        feed = feedparser.parse(url)
        
        if feed.bozo:
            return "Error: Invalid RSS feed or URL not accessible", pd.DataFrame()
        
        # Extract feed information
        feed_title = feed.feed.get('title', 'Unknown Feed')
        feed_description = feed.feed.get('description', 'No description available')
        
        # Extract entries
        entries = []
        for entry in feed.entries[:15]:  # Increased to 15 most recent entries
            entry_data = {
                'Title': entry.get('title', 'No title'),
                'Link': entry.get('link', 'No link'),
                'Published': entry.get('published', 'No date'),
                'Summary': entry.get('summary', 'No summary')[:300] + '...' if len(entry.get('summary', '')) > 300 else entry.get('summary', 'No summary')
            }
            entries.append(entry_data)
        
        # Create DataFrame for better display
        df = pd.DataFrame(entries)
        
        feed_info = f"**Feed:** {feed_title}\n**Description:** {feed_description}\n**Total Articles:** {len(entries)}\n**Feed URL:** {url}"
        
        return feed_info, df
        
    except Exception as e:
        return f"Error fetching RSS feed: {str(e)}", pd.DataFrame()

def search_entries(df, search_term):
    """Search through RSS entries"""
    if df.empty or not search_term:
        return df
    
    # Search in title and summary columns
    mask = df['Title'].str.contains(search_term, case=False, na=False) | \
           df['Summary'].str.contains(search_term, case=False, na=False)
    
    filtered_df = df[mask]
    return filtered_df

def clear_search(original_df):
    """Clear search and return original dataframe"""
    return original_df, ""

def load_preset_feed(feed_name):
    """Load a preset RSS feed"""
    if not feed_name:
        return "", pd.DataFrame(), pd.DataFrame()
    
    url = get_feed_url_by_name(feed_name)
    if url:
        feed_info, df = fetch_rss_feed(url)
        return url, feed_info, df, df
    return "", "Feed not found", pd.DataFrame(), pd.DataFrame()

# Create Gradio interface
with gr.Blocks(title="RSS Feed Helper", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📰 RSS Feed Helper")
    gr.Markdown("Choose from hundreds of curated RSS feeds or enter your own URL to fetch and display the latest articles.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Quick Select")
            preset_dropdown = gr.Dropdown(
                choices=get_all_feeds_list(),
                label="Choose from Curated Feeds",
                info="Select from our comprehensive collection of RSS feeds",
                interactive=True
            )
            load_preset_btn = gr.Button("📥 Load Selected Feed", variant="primary")
            
            gr.Markdown("### 🔗 Custom URL")
            url_input = gr.Textbox(
                label="RSS Feed URL",
                placeholder="https://example.com/rss.xml",
                info="Enter any valid RSS feed URL"
            )
            
            with gr.Row():
                fetch_btn = gr.Button("🔄 Fetch RSS Feed", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            gr.Markdown("### 🔍 Search Articles")
            search_input = gr.Textbox(
                label="Search Keywords",
                placeholder="Enter keywords to search in titles and summaries...",
                info="Search is case-insensitive"
            )
            
            with gr.Row():
                search_btn = gr.Button("🔍 Search", variant="secondary")
                clear_search_btn = gr.Button("❌ Clear Search", variant="secondary")
    
        with gr.Column(scale=2):
            feed_info = gr.Markdown("### Feed Information\nSelect a preset feed or enter an RSS URL to get started.")
            
            articles_table = gr.Dataframe(
                headers=['Title', 'Link', 'Published', 'Summary'],
                interactive=False,
                wrap=True,
                height=500,
                label="Articles"
            )
    
    # Store the original dataframe for searching
    original_df = gr.State(pd.DataFrame())
    
    # Event handlers
    load_preset_btn.click(
        fn=load_preset_feed,
        inputs=[preset_dropdown],
        outputs=[url_input, feed_info, articles_table, original_df]
    )
    
    fetch_btn.click(
        fn=fetch_rss_feed,
        inputs=[url_input],
        outputs=[feed_info, articles_table]
    ).then(
        fn=lambda df: df,
        inputs=[articles_table],
        outputs=[original_df]
    )
    
    search_btn.click(
        fn=search_entries,
        inputs=[original_df, search_input],
        outputs=[articles_table]
    )
    
    clear_search_btn.click(
        fn=clear_search,
        inputs=[original_df],
        outputs=[articles_table, search_input]
    )
    
    clear_btn.click(
        fn=lambda: ("### Feed Information\nSelect a preset feed or enter an RSS URL to get started.", pd.DataFrame(), pd.DataFrame(), "", ""),
        outputs=[feed_info, articles_table, original_df, search_input, url_input]
    )
    
    # Allow Enter key to trigger fetch
    url_input.submit(
        fn=fetch_rss_feed,
        inputs=[url_input],
        outputs=[feed_info, articles_table]
    ).then(
        fn=lambda df: df,
        inputs=[articles_table],
        outputs=[original_df]
    )
    
    # Allow Enter key to trigger search
    search_input.submit(
        fn=search_entries,
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
