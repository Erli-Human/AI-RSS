import feedparser
from gradio import interfaces, blocks
import gradio as gr
import threading

class RSSFeed:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.entries = []

    def fetch(self):
        parsed_feed = feedparser.parse(self.url)
        for entry in parsed_feed.entries:
            self.entries.append(entry)

# Sample function to refresh feeds
def refresh_feeds(feeds):
    for feed in feeds.values():
        feed.fetch()

# List of RSS feeds with categories
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "Science Daily - AI": "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        # Add more feeds here...
    },
    
    "💰 FINANCE & BUSINESS": {
        "Investing.com": "https://www.investing.com/rss/news.rss",
        # Add more feeds here...
    },
    
    "🔬 SCIENCE & PHYSICS": {
        "Phys.org": "https://phys.org/rss-feed/",
        # Add more feeds here...
    },
    
    "💻 TECHNOLOGY": {
        "TechCrunch": "https://techcrunch.com/feed/",
        # Add more feeds here...
    },
    
    "📰 GENERAL NEWS": {
        "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
        # Add more feeds here...
    },
    
    "🏈 SPORTS": {
        "ESPN": "https://www.espn.com/espn/rss/news",
        # Add more feeds here...
    },
    
    "🎬 ENTERTAINMENT": {
        "Entertainment Weekly": "https://ew.com/feed/",
        # Add more feeds here...
    },
    
    "🏥 HEALTH & MEDICINE": {
        "WebMD": "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
        # Add more feeds here...
    },
    
    "🔗 BLOCKCHAIN & CRYPTO": {
        "CoinTelegraph": "https://cointelegraph.com/rss",
        # Add more feeds here...
    },
    
    "📊 DATA SCIENCE": {
        "KDnuggets": "https://www.kdnuggets.com/feed",
        # Add more feeds here...
    },
    
    "🌍 WORLD NEWS": {
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        # Add more feeds here...
    },
    
    "🍔 FOOD & COOKING": {
        "Food Network": "https://www.foodnetwork.com/feeds/all-latest-recipes.xml",
        # Add more feeds here...
    },
    
    "🎨 DESIGN & CREATIVITY": {
        "Behance": "https://feeds.feedburner.com/behance/vorr",
        # Add more feeds here...
    },
    
    "🌱 ENVIRONMENT & SUSTAINABILITY": {
        "TreeHugger": "https://www.treehugger.com/feeds/rss/",
        # Add more feeds here...
    }
}

# Initialize feeds
class FeedManager:
    def __init__(self):
        self.feeds = {name: RSSFeed(name, url) for name, feeds in RSS_FEEDS.items() for url in feeds.values()}
    
    def refresh(self):
        threading.Thread(target=self.refresh_feeds).start()
        
    def refresh_feeds(self):
        # Placeholder function to fetch feeds
        print("Refreshing feeds...")
        # Implement actual fetching logic here if needed

def main():
    # Create a FeedManager instance
    feed_manager = FeedManager()
    
    # Define the UI components
    title_block = blocks.Textbox(label="Title")
    description_block = blocks.Textbox(label="Description")
    
    interface = gr.Interface(
        fn=lambda title, description: (title, description),  # Example function to display text
        inputs=[title_block, description_block],
        outputs=[blocks.Label(label="Feed Item"), blocks.Label(label="Feed Description")],
        title="RSS Reader",
        layout=gr.Layout(direction='rows', columns=1),
    )
    
    interface.launch(share=True)

if __name__ == "__main__":
    main()
