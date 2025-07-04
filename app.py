import gradio as gr
import feedparser

# Dictionary of categories and their RSS feeds
FEEDS = {
    "AI & Technology": [
        ("ScienceDaily - All", "https://www.sciencedaily.com/rss/all.xml"),
        ("ScienceDaily - Technology", "https://www.sciencedaily.com/rss/top/technology.xml"),
        ("O'Reilly Radar", "https://feeds.feedburner.com/oreilly-radar"),
        ("Google Blog AI", "https://blog.google/products/ai/rss"),
        ("OpenAI Blog", "https://openai.com/blog/rss.xml"),
        ("DeepMind Blog", "https://deepmind.com/blog/feed/basic/"),
        ("Google AI Blog", "https://ai.googleblog.com/feeds/posts/default"),
        ("Microsoft AI Blog", "https://blogs.microsoft.com/ai/feed/"),
        ("Machine Learning Mastery", "https://machinelearningmastery.com/feed/"),
        ("MarkTechPost", "https://www.marktechpost.com/feed/"),
        ("BAIR Blog", "https://bair.berkeley.edu/blog/feed.xml"),
        ("Distill", "https://distill.pub/rss.xml"),
        ("Unite.AI", "https://www.unite.ai/feed/"),
        ("AI News", "https://www.artificialintelligence-news.com/feed/"),
        ("VentureBeat AI", "https://venturebeat.com/ai/feed/"),
        ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
        ("IEEE Spectrum", "https://spectrum.ieee.org/rss/fulltext"),
    ],
    "Finance & Fintech": [
        ("Investing.com", "https://www.investing.com/rss/news.rss"),
        ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
        ("Fortune", "https://fortune.com/feed"),
        ("Forbes Business", "https://www.forbes.com/business/feed/"),
        ("Economic Times", "https://economictimes.indiatimes.com/rssfeedsdefault.cms"),
        ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
        ("Financial Samurai", "https://www.financialsamurai.com/feed/"),
        ("NerdWallet", "https://www.nerdwallet.com/blog/feed/"),
        ("Money Under 30", "https://www.moneyunder30.com/feed"),
    ],
    "Physics & Science": [
        ("Phys.org", "https://phys.org/rss-feed/"),
        ("Nature", "https://www.nature.com/nature.rss"),
        ("APS PRL", "https://feeds.aps.org/rss/recent/prl.xml"),
        ("Scientific American", "https://rss.sciam.com/ScientificAmerican-Global"),
        ("New Scientist", "https://www.newscientist.com/feed/home/"),
        ("Physics World", "https://physicsworld.com/feed/"),
        ("Symmetry Magazine", "https://www.symmetrymagazine.org/rss/all-articles.xml"),
        ("Space.com", "https://www.space.com/feeds/all"),
        ("NASA", "https://www.nasa.gov/rss/dyn/breaking_news.rss"),
        ("Sky & Telescope", "https://www.skyandtelescope.com/feed/"),
    ],
    "Technology": [
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("The Verge", "https://www.theverge.com/rss/index.xml"),
        ("Ars Technica", "https://arstechnica.com/feed/"),
        ("Wired", "https://www.wired.com/feed/rss"),
        ("Gizmodo", "https://gizmodo.com/rss"),
        ("Engadget", "https://www.engadget.com/rss.xml"),
        ("Hacker News", "https://news.ycombinator.com/rss"),
        ("Slashdot", "https://slashdot.org/slashdot.rss"),
        ("Reddit Technology", "https://www.reddit.com/r/technology/.rss"),
        ("The Next Web", "https://thenextweb.com/feed/"),
    ],
    "General News": [
        ("NY Times", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
        ("BBC News", "http://feeds.bbci.co.uk/news/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/world/rss"),
        ("CNN", "http://rss.cnn.com/rss/edition.rss"),
        ("Washington Post", "https://feeds.washingtonpost.com/rss/world"),
        ("Google News", "https://news.google.com/rss"),
        ("Reuters", "https://www.reuters.com/rssFeed/topNews"),
        ("WSJ", "https://www.wsj.com/xml/rss/3_7085.xml"),
    ],
    "Datanacci & Data Science": [
        ("Enterprise AI World", "https://www.enterpriseaiworld.com/RSS-Feeds"),
        ("AI Blog", "https://www.artificial-intelligence.blog/rss-feeds"),
        ("KDnuggets", "https://feeds.feedburner.com/kdnuggets-data-mining-analytics"),
        ("Analytics Vidhya", "https://www.analyticsvidhya.com/feed/"),
        ("Towards Data Science", "https://towardsdatascience.com/feed"),
        ("Data Science Central", "https://www.datasciencecentral.com/profiles/blog/feed"),
        ("KDnuggets (Main)", "https://www.kdnuggets.com/feed"),
        ("Machine Learning Mastery", "https://machinelearningmastery.com/feed/"),
    ],
    "Blockchain & Crypto": [
        ("Cointelegraph", "https://cointelegraph.com/rss"),
        ("Coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Decrypt", "https://decrypt.co/feed"),
        ("The Block", "https://www.theblockcrypto.com/rss.xml"),
        ("Bitcoin Magazine", "https://bitcoinmagazine.com/.rss/full/"),
        ("Crypto News", "https://www.crypto-news.net/feed/"),
    ]
}

def get_feed_entries(feed_url):
    feed = feedparser.parse(feed_url)
    if feed.bozo:
        return f"Failed to parse feed: {feed.bozo_exception}"
    entries = feed.entries[:5]  # Display first 5 entries
    result = ""
    for entry in entries:
        title = entry.get("title", "No title")
        link = entry.get("link", "#")
        summary = entry.get("summary", "")
        result += f"### [{title}]({link})\n{summary}\n\n"
    return result or "No entries found."

def show_feed(category, feed_name):
    # Find the feed URL from FEEDS dictionary
    url = None
    for name, link in FEEDS.get(category, []):
        if name == feed_name:
            url = link
            break
    if url:
        return get_feed_entries(url)
    return "Feed not found."

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="https://datanacci.carrd.co/assets/images/image01.png" alt="Datanacci" style="height: 60px;">
            <h1 style="margin: 0;">AI-RSS Feed Viewer</h1>
        </div>
        <p>Select a category and feed to view the latest headlines.</p>
        """
    )
    with gr.Row():
        category = gr.Dropdown(
            choices=list(FEEDS.keys()),
            label="Category",
            value="AI & Technology"
        )
        feed = gr.Dropdown(
            choices=[name for name, url in FEEDS["AI & Technology"]],
            label="Feed",
            value=FEEDS["AI & Technology"][0][0]
        )
    output = gr.Markdown()
    def update_feed_options(selected_category):
        return gr.Dropdown.update(
            choices=[name for name, url in FEEDS[selected_category]],
            value=FEEDS[selected_category][0][0]
        )
    category.change(update_feed_options, category, feed)
    # When feed is changed, show headlines
    gr.on(
        triggers=[category, feed],
        fn=lambda c, f: show_feed(c, f),
        inputs=[category, feed],
        outputs=output
    )
    # Also trigger on startup
    output.value = show_feed("AI & Technology", FEEDS["AI & Technology"][0][0])

if __name__ == "__main__":
    demo.launch()
