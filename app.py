import gradio as gr
import feedparser
import requests

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
}

def get_ticker_html():
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin,ethereum,dogecoin", "vs_currencies": "usd"}
        )
        prices = resp.json()
        btc = prices.get("bitcoin", {}).get("usd", "N/A")
        eth = prices.get("ethereum", {}).get("usd", "N/A")
        doge = prices.get("dogecoin", {}).get("usd", "N/A")
        nasdaq = "17,800"
        sp500 = "5,500"
        aapl = "210.50"
        return f"""
        <div style="overflow:hidden;background:linear-gradient(90deg,#22243A 0%,#36364e 100%);color:#f5f6fa;padding:8px 0;border-bottom:2px solid #0062ff;">
            <marquee style='font-size:1.08em;font-family:Segoe UI,Arial,sans-serif;letter-spacing:0.5px;'>
                <span style="color:#f7b731;">&#128176; BTC: <b>${btc}</b></span> &nbsp; | &nbsp;
                <span style="color:#45aaf2;">ETH: <b>${eth}</b></span> &nbsp; | &nbsp;
                <span style="color:#00b894;">NASDAQ: <b>{nasdaq}</b></span> &nbsp; | &nbsp;
                <span style="color:#636e72;">S&amp;P 500: <b>{sp500}</b></span> &nbsp; | &nbsp;
                <span style="color:#fd9644;">DOGE: <b>${doge}</b></span> &nbsp; | &nbsp;
                <span style="color:#4b7bec;">AAPL: <b>${aapl}</b></span>
            </marquee>
        </div>
        """
    except Exception:
        return """
        <div style="overflow:hidden;background:linear-gradient(90deg,#22243A 0%,#36364e 100%);color:#f5f6fa;padding:8px 0;border-bottom:2px solid #0062ff;">
            <marquee style='font-size:1.08em;font-family:Segoe UI,Arial,sans-serif;letter-spacing:0.5px;'>
                <span style="color:#f7b731;">&#128176; BTC: <b>$65,200</b></span> &nbsp; | &nbsp;
                <span style="color:#45aaf2;">ETH: <b>$3,500</b></span> &nbsp; | &nbsp;
                <span style="color:#00b894;">NASDAQ: <b>17,800</b></span> &nbsp; | &nbsp;
                <span style="color:#636e72;">S&amp;P 500: <b>5,500</b></span> &nbsp; | &nbsp;
                <span style="color:#fd9644;">DOGE: <b>$0.12</b></span> &nbsp; | &nbsp;
                <span style="color:#4b7bec;">AAPL: <b>$210.50</b></span>
            </marquee>
        </div>
        """

def get_feed_html(feed_url, num_entries=12):
    feed = feedparser.parse(feed_url)
    if feed.bozo:
        return f"<div class='rss-error'>Failed to parse feed: {feed.bozo_exception}</div>"
    entries = feed.entries[:num_entries]
    if not entries:
        return "<div class='rss-empty'>No entries found.</div>"
    html = "<div class='rss-grid'>"
    for entry in entries:
        title = entry.get("title", "No title")
        link = entry.get("link", "#")
        summary = entry.get("summary", "")
        published = entry.get("published", "")
        html += f"""
        <div class='rss-card'>
            <div class='rss-card-header'>
                <a href="{link}" target="_blank" class='rss-card-title'>{title}</a>
            </div>
            <div class='rss-card-summary'>{summary[:180]}{"..." if len(summary)>180 else ""}</div>
            <div class='rss-card-footer'>{published}</div>
        </div>
        """
    html += "</div>"
    return html

with gr.Blocks(css="""
.rss-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 24px 16px;
    margin-top: 28px;
    margin-bottom: 12px;
}
.rss-card {
    background: linear-gradient(105deg,#2c2f4a 0%,#232534 100%);
    border-radius: 17px;
    box-shadow: 0 2px 16px rgba(23,30,60,0.14);
    border: 1.5px solid #28305a;
    padding: 22px 18px 14px 18px;
    color: #f6faff;
    display: flex;
    flex-direction: column;
    min-height: 180px;
    transition: transform 0.18s;
    position: relative;
}
.rss-card:hover {
    transform: translateY(-4px) scale(1.018);
    box-shadow: 0 8px 32px rgba(0,98,255,0.14);
    border-color: #0062ff;
}
.rss-card-header {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 1.18em;
    font-weight: 600;
    margin-bottom: 0.6em;
    color: #fff;
}
.rss-card-title {
    color: #92b5ff;
    text-decoration: none;
    transition: color 0.12s;
}
.rss-card-title:hover {
    color: #fff;
    text-decoration: underline;
}
.rss-card-summary {
    font-size: 0.99em;
    color: #cbd3ea;
    margin-bottom: 1em;
}
.rss-card-footer {
    font-size: 0.93em;
    color: #5a86c6;
    margin-top: auto;
    text-align: right;
}
.rss-error, .rss-empty {
    color: #fff;
    background: #e74c3c;
    padding: 14px;
    border-radius: 12px;
    font-weight: bold;
    margin: 18px 0;
    text-align: center;
}
""") as demo:
    gr.HTML(get_ticker_html())
    gr.Markdown(
        """
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="https://datanacci.carrd.co/assets/images/image01.png" alt="Datanacci" style="height: 60px;">
            <h1 style="margin: 0;">AI-RSS Feed Viewer</h1>
        </div>
        <p style="font-size:1.1em;color:#7ec6ff;margin-top:0;">
            Select a category and feed to view the latest headlines, beautifully presented below.
        </p>
        """
    )
    with gr.Row():
        category = gr.Dropdown(
            choices=list(FEEDS.keys()),
            label="Category",
            value=list(FEEDS.keys())[0]
        )
        feed = gr.Dropdown(
            choices=[name for name, url in FEEDS[list(FEEDS.keys())[0]]],
            label="Feed",
            value=FEEDS[list(FEEDS.keys())[0]][0][0]
        )

    rss_cards = gr.HTML(label="Latest Headlines")

    def update_feeds(selected_category):
        return gr.Dropdown.update(
            choices=[name for name, url in FEEDS[selected_category]],
            value=FEEDS[selected_category][0][0]
        )

    def display_feed(selected_category, selected_feed):
        url = dict(FEEDS[selected_category])[selected_feed]
        return get_feed_html(url)

    category.change(
        fn=update_feeds,
        inputs=category,
        outputs=feed
    )
    category.change(
        fn=lambda c: display_feed(c, FEEDS[c][0][0]),
        inputs=category,
        outputs=rss_cards,
    )
    feed.change(
        fn=display_feed,
        inputs=[category, feed],
        outputs=rss_cards
    )

    # Initial load
    rss_cards.value = get_feed_html(FEEDS[list(FEEDS.keys())[0]][0][1])

if __name__ == "__main__":
    demo.launch()
