import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import gradio as gr
import feedparser
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama
import os

# --- Data Structures ---
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
GLOBAL_ARTICLE_CACHE: Dict[str, List[Article]] = {}
# Global variable to store available Ollama models
OLLAMA_MODELS: List[str] = []


# --- RSS Feed Functions ---
def fetch_rss_feed_single(url: str, feed_name: str, timeout: int = 10) -> FeedData:
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
            executor.submit(fetch_rss_feed_single, url, name): name
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

def get_article_display(feed_data: FeedData) -> str:
    if feed_data.status == "error":
        return f"<p style='color:red;'>Error fetching feed: {feed_data.error}</p>"
    if not feed_data.articles:
        return "<p>No articles found for this feed.</p>"

    html_output = "<ul>"
    for article in feed_data.articles:
        html_output += f"""
        <li>
            <strong><a href="{article.link}" target="_blank">{article.title}</a></strong>
            <br>
            <em>{article.feed_name}</em> - {article.published} (Author: {article.author})
            <br>
            {article.summary}
        </li>
        """
    html_output += "</ul>"
    return html_output

def display_rss_feed_category(category_name: str) -> gr.HTML:
    """Displays articles for a selected category and caches them."""
    global GLOBAL_ARTICLE_CACHE
    all_category_articles: List[Article] = []
    category_results = fetch_category_feeds_parallel(category_name)
    
    output_html = ""
    for feed_name, feed_data in category_results.items():
        output_html += f"<h3>{feed_name}</h3>"
        output_html += get_article_display(feed_data)
        if feed_data.status == "success":
            all_category_articles.extend(feed_data.articles)
    
    GLOBAL_ARTICLE_CACHE[category_name] = all_category_articles
    return gr.HTML(output_html)

def list_cached_categories() -> List[str]:
    """Returns a list of categories currently in the cache."""
    return list(GLOBAL_ARTICLE_CACHE.keys())


# --- Ollama Integration ---
def get_ollama_models() -> List[str]:
    """
    Fetches a list of available Ollama models.
    Handles the actual `ollama._types.ListResponse` object structure.
    """
    try:
        print(f"[{datetime.now()}] Attempting to list Ollama models...")
        models_info = ollama.list()
        print(f"[{datetime.now()}] Raw Ollama list response: {models_info}")

        if not hasattr(models_info, 'models'):
            print(f"[{datetime.now()}] Error: Ollama list response object missing 'models' attribute. Response: {models_info}")
            return ["Error: Ollama response malformed (missing 'models' attribute)."]

        model_list = models_info.models

        if not isinstance(model_list, list):
            print(f"[{datetime.now()}] Error: 'models' attribute is not a list. Type: {type(model_list)}. Value: {model_list}")
            return ["Error: Ollama response malformed ('models' attribute not a list)."]

        if not model_list:
            print(f"[{datetime.now()}] No Ollama models found in the response. Have you pulled any yet? (e.g., 'ollama pull llama2')")
            return ["No models found. Pull models like 'ollama pull gemma3n:e4b'."]

        models = []
        for i, model_entry in enumerate(model_list):
            if not hasattr(model_entry, 'model'):
                print(f"[{datetime.now()}] Warning: Model entry at index {i} missing 'model' attribute. Skipping. Entry: {model_entry}")
                continue
            
            models.append(model_entry.model)

        if not models:
            print(f"[{datetime.now()}] No valid model names extracted after processing Ollama list response.")
            return ["No valid model names extracted."]

        return sorted(list(set(models)))

    except requests.exceptions.ConnectionError:
        print(f"[{datetime.now()}] Error: Connection refused. Is Ollama server running on 127.0.0.1:11434 and accessible?")
        return ["Error: Ollama Server Not Running or Connection Refused."]
    except Exception as e:
        print(f"[{datetime.now()}] An unexpected error occurred while fetching Ollama models: {e}")
        return [f"Error: Could not fetch models. Details: {e}. Check console for more info."]

# Initial fetch of models when the app starts
ollama_available_models = get_ollama_models()
default_ollama_model = ollama_available_models[0] if ollama_available_models and not ollama_available_models[0].startswith("Error") else "No models available / Error fetching"

print(f"Default Ollama Model set to: {default_ollama_model}")
print(f"Available Ollama Models: {ollama_available_models}")

def generate_rss_summary_ollama(selected_category: str, user_query: str, ollama_model: str) -> str:
    """Generates insights from cached RSS articles using Ollama."""
    if "Error" in ollama_model:
        return f"Cannot generate insights: {ollama_model}. Please select a valid Ollama model."

    articles_to_summarize = GLOBAL_ARTICLE_CACHE.get(selected_category, [])

    if not articles_to_summarize:
        return f"No articles found in cache for category '{selected_category}'. Please fetch the feed first."

    # Prepare article data for the LLM prompt
    articles_text = "\n\n".join([
        f"Title: {article.title}\nSource: {article.feed_name}\nPublished: {article.published}\nSummary: {article.summary}"
        for article in articles_to_summarize
    ])

    system_prompt = (
        "You are an AI assistant specialized in summarizing news articles. "
        "Provide a concise and informative response based on the provided articles. "
        "If the user asks a specific question, try to answer it using the article content. "
        "If a question cannot be answered from the articles, state that clearly."
    )

    user_prompt = (
        f"Here are some recent articles from the '{selected_category}' category:\n\n"
        f"{articles_text}\n\n"
        f"Based on these articles, please respond to the following: \"{user_query}\""
    )

    try:
        response = ollama.chat(model=ollama_model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama model '{ollama_model}': {e}. Ensure the model is pulled and running."


# --- Gradio Interface ---

with gr.Blocks(title="Advanced RSS Feed Viewer") as demo: # Changed the main title
    gr.Markdown("# Advanced RSS Feed Viewer & AI Assistant") # Changed the main heading

    # The entire interface is now within this single tab
    with gr.Tab("RSS Feed Browser"):
        gr.Markdown("## Browse Latest News Feeds")
        with gr.Row():
            category_dropdown = gr.Dropdown(
                label="Select News Category",
                choices=list(RSS_FEEDS.keys()),
                value=list(RSS_FEEDS.keys())[0],
                interactive=True
            )
            fetch_category_button = gr.Button("Fetch Category Feeds")
        
        rss_articles_display = gr.HTML(label="Articles")

        gr.Markdown("---")
        gr.Markdown("## RSS Feed Insights (powered by Ollama)")
        with gr.Row():
            ollama_model_dropdown_rss = gr.Dropdown(
                label="Select Ollama Model",
                choices=ollama_available_models,
                value=default_ollama_model,
                interactive=True
            )
            cached_category_dropdown = gr.Dropdown(
                label="Select Cached Category for Insights",
                choices=[], # Initial empty, updated on fetch
                interactive=True
            )
        rss_ollama_query = gr.Textbox(label="Ask a question about the fetched articles:", placeholder="e.g., What are the main headlines?")
        rss_ollama_generate_button = gr.Button("Generate RSS Insights")
        rss_ollama_output = gr.Markdown(label="Ollama RSS Insights")

        # Link RSS fetching to UI
        fetch_category_button.click(
            display_rss_feed_category,
            inputs=[category_dropdown],
            outputs=[rss_articles_display]
        ).success(
            fn=list_cached_categories,
            inputs=[],
            outputs=[cached_category_dropdown]
        )
        
        # Link Ollama for RSS
        rss_ollama_generate_button.click(
            generate_rss_summary_ollama,
            inputs=[cached_category_dropdown, rss_ollama_query, ollama_model_dropdown_rss],
            outputs=rss_ollama_output
        )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=False)
