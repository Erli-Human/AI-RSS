import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
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
    feed_name: str = "" # Store the feed name with the article

@dataclass
class FeedData:
    status: str
    articles: List[Article]
    last_updated: str
    error: str = ""

# RSS Feed Sources (as defined previously)
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

# Global cache for fetched articles
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
                summary=entry.get('summary', 'No summary available'), # Keep full summary for Ollama
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

def fetch_all_feeds_parallel() -> Dict[str, Dict[str, FeedData]]:
    """Fetches all RSS feeds across all categories using parallel processing."""
    all_results: Dict[str, Dict[str, FeedData]] = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor: # Increased max_workers for more parallelism
        future_to_category = {}
        for category, feeds in RSS_FEEDS.items():
            futures_for_category = {
                executor.submit(fetch_rss_feed_single, url, name): name
                for name, url in feeds.items()
            }
            future_to_category[category] = futures_for_category

        for category, futures_for_category in future_to_category.items():
            category_results = {}
            for future in as_completed(futures_for_category):
                feed_name = futures_for_category[future]
                try:
                    category_results[feed_name] = future.result()
                except Exception as e:
                    category_results[feed_name] = FeedData(
                        status="error",
                        articles=[],
                        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        error=f"Processing error for {feed_name}: {str(e)}"
                    )
            all_results[category] = category_results
    return all_results

def get_article_preview(article: Article, preview_lines: int = 3) -> str:
    """Generates a concise preview of an article, including a clickable link and feed name."""
    # Ensure no HTML tags in the summary for plain text display and truncate
    clean_summary = gr.HTML(article.summary).value.replace('<p>', '').replace('</p>', '').strip()
    
    # Use Markdown for clickable link
    title_line = f"**[{article.title}]({article.link})**"
    
    # Clearly identify the feed name
    info_line = f"**Feed:** {article.feed_name} - {article.published} (Author: {article.author})"
    
    # Simple character-based truncation for summary preview
    max_chars_per_line = 90 
    summary_preview_words = []
    current_length = 0
    line_count = 0

    for word in clean_summary.split():
        if current_length + len(word) + 1 <= max_chars_per_line:
            summary_preview_words.append(word)
            current_length += len(word) + 1
        else:
            line_count += 1
            if line_count >= preview_lines - 1: # -1 because title and info already take two lines
                break
            summary_preview_words.append("\n" + word) # Add newline
            current_length = len(word) + 1
    
    summary_preview = " ".join(summary_preview_words).strip()
    if len(summary_preview) < len(clean_summary): # Check if original summary was longer
        summary_preview += "..." # Indicate truncation

    return f"{title_line}\n{info_line}\nSummary: {summary_preview}"


def update_all_feed_tabs() -> Tuple[Any, ...]:
    """
    Fetches all feeds and updates the Gradio Textbox components for each category tab.
    Returns a tuple of gr.Textbox components, one for each category.
    """
    global GLOBAL_ARTICLE_CACHE
    all_feed_data_by_category = fetch_all_feeds_parallel()
    
    outputs = []
    
    for category_name in RSS_FEEDS.keys():
        category_articles_for_cache: List[Article] = []
        feed_outputs_for_category = ""
        
        category_feeds = all_feed_data_by_category.get(category_name, {})
        
        for feed_name, feed_data in category_feeds.items():
            if feed_data.status == "error":
                feed_outputs_for_category += f"**{feed_name}**: Error: {feed_data.error}\n\n"
            elif not feed_data.articles:
                feed_outputs_for_category += f"**{feed_name}**: No articles found.\n\n"
            else:
                # Add all articles from this feed to the cache for the current category
                # This ensures Ollama has full summaries, even if we only display 5
                category_articles_for_cache.extend(feed_data.articles) 
                
                # Sort articles by published date (if available) and take the top 5
                # Ensure 'published' is parseable for sorting. If not, fallback.
                def get_sort_key(article):
                    try:
                        # Attempt to parse as the standard format we use
                        return datetime.strptime(article.published, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        try:
                            # Attempt to parse common RSS date formats
                            parsed_date = feedparser._parse_date(article.published)
                            if parsed_date:
                                return datetime(*parsed_date[:6])
                        except Exception:
                            pass
                        return datetime.min # Fallback for unparseable dates

                display_articles = sorted(feed_data.articles, key=get_sort_key, reverse=True)[:5]
                
                feed_content = "\n\n---\n\n".join([get_article_preview(art) for art in display_articles])
                feed_outputs_for_category += f"{feed_content}\n\n" # Removed bolding on feed_name here as it's in get_article_preview

        # Store all fetched articles (full content) for this category in the global cache
        GLOBAL_ARTICLE_CACHE[category_name] = category_articles_for_cache
        
        # Append the Textbox component update for the current category's tab.
        outputs.append(gr.Textbox(value=feed_outputs_for_category.strip()))
        
    return tuple(outputs)


def list_cached_categories() -> gr.Dropdown:
    """Returns an updated Gradio Dropdown component with cached categories."""
    current_cached_categories = list(GLOBAL_ARTICLE_CACHE.keys())
    # Preserve the current value if it's still in the choices, otherwise set to None
    return gr.Dropdown(choices=current_cached_categories, 
                       value=None) # Set initial value to None for clarity on refresh


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


def generate_rss_summary_ollama(selected_category: str, user_query: str, ollama_model: str, chat_history: List[List[str]]) -> Tuple[List[List[str]], str]:
    """Generates insights from cached RSS articles using Ollama, maintaining chat history."""
    
    if not selected_category: # Handle case where no category is selected initially
        response_text = "Please select a category from the 'Select Cached Category for Insights' dropdown."
        chat_history.append([user_query, response_text])
        return chat_history, ""

    if "Error" in ollama_model or "No models available" in ollama_model:
        error_message = f"Cannot generate insights: {ollama_model}. Please select a valid Ollama model and ensure Ollama server is running."
        chat_history.append([user_query, error_message])
        return chat_history, ""

    articles_to_summarize = GLOBAL_ARTICLE_CACHE.get(selected_category, [])

    if not articles_to_summarize:
        response_text = f"No articles found in cache for category '{selected_category}'. Please click 'Fetch All Feeds' first and select a category with fetched articles."
        chat_history.append([user_query, response_text])
        return chat_history, ""

    # Prepare article data for the LLM prompt
    # Providing more detail to the LLM than just the preview
    articles_text = "\n\n---\n\n".join([
        f"Title: {article.title}\nLink: {article.link}\nSource: {article.feed_name}\nPublished: {article.published}\nFull Summary: {article.summary}"
        for article in articles_to_summarize
    ])

    system_prompt = (
        "You are an AI assistant specialized in summarizing news articles and answering questions about them. "
        "Provide concise, informative, and relevant responses based *only* on the provided articles. "
        "If a question cannot be answered from the provided articles, state that you don't have enough information from the given context. "
        "Maintain a helpful and neutral tone. Prioritize the most recent and relevant information."
    )

    full_query_with_context = (
        f"Here are recent articles from the '{selected_category}' category. Use only the information provided in these articles to answer the user's questions:\n\n"
        f"{articles_text}\n\n"
        f"User's request: \"{user_query}\"\n\n"
        "Please provide your response based on these articles, maintaining the conversation flow."
    )

    messages = [{'role': 'system', 'content': system_prompt}]
    
    # Add previous chat history to maintain context
    for human_msg, ai_msg in chat_history:
        messages.append({'role': 'user', 'content': human_msg})
        messages.append({'role': 'assistant', 'content': ai_msg})
    
    messages.append({'role': 'user', 'content': full_query_with_context})

    try:
        response = ollama.chat(model=ollama_model, messages=messages)
        ai_response = response['message']['content']
        chat_history.append([user_query, ai_response])
        return chat_history, "" # Clear the textbox after sending
    except Exception as e:
        error_message = f"Error communicating with Ollama model '{ollama_model}': {e}. Ensure the model is pulled and running."
        chat_history.append([user_query, error_message])
        return chat_history, "" # Keep textbox cleared or return original query? Let's clear.


# --- Gradio Interface ---

with gr.Blocks(title="Advanced RSS Feed Viewer & AI Assistant") as demo:
    gr.Markdown("# Advanced RSS Feed Viewer & AI Assistant")

    # Define all the Textbox components that will be updated
    category_html_outputs: Dict[str, gr.Textbox] = {}

    with gr.Tab("RSS Feed Browser"):
        gr.Markdown("## Latest News Across Categories")
        fetch_all_button = gr.Button("Fetch All Feeds (This may take a moment)")
        
        with gr.Tabs() as category_tabs:
            for category_name in RSS_FEEDS.keys():
                with gr.Tab(category_name, id=f"tab_{category_name.replace(' ', '_').replace('&', 'and').lower()}"):
                    # Initialize Textbox components for each category
                    category_html_outputs[category_name] = gr.Textbox(
                        label=f"Articles for {category_name}",
                        lines=10, 
                        max_lines=20, 
                        interactive=False, 
                        value="Click 'Fetch All Feeds' to load articles..."
                    )

    with gr.Tab("AI News Insights"):
        gr.Markdown("## Get AI Insights from Fetched RSS Articles")
        with gr.Row():
            ollama_model_dropdown_rss = gr.Dropdown(
                label="Select Ollama Model",
                choices=ollama_available_models,
                value=default_ollama_model,
                interactive=True
            )
            # Initialize with all possible categories from RSS_FEEDS keys
            # Set value to None initially to avoid "value not in choices" error
            cached_category_dropdown = gr.Dropdown(
                label="Select Cached Category for Insights",
                choices=list(RSS_FEEDS.keys()), # Initial choices from RSS_FEEDS keys
                value=None, # Set initial value to None
                interactive=True
            )
        
        # Chat interface for Ollama
        chatbot = gr.Chatbot(label="Ollama Chat History", height=300)
        msg = gr.Textbox(label="Ask a question about the articles:", placeholder="e.g., What are the key trends in AI research?")
        clear = gr.ClearButton([msg, chatbot])

        # Link chat functionality
        msg.submit(generate_rss_summary_ollama, 
                   inputs=[cached_category_dropdown, msg, ollama_model_dropdown_rss, chatbot], 
                   outputs=[chatbot, msg])
        clear.click(lambda: (None, ""), inputs=None, outputs=[chatbot, msg])


    # Link the "Fetch All Feeds" button to update all category tabs and the cached categories dropdown
    fetch_all_button.click(
        update_all_feed_tabs,
        inputs=[],
        outputs=list(category_html_outputs.values()) # Pass all textbox components as outputs
    ).success(
        list_cached_categories, # This function will update the choices AND value of cached_category_dropdown
        inputs=[],
        outputs=[cached_category_dropdown]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=False)
