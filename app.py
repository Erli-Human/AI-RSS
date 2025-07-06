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
from bs4 import BeautifulSoup

# --- New Imports for TTS ---
import numpy as np
import soundfile as sf

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

# --- TTS Integration (Karoko/Kokoro TTS) ---
# IMPORTANT: This section is a placeholder for your actual Karoko TTS setup.
# You MUST adapt this based on the Karoko TTS library/repository you are using.

_KAROKO_TTS_AVAILABLE = False
# Example of where you might load your Karoko TTS model once at startup:
# karoko_model = None
# try:
#     # Assuming you have a Karoko TTS wrapper or direct model loading here
#     # from your_karoko_module import KarokoTTSModel, KarokoVoice
#     # karoko_model = KarokoTTSModel(model_path="path/to/your/kokoro.onnx", device="cpu")
#     # _KAROKO_TTS_AVAILABLE = True
#     print("Karoko/Kokoro TTS Placeholder: Ready to integrate your actual model.")
# except ImportError:
#     print("Karoko/Kokoro TTS library not found. Dummy audio will be used.")
# except Exception as e:
#     print(f"Error loading Karoko/Kokoro TTS model: {e}. Dummy audio will be used.")


# Define a list of available female voices from Karoko/Kokoro TTS.
# You will need to verify and update these with the actual voice IDs/names
# provided by your specific Karoko TTS setup.
FEMALE_VOICES = [
    "af_bella", "af_nicole", "af_sarah", "af_sky", # Common examples
    "bf_emma", "bf_isabella", # More examples
    "en_US_f001_kokoro", # A hypothetical generic female voice ID
    "female_default" # A generic fallback if specific names are unknown
]
# Set the default female voice, ensuring it's in the list
DEFAULT_FEMALE_VOICE = "af_bella" if "af_bella" in FEMALE_VOICES else (FEMALE_VOICES[0] if FEMALE_VOICES else "default")


def karoko_tts_generate_audio(text: str, voice_name: str = DEFAULT_FEMALE_VOICE) -> str:
    """
    Generates audio from text using Karoko TTS (or a dummy if not available).
    Returns a path to a temporary audio file.
    """
    cleaned_text = BeautifulSoup(text, 'html.parser').get_text().strip()
    if not cleaned_text:
        return "" # Return empty if no text

    if _KAROKO_TTS_AVAILABLE:
        try:
            print(f"Generating audio for: '{cleaned_text[:80]}...' with voice: {voice_name}")
            # --- YOUR ACTUAL KAROKO TTS INTEGRATION GOES HERE ---
            # This is where you would call your loaded Karoko TTS model.
            # Example (if 'karoko_model' was loaded globally and has a synthesize method):
            # audio_data_np, sample_rate = karoko_model.synthesize(cleaned_text, voice_id=voice_name)
            # ----------------------------------------------------

            # For demonstration without actual Karoko setup:
            sample_rate = 22050
            duration = min(len(cleaned_text) * 0.05, 15) # Max 15 seconds to prevent very long audios
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Generate a slightly more complex dummy audio
            audio_data_np = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
            if len(cleaned_text) > 50: # Add more complexity for longer texts
                audio_data_np += 0.1 * np.sin(2 * np.pi * 1200 * t + np.sin(t / 3))
            audio_data_np = audio_data_np.astype(np.float32)

            # Save to a temporary WAV file for Gradio
            temp_file_path = f"temp_tts_audio_{os.getpid()}_{np.random.randint(0, 100000)}.wav"
            sf.write(temp_file_path, audio_data_np, sample_rate)
            return temp_file_path

        except Exception as e:
            print(f"Error during Karoko TTS generation: {e}. Falling back to dummy audio.")
            # Fallback to a simple error tone if Karoko fails
            sample_rate = 22050
            duration = 2
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data_np = 0.6 * np.sin(2 * np.pi * 100 * t) # Simple error tone
            error_file_path = f"error_tts_audio_{os.getpid()}_{np.random.randint(0, 100000)}.wav"
            sf.write(error_file_path, audio_data_np.astype(np.float32), sample_rate)
            return error_file_path
    else:
        # Dummy audio generation if Karoko TTS is not enabled/available
        print("Karoko TTS not available, generating dummy audio.")
        sample_rate = 22050
        duration = min(len(cleaned_text) * 0.05, 15)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data_np = 0.5 * np.sin(2 * np.pi * 440 * t) # A simple sine wave
        dummy_file_path = f"dummy_tts_audio_{os.getpid()}_{np.random.randint(0, 100000)}.wav"
        sf.write(dummy_file_path, audio_data_np.astype(np.float32), sample_rate)
        return dummy_file_path

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
                summary=entry.get('summary', 'No summary available'), # Keep full summary for Ollama and TTS
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

# This function now generates HTML string for each category
def update_all_feed_tabs_and_render_articles() -> Tuple[Any, ...]:
    """
    Fetches all feeds, updates the global cache, and generates HTML strings
    for each category tab, including dynamically created article blocks with
    read-aloud buttons (via JavaScript calls).
    Returns a tuple of HTML strings, one for each category.
    """
    global GLOBAL_ARTICLE_CACHE
    all_feed_data_by_category = fetch_all_feeds_parallel()

    all_tab_html_outputs: List[str] = []

    for category_name in RSS_FEEDS.keys():
        category_articles_for_cache: List[Article] = []
        current_category_html = ""

        category_feeds = all_feed_data_by_category.get(category_name, {})

        if not category_feeds:
            current_category_html += f"<div>No feeds configured or found for {category_name}.</div>"
        else:
            for feed_name, feed_data in category_feeds.items():
                if feed_data.status == "error":
                    current_category_html += f"<div><strong>{feed_name}</strong>: Error: {feed_data.error}</div><br>"
                elif not feed_data.articles:
                    current_category_html += f"<div><strong>{feed_name}</strong>: No articles found.</div><br>"
                else:
                    category_articles_for_cache.extend(feed_data.articles)

                    def get_sort_key(article):
                        try:
                            return datetime.strptime(article.published, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            try:
                                parsed_date = feedparser._parse_date(article.published)
                                if parsed_date:
                                    return datetime(*parsed_date[:6])
                            except Exception:
                                pass
                            return datetime.min

                    display_articles = sorted(feed_data.articles, key=get_sort_key, reverse=True)[:5]

                    for idx, article in enumerate(display_articles):
                        # Generate a unique ID for the audio player for this article
                        article_unique_id = f"{category_name.replace(' ', '_').replace('&', 'and').lower()}_{feed_name.replace(' ', '_').replace('.', '').lower()}_article_{idx}"

                        # Sanitize summary for HTML display
                        clean_summary = BeautifulSoup(article.summary, 'html.parser').get_text().strip()

                        # Build the HTML for each article
                        current_category_html += f"""
                        <div style="border: 1px solid #eee; padding: 10px; margin-bottom: 15px; border-radius: 5px;">
                            <h3><a href="{article.link}" target="_blank" style="text-decoration: none; color: var(--link-text-color, #1F8BCA);">{article.title}</a></h3>
                            <p style="font-size: 0.9em; color: #666;">
                                <strong>Feed:</strong> {article.feed_name} |
                                <strong>Published:</strong> {article.published} |
                                <strong>Author:</strong> {article.author}
                            </p>
                            <p>{clean_summary}</p>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <button
                                    class="read-aloud-button"
                                    onclick="readAloudArticle('{article_unique_id}', '{article.title + ' ' + clean_summary.replace("'", "\\'").replace('"', '&quot;')}')"
                                    style="padding: 8px 12px; cursor: pointer; background-color: var(--button-primary-background-color); color: var(--button-primary-text-color); border: none; border-radius: 4px;"
                                >
                                    Read Aloud 🔊
                                </button>
                                <select id="voice-selector-{article_unique_id}" style="padding: 8px; border-radius: 4px; border: 1px solid #ddd;">
                                    {''.join([f'<option value="{v}" {"selected" if v == DEFAULT_FEMALE_VOICE else ""}>{v}</option>' for v in FEMALE_VOICES])}
                                </select>
                            </div>
                            <audio id="audio-player-{article_unique_id}" controls style="width: 100%; margin-top: 10px;"></audio>
                        </div>
                        <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));">
                        """
        GLOBAL_ARTICLE_CACHE[category_name] = category_articles_for_cache
        all_tab_html_outputs.append(current_category_html)

    return tuple(all_tab_html_outputs)


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
        # Append as a new message pair
        chat_history.append([user_query, response_text])
        return chat_history, ""

    if "Error" in ollama_model or "No models available" in ollama_model:
        error_message = f"Cannot generate insights: {ollama_model}. Please select a valid Ollama model and ensure Ollama server is running."
        # Append as a new message pair
        chat_history.append([user_query, error_message])
        return chat_history, ""

    articles_to_summarize = GLOBAL_ARTICLE_CACHE.get(selected_category, [])

    if not articles_to_summarize:
        response_text = f"No articles found in cache for category '{selected_category}'. Please click 'Fetch All Feeds' first and select a category with fetched articles."
        # Append as a new message pair
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
    # Chatbot history in Gradio is a list of lists: [[user_msg, ai_msg], ...]
    for human_msg, ai_msg in chat_history:
        messages.append({'role': 'user', 'content': human_msg})
    messages.append({'role': 'user', 'content': user_query}) # Only add current user query, context is built dynamically
    # The response will be appended after the ollama.chat call

    try:
        response = ollama.chat(model=ollama_model, messages=messages)
        ai_response = response['message']['content']
        # Append the new user query and AI response as a new pair to chat_history
        chat_history.append([user_query, ai_response])
        return chat_history, "" # Clear the textbox after sending
    except Exception as e:
        error_message = f"Error communicating with Ollama model '{ollama_model}': {e}. Ensure the model is pulled and running."
        # Append the new user query and error message as a new pair to chat_history
        chat_history.append([user_query, error_message])
        return chat_history, "" # Clear the textbox after sending

# --- Gradio Interface ---

with gr.Blocks(title="Advanced RSS Feed Viewer & AI Assistant") as demo:
    gr.Markdown("# Advanced RSS Feed Viewer & AI Assistant")

    # Define a list to hold the gr.HTML components for each tab's content.
    category_html_outputs: Dict[str, gr.HTML] = {}

    with gr.Tab("RSS Feed Browser"):
        gr.Markdown("## Latest News Across Categories")
        fetch_all_button = gr.Button("Fetch All Feeds (This may take a moment)")

        # Create a hidden Gradio Audio component to be the target for TTS
        # This will be shared by all dynamic HTML buttons
        hidden_audio_output = gr.Audio(visible=False, interactive=False, label="TTS Output")

        # Create a dummy Textbox that will receive the article text from JS,
        # which can then be passed to the TTS function. This is just an input
        # placeholder for the Python function called by JS.
        hidden_article_text_input = gr.Textbox(visible=False, interactive=False)
        hidden_voice_input = gr.Textbox(visible=False, interactive=False)

        # This function acts as a proxy that JavaScript calls.
        # It takes the article text and voice ID from JavaScript.
        # It then calls your actual Karoko TTS function.
        # It needs to return a Gradio Audio component (or a file path for it).
        # This function is defined *inside* the Blocks context.
        read_aloud_js_proxy_function = gr.Function(
            karoko_tts_generate_audio,
            inputs=[hidden_article_text_input, hidden_voice_input],
            outputs=[hidden_audio_output],
            api_name="read_aloud_api" # Give it an API name to call from JS
        )

        gr.Markdown("""
            <script>
            // JavaScript function to call the Gradio Python function for TTS
            async function readAloudArticle(article_id, article_content) {
                const voiceSelector = document.getElementById(`voice-selector-${article_id}`);
                const selectedVoice = voiceSelector ? voiceSelector.value : 'female_default'; // Get selected voice
                const audioPlayer = document.getElementById(`audio-player-${article_id}`);

                if (audioPlayer) {
                    audioPlayer.src = ''; // Clear previous audio
                    audioPlayer.pause();
                    audioPlayer.load();
                }

                // Call the Gradio API function
                // The `window.gradio_config.root` is needed to get the base URL if app is proxied.
                const response = await fetch(`${window.gradio_config.root}/run/read_aloud_api`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        data: [
                            article_content, // maps to hidden_article_text_input
                            selectedVoice    // maps to hidden_voice_input
                        ]
                    })
                });

                const jsonResponse = await response.json();
                const audioFilePath = jsonResponse.data[0].url; // Get the URL from Gradio's response

                if (audioPlayer && audioFilePath) {
                    audioPlayer.src = audioFilePath;
                    audioPlayer.play();
                } else {
                    console.error('Failed to get audio file path or audio player not found:', audioFilePath);
                }
            }
            </script>
        """)


        with gr.Tabs() as category_tabs:
            for category_name in RSS_FEEDS.keys():
                with gr.Tab(category_name, id=f"tab_{category_name.replace(' ', '_').replace('&', 'and').lower()}"):
                    # Use gr.HTML to display the dynamically generated content for each tab
                    # Initial value is a loading message
                    category_html_outputs[category_name] = gr.HTML(
                        value="<p>Click 'Fetch All Feeds' to load articles...</p>",
                        elem_id=f"html_output_{category_name.replace(' ', '_').replace('&', 'and').lower()}"
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
            cached_category_dropdown = gr.Dropdown(
                label="Select Cached Category for Insights",
                choices=list(RSS_FEEDS.keys()),
                value=None,
                interactive=True
            )

        chatbot = gr.Chatbot(label="Ollama Chat History", height=300, type="messages")
        msg = gr.Textbox(label="Ask a question about the articles:", placeholder="e.g., What are the key trends in AI research?")
        clear = gr.ClearButton([msg, chatbot])

        msg.submit(generate_rss_summary_ollama,
                   inputs=[cached_category_dropdown, msg, ollama_model_dropdown_rss, chatbot],
                   outputs=[chatbot, msg])
        clear.click(lambda: ([], ""), inputs=None, outputs=[chatbot, msg])

    # Link the "Fetch All Feeds" button to update all category tabs and the cached categories dropdown
    fetch_all_button.click(
        update_all_feed_tabs_and_render_articles,
        inputs=[],
        # The outputs are now the gr.HTML components
        outputs=list(category_html_outputs.values())
    ).success(
        list_cached_categories,
        inputs=[],
        outputs=[cached_category_dropdown]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=False)
