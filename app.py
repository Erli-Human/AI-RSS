import csv
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO, BytesIO
import base64
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import gradio as gr
import schedule
import time
import smtplib
from email.mime.text import MIMEText
import feedparser
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import ollama # Import the ollama library
import os

# --- Data Structures ---
@dataclass
class DailySummary:
    date: str
    total_sales: float
    total_quantity: int
    top_product: str
    region_sales: Dict[str, float] = field(default_factory=dict)
    average_transaction_value: float = 0.0

@dataclass
class SalesRecord:
    transaction_id: str
    date: datetime
    product_id: str
    product_name: str
    quantity: int
    price: float
    region: str

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


# --- Helper Functions (Sales Data) ---
def parse_sales_data(file_content: str) -> List[SalesRecord]:
    records = []
    f = StringIO(file_content)
    reader = csv.reader(f)
    header = next(reader) # Skip header
    for row in reader:
        try:
            records.append(SalesRecord(
                transaction_id=row[0],
                date=datetime.strptime(row[1], '%Y-%m-%d'),
                product_id=row[2],
                product_name=row[3],
                quantity=int(row[4]),
                price=float(row[5]),
                region=row[6]
            ))
        except (ValueError, IndexError) as e:
            print(f"Skipping malformed row: {row} - Error: {e}")
            continue
    return records

def analyze_data(records: List[SalesRecord]) -> Dict[str, Any]:
    df = pd.DataFrame([s.__dict__ for s in records])
    if df.empty:
        return {
            "total_sales": 0,
            "total_transactions": 0,
            "average_transaction_value": 0,
            "top_products": {}, # Changed from [] to {} for consistency with dict output
            "sales_by_region": {},
            "daily_summaries": []
        }

    df['total_price'] = df['quantity'] * df['price']
    df['date_only'] = df['date'].dt.date

    total_sales = df['total_price'].sum()
    total_transactions = df['transaction_id'].nunique()
    average_transaction_value = total_sales / total_transactions if total_transactions > 0 else 0

    top_products_df = df.groupby('product_name')['total_price'].sum().nlargest(5)
    top_products = top_products_df.to_dict()

    sales_by_region = df.groupby('region')['total_price'].sum().to_dict()

    daily_summaries_list = []
    for date, group in df.groupby('date_only'):
        daily_total_sales = group['total_price'].sum()
        daily_total_quantity = group['quantity'].sum()
        daily_top_product_series = group.groupby('product_name')['quantity'].sum().nlargest(1)
        daily_top_product = daily_top_product_series.index[0] if not daily_top_product_series.empty else "N/A"
        daily_region_sales = group.groupby('region')['total_price'].sum().to_dict()
        daily_avg_transaction = daily_total_sales / group['transaction_id'].nunique() if group['transaction_id'].nunique() > 0 else 0

        daily_summaries_list.append(DailySummary(
            date=str(date),
            total_sales=daily_total_sales,
            total_quantity=daily_total_quantity,
            top_product=daily_top_product,
            region_sales=daily_region_sales,
            average_transaction_value=daily_avg_transaction
        ))

    return {
        "total_sales": total_sales,
        "total_transactions": total_transactions,
        "average_transaction_value": average_transaction_value,
        "top_products": top_products,
        "sales_by_region": sales_by_region,
        "daily_summaries": daily_summaries_list
    }

def plot_data(records: List[SalesRecord]) -> str:
    df = pd.DataFrame([s.__dict__ for s in records])
    if df.empty:
        return "No data to plot."

    df['total_price'] = df['quantity'] * df['price']
    df['date_only'] = df['date'].dt.date

    # Plotting daily sales
    daily_sales = df.groupby('date_only')['total_price'].sum()
    plt.figure(figsize=(10, 6))
    daily_sales.plot(kind='line', marker='o')
    plt.title('Daily Total Sales')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_str}" />'

def send_email(to_address: str, subject: str, body: str, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = to_address

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"

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
        print(f"[{datetime.now()}] Raw Ollama list response: {models_info}") # THIS IS KEY FOR DEBUGGING

        # The ollama.list() returns an ollama._types.ListResponse object,
        # which has a 'models' attribute (a list of Model objects).
        if not hasattr(models_info, 'models'):
            print(f"[{datetime.now()}] Error: Ollama list response object missing 'models' attribute. Response: {models_info}")
            return ["Error: Ollama response malformed (missing 'models' attribute)."]

        model_list = models_info.models # Corrected: Access as an attribute

        # Ensure that the 'models' attribute is indeed a list
        if not isinstance(model_list, list):
            print(f"[{datetime.now()}] Error: 'models' attribute is not a list. Type: {type(model_list)}. Value: {model_list}")
            return ["Error: Ollama response malformed ('models' attribute not a list)."]

        if not model_list:
            print(f"[{datetime.now()}] No Ollama models found in the response. Have you pulled any yet? (e.g., 'ollama pull llama2')")
            return ["No models found. Pull models like 'ollama pull gemma3n:e4b'."]

        models = []
        for i, model_entry in enumerate(model_list):
            # Each model_entry is an ollama._types.Model object, access 'model' as an attribute
            if not hasattr(model_entry, 'model'): # Check for 'model' attribute, which holds the name string
                print(f"[{datetime.now()}] Warning: Model entry at index {i} missing 'model' attribute. Skipping. Entry: {model_entry}")
                continue
            
            models.append(model_entry.model) # Corrected: Access as an attribute

        if not models:
            print(f"[{datetime.now()}] No valid model names extracted after processing Ollama list response.")
            return ["No valid model names extracted."]

        return sorted(list(set(models))) # Use set for uniqueness, then sort

    except ConnectionRefusedError:
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


def generate_insights_ollama(sales_data_string: str, query: str, ollama_model: str) -> str:
    if "Error" in ollama_model: # Check if the selected model is an error message
        return f"Cannot generate insights: {ollama_model}. Please select a valid Ollama model."
    
    if not sales_data_string:
        return "No sales data uploaded. Please upload a CSV file first."

    records = parse_sales_data(sales_data_string)
    if not records:
        return "No valid sales records parsed from the uploaded data. Please check the CSV format."

    analysis_results = analyze_data(records)
    
    prompt_data = {
        "analysis_results": analysis_results,
        "user_query": query,
        "schema_of_analysis_results": """
        The 'analysis_results' dictionary contains:
        - total_sales: float, sum of all sales.
        - total_transactions: int, count of unique transactions.
        - average_transaction_value: float, total_sales / total_transactions.
        - top_products: dict, product_name -> total_price for top 5 products.
        - sales_by_region: dict, region -> total_price for each region.
        - daily_summaries: list of DailySummary objects, each containing:
            - date: str
            - total_sales: float
            - total_quantity: int
            - top_product: str
            - region_sales: dict
            - average_transaction_value: float
        """
    }
    
    system_prompt = (
        "You are an expert sales data analyst. Your task is to provide insightful responses "
        "based on the provided sales analysis results. Use the data to answer the user's query comprehensively. "
        "If the data doesn't directly answer the query, indicate that. Keep responses concise and actionable."
    )
    
    user_prompt = (
        f"Here are the sales analysis results:\n{json.dumps(prompt_data['analysis_results'], indent=2)}\n\n"
        f"This is the schema for the analysis results: {prompt_data['schema_of_analysis_results']}\n\n"
        f"Based on this data, answer the following question: \"{prompt_data['user_query']}\""
    )

    try:
        response = ollama.chat(model=ollama_model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama model '{ollama_model}': {e}. Ensure the model is pulled and running."

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

with gr.Blocks() as demo:
    gr.Markdown("# Integrated Data Analysis and News Dashboard")

    with gr.Tab("Sales Data Analysis"):
        with gr.Row():
            file_upload = gr.File(label="Upload CSV Sales Data", type="filepath")
            data_preview = gr.DataFrame(label="Data Preview")
            
        parse_button = gr.Button("Parse & Analyze Data")
        
        with gr.Accordion("Analysis Results", open=False):
            total_sales_output = gr.Number(label="Total Sales")
            total_transactions_output = gr.Number(label="Total Transactions")
            avg_transaction_output = gr.Number(label="Average Transaction Value")
            top_products_output = gr.JSON(label="Top 5 Products by Sales")
            sales_by_region_output = gr.JSON(label="Sales by Region")
            daily_summaries_output = gr.JSON(label="Daily Summaries")
            
        with gr.Accordion("Visualizations", open=False):
            plot_output = gr.HTML(label="Sales Trend Plot")
        
        gr.Markdown("## Sales Data Insights (powered by Ollama)")
        with gr.Row():
            ollama_model_dropdown_sales = gr.Dropdown(
                label="Select Ollama Model",
                choices=ollama_available_models,
                value=default_ollama_model,
                interactive=True
            )
            ollama_query = gr.Textbox(label="Ask a question about the sales data:", placeholder="e.g., What were the daily sales trends?")
        ollama_generate_button_sales = gr.Button("Generate Sales Insights")
        ollama_output_sales = gr.Markdown(label="Ollama Sales Insights")
        
    file_content_state = gr.State(None)
    parsed_records_state = gr.State([])

    def handle_file_upload(file_path: str):
        if file_path is None:
            return None, None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            df_preview = pd.read_csv(StringIO(file_content))
            return file_content, df_preview.head(5)
        except Exception as e:
            print(f"Error handling file upload: {e}")
            return None, pd.DataFrame({"Error": [f"Could not read file: {e}"]})

    def run_analysis(file_content: str):
        if file_content is None:
            return 0, 0, 0, {}, {}, [], "No file uploaded or file could not be read.", []

        records = parse_sales_data(file_content)
        if not records:
            return 0, 0, 0, {}, {}, [], "No valid records parsed. Check CSV format.", []

        analysis_results = analyze_data(records)
        plot_html = plot_data(records)

        daily_summaries_dict = [asdict(ds) for ds in analysis_results["daily_summaries"]]
            
        return (
            analysis_results["total_sales"],
            analysis_results["total_transactions"],
            analysis_results["average_transaction_value"],
            analysis_results["top_products"],
            analysis_results["sales_by_region"],
            daily_summaries_dict,
            plot_html,
            records
        )

    file_upload.upload(handle_file_upload, inputs=file_upload, outputs=[file_content_state, data_preview])
        
    parse_button.click(
        run_analysis,
        inputs=[file_content_state],
        outputs=[
            total_sales_output,
            total_transactions_output,
            avg_transaction_output,
            top_products_output,
            sales_by_region_output,
            daily_summaries_output,
            plot_output,
            parsed_records_state
        ]
    )

    ollama_generate_button_sales.click(
        generate_insights_ollama,
        inputs=[file_content_state, ollama_query, ollama_model_dropdown_sales],
        outputs=ollama_output_sales
    )

    # --- RSS Feed Browser Tab ---
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
            # This dropdown's choices are dynamic based on what's in the cache
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
            fn=list_cached_categories, # After fetching, update the cached categories dropdown
            inputs=[],
            outputs=[cached_category_dropdown]
        )
        
        # Link Ollama for RSS
        rss_ollama_generate_button.click(
            generate_rss_summary_ollama,
            inputs=[cached_category_dropdown, rss_ollama_query, ollama_model_dropdown_rss],
            outputs=rss_ollama_output
        )

    # --- Scheduler Tab ---
    with gr.Tab("Scheduled Tasks"):
        gr.Markdown("## Automated Reports & Notifications")
        with gr.Row():
            email_address_input = gr.Textbox(label="Recipient Email", placeholder="your_email@example.com")
            email_subject_input = gr.Textbox(label="Email Subject", placeholder="Daily Sales Report")
            smtp_server_input = gr.Textbox(label="SMTP Server", placeholder="smtp.example.com")
            smtp_port_input = gr.Number(label="SMTP Port", value=587)
            smtp_user_input = gr.Textbox(label="SMTP Username", placeholder="your_smtp_username")
            smtp_password_input = gr.Textbox(label="SMTP Password", type="password", placeholder="your_smtp_password")
            
        schedule_time_input = gr.Textbox(label="Schedule Time (HH:MM)", placeholder="e.g., 09:00 for 9 AM")
        schedule_button = gr.Button("Schedule Email Report")
        scheduler_status = gr.Textbox(label="Scheduler Status", interactive=False)

        # Scheduler thread management (important for background tasks in Gradio)
        scheduler_thread = None
        scheduler_stop_event = threading.Event()

        def start_scheduler_thread():
            global scheduler_thread
            if scheduler_thread is None or not scheduler_thread.is_alive():
                scheduler_stop_event.clear()
                scheduler_thread = threading.Thread(target=run_scheduler, args=(scheduler_stop_event,), daemon=True) # daemon=True allows thread to exit with main app
                scheduler_thread.start()
                return "Scheduler started."
            return "Scheduler is already running."

        def stop_scheduler_thread():
            global scheduler_thread
            if scheduler_thread and scheduler_thread.is_alive():
                scheduler_stop_event.set()
                scheduler_thread.join(timeout=5) # Wait for thread to finish, with a timeout
                return "Scheduler stopped."
            return "Scheduler is not running."

        def run_scheduler(stop_event: threading.Event):
            """Continuously runs scheduled jobs until stop_event is set."""
            while not stop_event.is_set():
                schedule.run_pending()
                time.sleep(1) # Check every second

        def schedule_email_report_action(
            email_address: str, subject: str, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str,
            schedule_time: str, records_from_state: List[SalesRecord] # Records passed from the state
        ):
            if not records_from_state:
                return "No sales data available to schedule report. Please upload and analyze data first."

            analysis_results = analyze_data(records_from_state)
            email_body = f"""
            Sales Report for {datetime.now().strftime('%Y-%m-%d')}

            Total Sales: ${analysis_results['total_sales']:.2f}
            Total Transactions: {analysis_results['total_transactions']}
            Average Transaction Value: ${analysis_results['average_transaction_value']:.2f}

            Top 5 Products:
            {json.dumps(analysis_results['top_products'], indent=2)}

            Sales by Region:
            {json.dumps(analysis_results['sales_by_region'], indent=2)}

            Daily Summaries:
            {json.dumps([asdict(ds) for ds in analysis_results['daily_summaries']], indent=2)}
            """
            
            # The actual job function that will be scheduled
            def email_job():
                print(f"Attempting to send scheduled email at {datetime.now().strftime('%H:%M:%S')}")
                send_result = send_email(email_address, subject, email_body, smtp_server, smtp_port, smtp_user, smtp_password)
                print(f"Scheduled email send result: {send_result}")
                # Note: Returning a value from a scheduled job isn't directly passed back to Gradio UI unless explicitly handled.
                # The print statements are for console feedback.

            schedule.clear('daily_report_job') # Clear any existing jobs with this tag
            try:
                schedule.every().day.at(schedule_time).do(email_job).tag('daily_report_job')
                start_scheduler_thread() # Ensure the scheduler thread is running
                return f"Email report scheduled for {schedule_time} daily. Scheduler is running in background."
            except Exception as e:
                return f"Failed to schedule email: {e}. Ensure time format is HH:MM."

        schedule_button.click(
            schedule_email_report_action,
            inputs=[
                email_address_input, email_subject_input, smtp_server_input, smtp_port_input,
                smtp_user_input, smtp_password_input, schedule_time_input, parsed_records_state
            ],
            outputs=scheduler_status
        )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=False)
