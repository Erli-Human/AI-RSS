import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import gradio as gr
import feedparser
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "rss_config.json"
HISTORY_PATH = "article_history.json"

RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "OpenAI Blog": "https://openai.com/blog/rss.xml",
        "Hugging Face Blog": "https://huggingface.co/blog/feed.xml"
    },
    "🌍 World News": {
        "BBC World News": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Global News": "https://globalnews.ca/feed/"
    },
    "📰 News": {
        "New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "CNN": "http://rss.cnn.com/rss/edition.rss",
        "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
        "USA Today": "https://rss.usatoday.com/usatoday-NewsTopStories",
        "Fox News": "https://feeds.foxnews.com/foxnews/latest",
        "NBC News": "https://feeds.nbcnews.com/nbcnews/public/news",
        "ABC News": "https://feeds.abcnews.com/abcnews/topstories",
        "CBS News": "https://www.cbsnews.com/latest/rss/main",
        "NPR": "https://feeds.npr.org/1001/rss.xml",
        "PBS NewsHour": "https://www.pbs.org/newshour/feeds/rss/headlines",
        "The Guardian": "https://www.theguardian.com/world/rss",
        "Los Angeles Times": "https://www.latimes.com/rss2.0.xml",
        "Chicago Tribune": "https://www.chicagotribune.com/rss2.0.xml",
        "Seattle Times": "https://www.seattletimes.com/rss-feeds/",
        "Denver Post": "https://www.denverpost.com/feed/",
        "Cleveland Plain Dealer": "https://www.cleveland.com/arc/outboundfeeds/rss/",
        "Pittsburgh Post-Gazette": "https://www.post-gazette.com/rss/"
    },
    "💻 Technology": {
        "TechCrunch": "https://techcrunch.com/feed/",
        "Wired": "https://www.wired.com/feed/rss",
        "The Verge": "https://www.theverge.com/rss/index.xml",
        "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
        "Engadget": "https://www.engadget.com/rss.xml",
        "Gizmodo": "https://gizmodo.com/rss",
        "TechRadar": "https://www.techradar.com/rss",
        "CNET": "https://www.cnet.com/rss/news/",
        "ZDNet": "https://www.zdnet.com/news/rss.xml",
        "Mashable Tech": "https://mashable.com/feeds/rss/tech",
        "VentureBeat": "https://venturebeat.com/feed/",
        "ReadWrite": "https://readwrite.com/feed/",
        "PCWorld": "https://www.pcworld.com/index.rss",
        "Tom's Hardware": "https://www.tomshardware.com/feeds/all",
        "AnandTech": "https://www.anandtech.com/rss/",
        "Slashdot": "https://rss.slashdot.org/Slashdot/slashdotMain",
        "Hacker News": "https://hnrss.org/frontpage",
        "GitHub Blog": "https://github.blog/feed/",
        "Stack Overflow Blog": "https://stackoverflow.blog/feed/",
        "Apple Newsroom": "https://www.apple.com/newsroom/rss-feed.rss",
        "Amazon Science": "https://www.amazon.science/index.rss",
        "Facebook Engineering": "https://engineering.fb.com/feed/",
        "Netflix Tech Blog": "https://netflixtechblog.com/feed",
        "Airbnb Engineering": "https://medium.com/feed/airbnb-engineering",
        "Spotify Engineering": "https://engineering.atspotify.com/feed/",
        "LinkedIn Engineering": "https://engineering.linkedin.com/blog.rss",
        "Dropbox Tech Blog": "https://dropbox.tech/feed",
        "Pinterest Engineering": "https://medium.com/feed/@Pinterest_Engineering",
        "Slack Engineering": "https://slack.engineering/feed/"
    },
    "⚽ Sports": {
        "ESPN": "https://www.espn.com/espn/rss/news",
        "Fox Sports": "https://www.foxsports.com/rss-feeds",
        "CBS Sports": "https://www.cbssports.com/rss/headlines",
        "Yahoo Sports": "https://sports.yahoo.com/rss/",
        "Sporting News": "https://www.sportingnews.com/us/rss",
        "MLB.com": "https://www.mlb.com/feeds/news/rss.xml",
        "NHL.com": "https://www.nhl.com/rss/news",
        "ESPN NFL": "https://www.espn.com/espn/rss/nfl/news",
        "ESPN NBA": "https://www.espn.com/espn/rss/nba/news",
        "ESPN MLB": "https://www.espn.com/espn/rss/mlb/news",
        "ESPN NHL": "https://www.espn.com/espn/rss/nhl/news",
        "Sky Sports": "https://www.skysports.com/rss/12040",
        "BBC Sport": "https://feeds.bbci.co.uk/sport/rss.xml",
        "Formula 1": "https://www.formula1.com/en/latest/all.xml",
        "Boxing News": "https://www.boxingnews24.com/feed/",
        "Pro Wrestling Torch": "https://www.pwtorch.com/feed/",
        "Wrestling Observer": "https://www.f4wonline.com/rss.xml"
    },
    "💼 Business": {
        "Financial Times": "https://www.ft.com/rss/home",
        "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "Forbes": "https://www.forbes.com/real-time/feed2/",
        "Fortune": "https://fortune.com/feed/",
        "Business Insider": "https://feeds.businessinsider.com/custom/all",
        "Fast Company": "https://www.fastcompany.com/rss.xml",
        "Inc.com": "https://www.inc.com/rss/homepage.xml",
        "Entrepreneur": "https://www.entrepreneur.com/latest.rss",
        "TechCrunch Business": "https://techcrunch.com/category/startups/feed/",
        "Seeking Alpha": "https://seekingalpha.com/feed.xml",
        "Money Magazine": "https://money.com/feed/",
        "SmartMoney": "https://www.smartmoney.com/rss/",
        "Morningstar": "https://www.morningstar.com/rss/news",
        "Financial Planning": "https://www.financial-planning.com/feed",
        "Employee Benefit News": "https://www.benefitnews.com/feed",
        "Accounting Today": "https://www.accountingtoday.com/feed",
        "Insurance Journal": "https://www.insurancejournal.com/rss/news/"
    },
    "🎵 Music": {
        "EDM.com": "https://edm.com/.rss/full/",
        "Rolling Stone Music": "https://www.rollingstone.com/music/feed/",
        "NME": "https://www.nme.com/feed/",
        "Stereogum": "https://www.stereogum.com/feed/",
        "Consequence of Sound": "https://consequence.net/feed/",
        "Spin": "https://www.spin.com/feed/",
        "Billboard": "https://www.billboard.com/feed/",
        "Paste Magazine": "https://www.pastemagazine.com/music/feed",
        "Brooklyn Vegan": "https://www.brooklynvegan.com/feed/",
        "Gorilla vs Bear": "https://www.gorillavsbear.net/feed/",
        "Hypebeast Music": "https://hypebeast.com/music/feed",
        "DJ Mag": "https://djmag.com/rss.xml",
        "Dancing Astronaut": "https://dancingastronaut.com/feed/",
        "HipHopDX": "https://hiphopdx.com/rss",
        "XXL": "https://www.xxlmag.com/feed/",
        "The Source": "https://thesource.com/feed/",
        "Loudwire": "https://loudwire.com/feed/",
        "Metal Injection": "https://metalinjection.net/feed",
        "Saving Country Music": "https://www.savingcountrymusic.com/feed/",
        "No Depression": "https://www.nodepression.com/feed/",
        "JazzTimes": "https://jazztimes.com/feed/",
        "Music Business Worldwide": "https://www.musicbusinessworldwide.com/feed/",
        "Hypebot": "https://www.hypebot.com/feed/",
        "Song Exploder": "https://songexploder.net/feed",
        "All Songs Considered": "https://feeds.npr.org/510019/podcast.xml"
    },
    "🎮 Gaming": {
        "Kotaku": "http://kotaku.com/vip.xml",
        "IGN": "https://feeds.ign.com/ign/all",
        "GameSpot": "https://www.gamespot.com/feeds/game-news/",
        "Polygon": "https://www.polygon.com/rss/index.xml",
        "PC Gamer": "https://www.pcgamer.com/rss/",
        "Game Informer": "https://www.gameinformer.com/rss.xml",
        "Eurogamer": "https://www.eurogamer.net/?format=rss",
        "Rock Paper Shotgun": "https://www.rockpapershotgun.com/feed",
        "VG247": "https://www.vg247.com/feed/",
        "Destructoid": "https://www.destructoid.com/feed/",
        "PlayStation Blog": "https://blog.playstation.com/feed/",
        "Xbox Wire": "https://news.xbox.com/en-us/feed/",
        "Nintendo Life": "https://www.nintendolife.com/feeds/latest",
        "Steam Blog": "https://store.steampowered.com/feeds/",
        "TIGSource": "https://forums.tigsource.com/index.php?type=rss;action=.xml",
        "TouchArcade": "https://toucharcade.com/feed/",
        "Pocket Gamer": "https://www.pocketgamer.com/rss/",
        "Esports Insider": "https://esportsinsider.com/feed/",
        "Dot Esports": "https://dotesports.com/feed",
        "Tom's Hardware": "https://www.tomshardware.com/feeds/all",
        "PC World": "https://www.pcworld.com/feed",
        "Giant Bombcast": "https://www.giantbomb.com/podcast-xml/giant-bombcast/"
    },
    "✈️ Travel": {
        "Nomadic Matt": "https://www.nomadicmatt.com/travel-blog/feed/",
        "The Blonde Abroad": "https://theblondeabroad.com/feed/",
        "Adventurous Kate": "https://www.adventurouskate.com/feed/",
        "One Mile at a Time": "https://onemileatatime.com/feed/",
        "The Points Guy": "https://thepointsguy.com/feed/",
        "Conde Nast Traveler": "https://www.cntraveler.com/feed/rss",
        "Frommer's": "https://www.frommers.com/feed",
        "View from the Wing": "https://viewfromthewing.com/feed/",
        "Thrifty Traveler": "https://thriftytraveler.com/feed/",
        "The Luxury Travel Expert": "https://theluxurytravelexpert.com/feed/",
        "A Luxury Travel Blog": "https://www.aluxurytravelblog.com/feed/",
        "Outside Magazine": "https://www.outsideonline.com/feed",
        "Adventure Journal": "https://adventure-journal.com/feed/",
        "Backpacker Magazine": "https://www.backpacker.com/feed",
        "Atlas Obscura": "https://www.atlasobscura.com/feeds/latest",
        "Roads & Kingdoms": "https://roadsandkingdoms.com/feed/",
        "Matador Network": "https://matadornetwork.com/feed/",
        "Family Travel Forum": "https://myfamilytravels.com/feed",
        "Ciao Bambino": "https://ciaobambino.com/feed/",
        "Eater": "https://www.eater.com/rss/index.xml",
        "Saveur": "https://www.saveur.com/feed/",
        "Migrationology": "https://migrationology.com/feed/",
        "Legal Nomads": "https://www.legalnomads.com/feed/",
        "Stuck in Customs": "https://stuckincustoms.com/feed/",
        "Skift": "https://skift.com/feed/",
    },
    "💡 Lifestyle": {
        "Men's Health": "https://www.menshealth.com/feed/",
        "Cup of Jo": "https://cupofjo.com/feed/",
        "A Beautiful Mess": "https://abeautifulmess.com/feed/",
        "Wit & Delight": "https://witanddelight.com/feed/",
        "Camille Styles": "https://camillestyles.com/feed/",
        "The Chriselle Factor": "https://thechrisellefactor.com/feed/",
        "Apartment Therapy": "https://www.apartmenttherapy.com/main.rss",
        "Design Milk": "https://design-milk.com/feed/",
        "Dezeen": "https://www.dezeen.com/feed/",
        "Architectural Digest": "https://www.architecturaldigest.com/feed/rss",
        "Who What Wear": "https://www.whowhatwear.com/rss",
        "The Blonde Salad": "https://www.theblondesalad.com/feed",
        "Allure": "https://www.allure.com/feed/rss",
        "Refinery29": "https://www.refinery29.com/rss.xml",
        "Goop": "https://goop.com/feed/",
        "Art of Manliness": "https://www.artofmanliness.com/feed/",
        "GQ": "https://www.gq.com/feed/rss",
        "Esquire": "https://www.esquire.com/rss/",
        "Primer Magazine": "https://www.primermagazine.com/feed",
        "Dappered": "https://dappered.com/feed/",
        "Scary Mommy": "https://www.scarymommy.com/feed/",
        "Motherly": "https://www.mother.ly/feed/",
        "The Minimalists": "https://www.theminimalists.com/feed/",
        "Becoming Minimalist": "https://www.becomingminimalist.com/feed/",
        "Tim Ferriss": "https://tim.blog/feed/",
        "James Clear": "https://jamesclear.com/feed",
        "Lifehacker": "https://lifehacker.com/rss",
        "Fast Company": "https://www.fastcompany.com/rss",
        "TED Blog": "https://blog.ted.com/feed/"
    },
    "🏡 Home & Garden": {
        "Rodale Institute": "https://rodaleinstitute.org/feed/",
        "Permaculture Magazine": "https://www.permaculture.co.uk/feed",
        "Gardenista": "https://www.gardenista.com/feed/",
        "Fine Gardening": "https://www.finegardening.com/feed",
        "Horticulture Magazine": "https://www.hortmag.com/feed",
        "Epic Gardening": "https://www.epicgardening.com/feed/",
        "Garden Betty": "https://www.gardenbetty.com/feed/",
        "Joe Gardener": "https://joegardener.com/feed/",
        "Garden Therapy": "https://gardentherapy.ca/feed/",
        "The Impatient Gardener": "https://www.theimpatientgardener.com/feed/",
        "You Grow Girl": "http://www.yougrowgirl.com/feed/",
        "Cold Climate Gardening": "https://www.coldclimategardening.com/feed/",
        "Floret Flowers": "https://www.floretflowers.com/feed/",
        "The Sill": "https://www.thesill.com/blogs/plants-101.atom",
        "Pistils Nursery": "https://pistilsnursery.com/blogs/journal.atom",
        "Urban Jungle Bloggers": "https://www.urbanjunglebloggers.com/feed/",
        "Homestead and Chill": "https://homesteadandchill.com/feed/",
        "Tenth Acre Farm": "https://www.tenthacrefarm.com/feed/",
        "The Prairie Homestead": "https://www.theprairiehomestead.com/feed",
        "Fresh Eggs Daily": "https://www.fresheggsdaily.blog/feeds/posts/default",
        "Family Handyman": "https://www.familyhandyman.com/feed/",
        "Bob Vila": "https://www.bobvila.com/feed",
        "Young House Love": "https://www.younghouselove.com/feed/",
        "Addicted 2 Decorating": "https://www.addicted2decorating.com/feed",
        "Remodelista": "https://www.remodelista.com/feed/",
        "Curbly": "https://www.curbly.com/rss",
        "Garden Rant": "https://gardenrant.com/feed",
        "The Graceful Gardener": "https://www.thegracefulgardener.com/feed/",
        "Urban Farm": "https://www.urbanfarmonline.com/rss"
    }
}

GPT2_MODEL_PATH = "gpt2_model.onnx"
GPT2_MODEL_URL = "https://huggingface.co/HyunjuJane/gpt2_model.onnx/resolve/main/gpt2_model.onnx"
GPT2_SESSION = None
GPT2_TOKENIZER = None
MODEL_AVAILABLE = False

# Check if model exists, if not download it
if not os.path.exists(GPT2_MODEL_PATH):
    logger.info(f"📥 GPT-2 model not found. Downloading from Hugging Face...")
    try:
        response = requests.get(GPT2_MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress
        with open(GPT2_MODEL_PATH, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='', flush=True)

        logger.info(f"\n✅ Model downloaded successfully to {GPT2_MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        logger.info(f"Please download manually from: {GPT2_MODEL_URL}")

# Load the model and tokenizer
if os.path.exists(GPT2_MODEL_PATH):
    try:
        logger.info(f"🔄 Loading GPT-2 model and tokenizer...")
        GPT2_SESSION = ort.InferenceSession(GPT2_MODEL_PATH)
        GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
        GPT2_TOKENIZER.pad_token = GPT2_TOKENIZER.eos_token
        MODEL_AVAILABLE = True
        logger.info(f"✅ GPT-2 model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"⚠️ Failed to load GPT-2 model: {e}")
        GPT2_SESSION = None
        MODEL_AVAILABLE = False

@dataclass
class Article:
    title: str
    link: str
    published: str
    summary: str
    feed_name: str
    author: str = ""
    fetched_at: str = datetime.utcnow().isoformat()

def load_json(path: str, default=[]):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, list) else default
    except:
        return default

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

def init_config():
    cfg = load_json(CONFIG_PATH)
    # Create a dict to track unique URLs and avoid duplicates
    url_to_feed = {}

    # First pass: collect existing feeds
    for f in cfg:
        if isinstance(f, dict) and "url" in f:
            url_to_feed[f["url"]] = f

    updated = False
    # Second pass: add new feeds from RSS_FEEDS
    for cat, feeds in RSS_FEEDS.items():
        for name, url in feeds.items():
            if url not in url_to_feed:
                feed_entry = {
                    "category": cat,
                    "feed_name": name,
                    "url": url,
                    "created": datetime.utcnow().isoformat(),
                    "key": f"{cat}_{name}"
                }
                url_to_feed[url] = feed_entry
                updated = True

    # Convert back to list, preserving deduplication
    cfg = list(url_to_feed.values())

    if updated:
        save_json(CONFIG_PATH, cfg)
    return cfg

def fetch_feed(url, name):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        return [
            Article(
                title=e.get("title", "No title"),
                link=e.get("link", ""),
                published=e.get("published", "Unknown"),
                summary=e.get("summary", "")[:300] + "...",
                author=e.get("author", ""),
                feed_name=name
            ) for e in feed.entries[:15]
        ]
    except requests.exceptions.ConnectionError as e:
        if "NameResolutionError" in str(e):
            logger.warning(f"Failed to resolve hostname for {name} at {url}. Please check your network connection and DNS settings.")
        else:
            logger.warning(f"Connection error while fetching feed {name} from {url}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch feed {name} from {url}: {e}")
        return []

def update_history():
    cfg = load_json(CONFIG_PATH)
    history = load_json(HISTORY_PATH)
    # Handle both old and new history formats
    links = set()
    for a in history:
        if isinstance(a, dict) and "link" in a:
            links.add(a["link"])

    new_articles = []
    # Track processed URLs to avoid duplicates
    processed_urls = set()

    with ThreadPoolExecutor(max_workers=8) as exe:
        # Deduplicate feeds by URL
        unique_feeds = {}
        for f in cfg:
            if isinstance(f, dict) and "url" in f and "feed_name" in f:
                url = f["url"]
                if url not in unique_feeds:
                    unique_feeds[url] = f

        fut2 = {exe.submit(fetch_feed, f["url"], f["feed_name"]): f for f in unique_feeds.values()}
        for fut in as_completed(fut2):
            for art in fut.result():
                if art.link not in links:
                    article_dict = asdict(art)
                    history.append(article_dict)
                    new_articles.append(article_dict)
                    links.add(art.link)
    
    if new_articles:
        # Sort all history by date, most recent first
        history.sort(key=lambda x: x.get("published", ""), reverse=True)
        save_json(HISTORY_PATH, history)
        return f"✅ {len(new_articles)} new articles. Total {len(history)}.", [a['link'] for a in new_articles]
    
    return f"ℹ️ No new articles. Total {len(history)}.", []

def generate_text(prompt: str) -> str:
    if not prompt:
        return "Please enter a prompt."
    if not GPT2_SESSION or not GPT2_TOKENIZER:
        return "🤖 AI Chat is currently unavailable. The GPT-2 model needs to be downloaded separately. For now, you can browse RSS feeds and search through articles."

    try:
        # Tokenize the input prompt
        inputs = GPT2_TOKENIZER(prompt, return_tensors="np", max_length=512, truncation=True, padding=True)
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        # Create position IDs
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        input_names = [inp.name for inp in GPT2_SESSION.get_inputs()]

        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        # The model requires past_key_values, but this is the first token, so we provide dummy/empty values.
        # Assuming GPT-2 small (12 layers, 12 heads, 64 embed size per head)
        num_layers = 12 
        for i in range(num_layers):
            key_name = f'past_key_values.{i}.key'
            value_name = f'past_key_values.{i}.value'
            if key_name in input_names and value_name in input_names:
                # Creating empty tensors with the expected rank (4) but a zero dimension for sequence length.
                empty_tensor = np.empty((1, 12, 0, 64), dtype=np.float32)
                onnx_inputs[key_name] = empty_tensor
                onnx_inputs[value_name] = empty_tensor

        # Run inference
        outputs = GPT2_SESSION.run(None, onnx_inputs)

        # Get logits (usually the first output)
        logits = outputs[0]

        # Get the predicted token (argmax of the last position)
        predicted_token_id = np.argmax(logits[0, -1, :])

        # Decode the predicted token
        predicted_text = GPT2_TOKENIZER.decode([predicted_token_id])

        return f"{prompt}{predicted_text}"

    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        logger.debug(f"Model inputs: {[inp.name for inp in GPT2_SESSION.get_inputs()]}")
        logger.debug(f"Model outputs: {[out.name for out in GPT2_SESSION.get_outputs()]}")

        return f"I understand you're asking about: '{prompt}'. However, I encountered an error. Please try browsing the RSS feeds instead." 

def create_feed_display(feed_name: str, feed_url: str, layout: str = "cards"):
    """Create a display for a single feed showing recent articles"""
    articles = fetch_feed(feed_url, feed_name)

    if not articles:
        return f"<h3>{feed_name}</h3><p><em>Unable to fetch articles or no articles available.</em></p>"

    # Escape HTML characters
    def escape_html(text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')

    html_content = f"<h3 style='margin-bottom: 20px;'>{feed_name}</h3>"

    if layout == "cards":
        # Card layout with grid
        html_content += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px;">'

        for article in articles:
            title = escape_html(article.title)
            summary = escape_html(article.summary)
            # Extract first 150 chars for card preview
            preview = summary[:150] + "..." if len(summary) > 150 else summary

            html_content += f"""
            <div style='border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; 
                        background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                        transition: transform 0.2s, box-shadow 0.2s; cursor: pointer;'
                 onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)';"
                 onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)';">
                <h4 style='margin: 0 0 12px 0; font-size: 1.1em; line-height: 1.3;'>
                    <a href='{article.link}' target='_blank' 
                       style='text-decoration: none; color: #1a73e8; display: block;'
                       onclick='event.stopPropagation();'>{title}</a>
                </h4>
                <p style='color: #666; font-size: 0.85em; margin: 0 0 12px 0;'>
                    📅 {article.published}
                </p>
                <p style='color: #333; font-size: 0.9em; line-height: 1.5; margin: 0;'>{preview}</p>
            </div>
            """

        html_content += '</div>'
    else:
        # List layout (original)
        for article in articles:
            title = escape_html(article.title)
            summary = escape_html(article.summary)

            html_content += f"""
            <div style='border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;'>
                <h4><a href='{article.link}' target='_blank' style='text-decoration: none; color: #1a73e8;'>{title}</a></h4>
                <p style='color: #666; font-size: 0.9em;'>Published: {article.published}</p>
                <p>{summary}</p>
            </div>
            """

    return html_content

def create_category_tab(category_name: str, feeds: Dict[str, str]):
    """Create a tab for a category with nested tabs for each feed"""
    with gr.Tab(category_name):
        # Add layout toggle for the category
        with gr.Row():
            layout_radio = gr.Radio(
                choices=["cards", "list"],
                value="cards",
                label="Layout",
                scale=1
            )

        with gr.Tabs():
            for feed_name, feed_url in feeds.items():
                with gr.Tab(feed_name):
                    # Create refresh button and display for this feed
                    with gr.Row():
                        refresh_btn = gr.Button(f"🔄 Refresh {feed_name}", scale=1)

                    feed_display = gr.HTML(value=create_feed_display(feed_name, feed_url, "cards"))

                    # Refresh functionality with closure to capture feed_name and feed_url
                    def make_refresh_fn(name, url):
                        def refresh_feed(layout):
                            return create_feed_display(name, url, layout)
                        return refresh_feed

                    # Update on refresh button click
                    refresh_btn.click(
                        fn=make_refresh_fn(feed_name, feed_url),
                        inputs=[layout_radio],
                        outputs=feed_display
                    )

                    # Update on layout change
                    layout_radio.change(
                        fn=make_refresh_fn(feed_name, feed_url),
                        inputs=[layout_radio],
                        outputs=feed_display
                    )

def create_app():
    def chat(history: List[Dict[str, str]], query: str) -> Tuple[List[Dict[str, str]], str]:
        if not query.strip():
            return history, ""
        ctx = load_json(HISTORY_PATH)
        # Use a larger portion of the history for context
        sys = {"role": "system", "content": f"CONTEXT:{json.dumps(ctx)[:8000]}..."}
        full = sys["content"] + "\n"
        for h in history:
            full += f"{h['role']}: {h['content']}\n"
        full += f"user: {query}\nassistant:"
        r = generate_text(full)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": r})
        return history, ""

    with gr.Blocks(title="Datanacci RSS Reader") as app:
        gr.Markdown("# Datanacci RSS Reader with ONNX GPT2")
        
        ARTICLES_PER_PAGE = 25

        # Helper function to update history display, now with pagination and highlighting
        def update_history_display(layout, page_num, new_article_links=None):
            if new_article_links is None:
                new_article_links = []

            history = load_json(HISTORY_PATH)
            total_articles = len(history)
            total_pages = (total_articles + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE or 1
            
            # Clamp page number
            page_num = max(1, min(page_num, total_pages))
            
            start_idx = (page_num - 1) * ARTICLES_PER_PAGE
            end_idx = start_idx + ARTICLES_PER_PAGE
            paginated_history = history[start_idx:end_idx]
            
            page_info = f"Page {page_num} of {total_pages}"
            
            cards_html_update = gr.update()
            table_update = gr.update()

            if layout == "cards":
                if not paginated_history:
                    cards_html = "<p>No articles in history. Click 'Fetch All RSS Feeds' to get started.</p>"
                else:
                    cards_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; margin-top: 20px;">'
                    for article in paginated_history:
                        if isinstance(article, dict):
                            title = article.get('title', 'No title').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                            link = article.get('link', '#')
                            published = article.get('published', 'Unknown')
                            summary = article.get('summary', '')[:200] + '...'
                            feed_name = article.get('feed_name', 'Unknown Feed')
                            
                            # Highlight new articles
                            card_style = 'border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'
                            if link in new_article_links:
                                card_style += 'background-color: #e3f2fd;' # Light blue background for new articles

                            cards_html += f'''
                            <div style='{card_style}'>
                                <div style='color: #1a73e8; font-size: 0.85em; margin-bottom: 8px;'>📰 {feed_name}</div>
                                <h4 style='margin: 0 0 12px 0; font-size: 1.1em; line-height: 1.3;'>
                                    <a href='{link}' target='_blank' style='text-decoration: none; color: #333;'>{title}</a>
                                </h4>
                                <p style='color: #666; font-size: 0.85em; margin: 0 0 12px 0;'>📅 {published}</p>
                                <p style='color: #555; font-size: 0.9em; line-height: 1.4; margin: 0;'>{summary}</p>
                            </div>
                            '''
                    cards_html += '</div>'
                cards_html_update = gr.update(value=cards_html, visible=True)
                table_update = gr.update(visible=False)
            else: # Table view
                df = pd.DataFrame(paginated_history)
                cards_html_update = gr.update(visible=False)
                table_update = gr.update(value=df, visible=True)
            
            return cards_html_update, table_update, gr.update(value=page_info), page_num

        # Combined refresh logic
        def refresh_history(layout, silent=False):
            status_msg, new_links = update_history()
            cards_update, table_update, page_info_update, page_num_update = update_history_display(layout, 1, new_links)
            
            if silent:
                return gr.update(value=""), cards_update, table_update, page_info_update, page_num_update

            return status_msg, cards_update, table_update, page_info_update, page_num_update

        with gr.Tabs():
            for category_name, feeds in RSS_FEEDS.items():
                create_category_tab(category_name, feeds)

            with gr.Tab("📊 All History"):
                page_state = gr.State(1)
                
                with gr.Row():
                    btn = gr.Button("🔄 Fetch All RSS Feeds", scale=1)
                    history_layout = gr.Radio(choices=["cards", "table"], value="cards", label="View", scale=1)
                
                status = gr.Markdown()
                
                with gr.Row(equal_height=True):
                    prev_btn = gr.Button("⬅️ Previous")
                    page_info = gr.Markdown("Page 1 of 1")
                    next_btn = gr.Button("Next ➡️")

                history_cards = gr.HTML(visible=True)
                history_table = gr.Dataframe(interactive=False, visible=False)

                def on_load():
                    return update_history_display("cards", 1, [])

                app.load(fn=on_load, outputs=[history_cards, history_table, page_info, page_state])

                btn.click(
                    fn=refresh_history, 
                    inputs=[history_layout], 
                    outputs=[status, history_cards, history_table, page_info, page_state]
                )
                
                def change_layout(layout):
                    # When changing layout, go back to page 1
                    return update_history_display(layout, 1, [])

                history_layout.change(
                    fn=change_layout, 
                    inputs=[history_layout], 
                    outputs=[history_cards, history_table, page_info, page_state]
                )
                
                def change_page(layout, current_page, direction):
                    new_page = current_page + direction
                    return update_history_display(layout, new_page, [])

                prev_btn.click(
                    fn=change_page,
                    inputs=[history_layout, page_state, gr.State(-1)],
                    outputs=[history_cards, history_table, page_info, page_state]
                )
                next_btn.click(
                    fn=change_page,
                    inputs=[history_layout, page_state, gr.State(1)],
                    outputs=[history_cards, history_table, page_info, page_state]
                )

            with gr.Tab("💬 Chat") as chat_tab:
                chatbot = gr.Chatbot(type="messages", value=[])
                txt = gr.Textbox(placeholder="Ask about the articles...")
                clr = gr.Button("Clear Chat")
                txt.submit(chat, [chatbot, txt], [chatbot, txt])
                clr.click(lambda: ([], ""), None, [chatbot, txt])

            with gr.Tab("⚙️ Config"):
                cfg_df = gr.Dataframe(value=pd.DataFrame(init_config()), interactive=True)
                def save_cfg(df):
                    save_json(CONFIG_PATH, df.to_dict("records"))
                cfg_df.change(save_cfg, cfg_df, None)

            # Add event handler for chat tab selection to refresh history
            def silent_refresh(layout):
                return refresh_history(layout, silent=True)
            
            chat_tab.select(
                fn=silent_refresh,
                inputs=[history_layout],
                outputs=[status, history_cards, history_table, page_info, page_state]
            )

    return app

if __name__ == "__main__":
    init_config()
    print("\n🚀 Starting Datanacci RSS Reader...")
    print("📌 The app will open in your browser at http://127.0.0.1:7860")
    print("📌 Press Ctrl+C to stop the server\n")
    create_app().launch(server_name="127.0.0.1", server_port=7860, share=False)
