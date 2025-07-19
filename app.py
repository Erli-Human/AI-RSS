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

# Constants
CONFIG_PATH = "rss_config.json"
HISTORY_PATH = "article_history.json"

RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "OpenAI Blog": "https://openai.com/blog/rss.xml",
        "Hugging Face Blog": "https://huggingface.co/blog/feed.xml"
    },
    "🚨 Breaking News": {
        "Reuters Top News": "http://feeds.reuters.com/reuters/topNews",
        "Associated Press": "https://apnews.com/hub/ap-top-news/rss"
    },
    "🌍 World News": {
        "Reuters World News": "http://feeds.reuters.com/Reuters/worldNews",
        "BBC World News": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Global News": "https://globalnews.ca/feed/"
    },
    "💻 Technology": {
        "TechCrunch": "https://techcrunch.com/feed/",
        "Wired": "https://www.wired.com/feed/rss"
    },
    "⚽ Sports": {
        "ESPN": "https://www.espn.com/espn/rss/news",
        "Olympic News": "https://olympics.com/en/rss/"
    },
    "💼 Business": {
        "Financial Times": "https://www.ft.com/rss/home",
        "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss"
    },
    "🎵 Music": {
        "Pitchfork Reviews": "http://pitchfork.com/rss/reviews/albums/",
        "Rolling Stone Music": "https://www.rollingstone.com/music/feed/",
        "NME": "https://www.nme.com/feed",
        "Stereogum": "https://www.stereogum.com/feed/",
        "The Fader": "https://www.thefader.com/rss",
        "Consequence of Sound": "https://consequence.net/feed/",
        "Spin": "https://www.spin.com/feed/",
        "Billboard": "https://www.billboard.com/feed/",
        "Paste Magazine": "https://www.pastemagazine.com/music/feed",
        "Brooklyn Vegan": "https://www.brooklynvegan.com/feed/",
        "Gorilla vs Bear": "https://www.gorillavsbear.net/feed/",
        "Hypebeast Music": "https://hypebeast.com/music/feed",
        "Resident Advisor": "https://ra.co/feed",
        "Mixmag": "https://mixmag.net/feed",
        "DJ Mag": "https://djmag.com/rss.xml",
        "Your EDM": "https://www.youredm.com/feed/",
        "Dancing Astronaut": "https://dancingastronaut.com/feed/",
        "EDM.com": "https://edm.com/feed",
        "Complex Music": "https://www.complex.com/music/rss",
        "HipHopDX": "https://hiphopdx.com/rss",
        "XXL": "https://www.xxlmag.com/feed/",
        "The Source": "https://thesource.com/feed/",
        "Loudwire": "https://loudwire.com/feed/",
        "Metal Injection": "https://metalinjection.net/feed",
        "Saving Country Music": "https://www.savingcountrymusic.com/feed",
        "No Depression": "https://www.nodepression.com/feed/",
        "JazzTimes": "https://jazztimes.com/feed/",
        "Gramophone": "https://www.gramophone.co.uk/feed",
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
        "GamesRadar": "https://www.gamesradar.com/rss/",
        "Destructoid": "https://www.destructoid.com/feed",
        "The Verge Gaming": "https://www.theverge.com/games/rss/index.xml",
        "PlayStation Blog": "https://blog.playstation.com/feed/",
        "Xbox Wire": "https://news.xbox.com/en-us/feed/",
        "Nintendo Life": "https://www.nintendolife.com/feeds/latest",
        "Steam Blog": "https://store.steampowered.com/feeds/",
        "Epic Games": "https://www.epicgames.com/site/en-US/rss",
        "Indie DB": "https://rss.indiedb.com/games/feed",
        "Gamasutra": "https://www.gamasutra.com/rss",
        "TIGSource": "https://forums.tigsource.com/index.php?type=rss;action=.xml",
        "Retro Gamer": "https://www.retrogamer.net/feed",
        "TouchArcade": "https://toucharcade.com/feed/",
        "Pocket Gamer": "https://www.pocketgamer.com/rss/",
        "Esports Insider": "https://esportsinsider.com/feed/",
        "The Esports Observer": "https://esportsobserver.com/feed/",
        "Dot Esports": "https://dotesports.com/feed",
        "Giant Bomb": "https://www.giantbomb.com/feeds/",
        "Waypoint": "https://www.vice.com/en/rss/section/games",
        "Tom's Hardware": "https://www.tomshardware.com/feeds/all",
        "PC World": "https://www.pcworld.com/feed",
        "Giant Bombcast": "https://www.giantbomb.com/podcast-xml/giant-bombcast/",
        "Kinda Funny Games": "https://kindafunnygames.libsyn.com/rss"
    },
    "✈️ Travel": {
        "Nomadic Matt": "https://www.nomadicmatt.com/travel-blog/feed/",
        "The Blonde Abroad": "https://theblondeabroad.com/feed/",
        "Expert Vagabond": "https://expertvagabond.com/feed/",
        "Adventurous Kate": "https://www.adventurouskate.com/feed/",
        "One Mile at a Time": "https://onemileatatime.com/feed/",
        "The Points Guy": "https://thepointsguy.com/feed/",
        "Lonely Planet": "https://www.lonelyplanet.com/news/feed/",
        "Travel + Leisure": "https://www.travelandleisure.com/feeds/",
        "Conde Nast Traveler": "https://www.cntraveler.com/feed/rss",
        "National Geographic Travel": "https://www.nationalgeographic.com/travel/feed/",
        "Afar": "https://www.afar.com/feed",
        "Budget Travel": "https://www.budgettravel.com/feed",
        "Frommer's": "https://www.frommers.com/feed",
        "Rick Steves": "https://www.ricksteves.com/feed",
        "View from the Wing": "https://viewfromthewing.com/feed/",
        "Thrifty Traveler": "https://thriftytraveler.com/feed/",
        "Scott's Cheap Flights": "https://scottscheapflights.com/feed/",
        "The Luxury Travel Expert": "https://theluxurytravelexpert.com/feed/",
        "Luxury Travel Diary": "https://www.luxurytraveldiary.com/feed/",
        "A Luxury Travel Blog": "https://www.aluxurytravelblog.com/feed/",
        "God Save the Points": "https://godsavethepoints.com/feed/",
        "Outside Magazine": "https://www.outsideonline.com/feed",
        "Adventure Journal": "https://adventure-journal.com/feed/",
        "The Adventure Blog": "https://www.theadventureblog.com/feeds/posts/default",
        "Backpacker Magazine": "https://www.backpacker.com/feed",
        "Atlas Obscura": "https://www.atlasobscura.com/feeds/latest",
        "Roads & Kingdoms": "https://roadsandkingdoms.com/feed/",
        "Matador Network": "https://matadornetwork.com/feed/",
        "The Culture Trip": "https://theculturetrip.com/feed/",
        "Family Travel Forum": "https://myfamilytravels.com/feed",
        "Travel with Kids": "https://www.travelwithkids.com.au/feed/",
        "Ciao Bambino": "https://ciaobambino.com/feed/",
        "Eater": "https://www.eater.com/rss/index.xml",
        "Saveur": "https://www.saveur.com/feed/",
        "Food & Wine": "https://www.foodandwine.com/feeds/",
        "Migrationology": "https://migrationology.com/feed/",
        "Legal Nomads": "https://www.legalnomads.com/feed/",
        "Travel Photography Blog": "https://www.travelphotographyblog.com/feed/",
        "Stuck in Customs": "https://stuckincustoms.com/feed/",
        "Skift": "https://skift.com/feed/",
        "Travel Weekly": "https://www.travelweekly.com/rss",
        "PhocusWire": "https://www.phocuswire.com/rss"
    },
    "💡 Lifestyle": {
        "Cup of Jo": "https://cupofjo.com/feed/",
        "Design*Sponge": "https://www.designsponge.com/feed",
        "A Beautiful Mess": "https://abeautifulmess.com/feed/",
        "The Everygirl": "https://theeverygirl.com/feed/",
        "Wit & Delight": "https://witanddelight.com/feed/",
        "Camille Styles": "https://camillestyles.com/feed/",
        "The Chriselle Factor": "https://thechrisellefactor.com/feed/",
        "Gal Meets Glam": "https://galmeetsglam.com/feed/",
        "Apartment Therapy": "https://www.apartmenttherapy.com/main.rss",
        "Design Milk": "https://design-milk.com/feed/",
        "Dezeen": "https://www.dezeen.com/feed/",
        "Dwell": "https://www.dwell.com/rss",
        "Architectural Digest": "https://www.architecturaldigest.com/feed/rss",
        "Better Homes & Gardens": "https://www.bhg.com/feeds/",
        "House Beautiful": "https://www.housebeautiful.com/rss/",
        "Elle Decor": "https://www.elledecor.com/rss/",
        "Man Repeller": "https://www.manrepeller.com/feed",
        "Who What Wear": "https://www.whowhatwear.com/rss",
        "The Blonde Salad": "https://www.theblondesalad.com/feed",
        "Atlantic-Pacific": "https://www.atlantic-pacific.com/feed/",
        "Song of Style": "https://www.songofstyle.com/feed",
        "Cupcakes and Cashmere": "https://cupcakesandcashmere.com/feed",
        "Into The Gloss": "https://intothegloss.com/feed/",
        "Byrdie": "https://www.byrdie.com/feeds/",
        "Allure": "https://www.allure.com/feed/rss",
        "Refinery29": "https://www.refinery29.com/rss.xml",
        "Well+Good": "https://www.wellandgood.com/feed/",
        "MindBodyGreen": "https://www.mindbodygreen.com/rss.xml",
        "Goop": "https://goop.com/feed/",
        "Art of Manliness": "https://www.artofmanliness.com/feed/",
        "GQ": "https://www.gq.com/feed/rss",
        "Esquire": "https://www.esquire.com/rss/",
        "Men's Health": "https://www.menshealth.com/feeds/",
        "Primer Magazine": "https://www.primermagazine.com/feed",
        "Dappered": "https://dappered.com/feed/",
        "Scary Mommy": "https://www.scarymommy.com/feed/",
        "Motherly": "https://www.mother.ly/feed/",
        "Parents": "https://www.parents.com/feeds/",
        "What to Expect": "https://www.whattoexpect.com/rss",
        "The Bump": "https://www.thebump.com/rss",
        "The Minimalists": "https://www.theminimalists.com/feed/",
        "Becoming Minimalist": "https://www.becomingminimalist.com/feed/",
        "Unclutterer": "https://unclutterer.com/feed/",
        "Zen Habits": "https://zenhabits.net/feed/",
        "Tim Ferriss": "https://tim.blog/feed/",
        "James Clear": "https://jamesclear.com/feed",
        "Lifehacker": "https://lifehacker.com/rss",
        "Fast Company": "https://www.fastcompany.com/rss",
        "TED Blog": "https://blog.ted.com/feed/"
    },
    "🏡 Home & Garden": {
        "Gardenista": "https://www.gardenista.com/feed/",
        "Garden Design": "https://www.gardendesign.com/feed/",
        "Fine Gardening": "https://www.finegardening.com/feed",
        "The Spruce": "https://www.thespruce.com/feeds/",
        "Better Homes & Gardens Gardening": "https://www.bhg.com/gardening/feed/",
        "Garden Gate Magazine": "https://www.gardengatemagazine.com/feed/",
        "Horticulture Magazine": "https://www.hortmag.com/feed",
        "Proven Winners": "https://www.provenwinners.com/feed",
        "Epic Gardening": "https://www.epicgardening.com/feed/",
        "Garden Betty": "https://www.gardenbetty.com/feed/",
        "Joe Gardener": "https://joegardener.com/feed/",
        "Garden Therapy": "https://gardentherapy.ca/feed/",
        "A Way to Garden": "https://awaytogarden.com/feed/",
        "The Impatient Gardener": "https://www.theimpatientgardener.com/feed/",
        "You Grow Girl": "http://www.yougrowgirl.com/feed/",
        "Cold Climate Gardening": "https://www.coldclimategardening.com/feed/",
        "Floret Flowers": "https://www.floretflowers.com/feed/",
        "Sierra Flower Finder": "https://www.sierraflowerfinder.com/feed",
        "American Meadows": "https://www.americanmeadows.com/blog/feed",
        "White Flower Farm": "https://www.whiteflowerfarm.com/blog/feed",
        "Houseplant Journal": "https://www.houseplantjournal.com/home/feed",
        "The Sill": "https://www.thesill.com/blogs/plants-101.atom",
        "Pistils Nursery": "https://pistilsnursery.com/blogs/journal.atom",
        "Urban Jungle Bloggers": "https://www.urbanjunglebloggers.com/feed/",
        "Homestead and Chill": "https://homesteadandchill.com/feed/",
        "Tenth Acre Farm": "https://www.tenthacrefarm.com/feed/",
        "The Prairie Homestead": "https://www.theprairiehomestead.com/feed",
        "Fresh Eggs Daily": "https://www.fresheggsdaily.blog/feeds/posts/default",
        "This Old House": "https://www.thisoldhouse.com/feeds/feed.xml",
        "Family Handyman": "https://www.familyhandyman.com/feed/",
        "Bob Vila": "https://www.bobvila.com/feed",
        "DIY Network": "https://www.diynetwork.com/feeds",
        "Young House Love": "https://www.younghouselove.com/feed/",
        "Addicted 2 Decorating": "https://www.addicted2decorating.com/feed",
        "Remodelista": "https://www.remodelista.com/feed/",
        "Houzz": "https://www.houzz.com/feeds",
        "Curbly": "https://www.curbly.com/rss",
        "Landscape Design Magazine": "https://landscapedesignmag.com/feed/",
        "Garden Rant": "https://gardenrant.com/feed",
        "The Graceful Gardener": "https://www.thegracefulgardener.com/feed/",
        "Organic Gardening Blog": "https://www.organicgardening.com/feed",
        "Rodale Institute": "https://rodaleinstitute.org/blog/feed/",
        "Permaculture Magazine": "https://www.permaculture.co.uk/rss.xml",
        "Mother Earth News": "https://www.motherearthnews.com/rss",
        "Grit Magazine": "https://www.grit.com/rss",
        "Urban Farm": "https://www.urbanfarmonline.com/rss"
    }
}

GPT2_MODEL_PATH = "gpt2_model.onnx"
GPT2_MODEL_URL = "https://huggingface.co/HyunjuJane/gpt2_model.onnx/resolve/main/gpt2_model.onnx"
GPT2_SESSION = None
MODEL_AVAILABLE = False

# Check if model exists, if not download it
if not os.path.exists(GPT2_MODEL_PATH):
    print("📥 GPT-2 model not found. Downloading from Hugging Face...")
    try:
        response = requests.get(GPT2_MODEL_URL, stream=True)
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
        
        print(f"\n✅ Model downloaded successfully to {GPT2_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print(f"Please download manually from: {GPT2_MODEL_URL}")

# Load the model
if os.path.exists(GPT2_MODEL_PATH):
    try:
        print("🔄 Loading GPT-2 model...")
        GPT2_SESSION = ort.InferenceSession(GPT2_MODEL_PATH)
        MODEL_AVAILABLE = True
        print(f"✅ GPT-2 model loaded successfully from {GPT2_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load GPT-2 model: {e}")
        GPT2_SESSION = None

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

def check_active_feeds(feeds):
    active_feeds = {}
    inactive_count = 0
    for category, items in feeds.items():
        for name, url in items.items():
            try:
                response = requests.head(url, timeout=5)
                if response.status_code == 200:
                    active_feeds.setdefault(category, {})[name] = url
                else:
                    inactive_count += 1
            except requests.RequestException:
                inactive_count += 1
    print(f"✅ Active feeds: {sum(len(v) for v in active_feeds.values())}")
    print(f"❌ Inactive feeds: {inactive_count}")
    return active_feeds

def init_config():
    active_feeds = check_active_feeds(RSS_FEEDS)
    cfg = []
    url_to_feed = {}
    for cat, feeds in active_feeds.items():
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
    cfg = list(url_to_feed.values())
    save_json(CONFIG_PATH, cfg)
    return cfg

def fetch_feed(url, name):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
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
            ) for e in feed.entries[:5]
        ]
    except:
        return []

def update_history():
    cfg = load_json(CONFIG_PATH)
    history = load_json(HISTORY_PATH)
    links = set(a["link"] for a in history if isinstance(a, dict) and "link" in a)
    
    new = 0
    with ThreadPoolExecutor(max_workers=8) as exe:
        unique_feeds = {f["url"]: f for f in cfg if isinstance(f, dict) and "url" in f}
        futures = {exe.submit(fetch_feed, f["url"], f["feed_name"]): f for f in unique_feeds.values()}
        
        for fut in as_completed(futures):
            for art in fut.result():
                if art.link not in links:
                    history.append(asdict(art))
                    links.add(art.link)
                    new += 1
    if new:
        history.sort(key=lambda x: x.get("published", ""), reverse=True)
        save_json(HISTORY_PATH, history)
        return f"✅ {new} new articles. Total {len(history)}."
    return f"ℹ️ No new articles. Total {len(history)}."

def generate_text(prompt: str) -> str:
    if not prompt:
        return "Please enter a prompt."
    if not GPT2_SESSION:
        return "🤖 AI Chat is currently unavailable. The GPT-2 model needs to be downloaded separately. For now, you can browse RSS feeds and search through articles."
    
    ids = np.array([ord(c) for c in prompt if ord(c) < 50257], dtype=np.int64).reshape(1, -1)
    try:
        inp = GPT2_SESSION.get_inputs()[0].name
        out = GPT2_SESSION.run(None, {inp: ids})
        return "".join(chr(i) for i in out[0][0] if i < 256)
    except Exception as e:
        return f"Error generating text: {e}"

def create_feed_display(feed_name: str, feed_url: str, layout: str = "cards"):
    articles = fetch_feed(feed_url, feed_name)
    if not articles:
        return f"<h3>{feed_name}</h3><p><em>Unable to fetch articles or no articles available.</em></p>"
    
    def escape_html(text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
    
    html_content = f"<h3 style='margin-bottom: 20px;'>{feed_name}</h3>"
    
    if layout == "cards":
        html_content += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px;">'
        for article in articles:
            title = escape_html(article.title)
            summary = escape_html(article.summary)
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
    with gr.Tab(category_name):
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
                    with gr.Row():
                        refresh_btn = gr.Button(f"🔄 Refresh {feed_name}", scale=1)
                    feed_display = gr.HTML(value=create_feed_display(feed_name, feed_url, "cards"))
                    
                    def make_refresh_fn(name, url):
                        def refresh_feed(layout):
                            return create_feed_display(name, url, layout)
                        return refresh_feed
                    
                    refresh_btn.click(
                        fn=make_refresh_fn(feed_name, feed_url),
                        inputs=[layout_radio],
                        outputs=feed_display
                    )
                    layout_radio.change(
                        fn=make_refresh_fn(feed_name, feed_url),
                        inputs=[layout_radio],
                        outputs=feed_display
                    )

def create_app():
    def chat(history: List[Dict[str, str]], query: str) -> Tuple[List[Dict[str, str]], None]:
        if not query.strip():
            return history, None
        ctx = load_json(HISTORY_PATH)
        sys = {"role": "system", "content": f"CONTEXT:{json.dumps(ctx)[:1000]}..."}
        full = sys["content"] + "\n"
        for h in history:
            full += f"{h['role']}: {h['content']}\n"
        full += f"user: {query}\nassistant:"
        r = generate_text(full)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": r})
        return history, None

    with gr.Blocks(title="Datanacci RSS Reader") as app:
        gr.Markdown("# Datanacci RSS Reader with ONNX GPT2")
        
        with gr.Tabs():
            active_feeds = check_active_feeds(RSS_FEEDS)
            for category_name, feeds in active_feeds.items():
                create_category_tab(category_name, feeds)
            
            with gr.Tab("📊 All History"):
                with gr.Row():
                    btn = gr.Button("🔄 Fetch All RSS Feeds", scale=1)
                    history_layout = gr.Radio(
                        choices=["cards", "table"],
                        value="cards",
                        label="View",
                        scale=1
                    )
                status = gr.Markdown()
                history_cards = gr.HTML(visible=True)
                history_table = gr.Dataframe(value=pd.DataFrame(load_json(HISTORY_PATH)), interactive=False, visible=False)
                
                def update_history_display(layout):
                    history = load_json(HISTORY_PATH)
                    if layout == "cards":
                        if not history:
                            cards_html = "<p>No articles in history. Click 'Fetch All RSS Feeds' to get started.</p>"
                        else:
                            cards_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; margin-top: 20px;">'
                            for article in history[:50]:
                                if isinstance(article, dict):
                                    title = article.get('title', 'No title').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                    link = article.get('link', '#')
                                    published = article.get('published', 'Unknown')
                                    summary = article.get('summary', '')[:200] + '...'
                                    feed_name = article.get('feed_name', 'Unknown Feed')
                                    cards_html += f"""
                                    <div style='border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px;
                                               background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                        <div style='color: #1a73e8; font-size: 0.85em; margin-bottom: 8px;'>📰 {feed_name}</div>
                                        <h4 style='margin: 0 0 12px 0; font-size: 1.1em; line-height: 1.3;'>
                                            <a href='{link}' target='_blank' style='text-decoration: none; color: #333;'>{title}</a>
                                        </h4>
                                        <p style='color: #666; font-size: 0.85em; margin: 0 0 12px 0;'>📅 {published}</p>
                                        <p style='color: #555; font-size: 0.9em; line-height: 1.4; margin: 0;'>{summary}</p>
                                    </div>
                                    """
                            cards_html += '</div>'
                            if len(history) > 50:
                                cards_html += f'<p style="text-align: center; margin-top: 20px; color: #666;">Showing 50 of {len(history)} articles</p>'
                        return gr.update(value=cards_html, visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(value=pd.DataFrame(history), visible=True)
                
                def ref():
                    s = update_history()
                    history = load_json(HISTORY_PATH)
                    cards_update, table_update = update_history_display(history_layout.value)
                    return s, cards_update, table_update
                
                initial_cards, initial_table = update_history_display("cards")
                history_cards.value = initial_cards['value']
                
                btn.click(ref, outputs=[status, history_cards, history_table])
                history_layout.change(
                    fn=lambda layout: update_history_display(layout),
                    inputs=[history_layout],
                    outputs=[history_cards, history_table]
                )
            
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(type="messages", value=[])
                txt = gr.Textbox(placeholder="Ask about the articles...")
                clr = gr.Button("Clear Chat")
                txt.submit(chat, [chatbot, txt], [chatbot, txt])
                clr.click(lambda: [], None, chatbot)
            
            with gr.Tab("⚙️ Config"):
                cfg_df = gr.Dataframe(value=pd.DataFrame(init_config()), interactive=True)
                def save_cfg(df):
                    save_json(CONFIG_PATH, df.to_dict("records"))
                cfg_df.change(save_cfg, cfg_df, None)

    return app

if __name__ == "__main__":
    init_config()
    create_app().launch()
