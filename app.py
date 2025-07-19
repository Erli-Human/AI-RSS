import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import gradio as gr
import feedparser
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import onnxruntime as ort

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
GPT2_SESSION = None
if os.path.exists(GPT2_MODEL_PATH):
    try:
        GPT2_SESSION = ort.InferenceSession(GPT2_MODEL_PATH)
    except:
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
    if not os.path.exists(path) or os.path.getsize(path)==0:
        return default
    try:
        data = json.load(open(path,"r",encoding="utf-8"))
        return data if isinstance(data,list) else default
    except:
        return default

def save_json(path: str, data):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2)

def init_config():
    cfg = load_json(CONFIG_PATH)
    urls = {f["url"] for f in cfg if isinstance(f,dict)}
    updated = False
    for cat, feeds in RSS_FEEDS.items():
        for name, url in feeds.items():
            if url not in urls:
                cfg.append({
                    "category": cat,
                    "feed_name": name,
                    "url": url,
                    "created": datetime.utcnow().isoformat(),
                    "key": f"{cat}_{name}"
                })
                updated = True
    cfg = [f for f in cfg if isinstance(f,dict)]
    if updated:
        save_json(CONFIG_PATH, cfg)
    return cfg

def fetch_feed(url, name):
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        return [
            Article(
                title=e.get("title","No title"),
                link=e.get("link",""),
                published=e.get("published","Unknown"),
                summary=e.get("summary","")[:300]+"...",
                author=e.get("author",""),
                feed_name=name
            ) for e in feed.entries
        ]
    except:
        return []

def update_history():
    cfg = load_json(CONFIG_PATH)
    history = load_json(HISTORY_PATH)
    links = {a["link"] for a in history if isinstance(a,dict)}
    new = 0
    with ThreadPoolExecutor(max_workers=8) as exe:
        fut2 = {exe.submit(fetch_feed,f["url"],f["feed_name"]):f for f in cfg if isinstance(f,dict)}
        for fut in as_completed(fut2):
            for art in fut.result():
                if art.link not in links:
                    history.append(asdict(art))
                    links.add(art.link)
                    new += 1
    if new:
        history.sort(key=lambda x: x.get("published",""), reverse=True)
        save_json(HISTORY_PATH, history)
        return f"✅ {new} new articles. Total {len(history)}."
    return f"ℹ️ No new articles. Total {len(history)}."

def generate_text(prompt: str) -> str:
    if not GPT2_SESSION or not prompt:
        return "Model unavailable or empty prompt."
    ids = np.array([ord(c) for c in prompt if ord(c)<50257], dtype=np.int64).reshape(1,-1)
    try:
        inp = GPT2_SESSION.get_inputs()[0].name
        out = GPT2_SESSION.run(None, {inp: ids})
        return "".join(chr(i) for i in out[0][0] if i<256)
    except Exception as e:
        return f"Error: {e}"

def create_app():
    def chat(history: List[Dict[str,str]], query: str) -> Tuple[List[Dict[str,str]],None]:
        if not query.strip(): return history, None
        ctx = load_json(HISTORY_PATH)
        sys = {"role":"system","content":f"CONTEXT:{json.dumps(ctx)[:1000]}..."}
        full = sys["content"]+"\n"
        for h in history:
            full += f"{h['role']}: {h['content']}\n"
        full += f"user: {query}\nassistant:"
        r = generate_text(full)
        history.append({"role":"user","content":query})
        history.append({"role":"assistant","content":r})
        return history, None

    with gr.Blocks() as app:
        gr.Markdown("# Datanacci RSS with ONNX GPT2")
        with gr.Tabs():
            with gr.TabItem("History"):
                btn = gr.Button("Fetch RSS")
                status = gr.Markdown()
                df = gr.Dataframe(value=pd.DataFrame(load_json(HISTORY_PATH)), interactive=False)
                def ref():
                    s = update_history()
                    return s, pd.DataFrame(load_json(HISTORY_PATH))
                btn.click(ref, outputs=[status, df])
            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(type="messages", value=[])
                txt = gr.Textbox(placeholder="Ask...")
                clr = gr.Button("Clear")
                txt.submit(chat, [chatbot, txt], [chatbot, txt])
                clr.click(lambda: [], None, chatbot)
            with gr.TabItem("Config"):
                cfg_df = gr.Dataframe(value=pd.DataFrame(init_config()), interactive=True)
                def save_cfg(df):
                    save_json(CONFIG_PATH, df.to_dict("records"))
                cfg_df.change(save_cfg, cfg_df, None)
    return app

if __name__=="__main__":
    init_config()
    create_app().launch()
