import gradio as gr
import feedparser
import requests
import os

OLLAMA_URL = "http://localhost:11434/api/chat"

def get_ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        models = [m['name'] for m in data.get('models', [])]
        if not models:
            models = ["llama3"]
        return models
    except Exception:
        return ["llama3"]

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

def get_feed_entries(feed_url, num_entries=5):
    feed = feedparser.parse(feed_url)
    if feed.bozo:
        return f"Failed to parse feed: {feed.bozo_exception}", ""
    entries = feed.entries[:num_entries]
    html = ""
    plain = ""
    for entry in entries:
        title = entry.get("title", "No title")
        link = entry.get("link", "#")
        summary = entry.get("summary", "")
        html += f"### [{title}]({link})\n{summary}\n\n"
        plain += f"{title}\n{summary}\n\n"
    return html or "No entries found.", plain

def ollama_chat(question, context, model="llama3", nlu=None, mmlu=None):
    system_prompt = "You are a helpful assistant answering questions about RSS feed news articles. Use only the provided feed content as your source."
    if nlu:
        system_prompt += f" Answer with an NLU (Natural Language Understanding) focus on: {nlu}."
    if mmlu:
        system_prompt += f" Include MMLU (Massive Multitask Language Understanding) reasoning for: {mmlu}."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        result = r.json()
        answer = result.get("message", {}).get("content", "No answer received.")
        return answer
    except Exception as e:
        return f"Error contacting Ollama: {e}"

def get_ticker_html():
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids":"bitcoin,ethereum,dogecoin", "vs_currencies":"usd"}
        )
        prices = resp.json()
        btc = prices.get("bitcoin", {}).get("usd", "N/A")
        eth = prices.get("ethereum", {}).get("usd", "N/A")
        doge = prices.get("dogecoin", {}).get("usd", "N/A")
        nasdaq = "17,800"
        sp500 = "5,500"
        aapl = "210.50"
        return f"""
        <div style="overflow:hidden;background:#111;color:#fff;padding:8px 0;">
            <marquee style='font-size:1.1em;'>
                &#128176; BTC: ${btc} &nbsp; | &nbsp; ETH: ${eth} &nbsp; | &nbsp; NASDAQ: {nasdaq} &nbsp; | &nbsp; S&amp;P 500: {sp500} &nbsp; | &nbsp; DOGE: ${doge} &nbsp; | &nbsp; AAPL: ${aapl}
            </marquee>
        </div>
        """
    except Exception:
        return """
        <div style="overflow:hidden;background:#111;color:#fff;padding:8px 0;">
            <marquee style='font-size:1.1em;'>
                &#128176; BTC: $65,200 &nbsp; | &nbsp; ETH: $3,500 &nbsp; | &nbsp; NASDAQ: 17,800 &nbsp; | &nbsp; S&amp;P 500: 5,500 &nbsp; | &nbsp; DOGE: $0.12 &nbsp; | &nbsp; AAPL: $210.50
            </marquee>
        </div>
        """

try:
    import soundfile as sf
    import numpy as np
    import torch
    import torchaudio
    import tempfile
    from TTS.api import TTS
    import whisper
    TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(TTS_MODEL)
    whisper_model = whisper.load_model("base")
    TTS_AVAILABLE = True
    WHISPER_AVAILABLE = True
except Exception as e:
    TTS_AVAILABLE = False
    WHISPER_AVAILABLE = False

def tts_speak(text, speaker_wav=None):
    if not TTS_AVAILABLE:
        return None, "TTS not available"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tts.tts_to_file(text=text, file_path=tmpfile.name, speaker_wav=speaker_wav)
        audio, sr = sf.read(tmpfile.name)
        return (sr, audio), None

def whisper_transcribe(audio):
    if not WHISPER_AVAILABLE:
        return None, "Whisper not available"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, audio[1], audio[0])
        result = whisper_model.transcribe(tmpfile.name)
        return result['text'], None

with gr.Blocks() as demo:
    gr.HTML(get_ticker_html())
    gr.Markdown(
        """
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="https://datanacci.carrd.co/assets/images/image01.png" alt="Datanacci" style="height: 60px;">
            <h1 style="margin: 0;">AI-RSS Feed Viewer + Chat</h1>
        </div>
        <p>Select a category and feed to view the latest headlines. Then ask questions using chat (text or voice!).</p>
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
        ollama_model = gr.Dropdown(
            choices=get_ollama_models(),
            label="Ollama Model",
            value=get_ollama_models()[0] if get_ollama_models() else "llama3"
        )
    headlines = gr.Markdown()
    chatbox = gr.Chatbot(label="Ollama Chat (about the selected feed)", type="messages")
    user_input = gr.Textbox(label="Ask a question (text)", placeholder="Type your question here and press Enter...")
    audio_input = gr.Audio(label="Or ask by voice")
    tts_button = gr.Button("🔊 Speak Answer (TTS)")
    tts_audio = gr.Audio(label="Ollama TTS Answer", autoplay=True)
    nlu = gr.Textbox(label="NLU focus (optional)", placeholder="e.g. sentiment, summarization")
    mmlu = gr.Textbox(label="MMLU focus (optional)", placeholder="e.g. reasoning, domain")
    context_state = gr.State("")
    chat_history = gr.State([])

    # CATEGORY CHANGE: update feeds, headlines, context, chat
    def category_changed(selected_category):
        feed_choices = [name for name, url in FEEDS[selected_category]]
        feed_default = feed_choices[0]
        html, plain = get_feed_entries(FEEDS[selected_category][0][1])
        return (
            gr.Dropdown.update(choices=feed_choices, value=feed_default),
            html,
            plain,
            []
        )
    category.change(
        fn=category_changed,
        inputs=category,
        outputs=[feed, headlines, context_state, chat_history]
    )

    # FEED CHANGE: update headlines, context, clear chat
    def feed_changed(selected_category, selected_feed):
        html, plain = get_feed_entries(FEEDS[selected_category][[n for n, _ in FEEDS[selected_category]].index(selected_feed)][1])
        return html, plain, []
    feed.change(
        fn=feed_changed,
        inputs=[category, feed],
        outputs=[headlines, context_state, chat_history]
    )

    def handle_chat(user_message, nlu, mmlu, context, history, ollama_model_):
        if not user_message.strip():
            return history, None
        if not context.strip():
            return history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "No feed context available."}
            ], None
        answer = ollama_chat(user_message, context, model=ollama_model_, nlu=nlu, mmlu=mmlu)
        history = (history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer}
        ]
        return history, answer

    user_input.submit(
        fn=handle_chat,
        inputs=[user_input, nlu, mmlu, context_state, chat_history, ollama_model],
        outputs=[chatbox, tts_audio]
    )

    def handle_audio(audio, nlu, mmlu, context, history, ollama_model_):
        if not WHISPER_AVAILABLE:
            return history + [
                {"role": "user", "content": "[Voice]"},
                {"role": "assistant", "content": "Whisper not available."}
            ], None
        transcript, err = whisper_transcribe(audio)
        if err:
            return history + [
                {"role": "user", "content": "[Voice]"},
                {"role": "assistant", "content": f"Whisper error: {err}"}
            ], None
        answer = ollama_chat(transcript, context, model=ollama_model_, nlu=nlu, mmlu=mmlu)
        history = (history or []) + [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": answer}
        ]
        return history, answer

    audio_input.change(
        fn=handle_audio,
        inputs=[audio_input, nlu, mmlu, context_state, chat_history, ollama_model],
        outputs=[chatbox, tts_audio]
    )

    def tts_from_last(history):
        if not TTS_AVAILABLE or not history:
            return None
        for msg in reversed(history):
            if msg["role"] == "assistant" and msg["content"]:
                audio, err = tts_speak(msg["content"])
                if err:
                    return None
                return audio
        return None

    tts_button.click(
        fn=tts_from_last,
        inputs=chat_history,
        outputs=tts_audio
    )

    html, plain = get_feed_entries(FEEDS["AI & Technology"][0][1])
    headlines.value = html
    context_state.value = plain
    chatbox.value = []

if __name__ == "__main__":
    demo.launch()
