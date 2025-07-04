import gradio as gr
import feedparser
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"  # Change if your Ollama endpoint is different
OLLAMA_MODEL = "llama3"  # Or another model you have

# ... (FEEDS dict here, unchanged) ...

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

def show_feed(category, feed_name):
    url = None
    for name, link in FEEDS.get(category, []):
        if name == feed_name:
            url = link
            break
    if url:
        return get_feed_entries(url)[0]
    return "Feed not found."

def ollama_chat(question, context):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant answering questions about RSS feed news articles. Use only the provided feed content as your source."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        result = r.json()
        # Ollama's streaming API may need collecting chunks; if so, adjust here.
        # For non-streaming, the answer is in result["message"]["content"]
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

with gr.Blocks() as demo:
    gr.HTML(get_ticker_html())
    gr.Markdown(
        """
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="https://datanacci.carrd.co/assets/images/image01.png" alt="Datanacci" style="height: 60px;">
            <h1 style="margin: 0;">AI-RSS Feed Viewer</h1>
        </div>
        <p>Select a category and feed to view the latest headlines. Ask questions about the feed using the chat below!</p>
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
    headlines = gr.Markdown()
    chat_input = gr.Textbox(label="Ask a question about this feed", placeholder="Type your question here...")
    chat_output = gr.Markdown()
    # Store feed context
    context_state = gr.State("")

    def update_feed_and_context(selected_category, selected_feed):
        html, plain = get_feed_entries(FEEDS[selected_category][[n for n, _ in FEEDS[selected_category]].index(selected_feed)][1])
        return html, plain

    category.change(
        fn=lambda c: gr.Dropdown.update(
            choices=[name for name, url in FEEDS[c]],
            value=FEEDS[c][0][0]
        ),
        inputs=category,
        outputs=feed
    )

    # Update headlines and context when feed or category changes
    def update_on_feed_change(selected_category, selected_feed):
        html, plain = get_feed_entries(FEEDS[selected_category][[n for n, _ in FEEDS[selected_category]].index(selected_feed)][1])
        return html, plain

    gr.on(
        triggers=[category, feed],
        fn=update_on_feed_change,
        inputs=[category, feed],
        outputs=[headlines, context_state]
    )

    # Chat button
    def handle_chat(question, context):
        if not question.strip():
            return "Please enter a question."
        if not context.strip():
            return "No feed context available."
        return ollama_chat(question, context)

    chat_input.submit(
        fn=handle_chat,
        inputs=[chat_input, context_state],
        outputs=chat_output
    )

    # Show initial feed
    html, plain = get_feed_entries(FEEDS["AI & Technology"][0][1])
    headlines.value = html
    context_state.value = plain

if __name__ == "__main__":
    demo.launch()
