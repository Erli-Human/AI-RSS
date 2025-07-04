import gradio as gr
import feedparser

def get_feed_entries(feed_url):
    feed = feedparser.parse(feed_url)
    if feed.bozo:
        return f"Failed to parse feed: {feed.bozo_exception}"
    entries = feed.entries[:5]  # Display first 5 entries
    result = ""
    for entry in entries:
        title = entry.get("title", "No title")
        link = entry.get("link", "#")
        summary = entry.get("summary", "")
        result += f"### [{title}]({link})\n{summary}\n\n"
    return result or "No entries found."

with gr.Blocks() as demo:
    gr.Markdown("# RSS Feed Viewer")
    feed_url = gr.Textbox(label="RSS Feed URL", placeholder="Enter RSS feed URL here")
    output = gr.Markdown()
    feed_url.change(get_feed_entries, feed_url, output)

if __name__ == "__main__":
    demo.launch()