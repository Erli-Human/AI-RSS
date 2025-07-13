import gradio as gr
import feedparser
from datetime import datetime
from dateutil import parser
import sqlite3  # For SQLite database storage

# Database setup (replace with your preferred method)
def create_table():
    conn = sqlite3.connect('news_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news (
                    title TEXT,
                    link TEXT,
                    summary TEXT,
                    published TEXT
                )''')
    conn.commit()
    conn.close()

def store_news(title, link, summary, published):
    conn = sqlite3.connect('news_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO news VALUES (?, ?, ?, ?)", (title, link, summary, published))
    conn.commit()
    conn.close()

# RSS Feed Parsing Function
def parse_rss_feed(url):
    """Fetches and parses an RSS feed."""
    try:
        feed = feedparser.parse(url)
        news_items = []
        for entry in feed.entries:
            title = entry.get('title', 'No Title')
            link = entry.get('link', 'No Link')
            summary = entry.get('summary', 'No Summary')  # Or use description if summary is missing
            published_date = entry.get('published')

            if published_date:
                try:
                    published = parser.parse(published_date).isoformat() # Convert to ISO format
                except Exception as e:
                    print(f"Error parsing date {published_date}: {e}")
                    published = datetime.now().isoformat()  # Use current time if parsing fails
            else:
                published = datetime.now().isoformat()

            news_items.append({
                'title': title,
                'link': link,
                'summary': summary,
                'published': published
            })
            store_news(title, link, summary, published) # Store in database
        return news_items
    except Exception as e:
        print(f"Error parsing feed {url}: {e}")
        return []

# Gradio Interface Function
def process_rss(url):
    """Processes the RSS feed and returns a formatted string."""
    news = parse_rss_feed(url)
    if not news:
        return "No news items found or error parsing feed."

    output_string = ""
    for item in news:
        output_string += f"<b>Title:</b> {item['title']}<br>"
        output_string += f"<b>Link:</b> <a href='{item['link']}'>{item['link']}</a><br>"
        output_string += f"<b>Summary:</b> {item['summary']}<br><br>"

    return output_string


# Gradio Interface Setup
if __name__ == '__main__':
    create_table() # Create the database table if it doesn't exist
    with gr.Blocks() as demo:
        gr.Markdown("# RSS Feed Indexer")
        url_input = gr.Textbox(label="Enter RSS Feed URL:")
        output_text = gr.HTML(label="News Items:")
        process_button = gr.Button("Process Feed")

        process_button.click(fn=process_rss, inputs=url_input, outputs=output_text)

    demo.launch()
