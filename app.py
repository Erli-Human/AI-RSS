import gradio as gr
from urllib.request import urlopen
from xml.etree import ElementTree as ET
import time

class RSSReader:
    def __init__(self, rss_feed_url):
        self.rss_feed_url = rss_feed_url
        self.rss_data = None
        self.load_rss()

    def load_rss(self):
        response = urlopen(self.rss_feed_url)
        self.rss_data = ET.parse(response).getroot()
    
    def get_entries(self):
        entries = []
        for item in self.rss_data.findall('.//item'):
            title = item.find('title').text
            link = item.find('link').text
            description = item.find('description').text
            pub_date = time.strptime(item.find('pubDate').text, "%a, %d %b %Y %H:%M:%S +0000")
            entries.append({
                "title": title,
                "link": link,
                "description": description,
                "date": time.strftime("%a, %d %b %Y %H:%M:%S", pub_date)
            })
        return entries

def display_rss_feed(rss_reader):
    # Get the RSS feed items
    entries = rss_reader.get_entries()
    
    # Create a simple UI using Gradio
    title = "RSS Feed Reader"
    description = "Select an RSS feed to view the latest articles."
    
    # Build the UI
    with gr.Blocks() as demo:
        gr.Markdown(description)
        
        selected_feed = gr.Dropdown(label="Select RSS Feed", value=list(RSS_FEEDS.keys())[0], choices=RSS_FEEDS.keys())
        output_area = gr.Textbox()

        @selected_feed.change()
        def update_feed(feed):
            # Load the corresponding RSS feed
            if feed:
                rss_reader = RSSReader(RSS_FEEDS[feed])
                entries = rss_reader.get_entries()
                
                # Create a text area to display the feed items
                output_area.value = "\n".join(f"- {entry['title']} ({entry['date']}) - [Read Article]({entry['link']})" for entry in entries)
            else:
                output_area.value = "No RSS Feed Selected"

        gr.Column([gr.Markdown(title), selected_feed, output_area])

    return demo

# Create a sample feed reader
rss_reader = RSSReader(RSS_FEEDS["🤖 AI & MACHINE LEARNING"]["Science Daily - AI"])
demo = display_rss_feed(rss_reader)

# Launch the application
if __name__ == "__main__":
    gr.Interface(demo, title="RSS Feed Reader").launch()
