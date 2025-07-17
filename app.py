import feedparser
import gradio as gr
from datetime import datetime

# Define a function to fetch articles from a given URL
def fetch_articles(url):
    try:
        # Parse the RSS feed
        feed = feedparser.parse(url)
        
        if feed.bozo:
            return "Error: Invalid RSS feed or URL not accessible", pd.DataFrame()
        
        # Extract feed information
        feed_title = feed.feed.get('title', 'Unknown Feed')
        feed_description = feed.feed.get('description', 'No description available')
        
        # Extract entries
        articles = []
        for entry in feed.entries[:15]:  # Increased to 15 most recent entries
            article_data = {
                'Title': entry.get('title', 'No title'),
                'Link': entry.get('link', 'No link'),
                'Published': entry.get('published', 'No date'),
                'Summary': entry.get('summary', 'No summary')[:300] + '...' if len(entry.get('summary', '')) > 300 else entry.get('summary', 'No summary')
            }
            articles.append(article_data)
        
        # Create DataFrame for better display
        df = pd.DataFrame(articles)
        
        feed_info = f"**Feed:** {feed_title}\n**Description:** {feed_description}\n**Total Articles:** {len(entries)}\n**Feed URL:** {url}"
        
        return feed_info, df
        
    except Exception as e:
        return f"Error fetching RSS feed: {str(e)}", pd.DataFrame()

# Define the Gradio interface
with gr.Blocks(title="RSS Feed Helper") as app:
    gr.Markdown("# 📰 RSS Feed Helper")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Quick Select")
            preset_dropdown = gr.Dropdown(
                choices=get_all_feeds_list(),
                label="Choose from Curated Feeds",
                info="Select from our comprehensive collection of RSS feeds",
                interactive=True
            )
            load_preset_btn = gr.Button("📥 Load Selected Feed", variant="primary")
            
            gr.Markdown("### 🔗 Custom URL")
            url_input = gr.Textbox(
                label="RSS Feed URL",
                placeholder="https://example.com/rss.xml",
                info="Enter any valid RSS feed URL"
            )
            
            with gr.Row():
                fetch_btn = gr.Button("🔄 Fetch RSS Feed", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            gr.Markdown("### 🔍 Search Articles")
            search_input = gr.Textbox(
                label="Search Keywords",
                placeholder="Enter keywords to search in titles and summaries...",
                info="Search is case-insensitive"
            )
            
            with gr.Row():
                search_btn = gr.Button("🔍 Search", variant="secondary")
                clear_search_btn = gr.Button("❌ Clear Search", variant="secondary")
    
        with gr.Column(scale=2):
            feed_info = gr.Markdown("### Feed Information\nSelect a preset feed or enter an RSS URL to get started.")
            
            articles_table = gr.Dataframe(
                headers=['Title', 'Link', 'Published', 'Summary'],
                interactive=False,
                wrap=True,
                height=500,
                label="Articles"
            )
    
    # Store the original dataframe for searching
    original_df = gr.State(pd.DataFrame())
    
    # Event handlers
    load_preset_btn.click(
        fn=load_preset_feed,
        inputs=[preset_dropdown],
        outputs=[url_input, feed_info, articles_table, original_df]
    )
    
    fetch_btn.click(
        fn=fetch_rss_feed,
        inputs=[url_input],
        outputs=[feed_info, articles_table]
    ).then(
        fn=lambda df: df,
        inputs=[articles_table],
        outputs=[original_df]
    )
    
    search_btn.click(
        fn=search_entries,
        inputs=[original_df, search_input],
        outputs=[articles_table]
    )
    
    clear_search_btn.click(
        fn=clear_search,
        inputs=[original_df],
        outputs=[articles_table, search_input]
    )
    
    clear_btn.click(
        fn=lambda: ("### Feed Information\nSelect a preset feed or enter an RSS URL to get started.", pd.DataFrame(), pd.DataFrame(), "", ""),
        outputs=[feed_info, articles_table, original_df, search_input, url_input]
    )
    
    # Allow Enter key to trigger fetch
    url_input.submit(
        fn=fetch_rss_feed,
        inputs=[url_input],
        outputs=[feed_info, articles_table]
    ).then(
        fn=lambda df: df,
        inputs=[articles_table],
        outputs=[original_df]
    )
    
    # Allow Enter key to trigger search
    search_input.submit(
        fn=search_entries,
        inputs=[original_df, search_input],
        outputs=[articles_table]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
