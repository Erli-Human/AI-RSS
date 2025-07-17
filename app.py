import gradio as gr

# Define the RSS feed sources
RSS_FEEDS = {
    "🤖 AI & MACHINE LEARNING": {
        "Science Daily - AI": "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        "Science Daily - Technology": "https://www.sciencedaily.com/rss/top/technology.xml",
        # Add more feeds here
    },
    
    "💰 FINANCE & BUSINESS": {
        "Investing.com": "https://www.investing.com/rss/news.rss",
        "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
        "Fortune": "https://fortune.com/feed",
        # Add more feeds here
    },
    
    "🔬 SCIENCE & PHYSICS": {
        "Phys.org": "https://phys.org/rss-feed/",
        "Nature": "https://www.nature.com/nature.rss",
        "Physical Review Letters": "https://feeds.aps.org/rss/recent/prl.xml",
        # Add more feeds here
    },
    
    "💻 TECHNOLOGY": {
        "TechCrunch": "https://techcrunch.com/feed/",
        "The Verge": "https://www.theverge.com/rss/index.xml",
        "Ars Technica": "https://arstechnica.com/feed/",
        # Add more feeds here
    },
    
    "🎬 ENTERTAINMENT": {
        "Variety": "https://variety.com/feed/",
        "The Hollywood Reporter": "https://www.hollywoodreporter.com/rss",
        "Rolling Stone": "https://rollingstone.com/feed/",
        # Add more feeds here
    },
    
    "afenment & CREATIVITY": {
        "Behance": "https://feeds.feedburner.com/behance/vorr",
        "Dribbble": "https://dribbble.com/shots/popular.rss",
        "Creative Bloq": "https://www.creativebloq.com/feed",
        # Add more feeds here
    },
    
    "🌱 ENVIRONMENT & SUSTAINABILITY": {
        "Green Tech Media": "https://www.greentechmedia.com/rss/all"
    }
}

# Define the Ollama models
OLLAMA_MODELS = ["No models found. Run ollama run <model_name>"]

# Create the Gradio interface
with gr.Blocks() as app:
    # Define the tabs for each category
    with gr.Tabs():
        # Tab for categories
        for category, feeds in RSS_FEEDS.items():
            with gr.TabItem(category):
                # Display articles from the current feed
                articles_html_output = gr.HTML(
                    value=format_category_feeds_html(category),
                    elem_id=f"articles_display_{category}"
                )
                
                # Refresh button to load new categories
                refresh_btn = gr.Button("🔄 Refresh Feeds", variant="primary")
                refresh_btn.click(
                    fn=format_category_feeds_html,
                    inputs=[gr.State(category)],
                    outputs=articles_html_output
                )

        # Tab for chat with feeds
        with gr.TabItem("💬 Chat with RSS Feeds"):
            gr.Markdown("### Ask questions about the loaded RSS feeds!")
            
            with gr.Row():
                chat_category_select = gr.Dropdown(
                    choices=list(RSS_FEEDS.keys()),
                    label="Select Category for Chat (Articles from this category will be used as context)",
                    interactive=True,
                    value=list(RSS_FEEDS.keys())[0] if RSS_FEEDS else None,
                    scale=1
                )
                ollama_model_dropdown = gr.Dropdown(
                    choices=OLLAMA_MODELS,
                    label="Select Ollama Model",
                    interactive=True,
                    value=OLLAMA_MODELS[0] if OLLAMA_MODELS else None,
                    scale=1
                )
                # Button to refresh model list in case new models are downloaded
                refresh_models_btn = gr.Button("Refresh Models", scale=0)
                
            chatbot = gr.Chatbot(label="RSS Chat")
            msg = gr.Textbox(label="Your Question", placeholder="e.g., What are the latest AI advancements?", container=False)
            clear = gr.Button("Clear Chat")

            msg.submit(
                chat_with_feeds,
                [chatbot, msg, chat_category_select, ollama_model_dropdown],
                [chatbot, msg]
            )
            clear.click(lambda: None, None, chatbot, queue=False) # Clears the chatbot
                
        # Settings Tab
        with gr.TabItem("⚙️ Settings"):
            gr.Markdown("### Application Settings")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Feed Sources")
                    feed_count = sum(len(feeds) for feeds in RSS_FEEDS.values())
                    gr.Markdown(f"**Total Categories:** {len(RSS_FEEDS)}")
                    gr.Markdown(f"**Total Feeds:** {feed_count}")
                    
                    for category, feeds in RSS_FEEDS.items():
                        gr.Markdown(f"**{category}:** {len(feeds)} feeds")

                with gr.Column():
                    gr.Markdown("#### System Info")
                    gr.Markdown(f"**Last Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    gr.Markdown("**Status:** Running")
                    gr.Markdown("**Version:** 1.0.0")
                    gr.Markdown("---")
                    gr.Markdown("#### Ollama Status")
                    ollama_status_display = gr.HTML(label="Ollama Server Status")

                # Function to check Ollama status
                def check_ollama_status():
                    try:
                        models = ollama.list()
                        num_models = len(models.get('models', []))
                        return f"<p style='color: green;'>✅ Ollama Server is Running!</p><p>Available Models: {num_models}</p>"
                    except Exception as e:
                        return f"<p style='color: red;'>❌ Ollama Server Not Reachable. Error: {e}</p><p>Please ensure Ollama is installed and running (`ollama serve`).</p>"

                app.load(
                    fn=check_ollama_status,
                    outputs=ollama_status_display
                )

# Run the Gradio application
if __name__ == "__main__":
    app.launch()
