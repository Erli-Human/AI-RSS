import csv
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO, BytesIO
import base64

class FeedAnalytics:
    """Analytics and export functionality for RSS feeds."""
    
    def __init__(self):
        self.feed_history = []
    
    def log_feed_check(self, category: str, feed_name: str, status: str, article_count: int):
        """Log feed check results for analytics."""
        self.feed_history.append({
            'timestamp': datetime.now(),
            'category': category,
            'feed_name': feed_name,
            'status': status,
            'article_count': article_count
        })
    
    def export_to_csv(self, results: Dict) -> str:
        """Export feed results to CSV format."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Category', 'Feed Name', 'Status', 'Articles', 'Last Updated', 'Error'])
        
        # Write data
        for category, feeds in results.items():
            for feed_name, feed_data in feeds.items():
                writer.writerow([
                    category,
                    feed_name,
                    feed_data.status,
                    len(feed_data.articles) if feed_data.status == 'success' else 0,
                    feed_data.last_updated,
                    feed_data.error if feed_data.status == 'error' else ''
                ])
        
        return output.getvalue()
    
    def export_to_json(self, results: Dict) -> str:
        """Export feed results to JSON format."""
        export_data = {}
        
        for category, feeds in results.items():
            export_data[category] = {}
            for feed_name, feed_data in feeds.items():
                export_data[category][feed_name] = {
                    'status': feed_data.status,
                    'articles': [asdict(article) for article in feed_data.articles],
                    'last_updated': feed_data.last_updated,
                    'error': feed_data.error if feed_data.status == 'error' else None
                }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def create_status_chart(self, results: Dict) -> str:
        """Create a status chart as base64 encoded image."""
        categories = []
        working_counts = []
        total_counts = []
        
        for category, feeds in results.items():
            categories.append(category.replace('🤖 ', '').replace('💻 ', '').replace('🔬 ', '')[:15])
            working = sum(1 for feed in feeds.values() if feed.status == 'success')
            total = len(feeds)
            working_counts.append(working)
            total_counts.append(total)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(categories))
        
        ax.bar([i - 0.2 for i in x], working_counts, 0.4, label='Working', color='#00b894')
        ax.bar([i + 0.2 for i in x], total_counts, 0.4, label='Total', color='#ddd')
        
        ax.set_xlabel('Categories')
        ax.set_ylabel('Number of Feeds')
        ax.set_title('RSS Feed Status by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"

# Add export functionality to the main app
def add_export_tab(app):
    """Add export functionality to the existing app."""
    
    analytics = FeedAnalytics()
    
    with gr.TabItem("📤 Export & Analytics"):
        gr.Markdown("### Export feed data and view analytics")
        
        with gr.Row():
            export_category = gr.Dropdown(
                choices=["All Categories"] + list(RSS_FEEDS.keys()),
                label="Select Category to Export",
                value="All Categories"
            )
            export_format = gr.Radio(
                choices=["CSV", "JSON"],
                label="Export Format",
                value="CSV"
            )
        
        export_btn = gr.Button("📊 Generate Export", variant="primary")
        
        with gr.Row():
            export_output = gr.File(label="Download Export")
            chart_output = gr.Image(label="Status Chart")
        
        def generate_export(category, format_type):
            # Fetch data
            if category == "All Categories":
                all_results = {}
                for cat in RSS_FEEDS.keys():
                    all_results[cat] = fetch_category_feeds_parallel(cat)
            else:
                all_results = {category: fetch_category_feeds_parallel(category)}
            
            # Generate export
            if format_type == "CSV":
                content = analytics.export_to_csv(all_results)
                filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            else:
                content = analytics.export_to_json(all_results)
                filename = f"rss_feeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Save file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Generate chart
            chart_image = analytics.create_status_chart(all_results)
            
            return filename, chart_image
        
        export_btn.click(
            fn=generate_export,
            inputs=[export_category, export_format],
            outputs=[export_output, chart_output]
        )

# Monitoring Script
def create_monitoring_script():
    """Create a separate monitoring script for continuous feed checking."""
    
    monitoring_script = '''
#!/usr/bin/env python3
"""
RSS Feed Monitoring Script
Runs continuous monitoring of RSS feeds and generates reports.
"""

import time
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import schedule

def check_all_feeds():
    """Check all feeds and generate status report."""
    print(f"🔍 Starting feed check at {datetime.now()}")
    
    all_results = {}
    for category in RSS_FEEDS.keys():
        print(f"  Checking {category}...")
        all_results[category] = fetch_category_feeds_parallel(category)
    
    # Generate report
    total_feeds = sum(len(feeds) for feeds in all_results.values())
    working_feeds = sum(
        sum(1 for feed in feeds.values() if feed.status == 'success')
        for feeds in all_results.values()
    )
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_feeds': total_feeds,
        'working_feeds': working_feeds,
        'success_rate': (working_feeds / total_feeds) * 100,
        'details': all_results
    }
    
    # Save report
    filename = f"feed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Check complete: {working_feeds}/{total_feeds} feeds working ({report['success_rate']:.1f}%)")
    
    # Alert if success rate is low
    if report['success_rate'] < 80:
        print(f"⚠️  WARNING: Success rate below 80%!")
        # Add email notification here if needed
    
    return report

def main():
    """Main monitoring loop."""
    print("🚀 RSS Feed Monitor Started")
    print("📊 Monitoring", sum(len(feeds) for feeds in RSS_FEEDS.values()), "feeds")
    
    # Schedule checks
    schedule.every(30).minutes.do(check_all_feeds)
    schedule.every().day.at("09:00").do(check_all_feeds)
    
    # Run initial check
    check_all_feeds()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
'''
    
    with open("rss_monitor.py", "w") as f:
        f.write(monitoring_script)
    
    print("📝 Created monitoring script: rss_monitor.py")

# Docker Configuration
def create_docker_files():
    """Create Docker configuration for easy deployment."""
    
    dockerfile = '''
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "rss_viewer.py"]
'''
    
    docker_compose = '''
version: '3.8'

services:
  rss-viewer:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
    restart: unless-stopped
'''
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("🐳 Created Docker configuration files")

if __name__ == "__main__":
    # Create additional files
    create_monitoring_script()
    create_docker_files()
    
    # Launch main application
    print("🚀 Starting Advanced RSS Feed Viewer...")
    app = create_enhanced_rss_viewer()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False
    )
