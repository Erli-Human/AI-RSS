import feedparser
import requests
from bs4 import BeautifulSoup
import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")


def fetch_news(url):
    """Fetches news articles from an RSS feed URL."""
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            article = {
                'title': entry.title,
                'link': entry.link,
                'summary': entry.summary,  # Or use description if summary is missing
                'published': datetime.datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else None
            }
            articles.append(article)
        return articles
    except Exception as e:
        print(f"Error fetching news from {url}: {e}")
        return []

def scrape_content(url):
    """Scrapes the full content of an article from its URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Adapt this selector to the specific website's HTML structure!
        article_text = soup.find('div', class_='article-body').get_text() # Example: find a div with class "article-body"
        return article_text
    except Exception as e:
        print(f"Error scraping content from {url}: {e}")
        return None

def process_articles(articles):
    """Processes articles, potentially scraping full content."""
    for article in articles:
        if not article['summary']:  # If summary is empty, try to scrape
            full_content = scrape_content(article['link'])
            if full_content:
                article['summary'] = full_content
    return articles

def get_news_from_urls(urls):
    """Fetches and processes news from multiple URLs."""
    all_articles = []
    for url in urls:
        articles = fetch_news(url)
        if articles:
            processed_articles = process_articles(articles)
            all_articles.extend(processed_articles)
    return all_articles

# Example usage (for testing):
if __name__ == '__main__':
    urls = [
        "https://www.bbc.com/news/rss.xml",  # Replace with your desired RSS feeds
        "https://feeds.bbci.co.uk/news/world/rss.xml"
    ]
    news_articles = get_news_from_urls(urls)
    for article in news_articles:
        print(f"Title: {article['title']}")
        print(f"Summary: {article['summary'][:100]}...") # Print first 100 chars of summary
        print("-" * 20)
