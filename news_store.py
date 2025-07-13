import json
import os

NEWS_FILE = "news_data.json"  # File to store news articles

def save_news(articles):
    """Saves news articles to a JSON file."""
    try:
        with open(NEWS_FILE, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)  # Use indent for readability
        print("News saved successfully.")
    except Exception as e:
        print(f"Error saving news to file: {e}")

def load_news():
    """Loads news articles from a JSON file."""
    try:
        with open(NEWS_FILE, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        return articles
    except FileNotFoundError:
        print("News file not found.  Returning empty list.")
        return []
    except Exception as e:
        print(f"Error loading news from file: {e}")
        return []

if __name__ == '__main__':
    # Example usage (for testing):
    from news_fetcher import get_news_from_urls  # Import the fetcher function

    urls = [
        "https://www.bbc.com/news/rss.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml"
    ]
    news_articles = get_news_from_urls(urls)
    save_news(news_articles)

    loaded_articles = load_news()
    print(f"Loaded {len(loaded_articles)} articles from file.")
