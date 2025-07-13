import feedparser
from bs4 import BeautifulSoup
import requests
import json
from datetime import datetime
from dateutil import parser

def scrape_content(url):
    """Scrapes the content from a given URL using Beautiful Soup."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Example selectors - adjust these based on the website's HTML structure
        articles = soup.find_all('article') # Adjust this selector!

        news_items = []
        for article in articles:
            try:
                title_element = article.find('h2') or article.find('h1')  # Try h2 first, then h1
                if not title_element:
                    continue # Skip if no title is found

                title = title_element.text.strip()
                link_element = article.find('a')
                if link_element:
                    link = link_element['href']
                    # Handle relative links
                    if not link.startswith("http"):
                        link = url + link  # Assuming the base URL is correct

                    summary_element = article.find('p') # Adjust this selector!
                    summary = summary_element.text.strip() if summary_element else "No summary available."

                    news_items.append({
                        'title': title,
                        'link': link,
                        'summary': summary,
                        'published': datetime.now().isoformat()  # Use current time as a placeholder
                    })
            except Exception as e:
                print(f"Error processing article on {url}: {e}")

        return news_items

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return []


def get_news_from_urls(urls):
    """Fetches news from multiple URLs and combines the results."""
    all_news = []
    for url in urls:
        news = scrape_content(url)
        all_news.extend(news)
    return all_news

def save_news(news, filename="news_data.json"):
    """Saves news articles to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(news, f, indent=4)

def load_news(filename="news_data.json"):
    """Loads news articles from a JSON file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
