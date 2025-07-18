import gradio as gr
import yt_dlp
import os
import json
import base64
import io
import torch
import time
import subprocess
import whisper
import re
import cv2
import numpy as np
import librosa
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import tempfile
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import threading
import queue
import asyncio
import aiohttp
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from urllib.parse import urljoin, urlparse
import logging
from dataclasses import dataclass
from pathlib import Path
import hashlib
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# UNIFIED DATA MODELS & DATABASE
# ================================

@dataclass
class VideoMetadata:
    """Video metadata structure"""
    title: str
    url: str
    duration: float
    transcript: str = ""
    objects_detected: List[Dict] = None
    analysis_complete: bool = False
    created_at: str = ""

@dataclass
class RSSArticle:
    """RSS article structure"""
    title: str
    link: str
    description: str
    published: str
    category: str
    content: str = ""
    video_links: List[str] = None

class UnifiedDatabase:
    """Unified database for both video and RSS data"""
    
    def __init__(self, db_path="datanacci_platform.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with tables for both video and RSS data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Video metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                duration REAL,
                transcript TEXT,
                objects_detected TEXT,
                analysis_complete BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # RSS articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rss_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                link TEXT UNIQUE NOT NULL,
                description TEXT,
                published TEXT,
                category TEXT,
                content TEXT,
                video_links TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Cross-reference table for video-article relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_article_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                article_id INTEGER,
                similarity_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (id),
                FOREIGN KEY (article_id) REFERENCES rss_articles (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_video(self, video_metadata: VideoMetadata):
        """Save video metadata to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO videos 
            (title, url, duration, transcript, objects_detected, analysis_complete)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            video_metadata.title,
            video_metadata.url,
            video_metadata.duration,
            video_metadata.transcript,
            json.dumps(video_metadata.objects_detected) if video_metadata.objects_detected else None,
            video_metadata.analysis_complete
        ))
        
        conn.commit()
        conn.close()
    
    def save_articles(self, articles: List[RSSArticle]):
        """Save RSS articles to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            cursor.execute('''
                INSERT OR REPLACE INTO rss_articles 
                (title, link, description, published, category, content, video_links)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.title,
                article.link,
                article.description,
                article.published,
                article.category,
                article.content,
                json.dumps(article.video_links) if article.video_links else None
            ))
        
        conn.commit()
        conn.close()
    
    def search_content(self, query: str) -> Dict[str, List]:
        """Universal search across both videos and articles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search videos
        cursor.execute('''
            SELECT title, url, transcript FROM videos 
            WHERE title LIKE ? OR transcript LIKE ?
        ''', (f'%{query}%', f'%{query}%'))
        videos = cursor.fetchall()
        
        # Search articles
        cursor.execute('''
            SELECT title, link, description, content FROM rss_articles 
            WHERE title LIKE ? OR description LIKE ? OR content LIKE ?
        ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
        articles = cursor.fetchall()
        
        conn.close()
        
        return {
            'videos': [{'title': v[0], 'url': v[1], 'transcript': v[2]} for v in videos],
            'articles': [{'title': a[0], 'link': a[1], 'description': a[2], 'content': a[3]} for a in articles]
        }

# ================================
# MEDIA PROCESSING CORE
# ================================

class MediaProcessingCore:
    """Core media processing functionality from original app.py"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.whisper_model = None
        self.yolo_model = None
        self.temp_dir = tempfile.mkdtemp()
        
    def _get_optimal_device(self):
        """Determine optimal device for processing"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_whisper_model(self, model_size="base"):
        """Load Whisper model for transcription"""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(model_size, device=self.device)
        return self.whisper_model
    
    def load_yolo_model(self):
        """Load YOLO model for object detection"""
        if self.yolo_model is None:
            self.yolo_model = YOLO('yolo11n.pt')
        return self.yolo_model
    
    def download_video(self, url: str, quality: str = "best") -> Tuple[str, Dict]:
        """Download video using yt-dlp with GPU acceleration"""
        try:
            output_path = os.path.join(self.temp_dir, "%(title)s.%(ext)s")
            
            ydl_opts = {
                'format': quality,
                'outtmpl': output_path,
                'extractaudio': False,
                'audioformat': 'mp3',
                'embed_subs': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
                return filename, {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0)
                }
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            return None, {}
    
    def transcribe_video(self, video_path: str) -> str:
        """Transcribe video using Whisper"""
        try:
            model = self.load_whisper_model()
            result = model.transcribe(video_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""
    
    def detect_objects(self, video_path: str) -> List[Dict]:
        """Detect objects in video using YOLO"""
        try:
            model = self.load_yolo_model()
            cap = cv2.VideoCapture(video_path)
            detections = []
            
            frame_count = 0
            while cap.read()[0] and frame_count < 100:  # Sample frames
                ret, frame = cap.read()
                if ret and frame_count % 10 == 0:  # Every 10th frame
                    results = model(frame)
                    for result in results:
                        for box in result.boxes:
                            detection = {
                                'timestamp': frame_count / 30.0,  # Assuming 30fps
                                'class': model.names[int(box.cls)],
                                'confidence': float(box.conf),
                                'bbox': box.xyxy.tolist()[0]
                            }
                            detections.append(detection)
                frame_count += 1
            
            cap.release()
            return detections
        except Exception as e:
            logger.error(f"Object detection error: {str(e)}")
            return []

# ================================
# RSS INTELLIGENCE CORE
# ================================

class RSSIntelligenceCore:
    """RSS processing functionality from original app2.py"""
    
    def __init__(self):
        self.rss_feeds = {
            "AI & Machine Learning": [
                "https://feeds.feedburner.com/oreilly/radar",
                "https://machinelearningmastery.com/feed/",
                "https://towardsdatascience.com/feed",
                "https://blog.openai.com/rss/",
                "https://deepmind.com/blog/feed/basic/",
            ],
            "Technology": [
                "https://feeds.feedburner.com/TechCrunch",
                "https://www.theverge.com/rss/index.xml",
                "https://feeds.arstechnica.com/arstechnica/index",
                "https://www.wired.com/feed/rss",
                "https://feeds.feedburner.com/venturebeat/SZYF",
            ],
            "Science": [
                "https://feeds.nature.com/nature/rss/current",
                "https://www.science.org/rss/news_current.xml",
                "https://feeds.feedburner.com/sciencedaily",
                "https://www.newscientist.com/feed/home/",
            ],
            "Finance": [
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://feeds.reuters.com/reuters/businessNews",
                "https://feeds.feedburner.com/zerohedge/feed",
                "https://www.ft.com/rss/home",
            ],
            "Cybersecurity": [
                "https://feeds.feedburner.com/TheHackersNews",
                "https://krebsonsecurity.com/feed/",
                "https://www.darkreading.com/rss.xml",
                "https://feeds.feedburner.com/securityweek",
            ],
            "Space & Astronomy": [
                "https://www.nasa.gov/rss/dyn/breaking_news.rss",
                "https://feeds.space.com/space/news",
                "https://www.universetoday.com/feed/",
            ],
            "Health & Medicine": [
                "https://feeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
                "https://www.medicalnewstoday.com/rss",
                "https://feeds.reuters.com/reuters/health",
            ],
            "Climate & Environment": [
                "https://feeds.feedburner.com/climatecentral/djOO",
                "https://www.epa.gov/newsreleases/rss.xml",
                "https://feeds.reuters.com/reuters/environment",
            ],
            "Gaming": [
                "https://feeds.feedburner.com/ign/games-all",
                "https://www.gamespot.com/feeds/mashup/",
                "https://kotaku.com/rss",
            ],
            "Startups": [
                "https://feeds.feedburner.com/venturebeat/SZYF",
                "https://techcrunch.com/startups/feed/",
                "https://feeds.feedburner.com/entrepreneur/latest",
            ]
        }
        self.articles_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def fetch_feed(self, url: str, category: str) -> List[RSSArticle]:
        """Fetch and parse RSS feed"""
        try:
            feed = feedparser.parse(url)
            articles = []
            
            for entry in feed.entries[:10]:  # Limit to 10 articles per feed
                # Extract video links from content
                video_links = self._extract_video_links(entry.get('description', ''))
                
                article = RSSArticle(
                    title=entry.get('title', 'No Title'),
                    link=entry.get('link', ''),
                    description=entry.get('description', ''),
                    published=entry.get('published', ''),
                    category=category,
                    content=entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '',
                    video_links=video_links
                )
                articles.append(article)
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching feed {url}: {str(e)}")
            return []
    
    def _extract_video_links(self, content: str) -> List[str]:
        """Extract video URLs from article content"""
        video_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:www\.)?vimeo\.com/\d+',
            r'https?://(?:www\.)?dailymotion\.com/video/[\w-]+',
        ]
        
        video_links = []
        for pattern in video_patterns:
            matches = re.findall(pattern, content)
            video_links.extend(matches)
        
        return list(set(video_links))  # Remove duplicates
    
    def fetch_all_feeds(self) -> Dict[str, List[RSSArticle]]:
        """Fetch all RSS feeds concurrently"""
        all_articles = {}
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_category = {}
            
            for category, urls in self.rss_feeds.items():
                for url in urls:
                    future = executor.submit(self.fetch_feed, url, category)
                    future_to_category[future] = category
            
            for future in as_completed(future_to_category):
                category = future_to_category[future]
                try


continue

                    articles = future.result()
                    if category not in all_articles:
                        all_articles[category] = []
                    all_articles[category].extend(articles)
                except Exception as e:
                    logger.error(f"Error processing category {category}: {str(e)}")
        
        # Cache articles for chat functionality
        self.articles_cache = all_articles
        return all_articles
    
    def search_articles(self, query: str) -> List[RSSArticle]:
        """Search articles by keyword"""
        results = []
        query_lower = query.lower()
        
        for category, articles in self.articles_cache.items():
            for article in articles:
                if (query_lower in article.title.lower() or 
                    query_lower in article.description.lower() or
                    query_lower in article.content.lower()):
                    results.append(article)
        
        return results

# ================================
# CROSS-PLATFORM INTEGRATION
# ================================

class CrossPlatformIntegration:
    """Integration layer for cross-platform features"""
    
    def __init__(self, media_core: MediaProcessingCore, rss_core: RSSIntelligenceCore, database: UnifiedDatabase):
        self.media_core = media_core
        self.rss_core = rss_core
        self.database = database
    
    def find_related_content(self, query: str) -> Dict:
        """Find related content across both video and RSS data"""
        search_results = self.database.search_content(query)
        rss_results = self.rss_core.search_articles(query)
        
        return {
            'videos': search_results['videos'],
            'articles': search_results['articles'],
            'rss_live': [
                {
                    'title': article.title,
                    'link': article.link,
                    'description': article.description,
                    'category': article.category,
                    'video_links': article.video_links
                } for article in rss_results
            ]
        }
    
    def process_video_from_rss(self, video_url: str, article_title: str) -> Dict:
        """Process video found in RSS article"""
        try:
            # Download video
            video_path, metadata = self.media_core.download_video(video_url)
            if not video_path:
                return {'error': 'Failed to download video'}
            
            # Transcribe and analyze
            transcript = self.media_core.transcribe_video(video_path)
            objects = self.media_core.detect_objects(video_path)
            
            # Save to database
            video_metadata = VideoMetadata(
                title=metadata.get('title', article_title),
                url=video_url,
                duration=metadata.get('duration', 0),
                transcript=transcript,
                objects_detected=objects,
                analysis_complete=True
            )
            self.database.save_video(video_metadata)
            
            return {
                'success': True,
                'transcript': transcript,
                'objects_count': len(objects),
                'duration': metadata.get('duration', 0)
            }
        except Exception as e:
            logger.error(f"Error processing video from RSS: {str(e)}")
            return {'error': str(e)}
    
    def generate_intelligence_report(self) -> Dict:
        """Generate unified intelligence report"""
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()
        
        # Video statistics
        cursor.execute('SELECT COUNT(*), AVG(duration) FROM videos WHERE analysis_complete = TRUE')
        video_stats = cursor.fetchone()
        
        # Article statistics
        cursor.execute('SELECT category, COUNT(*) FROM rss_articles GROUP BY category')
        article_stats = cursor.fetchall()
        
        # Recent activity
        cursor.execute('''
            SELECT title, created_at FROM videos 
            WHERE analysis_complete = TRUE 
            ORDER BY created_at DESC LIMIT 5
        ''')
        recent_videos = cursor.fetchall()
        
        cursor.execute('''
            SELECT title, category, created_at FROM rss_articles 
            ORDER BY created_at DESC LIMIT 10
        ''')
        recent_articles = cursor.fetchall()
        
        conn.close()
        
        return {
            'video_stats': {
                'total_processed': video_stats[0] or 0,
                'avg_duration': video_stats[1] or 0
            },
            'article_stats': {category: count for category, count in article_stats},
            'recent_videos': [{'title': v[0], 'date': v[1]} for v in recent_videos],
            'recent_articles': [{'title': a[0], 'category': a[1], 'date': a[2]} for a in recent_articles]
        }

# ================================
# UNIFIED DATANACCI PLATFORM
# ================================

class DatanacciPlatform:
    """Main unified platform class"""
    
    def __init__(self):
        self.database = UnifiedDatabase()
        self.media_core = MediaProcessingCore()
        self.rss_core = RSSIntelligenceCore()
        self.integration = CrossPlatformIntegration(self.media_core, self.rss_core, self.database)
        self.chat_history = []
    
    # ================================
    # MEDIA PROCESSING INTERFACE
    # ================================
    
    def process_video_url(self, url: str, quality: str = "best") -> Tuple[str, str, str]:
        """Process video URL - download, transcribe, analyze"""
        try:
            # Download video
            video_path, metadata = self.media_core.download_video(url, quality)
            if not video_path:
                return "❌ Failed to download video", "", ""
            
            status = f"✅ Downloaded: {metadata.get('title', 'Unknown')}\n"
            status += f"Duration: {metadata.get('duration', 0):.1f} seconds\n"
            
            # Transcribe
            transcript = self.media_core.transcribe_video(video_path)
            status += f"✅ Transcription complete ({len(transcript)} characters)\n"
            
            # Object detection
            objects = self.media_core.detect_objects(video_path)
            status += f"✅ Object detection complete ({len(objects)} detections)\n"
            
            # Save to database
            video_metadata = VideoMetadata(
                title=metadata.get('title', 'Unknown'),
                url=url,
                duration=metadata.get('duration', 0),
                transcript=transcript,
                objects_detected=objects,
                analysis_complete=True
            )
            self.database.save_video(video_metadata)
            
            # Generate object summary
            object_summary = self._generate_object_summary(objects)
            
            return status, transcript, object_summary
            
        except Exception as e:
            return f"❌ Error: {str(e)}", "", ""
    
    def _generate_object_summary(self, objects: List[Dict]) -> str:
        """Generate summary of detected objects"""
        if not objects:
            return "No objects detected."
        
        # Count objects by class
        object_counts = {}
        for obj in objects:
            class_name = obj['class']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Sort by frequency
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        
        summary = "🔍 **Objects Detected:**\n\n"
        for obj_class, count in sorted_objects[:10]:  # Top 10
            summary += f"• {obj_class}: {count} detections\n"
        
        return summary
    
    # ================================
    # RSS INTELLIGENCE INTERFACE
    # ================================
    
    def refresh_rss_feeds(self) -> str:
        """Refresh all RSS feeds"""
        try:
            articles_by_category = self.rss_core.fetch_all_feeds()
            
            # Save to database
            all_articles = []
            for category, articles in articles_by_category.items():
                all_articles.extend(articles)
            
            self.database.save_articles(all_articles)
            
            # Generate summary
            total_articles = len(all_articles)
            categories = len(articles_by_category)
            
            summary = f"✅ **RSS Refresh Complete**\n\n"
            summary += f"📊 **Summary:**\n"
            summary += f"• Total Articles: {total_articles}\n"
            summary += f"• Categories: {categories}\n\n"
            summary += f"📰 **By Category:**\n"
            
            for category, articles in articles_by_category.items():
                summary += f"• {category}: {len(articles)} articles\n"
            
            return summary
            
        except Exception as e:
            return f"❌ Error refreshing feeds: {str(e)}"
    
    def get_category_articles(self, category: str) -> str:
        """Get articles for specific category"""
        if category not in self.rss_core.articles_cache:
            return f"No articles found for category: {category}"
        
        articles = self.rss_core.articles_cache[category][:10]  # Limit to 10
        
        output = f"📰 **{category} - Latest Articles**\n\n"
        
        for i, article in enumerate(articles, 1):
            output += f"**{i}. {article.title}**\n"
            output += f"🔗 [Read More]({article.link})\n"
            output += f"📝 {article.description[:200]}...\n"
            
            if article.video_links:
                output += f"🎥 Video Links Found: {len(article.video_links)}\n"
            
            output += f"📅 {article.published}\n\n"
            output += "---\n\n"
        
        return output
    
    # ================================
    # UNIVERSAL CHAT INTERFACE
    # ================================
    
    def universal_chat(self, message: str, history: List) -> Tuple[str, List]:
        """Universal chat across both video and RSS content"""
        try:
            # Search across all content
            results = self.integration.find_related_content(message)
            
            response = f"🔍 **Search Results for: '{message}'**\n\n"
            
            # Video results
            if results['videos']:
                response += "🎬 **Video Content:**\n"
                for video in results['videos'][:3]:  # Top 3
                    response += f"• **{video['title']}**\n"
                    response += f"  📝 Transcript excerpt: {video['transcript'][:150]}...\n\n"
            
            # Article results from database
            if results['articles']:
                response += "📰 **Archived Articles:**\n"
                for article in results['articles'][:3]:  # Top 3
                    response += f"• **{article['title']}**\n"
                    response += f"  📝 {article['description'][:150]}...\n\n"
            
            # Live RSS results
            if results['rss_live']:
                response += "🔴 **Live RSS Results:**\n"
                for article in results['rss_live'][:3]:  # Top 3
                    response += f"• **{article['title']}** ({article['category']})\n"
                    response += f"  📝 {article['description'][:150]}...\n"
                    if article['video_links']:
                        response += f"  🎥 Contains {len(article['video_links'])} video link(s)\n"
                    response += "\n"
            
            if not any([results['videos'], results['articles'], results['rss_live']]):
                response = f"🤔 No content found matching '{message}'. Try different keywords or refresh RSS feeds."
            
            # Update history
            history.append([message, response])
            
            return response, history
            
        except Exception as e:
            error_response = f"❌ Error in chat: {str(e)}"
            history.append([message, error_response])
            return error_response, history
    
    # ================================
    # INTELLIGENCE HUB
    # ================================
    
    def generate_intelligence_dashboard(self) -> str:
        """Generate unified intelligence dashboard"""
        try:
            report = self.integration.generate_intelligence_report()
            
            dashboard = "📊 **Datanacci Intelligence Hub**\n\n"
            
            # Video Statistics
            dashboard += "🎬 **Video Processing Stats:**\n"
            dashboard += f"• Total Videos Processed: {report['video_stats']['total_processed']}\n"
            dashboard += f"• Average Duration: {report['video_stats']['avg_duration']:.1f} seconds\n\n"
            
            # Article Statistics
            dashboard += "📰 **RSS Intelligence Stats:**\n"
            total_articles = sum(report['article_stats'].values())
            dashboard += f"• Total Articles Tracked: {total_articles}\n"
            dashboard += f"• Active Categories: {len(report['article_stats'])}\n\n"
            
            dashboard += "📊 **Articles by Category:**\n"
            for category, count in sorted(report['article_stats'].items(), key=lambda x: x[1], reverse=True):
                dashboard += f"• {category}: {count}\n"
            dashboard += "\n"
            
            # Recent Activity
            dashboard += "🕒 **Recent Video Processing:**\n"
            if report['recent_videos']:
                for video in report['recent_videos']:
                    dashboard += f"• {video['title']} ({video['date']})\n"
            else:
                dashboard += "• No recent video processing\n"
            dashboard += "\n"
            
            dashboard += "🕒 **Recent Articles:**\n"
            if report['recent_articles']:
                for article in report['recent_articles'][:5]:
                    dashboard += f"• {article['title']} ({article['category']})\n"
            else:
                dashboard += "• No recent articles\n"
            
            return dashboard
            
        except Exception as e:
            return f"❌ Error generating dashboard: {str(e)}"
    
    # ================================
    # VIDEO FROM RSS PROCESSING
    # ================================
    
    def process_rss_video(self, article_title: str, video_url: str) -> str:
        """Process video found in RSS article"""
        try:
            result = self.integration.process_video_from_rss(video_url, article_title)
            
            if 'error' in result:
                return f"❌ Error processing video: {result['error']}"
            
            response = f"✅ **Video Processed from RSS Article**\n\n"
            response += f"📰 **Source Article:** {article_title}\n"
            response += f"🎬 **Video URL:** {video_url}\n"
            response += f"⏱️ **Duration:** {result['duration']:.1f} seconds\n"
            response += f"🔍 **Objects Detected:** {result['objects_count']}\n\n"
            response += f"📝 **Transcript Preview:**\n{result['transcript'][:300]}...\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"

# ================================
# GRADIO INTERFACE
# ================================

def create_gradio_interface():
    """Create the unified Gradio interface"""
    
    # Initialize platform
    platform = DatanacciPlatform()
    
    # Custom CSS for enhanced UI
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .status-box {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background: #f8f9fa;
        border-left: 4px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Datanacci Platform", theme=gr.themes.Soft()) as app:
        
        # Main Header
        gr.HTML("""
        <div class="main-header">
            <h1>🚀 Datanacci Unified Platform</h1>
            <p>Advanced Media Processing & RSS Intelligence Hub</p>
        </div>
        """)
        
        with gr.Tabs():
            
            # ================================
            # STREAMDL TAB - Video Processing
            # ================================
            with gr.Tab("🎬 StreamDL", id="streamdl"):
                gr.Markdown("## 🎬 Advanced Video Processing")
                gr.Markdown("Download, transcribe, and analyze videos with AI-powered insights.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        video_url = gr.Textbox(
                            label="🔗 Video URL",
                            placeholder="Enter YouTube, Vimeo, or other video URL...",
                            lines=1
                        )
                        quality_choice = gr.Dropdown(
                            choices=["best", "worst", "720p", "480p", "360p"],
                            value="best",
                            label="📺 Quality"
                        )
                        process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Features")
                        gr.Markdown("""
                        - 🎥 **Video Download** with GPU acceleration
                        - 🎤 **AI Transcription** using Whisper
                        - 🔍 **Object Detection** with YOLO11
                        - 💾 **Database Storage** for future reference
                        - 🔗 **Cross-Platform Integration**
                        """)
                
                with gr.Row():
                    with gr.Column():
                        processing_status = gr.Textbox(
                            label="📊 Processing Status",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column():
                        transcript_output = gr.Textbox(
                            label="📝 Transcript",
                            lines=8,
                            interactive=False
                        )
                
                object_analysis = gr.Textbox(
                    label="🔍 Object Detection Analysis",
                    lines=6,
                    interactive=False
                )
                
                process_btn.click(
                    fn=platform.process_video_url,
                    inputs=[video_url, quality_choice],
                    outputs=[processing_status, transcript_output, object_analysis]
                )
            
            # ================================
            # RSS INTELLIGENCE TAB
            # ================================
            with gr.Tab("📰 RSS Intelligence", id="rss"):
                gr.Markdown("## 📰 RSS Intelligence Hub")
                gr.Markdown("Monitor and analyze RSS feeds across multiple categories with AI insights.")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh All Feeds", variant="primary", size="lg")
                    category_dropdown = gr.Dropdown(
                        choices=list(platform.rss_core.rss_feeds.keys()),
                        label="📂 Select Category",
                        value="AI & Machine Learning"
                    )
                    view_category_btn = gr.Button("👀 View Category", variant="secondary")
                
                with gr.Row():
                    with gr.Column():
                        rss_status = gr.Textbox(
                            label="📊 RSS Status",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.Column():
                        category_articles = gr.Textbox(
                            label="📰 Category Articles",
                            lines=15,
                            interactive=False
                        )
                
                with gr.Row():
                    gr.Markdown("### 🎯 RSS Categories")
                    with gr.Column():
                        gr.Markdown("""
                        - 🤖 **AI & Machine Learning**
                        - 💻 **Technology**
                        - 🔬 **Science**
                        - 💰 **Finance**
                        - 🔒 **Cybersecurity**
                        """)
                    with gr.Column():
                        gr.Markdown("""
                        - 🚀 **Space & Astronomy**
                        - 🏥 **Health & Medicine**
                        - 🌍 **Climate & Environment**
                        - 🎮 **Gaming**
                        - 🏢 **Startups**
                        """)
                
                refresh_btn.click(
                    fn=platform.refresh_rss_feeds,
                    outputs=[rss_status]
                )
                
                view_category_btn.click(
                    fn=platform.get_category_articles,
                    inputs=[category_dropdown],
                    outputs=[category_articles]
                )
            
            # ================================
            # UNIVERSAL CHAT TAB
            # ================================
            with gr.Tab("💬 Universal Chat", id="chat"):
                gr.Markdown("## 💬 Universal Intelligence Chat")
                gr.Markdown("Chat with both your video transcripts and RSS articles using AI-powered search.")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="🤖 Datanacci Assistant",
                            height=500,
                            show_label=True
                        )
                        
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="💭 Ask anything about your videos or RSS content...",
                                placeholder="e.g., 'Find videos about AI' or 'Show me tech news'",
                                lines=2,
                                scale=4
                            )
                            chat_btn = gr.Button("💬 Send", variant="primary", scale=1)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Chat Features")
                        gr.Markdown("""
                        - 🔍 **Universal Search** across all content
                        - 🎬 **Video Transcript** queries
                        - 📰 **RSS Article** search
                        - 🔗 **Cross-Reference** related content
                        - 🤖 **AI-Powered** responses
                        """)
                        
                        gr.Markdown("### 💡 Example Queries")
                        gr.Markdown("""
                        - *"Find videos about machine learning"*
                        - *"Show me recent AI news"*
                        - *"What tech articles mention GPT?"*
                        - *"Videos with object detection"*
                        - *"Climate change articles"*
                        """)
                
                def chat_wrapper(message, history):
                    return platform.universal_chat(message, history)
                
                chat_btn.click(
                    fn=chat_wrapper,
                    inputs=[chat_input, chatbot],
                    outputs=[chat_input, chatbot]
                )
                
                chat_input.submit(
                    fn=chat_wrapper,
                    inputs=[chat_input, chatbot],
                    outputs=[chat_input, chatbot]
                )
            
            # ================================
            # INTELLIGENCE HUB TAB
            # ================================
            with gr.Tab("📊 Intelligence Hub", id="intelligence"):
                gr.Markdown("## 📊 Unified Intelligence Dashboard")
                gr.Markdown("Comprehensive analytics and insights across all platform activities.")
                
                with gr.Row():
                    generate_report_btn = gr.Button("📊 Generate Intelligence Report", variant="primary", size="lg")
                    export_data_btn = gr.Button("💾 Export Data", variant="secondary")
                
                intelligence_dashboard = gr.Textbox(
                    label="📊 Intelligence Dashboard",
                    lines=20,
                    interactive=False
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎯 Dashboard Features")
                        gr.Markdown("""
                        - 📈 **Processing Statistics**
                        - 🎬 **Video Analytics**
                        - 📰 **RSS Insights**
                        - 🕒 **Recent Activity**
                        - 🔗 **Cross-Platform Correlations**
                        """)
                    
                    with gr.Column():
                        gr.Markdown("### 📊 Available Metrics")
                        gr.Markdown("""
                        - **Total Videos Processed**
                        - **Average Video Duration**
                        - **Articles by Category**
                        - **Object Detection Stats**
                        - **Content Trends**
                        """)
                
                generate_report_btn.click(
                    fn=platform.generate_intelligence_dashboard,
                    outputs=[intelligence_dashboard]
                )
            
            # ================================
            # RSS-VIDEO INTEGRATION TAB
            # ================================
            with gr.Tab("🔗 RSS-Video Integration", id="integration"):
                gr.Markdown("## 🔗 RSS-Video Integration Hub")
                gr.Markdown("Process videos found in RSS articles with full AI analysis.")
                
                with gr.Row():
                    with gr.Column():
                        article_title_input = gr.Textbox(
                            label="📰 Article Title",
                            placeholder="Enter the RSS article title...",
                            lines=2
                        )
                        video_url_input = gr.Textbox(
                            label="🎥 Video URL from Article",
                            placeholder="Enter video URL found in RSS article...",
                            lines=1
                        )
                        process_rss_video_btn = gr.Button("🚀 Process RSS Video", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("### 🎯 Integration Features")
                        gr.Markdown("""
                        - 🔍 **Auto-detect** videos in RSS articles
                        - 🎬 **Full Video Processing** pipeline
                        - 📝 **Transcript Generation**
                        - 🔍 **Object Detection**
                        - 🔗 **Cross-Reference** with articles
                        """)
                
                rss_video_output = gr.Textbox(
                    label="📊 Processing Results",
                    lines=15,
                    interactive=False
                )
                
                with gr.Row():
                    gr.Markdown("### 💡 How It Works")
                    gr.Markdown("""
                    1. **RSS Monitoring**: Platform automatically detects video links in RSS articles
                    2. **Video Processing**: Downloads and analyzes videos using full AI pipeline
                    3. **Cross-Reference**: Links video analysis with original RSS article
                    4. **Unified Search**: Makes content searchable across both systems
                    5. **Intelligence**: Provides insights connecting news and video content
                    """)
                
                process_rss_video_btn.click(
                    fn=platform.process_rss_video,
                    inputs=[article_title_input, video_url_input],
                    outputs=[rss_video_output]
                )
            
            # ================================
            # SETTINGS & INFO TAB
            # ================================
            with gr.Tab("⚙️ Settings", id="settings"):
                gr.Markdown("## ⚙️ Platform Settings & Information")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🖥️ System Information")
                        device_info = f"**Processing Device:** {platform.media_core.device.upper()}"
                        gr.Markdown(device_info)
                        
                        gr.Markdown("### 📊 Database Status")
                        gr.Markdown("**Database:** SQLite (datanacci_platform.db)")
                        gr.Markdown("**Tables:** videos, rss_articles, video_article_relations")
                        
                        gr.Markdown("### 🔧 Configuration")
                        gr.Markdown("""
                        - **Whisper Model:** Base (adjustable)
                        - **YOLO Model:** YOLOv11n
                        - **RSS Refresh:** Manual/Automatic
                        - **GPU Acceleration:** Auto-detected
                        """)
                    
                    with gr.Column():
                        gr.Markdown("### 📚 Platform Capabilities")
                        gr.Markdown("""
                        **🎬 Video Processing:**
                        - YouTube, Vimeo, and other platforms
                        - GPU-accelerated downloading
                        - AI transcription with Whisper
                        - Real-time object detection
                        - Smart video analysis
                        
                        **📰 RSS Intelligence:**
                        - 10+ categories, 40+ feeds
                        - Parallel processing
                        - Article caching and search
                        - Video link extraction
                        - Real-time updates
                        
                        **🔗 Integration:**
                        - Cross-platform search
                        - Unified chat interface
                        - Intelligence dashboard
                        - Automated workflows
                        - Data correlation
                        """)
                
                with gr.Row():
                    gr.Markdown("### 🚀 Version Information")
                    gr.Markdown("""
                    **Datanacci Platform v2.0**
                    - Unified Media & RSS Intelligence
                    - Built with Gradio, PyTorch, Whisper, YOLO
                    - Cross-platform AI integration
                    - Real-time processing capabilities
                    """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3>🚀 Datanacci Platform - Unified Intelligence Hub</h3>
            <p>Advanced Media Processing • RSS Intelligence • AI-Powered Insights</p>
        </div>
        """)
    
    return app

# ================================
# MAIN APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    # Create and launch the application
    app = create_gradio_interface()
    
    # Launch with configuration
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        quiet=False,
        inbrowser=True
    )
