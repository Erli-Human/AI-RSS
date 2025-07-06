import csv
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import io
from io import StringIO, BytesIO
import base64
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import gradio as gr
import schedule
import time
import smtplib
from email.mime.text import MIMEText
import feedparser
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import ollama # Import the ollama library
import os

# --- Data Structures (assuming these are defined elsewhere in your app.py) ---
@dataclass
class DailySummary:
    date: str
    total_sales: float
    total_quantity: int
    top_product: str
    region_sales: Dict[str, float] = field(default_factory=dict)
    average_transaction_value: float = 0.0

@dataclass
class SalesRecord:
    transaction_id: str
    date: datetime
    product_id: str
    product_name: str
    quantity: int
    price: float
    region: str

# --- Helper Functions (assuming these are defined elsewhere in your app.py) ---
# Placeholder for your existing functions
def parse_sales_data(file_content: str) -> List[SalesRecord]:
    # Your existing implementation
    records = []
    f = StringIO(file_content)
    reader = csv.reader(f)
    header = next(reader) # Skip header
    for row in reader:
        try:
            records.append(SalesRecord(
                transaction_id=row[0],
                date=datetime.strptime(row[1], '%Y-%m-%d'),
                product_id=row[2],
                product_name=row[3],
                quantity=int(row[4]),
                price=float(row[5]),
                region=row[6]
            ))
        except (ValueError, IndexError) as e:
            print(f"Skipping malformed row: {row} - Error: {e}")
            continue
    return records

def analyze_data(records: List[SalesRecord]) -> Dict[str, Any]:
    # Your existing implementation for sales analysis
    df = pd.DataFrame([s.__dict__ for s in records])
    if df.empty:
        return {
            "total_sales": 0,
            "total_transactions": 0,
            "average_transaction_value": 0,
            "top_products": [],
            "sales_by_region": {},
            "daily_summaries": []
        }

    df['total_price'] = df['quantity'] * df['price']
    df['date_only'] = df['date'].dt.date

    total_sales = df['total_price'].sum()
    total_transactions = df['transaction_id'].nunique()
    average_transaction_value = total_sales / total_transactions if total_transactions > 0 else 0

    top_products_df = df.groupby('product_name')['total_price'].sum().nlargest(5)
    top_products = top_products_df.to_dict()

    sales_by_region = df.groupby('region')['total_price'].sum().to_dict()

    daily_summaries_list = []
    for date, group in df.groupby('date_only'):
        daily_total_sales = group['total_price'].sum()
        daily_total_quantity = group['quantity'].sum()
        daily_top_product_series = group.groupby('product_name')['quantity'].sum().nlargest(1)
        daily_top_product = daily_top_product_series.index[0] if not daily_top_product_series.empty else "N/A"
        daily_region_sales = group.groupby('region')['total_price'].sum().to_dict()
        daily_avg_transaction = daily_total_sales / group['transaction_id'].nunique() if group['transaction_id'].nunique() > 0 else 0

        daily_summaries_list.append(DailySummary(
            date=str(date),
            total_sales=daily_total_sales,
            total_quantity=daily_total_quantity,
            top_product=daily_top_product,
            region_sales=daily_region_sales,
            average_transaction_value=daily_avg_transaction
        ))

    return {
        "total_sales": total_sales,
        "total_transactions": total_transactions,
        "average_transaction_value": average_transaction_value,
        "top_products": top_products,
        "sales_by_region": sales_by_region,
        "daily_summaries": daily_summaries_list
    }

def plot_data(records: List[SalesRecord]) -> str:
    # Your existing implementation for plotting
    df = pd.DataFrame([s.__dict__ for s in records])
    if df.empty:
        return "No data to plot."

    df['total_price'] = df['quantity'] * df['price']
    df['date_only'] = df['date'].dt.date

    # Plotting daily sales
    daily_sales = df.groupby('date_only')['total_price'].sum()
    plt.figure(figsize=(10, 6))
    daily_sales.plot(kind='line', marker='o')
    plt.title('Daily Total Sales')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_str}" />'

def send_email(to_address: str, subject: str, body: str, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str):
    # Your existing implementation for sending emails
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = to_address

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"

def fetch_rss_feed(url: str) -> List[Dict[str, str]]:
    # Your existing implementation for RSS feed fetching
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            articles.append({
                'title': entry.title,
                'link': entry.link,
                'published': entry.published if hasattr(entry, 'published') else 'N/A',
                'summary': entry.summary if hasattr(entry, 'summary') else 'No summary available.'
            })
        return articles
    except Exception as e:
        print(f"Error fetching RSS feed from {url}: {e}")
        return []

# --- Ollama Integration ---

# This is the function with the primary fix
def get_ollama_models() -> List[str]:
    """
    Fetches a list of available Ollama models.
    Includes robust error handling and debugging print statements.
    """
    try:
        print(f"[{datetime.now()}] Attempting to list Ollama models...")
        models_info = ollama.list()
        print(f"[{datetime.now()}] Raw Ollama list response: {models_info}") # CRUCIAL DEBUG OUTPUT

        # Check if models_info is a dictionary
        if not isinstance(models_info, dict):
            print(f"[{datetime.now()}] Error: Ollama list returned unexpected type: {type(models_info)}. Expected a dictionary.")
            return ["Error: Ollama response malformed (not a dict)."]

        # Check if 'models' key exists
        if 'models' not in models_info:
            print(f"[{datetime.now()}] Error: Ollama list response missing 'models' key. Response: {models_info}")
            return ["Error: Ollama response malformed (missing 'models' key)."]

        model_list = models_info['models']

        # Check if 'models' value is a list
        if not isinstance(model_list, list):
            print(f"[{datetime.now()}] Error: 'models' value is not a list. Type: {type(model_list)}. Value: {model_list}")
            return ["Error: Ollama response malformed ('models' value not a list)."]

        if not model_list:
            print(f"[{datetime.now()}] No Ollama models found in the response. Have you pulled any yet? (e.g., 'ollama pull llama2')")
            return ["No models found. Pull models like 'ollama pull gemma3n:e4b'."]

        models = []
        for i, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                print(f"[{datetime.now()}] Warning: Model entry at index {i} is not a dictionary. Skipping. Entry: {model_entry}")
                continue
            if 'name' in model_entry:
                models.append(model_entry['name'])
            else:
                print(f"[{datetime.now()}] Error: Model entry at index {i} missing 'name' key. Entry: {model_entry}")
                # You could choose to append a placeholder or skip, depending on desired behavior
                # For now, let's skip it if 'name' is truly missing to avoid bad entries in dropdown
                continue
        
        if not models:
            print(f"[{datetime.now()}] No valid model names extracted after processing Ollama list response.")
            return ["No valid model names extracted."]

        return sorted(list(set(models))) # Use set for uniqueness, then sort

    except ConnectionRefusedError:
        print(f"[{datetime.now()}] Error: Connection refused. Is Ollama server running on 127.0.0.1:11434 and accessible?")
        return ["Error: Ollama Server Not Running or Connection Refused."]
    except Exception as e:
        print(f"[{datetime.now()}] An unexpected error occurred while fetching Ollama models: {e}")
        # Return a more informative error for the UI
        return [f"Error: Could not fetch models. Details: {e}. Check console for more info."]

# Initial fetch of models when the app starts
# This will try to populate the dropdown.
ollama_available_models = get_ollama_models()
# Set a default model. Use the first available, or a fallback message.
default_ollama_model = ollama_available_models[0] if ollama_available_models and not ollama_available_models[0].startswith("Error") else "No models available / Error fetching"
# You might want to filter out error messages from default_ollama_model if it's selected as actual model
# For now, if default_ollama_model contains an error message, it will be the selected one
# and the user will have to select a valid model.

print(f"Default Ollama Model set to: {default_ollama_model}")
print(f"Available Ollama Models: {ollama_available_models}")

def generate_insights_ollama(sales_data_string: str, query: str, ollama_model: str) -> str:
    if "Error" in ollama_model:
        return f"Cannot generate insights: {ollama_model}. Please select a valid Ollama model."
    
    records = parse_sales_data(sales_data_string)
    analysis_results = analyze_data(records)
    
    # Prepare a prompt for Ollama
    prompt_data = {
        "analysis_results": analysis_results,
        "user_query": query,
        "schema_of_analysis_results": """
        The 'analysis_results' dictionary contains:
        - total_sales: float, sum of all sales.
        - total_transactions: int, count of unique transactions.
        - average_transaction_value: float, total_sales / total_transactions.
        - top_products: dict, product_name -> total_price for top 5 products.
        - sales_by_region: dict, region -> total_price for each region.
        - daily_summaries: list of DailySummary objects, each containing:
            - date: str
            - total_sales: float
            - total_quantity: int
            - top_product: str
            - region_sales: dict
            - average_transaction_value: float
        """
    }
    
    system_prompt = (
        "You are an expert sales data analyst. Your task is to provide insightful responses "
        "based on the provided sales analysis results. Use the data to answer the user's query comprehensively. "
        "If the data doesn't directly answer the query, indicate that. Keep responses concise and actionable."
    )
    
    user_prompt = (
        f"Here are the sales analysis results:\n{json.dumps(prompt_data['analysis_results'], indent=2)}\n\n"
        f"This is the schema for the analysis results: {prompt_data['schema_of_analysis_results']}\n\n"
        f"Based on this data, answer the following question: \"{prompt_data['user_query']}\""
    )

    try:
        response = ollama.chat(model=ollama_model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama model '{ollama_model}': {e}. Ensure the model is pulled and running."

# --- Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown("# Sales Data Analysis Dashboard")

    with gr.Tab("Data Upload & Analysis"):
        with gr.Row():
            file_upload = gr.File(label="Upload CSV Sales Data", type="file")
            data_preview = gr.DataFrame(label="Data Preview")
        
        parse_button = gr.Button("Parse & Analyze Data")
        
        with gr.Accordion("Analysis Results", open=False):
            total_sales_output = gr.Number(label="Total Sales")
            total_transactions_output = gr.Number(label="Total Transactions")
            avg_transaction_output = gr.Number(label="Average Transaction Value")
            top_products_output = gr.JSON(label="Top 5 Products by Sales")
            sales_by_region_output = gr.JSON(label="Sales by Region")
            daily_summaries_output = gr.JSON(label="Daily Summaries")
        
        with gr.Accordion("Visualizations", open=False):
            plot_output = gr.HTML(label="Sales Trend Plot")

    with gr.Tab("Ollama Insights"):
        with gr.Row():
            # Use the dynamically fetched models for the dropdown
            ollama_model_dropdown = gr.Dropdown(
                label="Select Ollama Model",
                choices=ollama_available_models,
                value=default_ollama_model, # Set default value
                interactive=True
            )
            ollama_query = gr.Textbox(label="Ask a question about the sales data:", placeholder="e.g., What were the daily sales trends?")
        ollama_generate_button = gr.Button("Generate Ollama Insights")
        ollama_output = gr.Markdown(label="Ollama Insights")
    
    # Assuming 'file_content_state' is a way to pass data between tabs without re-uploading
    # This needs to be defined if not already. Let's assume a State component.
    file_content_state = gr.State(None)
    parsed_records_state = gr.State([]) # To store parsed SalesRecord objects
    
    def handle_file_upload(file):
        if file is None:
            return None, None
        file_content = file.read().decode('utf-8')
        df_preview = pd.read_csv(StringIO(file_content))
        return file_content, df_preview.head(5) # Pass raw content and a preview

    def run_analysis(file_content: str):
        if file_content is None:
            return 0, 0, 0, {}, {}, [], "No file uploaded.", []

        records = parse_sales_data(file_content)
        analysis_results = analyze_data(records)
        plot_html = plot_data(records)

        # Convert DailySummary objects to dictionaries for JSON output
        daily_summaries_dict = [ds.__dict__ for ds in analysis_results["daily_summaries"]]
        
        return (
            analysis_results["total_sales"],
            analysis_results["total_transactions"],
            analysis_results["average_transaction_value"],
            analysis_results["top_products"],
            analysis_results["sales_by_region"],
            daily_summaries_dict,
            plot_html,
            records # Pass records to state for Ollama insights
        )


    file_upload.upload(handle_file_upload, inputs=file_upload, outputs=[file_content_state, data_preview])
    
    parse_button.click(
        run_analysis,
        inputs=[file_content_state],
        outputs=[
            total_sales_output,
            total_transactions_output,
            avg_transaction_output,
            top_products_output,
            sales_by_region_output,
            daily_summaries_output,
            plot_output,
            parsed_records_state # Output records to state for later use
        ]
    )

    ollama_generate_button.click(
        generate_insights_ollama,
        inputs=[file_content_state, ollama_query, ollama_model_dropdown], # Use file_content_state
        outputs=ollama_output
    )

    # --- Scheduler Tab (if you have one) ---
    with gr.Tab("Scheduled Tasks"):
        with gr.Row():
            email_address_input = gr.Textbox(label="Recipient Email", placeholder="your_email@example.com")
            email_subject_input = gr.Textbox(label="Email Subject", placeholder="Daily Sales Report")
            smtp_server_input = gr.Textbox(label="SMTP Server", placeholder="smtp.example.com")
            smtp_port_input = gr.Number(label="SMTP Port", value=587)
            smtp_user_input = gr.Textbox(label="SMTP Username", placeholder="your_smtp_username")
            smtp_password_input = gr.Textbox(label="SMTP Password", type="password", placeholder="your_smtp_password")
        
        schedule_time_input = gr.Textbox(label="Schedule Time (HH:MM)", placeholder="e.g., 09:00 for 9 AM")
        schedule_button = gr.Button("Schedule Email Report")
        scheduler_status = gr.Textbox(label="Scheduler Status", interactive=False)

        # RSS Feed Fetcher
        rss_url_input = gr.Textbox(label="RSS Feed URL", placeholder="e.g., https://www.nytimes.com/services/xml/rss/nyt/HomePage.xml")
        fetch_rss_button = gr.Button("Fetch RSS Feed")
        rss_output = gr.JSON(label="RSS Feed Articles")
        
        # Scheduler thread management
        scheduler_thread = None
        scheduler_stop_event = threading.Event()

        def start_scheduler_thread():
            global scheduler_thread
            if scheduler_thread is None or not scheduler_thread.is_alive():
                scheduler_stop_event.clear()
                scheduler_thread = threading.Thread(target=run_scheduler, args=(scheduler_stop_event,))
                scheduler_thread.start()
                return "Scheduler started."
            return "Scheduler is already running."

        def stop_scheduler_thread():
            global scheduler_thread
            if scheduler_thread and scheduler_thread.is_alive():
                scheduler_stop_event.set()
                scheduler_thread.join(timeout=5) # Give it a moment to stop
                return "Scheduler stopped."
            return "Scheduler is not running."

        def run_scheduler(stop_event: threading.Event):
            while not stop_event.is_set():
                schedule.run_pending()
                time.sleep(1) # Check every second

        def schedule_email_report_action(
            email_address: str, subject: str, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str,
            schedule_time: str, records_from_state: List[SalesRecord] # Get records from state
        ):
            if not records_from_state:
                return "No sales data available to schedule report. Please upload and analyze data first."

            analysis_results = analyze_data(records_from_state)
            email_body = f"""
            Sales Report for {datetime.now().strftime('%Y-%m-%d')}

            Total Sales: ${analysis_results['total_sales']:.2f}
            Total Transactions: {analysis_results['total_transactions']}
            Average Transaction Value: ${analysis_results['average_transaction_value']:.2f}

            Top 5 Products:
            {json.dumps(analysis_results['top_products'], indent=2)}

            Sales by Region:
            {json.dumps(analysis_results['sales_by_region'], indent=2)}

            Daily Summaries:
            {json.dumps([ds.__dict__ for ds in analysis_results['daily_summaries']], indent=2)}
            """
            
            def job():
                print(f"Attempting to send scheduled email at {datetime.now().strftime('%H:%M:%S')}")
                send_result = send_email(email_address, subject, email_body, smtp_server, smtp_port, smtp_user, smtp_password)
                print(f"Scheduled email send result: {send_result}")
                return send_result

            # Clear existing jobs to prevent duplicates if scheduled multiple times
            schedule.clear('daily_report_job') 
            schedule.every().day.at(schedule_time).do(job).tag('daily_report_job')
            start_scheduler_thread() # Ensure scheduler thread is running
            return f"Email report scheduled for {schedule_time} daily."

        schedule_button.click(
            schedule_email_report_action,
            inputs=[
                email_address_input, email_subject_input, smtp_server_input, smtp_port_input,
                smtp_user_input, smtp_password_input, schedule_time_input, parsed_records_state # Use parsed_records_state
            ],
            outputs=scheduler_status
        )
        
        fetch_rss_button.click(
            fetch_rss_feed,
            inputs=rss_url_input,
            outputs=rss_output
        )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=False)
