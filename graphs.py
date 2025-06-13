from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Annotated, Dict, List, Union, Optional, Any
from datetime import datetime, timedelta
import dateparser
import os
import json
import re
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from functools import reduce
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.figure import Figure
import io
import base64
from prompt import description

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

with open("1.json", "r") as f:
    FIELD_METADATA = json.load(f)  # Expected: {"Pump_1_EM_1_VRY": "Voltage between R and Y...", ...}

# Convert to list of dicts for prompt clarity
FIELD_LIST = [{"key": k, "description": v or f"Telemetry field: {k}"} for k, v in FIELD_METADATA.items()]


# --- Graph Generation Functions ---
def detect_graph_request(user_query: str) -> dict:
    """
    Detect if user is asking for graphs/charts and what type
    """
    query_lower = user_query.lower()
    
    graph_keywords = [
        'graph', 'chart', 'plot', 'visualize', 'show', 'display',
        'trend', 'pattern', 'over time', 'history', 'compare',
        'insights', 'analysis', 'dashboard', 'visual'
    ]
    
    # Check for graph request
    needs_graph = any(keyword in query_lower for keyword in graph_keywords)
    
    # Determine graph type based on query
    graph_type = 'line'  # default
    
    if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
        graph_type = 'comparison'
    elif any(word in query_lower for word in ['distribution', 'histogram', 'frequency']):
        graph_type = 'histogram'
    elif any(word in query_lower for word in ['correlation', 'relationship']):
        graph_type = 'scatter'
    elif any(word in query_lower for word in ['daily', 'hourly', 'monthly', 'summary']):
        graph_type = 'aggregated'
    elif any(word in query_lower for word in ['status', 'current', 'latest']):
        graph_type = 'gauge'
    
    return {
        'needs_graph': needs_graph,
        'graph_type': graph_type
    }

def create_time_series_plot(df: pd.DataFrame, selected_fields: List[str], title: str = "Time Series Data") -> str:
    """
    Create a time series plot and return as base64 string
    """
    try:
        # Convert timestamp to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter numeric columns only
        numeric_cols = [col for col in df.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols:
            return None
            
        # Create figure
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 4*len(numeric_cols)), sharex=True)
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols):
            # Remove NaN values for plotting
            plot_data = df[['timestamp', col]].dropna()
            
            if len(plot_data) > 0:
                axes[i].plot(plot_data['timestamp'], plot_data[col], marker='o', markersize=3, linewidth=2)
                axes[i].set_ylabel(col, fontsize=12)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='y', labelsize=10)
                
                # Add trend line if enough data points
                if len(plot_data) > 2:
                    z = np.polyfit(range(len(plot_data)), plot_data[col], 1)
                    p = np.poly1d(z)
                    axes[i].plot(plot_data['timestamp'], p(range(len(plot_data))), 
                               "--", alpha=0.7, color='red', label='Trend')
                    axes[i].legend()
        
        # Format x-axis
        if axes:
            axes[-1].set_xlabel('Time', fontsize=12)
            axes[-1].tick_params(axis='x', rotation=45, labelsize=10)
            
            # Format date display
            if len(df) > 1:
                axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//10)))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error creating time series plot: {e}")
        return None

def create_comparison_plot(df: pd.DataFrame, selected_fields: List[str], title: str = "Comparison Chart") -> str:
    """
    Create a comparison plot for multiple parameters
    """
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        numeric_cols = [col for col in df.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) < 2:
            return create_time_series_plot(df, selected_fields, title)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: All parameters on same axis (normalized)
        for col in numeric_cols:
            plot_data = df[['timestamp', col]].dropna()
            if len(plot_data) > 0:
                # Normalize to 0-1 scale for comparison
                normalized_data = (plot_data[col] - plot_data[col].min()) / (plot_data[col].max() - plot_data[col].min() + 1e-10)
                ax1.plot(plot_data['timestamp'], normalized_data, marker='o', markersize=2, label=col, linewidth=2)
        
        ax1.set_title('Normalized Comparison (0-1 scale)', fontsize=14)
        ax1.set_ylabel('Normalized Value', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Latest values bar chart
        latest_values = df.iloc[-1] if len(df) > 0 else df.iloc[0]
        latest_numeric = {col: latest_values[col] for col in numeric_cols if pd.notna(latest_values[col])}
        
        if latest_numeric:
            bars = ax2.bar(latest_numeric.keys(), latest_numeric.values(), color=sns.color_palette("husl", len(latest_numeric)))
            ax2.set_title('Latest Values', fontsize=14)
            ax2.set_ylabel('Value', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, latest_numeric.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        return None

def create_aggregated_plot(df: pd.DataFrame, selected_fields: List[str], title: str = "Aggregated Analysis") -> str:
    """
    Create aggregated plots (hourly/daily summaries)
    """
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        numeric_cols = [col for col in df.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols:
            return None
        
        # Determine aggregation level based on data span
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600  # hours
        
        if time_span > 48:  # More than 2 days, aggregate by day
            df['period'] = df['timestamp'].dt.date
            period_label = 'Date'
        elif time_span > 2:  # More than 2 hours, aggregate by hour
            df['period'] = df['timestamp'].dt.floor('H')
            period_label = 'Hour'
        else:  # Aggregate by 15 minutes
            df['period'] = df['timestamp'].dt.floor('15T')
            period_label = '15-Min Interval'
        
        # Create aggregated data
        agg_data = df.groupby('period')[numeric_cols].agg(['mean', 'min', 'max']).reset_index()
        
        # Create subplots
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 4*len(numeric_cols)), sharex=True)
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols):
            # Plot mean with error bars (min-max range)
            axes[i].plot(agg_data['period'], agg_data[(col, 'mean')], 
                        marker='o', markersize=4, linewidth=2, label='Average')
            axes[i].fill_between(agg_data['period'], 
                                agg_data[(col, 'min')], 
                                agg_data[(col, 'max')], 
                                alpha=0.2, label='Min-Max Range')
            
            axes[i].set_ylabel(f'{col}', fontsize=12)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        if axes:
            axes[-1].set_xlabel(period_label, fontsize=12)
            axes[-1].tick_params(axis='x', rotation=45, labelsize=10)
        
        plt.suptitle(f'{title} - {period_label} Aggregation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error creating aggregated plot: {e}")
        return None

def create_gauge_chart(df: pd.DataFrame, selected_fields: List[str], title: str = "Current Status") -> str:
    """
    Create gauge-style charts for current values
    """
    try:
        numeric_cols = [col for col in df.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols or len(df) == 0:
            return None
        
        # Get latest values
        latest_row = df.iloc[-1]
        
        # Create subplot grid
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i >= len(axes):
                break
                
            current_value = latest_row[col] if pd.notna(latest_row[col]) else 0
            
            # Create simple bar gauge
            axes[i].barh([0], [current_value], height=0.5, 
                        color=sns.color_palette("husl", len(numeric_cols))[i])
            axes[i].set_xlim(0, max(current_value * 1.2, 1))
            axes[i].set_ylim(-0.5, 0.5)
            axes[i].set_title(f'{col}\n{current_value:.2f}', fontsize=12, fontweight='bold')
            axes[i].set_yticks([])
            axes[i].grid(True, alpha=0.3)
            
            # Add value text
            axes[i].text(current_value/2, 0, f'{current_value:.2f}', 
                        ha='center', va='center', fontweight='bold', color='white')
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error creating gauge chart: {e}")
        return None

def generate_graph(df: pd.DataFrame, selected_fields: List[str], graph_type: str, user_query: str) -> Optional[str]:
    """
    Generate appropriate graph based on type and return base64 string
    """
    if len(df) == 0:
        return None
    
    title = f"Analysis for: {user_query[:50]}..."
    
    try:
        if graph_type == 'comparison':
            return create_comparison_plot(df, selected_fields, title)
        elif graph_type == 'aggregated':
            return create_aggregated_plot(df, selected_fields, title)
        elif graph_type == 'gauge':
            return create_gauge_chart(df, selected_fields, title)
        else:  # default to time series
            return create_time_series_plot(df, selected_fields, title)
    except Exception as e:
        print(f"Error generating graph: {e}")
        return None

def save_graph_as_file(image_base64: str, filename: str = "analysis_chart.png") -> str:
    """
    Save base64 image to file and return filename
    """
    try:
        image_data = base64.b64decode(image_base64)
        with open(filename, 'wb') as f:
            f.write(image_data)
        return filename
    except Exception as e:
        print(f"Error saving graph: {e}")
        return None

# --- Step 1: Extract Dates ---
def extract_dates_from_query(user_query: str) -> dict:
    """
    Extract dates with local fallback to reduce API calls
    """
    # Try local date extraction first
    query_lower = user_query.lower()
    current_time = datetime.utcnow()
    
    # Define common date patterns
    if 'today' in query_lower:
        start_date = current_time.replace(hour=0, minute=0, second=0)
        end_date = current_time.replace(hour=23, minute=59, second=59)
    elif 'yesterday' in query_lower:
        yesterday = current_time - timedelta(days=1)
        start_date = yesterday.replace(hour=0, minute=0, second=0)
        end_date = yesterday.replace(hour=23, minute=59, second=59)
    elif 'last week' in query_lower or 'past week' in query_lower:
        start_date = (current_time - timedelta(days=7)).replace(hour=0, minute=0, second=0)
        end_date = current_time.replace(hour=23, minute=59, second=59)
    elif 'last month' in query_lower or 'past month' in query_lower:
        start_date = (current_time - timedelta(days=30)).replace(hour=0, minute=0, second=0)
        end_date = current_time.replace(hour=23, minute=59, second=59)
    else:
        # Try API with rate limiting protection
        try:
            import time
            time.sleep(0.5)  # Small delay
            
            prompt = f"""Extract start and end dates from this query: "{user_query}"
Current time (UTC): {datetime.utcnow().strftime('%d %B %Y %H:%M:%S')} UTC

Return ONLY a JSON object in this format:
{{
  "start_date": "YYYY-MM-DD HH:MM:SS",
  "end_date": "YYYY-MM-DD HH:MM:SS"
}}

Follow rules: All times must be in UTC. Use 00:00:00 and 23:59:59 for default bounds.
"""
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt, generation_config={
                "temperature": 0.1,
                "max_output_tokens": 100
            })
            text = response.text.strip()

            try:
                result = json.loads(text)
                start_date = datetime.strptime(result["start_date"], "%Y-%m-%d %H:%M:%S")
                end_date = datetime.strptime(result["end_date"], "%Y-%m-%d %H:%M:%S")
            except:
                match = re.search(r'\{[\s\S]*?\}', text)
                result = json.loads(match.group()) if match else None
                if result:
                    start_date = datetime.strptime(result["start_date"], "%Y-%m-%d %H:%M:%S")
                    end_date = datetime.strptime(result["end_date"], "%Y-%m-%d %H:%M:%S")
                else:
                    raise Exception("Could not parse dates")
        except:
            # Default to today
            start_date = current_time.replace(hour=0, minute=0, second=0)
            end_date = current_time.replace(hour=23, minute=59, second=59)
    
    return {
        "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S")
    }

def fallback_dates():
    current_time = datetime.utcnow()
    return {
        "start_date": current_time.replace(hour=0, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": current_time.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%d %H:%M:%S"),
    }

def normalize_dates(start_str, end_str):
    current_year = datetime.utcnow().year
    start_dt = dateparser.parse(start_str)
    end_dt = dateparser.parse(end_str)

    if start_dt.year == 1900: start_dt = start_dt.replace(year=current_year)
    if end_dt.year == 1900: end_dt = end_dt.replace(year=current_year)

    return start_dt.replace(hour=0, minute=0, second=0), end_dt.replace(hour=23, minute=59, second=59)

def to_epoch_millis(dt): return int(dt.timestamp() * 1000)

# --- Step 2: Select Relevant Fields with Fallback ---
def select_fields_from_query(user_query: str, field_list: List[Dict[str, str]]) -> List[str]:
    """
    Select relevant fields with local fallback to avoid API rate limits
    """
    # First try local keyword matching to reduce API calls
    query_lower = user_query.lower()
    matched_fields = []
    
    # Define keyword mappings for common queries
    keyword_mappings = {
        'pump_1': ['pump_1', 'pump1'],
        'pump_2': ['pump_2', 'pump2'],
        'pump_3': ['pump_3', 'pump3'],
        'voltage': ['voltage', 'vry', 'vyb', 'vbr', 'volt'],
        'current': ['current', 'iry', 'iyb', 'ibr', 'amp'],
        'power': ['power', 'watt', 'kw', 'energy'],
        'temperature': ['temp', 'temperature', 'heat'],
        'pressure': ['pressure', 'press', 'bar', 'psi'],
        'flow': ['flow', 'rate', 'gpm', 'lpm'],
        'frequency': ['freq', 'frequency', 'hz'],
        'speed': ['speed', 'rpm', 'rotation'],
        'running': ['running', 'status', 'state', 'on', 'off'],
        'motor': ['motor', 'em'],
        'vibration': ['vibration', 'vib'],
        'level': ['level', 'tank', 'reservoir']
    }
    
    # Find matching fields based on keywords
    for category, keywords in keyword_mappings.items():
        if any(keyword in query_lower for keyword in keywords):
            for field in field_list:
                field_key_lower = field["key"].lower()
                field_desc_lower = (field["description"] or "").lower()  # Handle None descriptions
                
                # Check if field matches the category
                if any(keyword in field_key_lower or keyword in field_desc_lower for keyword in keywords):
                    if field["key"] not in matched_fields:
                        matched_fields.append(field["key"])
    
    # If we found matches through keywords, use them
    if matched_fields:
        return matched_fields[:15]  # Limit to reasonable number
    
    # Fallback: Try API with rate limiting protection
    try:
        import time
        time.sleep(1)  # Small delay to help with rate limiting
        
        prompt = f"""From the following list of telemetry fields, select the most relevant ones to answer the query: "{user_query}"

        Return ONLY a JSON array of keys.

        Telemetry Fields:   
        {json.dumps(field_list[:50], indent=2)}  # Limit fields to reduce token usage
        """
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, generation_config={
            "temperature": 0.1,
            "max_output_tokens": 200
        })
        raw = response.text.strip()
        api_result = json.loads(raw) if raw.startswith("[") else json.loads(re.search(r"\[.*?\]", raw, re.DOTALL).group())
        return api_result
        
    except Exception as e:
        print("API field selection failed, using keyword fallback:", str(e))
        
        # Ultimate fallback: return fields that contain query keywords
        fallback_fields = []
        for field in field_list:
            # Check if any word from query appears in field key or description
            query_words = [word for word in query_lower.split() if len(word) > 2]
            for word in query_words:
                field_desc = field.get("description") or ""  # Handle None descriptions
                if (word in field["key"].lower() or word in field_desc.lower()) and field["key"] not in fallback_fields:
                    fallback_fields.append(field["key"])
        
        # If still no matches, return first 10 fields
        return fallback_fields[:10] if fallback_fields else [f["key"] for f in field_list[:10]]

# --- Step 3: API Call ---
def fetch_data_from_api(url: str) -> dict:
    base_url = "http://13.71.23.55:8080"
    device_id = "fa2bef00-ed29-11ef-8baf-55643890fc3e"
    username = "jwil@tenantadmin.com"
    password = "cimcon@123"

    try:
        auth_resp = requests.post(f"{base_url}/api/auth/login", json={"username": username, "password": password})
        auth_resp.raise_for_status()
        token = auth_resp.json()["token"]
    except Exception as e:
        raise Exception(f"Auth failed: {e}")

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    try:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 401:
            auth_resp = requests.post(f"{base_url}/api/auth/login", json={"username": username, "password": password})
            token = auth_resp.json()["token"]
            headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise Exception(f"API fetch failed: {e}")

# --- Enhanced Natural Language Generation ---
def generate_comprehensive_answer(user_query: str, df: pd.DataFrame, selected_fields: List[str], 
                                 graph_info: dict = None) -> str:
    """
    Generate a focused, accurate answer based on what the user actually asked
    """
    query_lower = user_query.lower()

    try:
        if len(df) == 0:
            return "No data found for your query. Please check the time period or field names."

        df_sorted = df.sort_values('timestamp')
        latest_data = df_sorted.tail(1)

        # Prepare concise data summary based on query type
        if any(word in query_lower for word in ['current', 'now', 'latest', 'status']):
            data_summary = latest_data.to_string(index=False)
        elif any(word in query_lower for word in ['trend', 'change', 'increase', 'decrease']):
            first_few = df_sorted.head(3)
            last_few = df_sorted.tail(3)
            data_summary = f"FIRST VALUES:\n{first_few.to_string(index=False)}\n\nLATEST VALUES:\n{last_few.to_string(index=False)}"
        else:
            data_summary = df_sorted.head(10).to_string(index=False)

        # Simple numerical analysis
        insights = []
        numeric_cols = [col for col in df.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col])]

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 1:
                latest_val = col_data.iloc[-1]
                first_val = col_data.iloc[0]
                avg_val = col_data.mean()
                if abs(latest_val - first_val) > abs(avg_val * 0.05):
                    trend = "increased" if latest_val > first_val else "decreased"
                    insights.append(f"{col} {trend} from {first_val:.2f} to {latest_val:.2f}")
                else:
                    insights.append(f"{col} remained stable around {avg_val:.2f}")
    except Exception as e:
        data_summary = df.head(5).to_string(index=False) if len(df) > 0 else "No data available"
        insights = []

    graph_text = ""
    if graph_info and graph_info.get('generated'):
        graph_text = f"\n\nA visual chart has been generated and saved as '{graph_info.get('filename', 'analysis_chart.png')}' showing the data trends and patterns."

    enhanced_prompt = f"""
You are TwinBot, a SCADA system assistant. Answer the user's question directly and accurately based on the actual data provided.

USER QUERY: "{user_query}"

AVAILABLE DATA FROM TIME PERIOD:
{data_summary}

ANALYSIS SUMMARY:
{chr(10).join(insights) if insights else "Data shows consistent readings."}
{graph_text}

INSTRUCTIONS:
- Answer EXACTLY what the user asked - no more, no less
- Use the actual data values from the dataset
- If user asks for specific time period, focus on that period
- If user asks about trends, describe the actual trend you see
- If user asks about current status, use the latest values
- If user asks about problems/issues, only mention them if data actually shows problems
- Keep response concise and directly relevant to the query
- Use simple language but be technically accurate
- Don't add extra information the user didn't ask for
- If a chart was generated, mention it briefly

Base your entire response on the actual data provided above. If the data doesn't support a claim, don't make it.
"""

    try:
        import time
        time.sleep(1)  # avoid Gemini rate limit

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(enhanced_prompt)
        return response.text.strip() if hasattr(response, "text") else "No answer generated."

    except Exception as e:
        return "Could not generate natural language answer due to an error."

