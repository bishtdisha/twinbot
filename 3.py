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
from prompt import description


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

with open("1.json", "r") as f:
    FIELD_METADATA = json.load(f)  # Expected: {"Pump_1_EM_1_VRY": "Voltage between R and Y...", ...}

# Convert to list of dicts for prompt clarity
FIELD_LIST = [{"key": k, "description": v or f"Telemetry field: {k}"} for k, v in FIELD_METADATA.items()]


# --- Step 1: Extract Dates ---
# --- Step 1: Extract Dates with Fallback ---
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
def generate_comprehensive_answer(user_query: str, df: pd.DataFrame, selected_fields: List[str]) -> str:
    """
    Generate a focused, accurate answer based on what the user actually asked
    """
    # Analyze the query type to provide targeted responses
    query_lower = user_query.lower()
    
    try:
        # Get the most recent and relevant data
        if len(df) == 0:
            return "No data found for your query. Please check the time period or field names."
        
        # Sort by timestamp to get latest data
        df_sorted = df.sort_values('timestamp')
        latest_data = df_sorted.tail(1)
        
        # Prepare concise data summary based on query type
        if any(word in query_lower for word in ['current', 'now', 'latest', 'status']):
            # For current status queries, focus on latest values
            data_summary = latest_data.to_string(index=False)
        elif any(word in query_lower for word in ['trend', 'change', 'increase', 'decrease']):
            # For trend queries, show first and last few rows
            first_few = df_sorted.head(3)
            last_few = df_sorted.tail(3)
            data_summary = f"FIRST VALUES:\n{first_few.to_string(index=False)}\n\nLATEST VALUES:\n{last_few.to_string(index=False)}"
        else:
            # For general queries, show representative sample
            data_summary = df_sorted.head(10).to_string(index=False)
        
        # Generate simple insights based on numerical data
        insights = []
        numeric_cols = [col for col in df.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col])]
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 1:
                latest_val = col_data.iloc[-1]
                first_val = col_data.iloc[0]
                avg_val = col_data.mean()
                
                # Simple trend analysis
                if abs(latest_val - first_val) > abs(avg_val * 0.05):  # 5% change threshold
                    trend = "increased" if latest_val > first_val else "decreased"
                    insights.append(f"{col} {trend} from {first_val:.2f} to {latest_val:.2f}")
                else:
                    insights.append(f"{col} remained stable around {avg_val:.2f}")
    
    except Exception as e:
        data_summary = df.head(5).to_string(index=False) if len(df) > 0 else "No data available"
        insights = []
    
    # Create enhanced prompt
    enhanced_prompt = f"""
    You are TwinBot, a SCADA system assistant. Answer the user's question directly and accurately based on the actual data provided.

    USER QUERY: "{user_query}"

    AVAILABLE DATA FROM TIME PERIOD:
    {data_summary}

    ANALYSIS SUMMARY:
    {chr(10).join(insights) if insights else "Data shows consistent readings"}

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

    Base your entire response on the actual data provided above. If the data doesn't support a claim, don't make it.
    """

    try:
        import time
        time.sleep(1)  # Rate limiting protection
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(enhanced_prompt, generation_config={
            "temperature": 0.2,  # Lower temperature for more factual responses
            "max_output_tokens": 300  # Shorter responses to save quota
        })
        return response.text.strip()
    except Exception as e:
        # Better fallback response based on data analysis
        if len(df) > 0:
            # Try to answer basic questions without API
            query_lower = user_query.lower()
            
            if 'running' in query_lower or 'status' in query_lower:
                # For running status, check if we have status fields
                status_cols = [col for col in df.columns if 'status' in col.lower() or 'running' in col.lower()]
                if status_cols:
                    latest_status = df[status_cols[0]].iloc[-1]
                    return f"Based on the latest data, the status shows: {latest_status}"
                else:
                    return "I found data but couldn't determine the running status. Please check the saved CSV file for details."
            
            elif 'voltage' in query_lower:
                voltage_cols = [col for col in df.columns if 'voltage' in col.lower() or 'vry' in col.lower()]
                if voltage_cols:
                    latest_voltage = df[voltage_cols[0]].iloc[-1]
                    return f"The latest voltage reading is {latest_voltage}V"
            
            elif 'current' in query_lower:
                current_cols = [col for col in df.columns if 'current' in col.lower() or 'iry' in col.lower()]
                if current_cols:
                    latest_current = df[current_cols[0]].iloc[-1]
                    return f"The latest current reading is {latest_current}A"
            
            # Generic fallback
            return f"I found {len(df)} data points for your query from {df['timestamp'].min()} to {df['timestamp'].max()}. The detailed data has been saved to 'output_dataframe.csv'."
        else:
            return "No data was found for your query. This might be because the requested equipment (like Pump 2) is not available in the current dataset, or the time period has no data."
    
    # Create enhanced prompt
    enhanced_prompt = f"""
    You are TwinBot, a SCADA system assistant. Answer the user's question directly and accurately based on the actual data provided.

    USER QUERY: "{user_query}"

    AVAILABLE DATA FROM TIME PERIOD:
    {data_summary}

    ANALYSIS SUMMARY:
    {chr(10).join(insights) if insights else "Data shows consistent readings"}

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

    Base your entire response on the actual data provided above. If the data doesn't support a claim, don't make it.
    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(enhanced_prompt, generation_config={
            "temperature": 0.3,
            "max_output_tokens": 1000
        })
        return response.text.strip()
    except Exception as e:
        return f"I found {len(df)} data points for your query, but I'm having trouble generating a detailed analysis right now. The data has been saved to 'output_dataframe.csv' for your review."

# --- Step 4: Enhanced LangGraph Node ---
class State(TypedDict):
    user_query: str
    start_date: str
    end_date: str
    startTs: int
    endTs: int
    final_api: str
    api_response: dict
    natural_answer: str  # This will be the main output

def build_langgraph_workflow():
    state = State

    def process_node(input: Dict[str, Any]) -> Dict[str, Any]:
        user_query = input["user_query"]

        # Dates
        result = extract_dates_from_query(user_query)
        start_dt, end_dt = normalize_dates(result["start_date"], result["end_date"])
        start_ts = to_epoch_millis(start_dt)
        end_ts = to_epoch_millis(end_dt)

        # Fields
        selected_keys = select_fields_from_query(user_query, FIELD_LIST)
        keys_param = ",".join(selected_keys)

        # API
        base_url = (
            "http://13.71.23.55:8080/api/plugins/telemetry/DEVICE/"
            "fa2bef00-ed29-11ef-8baf-55643890fc3e/values/timeseries"
        )
        final_url = f"{base_url}?keys={keys_param}&startTs={start_ts}&endTs={end_ts}&agg=NONE&orderBy=ASC&limit=10000"

        try:
            api_response = fetch_data_from_api(final_url)
            dfs = []

            for key, values in api_response.items():
                df = pd.DataFrame(values)
                df['timestamp'] = pd.to_datetime(df['ts'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
                df = df.rename(columns={'value': key})[['timestamp', key]]
                dfs.append(df)

            final_df = reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='outer'), dfs)
            final_df = final_df.sort_values('timestamp')
            final_df.to_csv("output_dataframe.csv", index=False)

            # Generate comprehensive natural language answer
            natural_answer = generate_comprehensive_answer(user_query, final_df, selected_keys)

            return {
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "startTs": start_ts,
                "endTs": end_ts,
                "final_api": final_url,
                "api_response": final_df.to_dict(orient="records"),
                "natural_answer": natural_answer  # Main output for the user
            }

        except Exception as e:
            error_message = f"I encountered an issue while retrieving data for your query '{user_query}'. {str(e)} Please try rephrasing your question or check if the system is available."
            
            return {
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "startTs": start_ts,
                "endTs": end_ts,
                "final_api": final_url,
                "api_response": {"error": str(e)},
                "natural_answer": error_message
            }

    builder = StateGraph(state)
    builder.add_node("process", RunnableLambda(process_node))
    builder.set_entry_point("process")
    builder.set_finish_point("process")
    return builder.compile()

# --- Usage Example ---
def run_twinbot(user_query: str) -> str:
    """
    Main function to run TwinBot and get natural language response
    """
    workflow = build_langgraph_workflow()
    result = workflow.invoke({"user_query": user_query})
    
    # Return the natural language answer as the primary output
    return result.get("natural_answer", "I couldn't process your request at this time.")

# Example usage:
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What is the current tail-end pressure?", 
        "Which pump is currently running and what is its running frequency?",
        "What is the water level in the sump tank?", 
        "How much water was supplied today till now?", 
        "Is any pump running outside its ideal range?",
        "What is the real-time flow rate from both outlet lines?",
        "What is the current power consumption of Pump 1?",
        "How long has Pump 2 been running today?",
        "Are there any current alerts?",
        "What is the VFD frequency and tail-end pressure right now?",
        "Show daily water consumption for the last 7 days."
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Answer: {run_twinbot(query)}")
        print("-" * 80)