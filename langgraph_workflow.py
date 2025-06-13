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

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_dates_from_query(user_query: str) -> dict:  
    prompt = f"""Extract start and end dates from this query: "{user_query}"
Current time (UTC): {datetime.utcnow().strftime('%d %B %Y %H:%M:%S')} UTC

Return ONLY a JSON object in this format:
{{
  "start_date": "YYYY-MM-DD HH:MM:SS",
  "end_date": "YYYY-MM-DD HH:MM:SS"
}}

Rules:
1. ALL times must be in UTC. Do NOT use any other timezone. Do NOT use local time.
2. Use current year if year not specified.
3. Start time: 00:00:00, End time: 23:59:59 unless otherwise specified.
4. For "last X days": start = (now - X days) at the current UTC time, end = now (current UTC time).
5. For "last X hours": start = (now - X hours) at the current UTC time, end = now (current UTC time).
6. For "yesterday": use yesterday's date in UTC.
7. For date ranges: parse both dates in UTC.
8. All times in UTC. Do NOT use IST or any other timezone.

Example patterns:
- "last 10 days" → start: (now-10d) at current UTC time, end: now (current UTC time)
- "last 24 hours" → start: (now-24h) at current UTC time, end: now (current UTC time)
- "last 2 hours" → start: (now-2h) at current UTC time, end: now (current UTC time)
- "yesterday" → start: yesterday 00:00:00 UTC, end: yesterday 23:59:59 UTC
- "1st May to 5th May" → start: May 1 00:00:00 UTC, end: May 5 23:59:59 UTC

IMPORTANT: All output times must be in UTC. Do NOT use IST or any other timezone.
"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
            }
        )
        
        response_text = response.text.strip()
        print(f"Raw model response: {response_text}")  # Debug log
        
        try:
            result = json.loads(response_text)
            print(f"Successfully parsed JSON: {result}")  # Debug log
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")  # Debug log
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group()
                print(f"Extracted JSON string: {json_str}")  # Debug log
                return json.loads(json_str)
            
    except Exception as e:
        print(f"Exception occurred: {e}")  # Debug log
    
    # Fallback to current day if all else fails
    current_time = datetime.utcnow()
    start_time = current_time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    end_time = current_time.replace(hour = 23, minute = 59, second = 59, microsecond = 999999)
    
    return {
        "start_date": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_time.strftime("%Y-%m-%d %H:%M:%S")
    }

def normalize_dates(start_str, end_str):
    current_year = datetime.utcnow().year

    start_dt = dateparser.parse(start_str, settings = {'PREFER_DATES_FROM': 'past'})
    end_dt = dateparser.parse(end_str, settings = {'PREFER_DATES_FROM': 'future'})

    if start_dt.year == 1900:
        start_dt = start_dt.replace(year = current_year)
    if end_dt.year == 1900:
        end_dt = end_dt.replace(year = current_year)

    if start_dt.hour == 0 and start_dt.minute == 0:
        start_dt = start_dt.replace(hour = 0, minute = 0, second = 0)
    if end_dt.hour == 0 and end_dt.minute == 0:
        end_dt = end_dt.replace(hour = 23, minute = 59, second = 59)

    return start_dt, end_dt

def to_epoch_millis(dt):
    return int(dt.timestamp() * 1000)

class State(TypedDict):
    user_query: str
    start_date: str
    end_date: str
    startTs: int
    endTs: int
    final_api: str
    api_response: dict

def fetch_data_from_api(url: str) -> dict:
    """
    Fetch data from the API with authentication and retry logic
    """
    base_url = "http://13.71.23.55:8080"
    device_id = "fa2bef00-ed29-11ef-8baf-55643890fc3e"
    username = "jwil@tenantadmin.com"
    password = "cimcon@123"
    static_keys = [
        "dailyConsumption", "dailyConsumption1", "dailyConsumption2", "dailyConsumption3",
        "flow", "Flow_P1", "Flow_P2", "1_Flow_ls", "2_Flow_ls", "3_Flow_ls",
        "1_Flow_m3h", "2_Flow_m3h", "3_Flow_m3h"
    ]

    # Authenticate
    try:
        auth_resp = requests.post(
            f"{base_url}/api/auth/login",   
            json={"username": username, "password": password}
        )
        auth_resp.raise_for_status()
        token = auth_resp.json()["token"]
    except Exception as e:
        print(f"Authentication failed: {e}")
        raise

    # Headers with token
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # API Call
    try:
        response = requests.get(url, headers = headers, timeout = 300)
        if response.status_code == 401: 
            # Retry once if token expired
            auth_resp = requests.post(
                f"{base_url}/api/auth/login",
                json={"username": username, "password": password}
            )
            token = auth_resp.json()["token"]
            headers["Authorization"] = f"Bearer {token}"
            response = requests.get(url, headers = headers, timeout = 300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API request failed: {e}")
        raise

def build_langgraph_workflow():
    state = State

    def process_node(input):
        # Step 1: Extract dates from query
        result = extract_dates_from_query(input["user_query"])
        start_date = result["start_date"]
        end_date = result["end_date"]

        # Step 2: Convert dates to epoch milliseconds
        start_dt, end_dt = normalize_dates(start_date, end_date)
        start_ts = to_epoch_millis(start_dt)
        end_ts = to_epoch_millis(end_dt)

        # Step 3: Compose final API URL
        base_url = (
            "http://13.71.23.55:8080/api/plugins/telemetry/DEVICE/"
            "fa2bef00-ed29-11ef-8baf-55643890fc3e/values/timeseries"
        )
        keys = (
            "dailyConsumption,dailyConsumption1,dailyConsumption2,dailyConsumption3,"
            "flow,Flow_P1,Flow_P2,1_Flow_ls,2_Flow_ls,3_Flow_ls,"
            "1_Flow_m3h,2_Flow_m3h,3_Flow_m3h"
        )
        final_url = (
            f"{base_url}?keys={keys}&startTs={start_ts}&endTs={end_ts}"
            "&agg=NONE&orderBy=ASC&limit=10000&useStrictDataTypes=false"
        )

        # Step 4: Fetch data from API
        try:
            api_response = fetch_data_from_api(final_url)
            
            # Convert the data to DataFrame format
            dfs = []
            for key, values in api_response.items():
                # Convert timestamp to datetime and adjust to UTC/local time
                df = pd.DataFrame(values)
                # First convert to datetime
                df['timestamp'] = pd.to_datetime(df['ts'], unit = 'ms')
                # Convert to UTC by adding the timezone information
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                # Convert to local timezone (GMT+05:30)
                df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')
                # Format the timestamp to match required format
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                df = df.rename(columns={'value': key})
                df = df[['timestamp', key]]  # Keep only timestamp and value columns
                dfs.append(df)
            
            # Merge all dataframes on timestamp
            final_df = dfs[0]
            for df in dfs[1:]:
                final_df = pd.merge(final_df, df, on = 'timestamp', how = 'outer')
            
            # Sort by timestamp
            final_df = final_df.sort_values('timestamp')
            
            # Save DataFrame to CSV with UTC timestamps
            output_file = "output_dataframe.csv"
            final_df.to_csv(output_file, index = False)
            print(f"\nDataFrame saved to {output_file}")
            
            # Print the DataFrame with UTC timestamps
            print("\n=== Telemetry Data in DataFrame Format (UTC) ===")
            print("\nFirst few rows:")
            print(final_df.head())
            print("\nDataFrame Info:")
            print(final_df.info())
            print("\n===========================")
            
            # Store the DataFrame in the state
            api_response = final_df.to_dict(orient = 'records')
            
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            api_response = {"error": str(e)}
            print(f"\nError: {str(e)}")

        # Return all state fields including API response
        return {
            "start_date": start_date,
            "end_date": end_date,
            "startTs": start_ts,
            "endTs": end_ts,
            "final_api": final_url,
            "api_response": api_response
        }

    builder = StateGraph(state)
    builder.add_node("process", RunnableLambda(process_node))
    builder.set_entry_point("process")
    builder.set_finish_point("process")

    return builder.compile()