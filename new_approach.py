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


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

with open("1.json", "r") as f:
    FIELD_METADATA = json.load(f)  # Expected: {"Pump_1_EM_1_VRY": "Voltage between R and Y...", ...}

# Convert to list of dicts for prompt clarity
FIELD_LIST = [{"key": k, "description": v} for k, v in FIELD_METADATA.items()]


# --- Step 1: Extract Dates ---
def extract_dates_from_query(user_query: str) -> dict:
    prompt = f"""Extract start and end dates from this query: "{user_query}"
Current time (UTC): {datetime.utcnow().strftime('%d %B %Y %H:%M:%S')} UTC

Return ONLY a JSON object in this format:
{{
  "start_date": "YYYY-MM-DD HH:MM:SS",
  "end_date": "YYYY-MM-DD HH:MM:SS"
}}

Follow rules: All times must be in UTC. Use 00:00:00 and 23:59:59 for default bounds.
"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, generation_config={"temperature": 0.1})
        text = response.text.strip()

        try:
            return json.loads(text)
        except:
            match = re.search(r'\{[\s\S]*?\}', text)
            return json.loads(match.group()) if match else fallback_dates()
    except:
        return fallback_dates()

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

# --- Step 2: Select Relevant Fields ---
def select_fields_from_query(user_query: str, field_list: List[Dict[str, str]]) -> List[str]:
    prompt = f"""From the following list of telemetry fields, select the most relevant ones to answer the query: "{user_query}"

    Return ONLY a JSON array of keys.

    Telemetry Fields:   
    {json.dumps(field_list, indent=2)}
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, generation_config={"temperature": 0.1})
        raw = response.text.strip()
        return json.loads(raw) if raw.startswith("[") else json.loads(re.search(r"\[.*?\]", raw, re.DOTALL).group())
    except Exception as e:
        print("Error in field selection:", e)
        return [f["key"] for f in field_list[:10]]

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

# --- Step 4: LangGraph Node ---
class State(TypedDict):
    user_query: str
    start_date: str
    end_date: str
    startTs: int
    endTs: int
    final_api: str
    api_response: dict

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

            # === Natural Language Summary ===
            summary_prompt = f"""
            You are TwinBot, a helpful assistant that explains telemetry data in simple language for non-technical users.

            The user query was:
            "{user_query}"

            Here is the data retrieved from the SCADA system (first few rows only):
            {final_df.head(10).to_string(index=False)}

            Based on the query and the data, generate a brief summary in plain English. Be concise, user-friendly, and avoid technical jargon. Explain what the data means.
            """

            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                summary_response = model.generate_content(summary_prompt)
                natural_summary = summary_response.text.strip()
            except Exception as e:
                natural_summary = "No summary available due to an error."

            return {
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "startTs": start_ts,
                "endTs": end_ts,
                "final_api": final_url,
                "api_response": final_df.to_dict(orient="records"),
                "natural_summary": natural_summary
            }

        except Exception as e:
            return {
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "startTs": start_ts,
                "endTs": end_ts,
                "final_api": final_url,
                "api_response": {"error": str(e)},
                "natural_summary": "Something went wrong while processing your request."
            }


    builder = StateGraph(state)
    builder.add_node("process", RunnableLambda(process_node))
    builder.set_entry_point("process")
    builder.set_finish_point("process")
    return builder.compile()