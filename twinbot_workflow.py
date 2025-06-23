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

# Custom prompt builder
from prompt import (
    build_date_extraction_prompt,
    build_field_selection_prompt,
    build_analysis_prompt
)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

with open("fields.txt", "r") as f:
    lines = f.readlines()

FIELD_METADATA = {}
for line in lines:
    if ':' in line:
        key, desc = line.strip().split(":", 1)
        FIELD_METADATA[key.strip()] = desc.strip()

FIELD_LIST = [{"key": k, "description": v or f"Telemetry field: {k}"} for k, v in FIELD_METADATA.items()]

def extract_dates_from_query(user_query: str) -> dict:
    query_lower = user_query.lower()
    current_time = datetime.utcnow()

    days_match = re.search(r'(last|past)\s+(\d+)\s+days?', query_lower)
    if days_match:
        num_days = int(days_match.group(2))
        start_date = (current_time - timedelta(days=num_days)).replace(hour=0, minute=0, second=0)
        end_date = current_time.replace(hour=23, minute=59, second=59)
        return {
            "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }

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
        try:
            import time
            time.sleep(0.5)

            prompt = build_date_extraction_prompt(user_query)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt, generation_config={
                "temperature": 0.1,
                "max_output_tokens": 100
            })
            raw_text = response.text.strip()

            try:
                result = json.loads(raw_text)
            except:
                match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
                result = json.loads(match.group()) if match else fallback_dates()

            return {
                "start_date": result["start_date"],
                "end_date": result["end_date"]
            }

        except Exception:
            return fallback_dates()

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
    start_dt = dateparser.parse(start_str)
    end_dt = dateparser.parse(end_str)

    if not start_dt or start_dt.year == 1900:
        start_dt = datetime.utcnow().replace(hour=0, minute=0, second=0)
    if not end_dt or end_dt.year == 1900:
        end_dt = datetime.utcnow().replace(hour=23, minute=59, second=59)

    return start_dt.replace(hour=0, minute=0, second=0), end_dt.replace(hour=23, minute=59, second=59)

def to_epoch_millis(dt):
    return int(dt.timestamp() * 1000)

def select_fields_from_query(user_query: str, field_list: List[Dict[str, str]]) -> List[str]:
    query_lower = user_query.lower()
    matched_fields = []

    keyword_mappings = {
        'pump_1': ['pump_1', 'pump1'],
        'pump_2': ['pump_2', 'pump2'],
        'voltage': ['voltage', 'vry', 'vyb', 'vbr'],
        'current': ['current', 'iry', 'iyb', 'ibr'],
        'power': ['power', 'kw'],
        'pressure': ['pressure', 'bar', 'psi'],
        'flow': ['flow', 'lpm'],
        'frequency': ['freq', 'hz'],
        'speed': ['speed', 'rpm'],
        'running': ['status', 'running', 'on', 'off']
    }

    for category, keywords in keyword_mappings.items():
        if any(k in query_lower for k in keywords):
            for field in field_list:
                if any(k in field["key"].lower() or (field["description"] or "").lower() for k in keywords):
                    if field["key"] not in matched_fields:
                        matched_fields.append(field["key"])

    if matched_fields:
        return matched_fields[:15]

    try:
        import time
        time.sleep(1)

        prompt = build_field_selection_prompt(user_query, field_list)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, generation_config={
            "temperature": 0.1,
            "max_output_tokens": 200
        })
        raw = response.text.strip()
        api_result = json.loads(raw) if raw.startswith("[") else json.loads(re.search(r"\[.*?\]", raw, re.DOTALL).group())
        return [item["key"] for item in api_result]
    except:
        fallback = []
        for field in field_list:
            query_words = [w for w in query_lower.split() if len(w) > 2]
            for w in query_words:
                if (w in field["key"].lower() or w in (field["description"] or "").lower()) and field["key"] not in fallback:
                    fallback.append(field["key"])
        return fallback[:10] if fallback else [f["key"] for f in field_list[:10]]

def fetch_data_from_api(url: str) -> dict:
    base_url = "http://13.71.23.55:8080"
    device_id = "fa2bef00-ed29-11ef-8baf-55643890fc3e"
    username = "jwil@tenantadmin.com"
    password = "cimcon@123"

    try:
        auth_resp = requests.post(f"{base_url}/api/auth/login", json={"username": username, "password": password})
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
        return resp.json()
    except Exception as e:
        raise Exception(f"API fetch failed: {e}")

def generate_comprehensive_answer(user_query: str, df: pd.DataFrame, selected_fields: List[str]) -> str:
    query_lower = user_query.lower()
    if df.empty:
        return "No data found for your query. Please check the time period or field names."

    df_sorted = df.sort_values('timestamp')
    latest_data = df_sorted.tail(1)

    if any(word in query_lower for word in ['current', 'now', 'latest', 'status']):
        data_summary = latest_data.to_string(index=False)
    elif any(word in query_lower for word in ['trend', 'increase', 'decrease']):
        first_few = df_sorted.head(3)
        last_few = df_sorted.tail(3)
        data_summary = f"FIRST VALUES:\n{first_few.to_string(index=False)}\n\nLATEST VALUES:\n{last_few.to_string(index=False)}"
    else:
        data_summary = df_sorted.head(10).to_string(index=False)

    insights = []
    for col in [c for c in df.columns if c != 'timestamp' and pd.api.types.is_numeric_dtype(df[c])]:
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

    try:
        prompt = build_analysis_prompt(user_query, data_summary, "\n".join(insights))
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, generation_config={
            "temperature": 0.2,
            "max_output_tokens": 300
        })
        return response.text.strip()
    except:
        return f"I found {len(df)} data points. Please check output_dataframe.csv for details."

class State(TypedDict):
    user_query: str
    start_date: str
    end_date: str
    startTs: int
    endTs: int
    final_api: str
    api_response: dict
    natural_answer: str

def build_langgraph_workflow():
    state = State

    def process_node(input: Dict[str, Any]) -> Dict[str, Any]:
        user_query = input["user_query"]

        result = extract_dates_from_query(user_query)
        start_dt, end_dt = normalize_dates(result["start_date"], result["end_date"])
        start_ts = to_epoch_millis(start_dt)
        end_ts = to_epoch_millis(end_dt)

        selected_keys = select_fields_from_query(user_query, FIELD_LIST)
        keys_param = ",".join(selected_keys)

        final_url = (
            f"http://13.71.23.55:8080/api/plugins/telemetry/DEVICE/"
            f"fa2bef00-ed29-11ef-8baf-55643890fc3e/values/timeseries"
            f"?keys={keys_param}&startTs={start_ts}&endTs={end_ts}&agg=NONE&orderBy=ASC&limit=10000"
        )

        try:
            api_response = fetch_data_from_api(final_url)
            dfs = []
            expected_keys = set(selected_keys)

            for key, values in api_response.items():
                if key not in expected_keys:
                    continue
                df = pd.DataFrame(values)
                df['timestamp'] = pd.to_datetime(df['ts'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
                df = df.rename(columns={'value': key})[['timestamp', key]]
                dfs.append(df)

            final_df = pd.DataFrame(columns=["timestamp"] + selected_keys) if not dfs else reduce(lambda l, r: pd.merge(l, r, on='timestamp', how='outer'), dfs)
            final_df = final_df.sort_values('timestamp')
            final_df.to_csv("output_dataframe.csv", index=False)

            natural_answer = generate_comprehensive_answer(user_query, final_df, selected_keys)

            return {
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "startTs": start_ts,
                "endTs": end_ts,
                "final_api": final_url,
                "api_response": final_df.to_dict(orient="records"),
                "natural_answer": natural_answer
            }

        except Exception as e:
            return {
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "startTs": start_ts,
                "endTs": end_ts,
                "final_api": final_url,
                "api_response": {"error": str(e)},
                "natural_answer": f"I encountered an issue while retrieving data: {str(e)}"
            }

    builder = StateGraph(state)
    builder.add_node("process", RunnableLambda(process_node))
    builder.set_entry_point("process")
    builder.set_finish_point("process")
    return builder.compile()

def run_twinbot(user_query: str) -> str:
    workflow = build_langgraph_workflow()
    result = workflow.invoke({"user_query": user_query})
    return result.get("natural_answer", "I couldn't process your request.")