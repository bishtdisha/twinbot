from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Dict, List, Any
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
from twinbot_prompts import DATE_EXTRACTION_PROMPT, FIELD_SELECTION_PROMPT, get_nlg_prompt

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

# --- Step 1: Extract Dates with Fallback ---
def extract_dates_from_query(user_query: str) -> dict:
    """
    Extracts start and end dates from the user query using rules and Gemini fallback.
    Returns a dictionary with UTC-formatted datetime strings.
    If no time period is mentioned, defaults to the latest data point.
    """
    query_lower = user_query.lower()
    current_time = datetime.utcnow()

    # 1. Handle phrases like "last 7 days"
    days_match = re.search(r'(last|past)\s+(\d+)\s+days?', query_lower)
    if days_match:
        num_days = int(days_match.group(2))
        start_date = (current_time - timedelta(days=num_days)).replace(hour=0, minute=0, second=0)
        end_date = current_time.replace(hour=23, minute=59, second=59)
        return {
            "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }

    # 2. Handle specific common keywords
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
    elif re.search(r'(today|yesterday|last|past|week|month|days?)', query_lower):
        # fallback for any other time-related word
        # let LLM handle it
        try:
            import time
            time.sleep(0.5)
            prompt = DATE_EXTRACTION_PROMPT.format(user_query=user_query, current_utc=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
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
    else:
        # No time period mentioned: default to latest data point
        # Set start and end date to current time (or a very small window)
        latest_time = current_time
        # Use a 5-minute window to ensure at least one data point is fetched
        start_date = (latest_time - timedelta(minutes=5)).replace(second=0, microsecond=0)
        end_date = latest_time.replace(second=59, microsecond=999999)
        return {
            "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }

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
    """
    Fixes dateparser issues like assigning year 1900.
    Returns datetime objects at UTC 00:00:00 and 23:59:59.
    """
    current_year = datetime.utcnow().year
    start_dt = dateparser.parse(start_str)
    end_dt = dateparser.parse(end_str)

    if not start_dt or start_dt.year == 1900:
        start_dt = datetime.utcnow().replace(hour=0, minute=0, second=0)
    if not end_dt or end_dt.year == 1900:
        end_dt = datetime.utcnow().replace(hour=23, minute=59, second=59)

    return start_dt.replace(hour=0, minute=0, second=0), end_dt.replace(hour=23, minute=59, second=59)

def to_epoch_millis(dt):
    return int(dt.timestamp() * 1000)

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
        'tail_end_pressure': ['tail end pressure', 'tail-end pressure', '3_pressure', 'sensor 3'],
        'outlet_pressure': ['outlet pressure', 'outlet_pressure', 'pressure at outlet', 'outlet press'],
        'inlet_pressure': ['inlet pressure', 'inlet_pressure', 'pressure at inlet', 'inlet press'],
        'pump_1': ['pump_1', 'pump1', 'pump 1', 'first pump', 'main pump 1'],
        'pump_2': ['pump_2', 'pump2', 'pump 2', 'second pump', 'main pump 2'],
        'pump_3': ['pump_3', 'pump3', 'pump 3', 'third pump', 'main pump 3'],
        'voltage': ['voltage', 'vry', 'vyb', 'vbr', 'volt'],
        'power': ['power', 'watt', 'kw', 'energy', 'consumption', 'kwh', 'load'],
        'temperature': ['temp', 'temperature', 'heat', 'thermal'],
        'pressure': ['pressure', 'press', 'bar', 'psi', 'head'],
        'flow': ['flow', 'rate', 'gpm', 'lpm', 'flowrate', 'water flow', 'water rate', 'discharge', 'inlet flow', 'outlet flow'],
        'frequency': ['freq', 'frequency', 'hz', 'run frequency', 'operating frequency'],
        'speed': ['speed', 'rpm', 'rotation'],
        'running': ['running', 'status', 'state', 'on', 'off', 'active', 'operating', 'working'],
        'motor': ['motor', 'em', 'engine'],
        'vibration': ['vibration', 'vib'],
        'level': ['level', 'tank', 'reservoir', 'water level', 'liquid level'],
        'chlorine': ['chlorine', 'cl', 'chlorination', 'chlorine level', 'inlet chlorine', 'outlet chlorine', 'tail end chlorine'],
        'water': ['water', 'water flow', 'water pressure', 'water level'],
    }
    
    # Special case: if user query contains 'outlet 1' and 'pressure', map to '1_Pressure'
    if ('outlet 1' in query_lower or 'outlet-1' in query_lower or 'outlet1' in query_lower) and 'pressure' in query_lower:
        for field in field_list:
            if field['key'].lower() == '1_pressure' or 'outlet pt-1' in (field['description'] or '').lower():
                return [{'key': field['key'], 'user_term': 'outlet 1 pressure'}]

    # Special case: if user query contains 'outlet 2' and 'pressure', map to '2_Pressure'
    if ('outlet 2' in query_lower or 'outlet-2' in query_lower or 'outlet2' in query_lower) and 'pressure' in query_lower:
        for field in field_list:
            if field['key'].lower() == '2_pressure' or 'outlet pt-2' in (field['description'] or '').lower():
                return [{'key': field['key'], 'user_term': 'outlet 2 pressure'}]

    # Special case: if user query contains 'tail end pressure', always map to '3_Pressure'
    if 'tail end pressure' in query_lower or 'tail-end pressure' in query_lower:
        for field in field_list:
            if field['key'].lower() == '3_pressure' or 'tail-end pressure' in (field['description'] or '').lower():
                return [{'key': field['key'], 'user_term': 'tail end pressure'}]

    # Special case: if user asks for energy consumption by any pump, prioritize dailyConsumption fields
    if (('energy' in query_lower or 'consumption' in query_lower) and 'pump' in query_lower):
        energy_fields = []
        for field in field_list:
            key_lower = field['key'].lower()
            if 'dailyconsumption' in key_lower or ('kwh' in key_lower and 'pump' in key_lower):
                energy_fields.append({'key': field['key'], 'user_term': 'daily energy consumption'})
        if energy_fields:
            return energy_fields

    # Special case: if user query is about total water supplied today
    if (('total water supplied' in query_lower or 'how much water supplied' in query_lower or 'water supplied today' in query_lower or ('water' in query_lower and 'supplied' in query_lower and 'today' in query_lower))
        and not any(x in query_lower for x in ['outlet 1', 'outlet-1', 'outlet1', 'pump 1', 'pump-1', 'pump1', 'outlet 2', 'outlet-2', 'outlet2', 'pump 2', 'pump-2', 'pump2'])):
        for field in field_list:
            if field['key'].lower() == 'dpr_total_water':
                return [{'key': field['key'], 'user_term': 'total water supplied today'}]

    # Special case: water supplied from outlet 1 or pump 1
    if (('water supplied' in query_lower or 'water delivered' in query_lower or 'water from' in query_lower or 'water at' in query_lower) and (
        'outlet 1' in query_lower or 'outlet-1' in query_lower or 'outlet1' in query_lower or 'pump 1' in query_lower or 'pump-1' in query_lower or 'pump1' in query_lower)):
        for field in field_list:
            if field['key'].lower() == 'pump_1_water_supplied':
                return [{'key': field['key'], 'user_term': 'water supplied from outlet 1'}]

    # Special case: water supplied from outlet 2 or pump 2
    if (('water supplied' in query_lower or 'water delivered' in query_lower or 'water from' in query_lower or 'water at' in query_lower) and (
        'outlet 2' in query_lower or 'outlet-2' in query_lower or 'outlet2' in query_lower or 'pump 2' in query_lower or 'pump-2' in query_lower or 'pump2' in query_lower)):
        for field in field_list:
            if field['key'].lower() == 'pump_2_water_supplied':
                return [{'key': field['key'], 'user_term': 'water supplied from outlet 2'}]

    # Special case: if user query is about total energy supplied today
    if (('total energy supplied' in query_lower or 'how much energy supplied' in query_lower or 'energy supplied today' in query_lower or ('energy' in query_lower and 'supplied' in query_lower and 'today' in query_lower))
        and not any(x in query_lower for x in ['pump 1', 'pump-1', 'pump1', 'pump 2', 'pump-2', 'pump2'])):
        for field in field_list:
            if field['key'].lower() == 'dpr_total_energy':
                return [{'key': field['key'], 'user_term': 'total energy supplied today'}]

    # Special case: energy consumed from pump 1
    if (('energy consumed' in query_lower or 'energy used' in query_lower or 'energy from' in query_lower or 'energy by' in query_lower or 'energy at' in query_lower) and (
        'pump 1' in query_lower or 'pump-1' in query_lower or 'pump1' in query_lower)):
        for field in field_list:
            if field['key'].lower() == 'dailyconsumption':
                return [{'key': field['key'], 'user_term': 'energy consumed from pump 1'}]

    # Special case: energy consumed from pump 2
    if (('energy consumed' in query_lower or 'energy used' in query_lower or 'energy from' in query_lower or 'energy by' in query_lower or 'energy at' in query_lower) and (
        'pump 2' in query_lower or 'pump-2' in query_lower or 'pump2' in query_lower)):
        for field in field_list:
            if field['key'].lower() == 'dailyconsumptionn':
                return [{'key': field['key'], 'user_term': 'energy consumed from pump 2'}]

    # Special case: if user query is about total energy consumed today (robust synonyms)
    if (
        (
            'total energy consumed' in query_lower or
            'how much energy consumed' in query_lower or
            'energy consumed today' in query_lower or
            ('energy' in query_lower and 'consumed' in query_lower and 'today' in query_lower) or
            ('energy' in query_lower and 'today' in query_lower and 'consumed' in query_lower) or
            ('energy' in query_lower and 'today' in query_lower and 'used' in query_lower) or
            ('energy' in query_lower and 'used' in query_lower and 'today' in query_lower) or
            ('energy' in query_lower and 'today' in query_lower)
        )
        and not any(x in query_lower for x in ['pump 1', 'pump-1', 'pump1', 'pump 2', 'pump-2', 'pump2'])
    ):
        # Prefer DPR_Total_Energy if present
        for field in field_list:
            if field['key'].lower() == 'dpr_total_energy':
                return [{'key': field['key'], 'user_term': 'total energy consumed today'}]
        # Fallback: sum dailyConsumption and dailyConsumptionn if both present
        keys = []
        for field in field_list:
            if field['key'].lower() == 'dailyconsumption':
                keys.append({'key': field['key'], 'user_term': 'energy consumed by pump 1 today'})
            if field['key'].lower() == 'dailyconsumptionn':
                keys.append({'key': field['key'], 'user_term': 'energy consumed by pump 2 today'})
        if keys:
            return keys

    # Special case: if user query contains 'tail end chlorine', map to '2_Chlorine'
    if 'tail end chlorine' in query_lower or 'tail-end chlorine' in query_lower or 'chlorine at tail end' in query_lower or 'chlorine at tail-end' in query_lower or 'chlorine 2 meter' in query_lower or '2 chlorine' in query_lower or 'chlorine 2' in query_lower:
        for field in field_list:
            if field['key'].lower() == '2_chlorine' or 'tail end chlorine' in (field['description'] or '').lower():
                return [{'key': field['key'], 'user_term': 'tail end chlorine'}]
    # Special case: if user query contains 'inlet chlorine', map to '1_Chlorine'
    if 'inlet chlorine' in query_lower or 'chlorine at inlet' in query_lower:
        for field in field_list:
            if field['key'].lower() == '1_chlorine' or 'inlet chlorine' in (field['description'] or '').lower():
                return [{'key': field['key'], 'user_term': 'inlet chlorine'}]

    # Special case: flow rate of water from outlet 1
    if (
        ('flow rate' in query_lower or 'water flow' in query_lower or 'flow of water' in query_lower or 'flowrate' in query_lower or 'flow rate of water' in query_lower) and
        ('outlet 1' in query_lower or 'outlet-1' in query_lower or 'outlet1' in query_lower)
    ):
        for field in field_list:
            if field['key'].lower() == '1_flow_m3h':
                return [{'key': field['key'], 'user_term': 'flow rate of water from outlet 1'}]
    # Special case: flow rate of water from outlet 2
    if (
        ('flow rate' in query_lower or 'water flow' in query_lower or 'flow of water' in query_lower or 'flowrate' in query_lower) and
        ('outlet 2' in query_lower or 'outlet-2' in query_lower or 'outlet2' in query_lower)
    ):
        for field in field_list:
            if field['key'].lower() == '2_flow_m3h':
                return [{'key': field['key'], 'user_term': 'flow rate of water from outlet 2'}]
    # Special case: flow rate of water from inlet
    if (
        ('flow rate' in query_lower or 'water flow' in query_lower or 'flow of water' in query_lower or 'flowrate' in query_lower) and
        ('inlet' in query_lower)
    ):
        for field in field_list:
            if field['key'].lower() == '3_flow_m3h':
                return [{'key': field['key'], 'user_term': 'flow rate of water from inlet'}]

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
        prompt = FIELD_SELECTION_PROMPT.format(
            user_query=user_query,
            field_list=json.dumps(field_list, indent=2),
            keyword_mappings=json.dumps(list(keyword_mappings.items()), indent=2)
        )
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, generation_config={
            "temperature": 0.1,
            "max_output_tokens": 200
        })
        raw = response.text.strip()
        api_result = json.loads(raw) if raw.startswith("[") else json.loads(re.search(r"\[.*?\]", raw, re.DOTALL).group())
        return api_result
    except Exception:
        pass  # Silently fallback with no console output

        
        # Ultimate fallback: return fields that contain query keywords
        fallback_fields = []
        for field in field_list:
            # Check if any word from query appears in field key or description
            query_words = [word for word in query_lower.split() if len(word) > 2]
            for word in query_words:
                field_desc = field.get("description") or ""  # Handle None descriptions
                if (word in field["key"].lower() or word in field_desc.lower()) and field["key"] not in fallback_fields:
                    fallback_fields.append(field["key"])
        
        # If still no matches, return empty list instead of random fields
        return fallback_fields if fallback_fields else []

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
def generate_comprehensive_answer(user_query: str, df: pd.DataFrame, selected_fields: list, user_term_map: dict = None) -> str:
    """
    Generate a focused, concise answer based only on the selected fields and user query.
    The answer should be a clear, single sentence (not too detailed, not too simple).
    Always append the correct SI unit after any numerical value, based on the field type.
    """
    user_term_map = user_term_map or {}
    if len(df) == 0:
        return "No data found for your query. Please check the time period or field names."
    # Only keep selected fields (plus timestamp)
    fields_to_keep = ['timestamp'] + [f for f in selected_fields if f in df.columns]
    df = df[fields_to_keep]

    query_lower = user_query.lower()
    # Special logic for pump running/working queries
    if (
        ('which pump' in query_lower or 'pump is running' in query_lower or 'pump is working' in query_lower or 'pump is on' in query_lower or 'which pump is on' in query_lower or 'which pump is working' in query_lower or 'which pump is running' in query_lower or 'which pump running' in query_lower or 'which pump working' in query_lower or 'which pump on' in query_lower)
        and not any(x in query_lower for x in ['pump 1', 'pump-1', 'pump1', 'pump 2', 'pump-2', 'pump2'])
    ):
        # Check both Pump_1_On and Pump_2_On if present
        running_pumps = []
        for pump_key, pump_name in [('Pump_1_On', 'Pump 1'), ('Pump_2_On', 'Pump 2')]:
            if pump_key in df.columns:
                latest_row = df.dropna(subset=[pump_key]).sort_values('timestamp').iloc[-1]
                if str(latest_row[pump_key]) == '1':
                    running_pumps.append(pump_name)
        if running_pumps:
            return f"Currently running: {', '.join(running_pumps)}."
        else:
            return "No pumps are currently running."
    # If user asks for a specific pump
    for pump_num, pump_key in [('1', 'Pump_1_On'), ('2', 'Pump_2_On')]:
        if any(x in query_lower for x in [f'pump {pump_num}', f'pump-{pump_num}', f'pump{pump_num}']) and pump_key in df.columns:
            latest_row = df.dropna(subset=[pump_key]).sort_values('timestamp').iloc[-1]
            if str(latest_row[pump_key]) == '1':
                return f"Pump {pump_num} is currently running."
            else:
                return f"Pump {pump_num} is currently stopped."
    # Default: previous logic
    main_field = next((f for f in selected_fields if f in df.columns and f != 'timestamp'), None)
    if not main_field:
        return "No relevant data found for your query."
    # Get the latest value for the main field
    latest_row = df.dropna(subset=[main_field]).sort_values('timestamp').iloc[-1]
    latest_val = latest_row[main_field]
    user_term = user_term_map.get(main_field, main_field.replace('_', ' '))

    # Determine SI unit based on field key or user_term
    def get_si_unit(field_key, user_term):
        key = field_key.lower()
        term = user_term.lower()
        if any(x in key for x in ['pressure']) or 'pressure' in term:
            return 'bar'
        if any(x in key for x in ['flow_m3h']) or 'flow rate' in term:
            return 'm^3/h'
        if any(x in key for x in ['water_supplied', 'totalizer', 'dpr_total_water', 'actualdailyconsumption', 'total_distributed_water', 'total_incoming_water', 'dpr_total_water']) or 'water' in term:
            return 'm^3'
        if any(x in key for x in ['energy', 'consumption', 'kwh', 'dailyconsumption', 'monthlyconsumption', 'weeklyconsumption']) or 'energy' in term:
            return 'kWh'
        if 'chlorine' in key or 'chlorine' in term:
            return 'ppm'
        if 'level' in key or 'sump' in key or 'tank' in key or 'reservoir' in key or 'lakh litre' in term:
            return 'lakh litres'
        return ''

    si_unit = get_si_unit(main_field, user_term)
    # Compose a concise, clear answer with SI unit
    if si_unit:
        return f"The current {user_term} is {latest_val} {si_unit}."
    else:
        return f"The current {user_term} is {latest_val}."

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

    # Replace technical error messages with a user-friendly message
    USER_FRIENDLY_NO_DATA_MSG = "Sorry, I couldn't find any data to answer your question. Please check your query or try a different time period."

    def process_node(input: Dict[str, Any]) -> Dict[str, Any]:
        user_query = input["user_query"]

        # Dates
        result = extract_dates_from_query(user_query)
        start_dt, end_dt = normalize_dates(result["start_date"], result["end_date"])
        start_ts = to_epoch_millis(start_dt)
        end_ts = to_epoch_millis(end_dt)

        # Fields
        api_field_result = select_fields_from_query(user_query, FIELD_LIST)

        # If the result is a list of lists (multi-intent), flatten to first group
        if api_field_result and isinstance(api_field_result[0], list):
            api_field_result = api_field_result[0]

        # Check if no fields were selected
        if not api_field_result:
            return {
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "startTs": start_ts,
                "endTs": end_ts,
                "final_api": "No API call made - no matching fields found",
                "api_response": {},
                "natural_answer": USER_FRIENDLY_NO_DATA_MSG
            }

        # Handle key-user_term mapping
        if isinstance(api_field_result[0], dict) and 'key' in api_field_result[0]:
            selected_keys = [item['key'] for item in api_field_result]
            user_term_map = {
                item['key']: item.get('user_term', FIELD_METADATA.get(item['key'], item['key']))
                for item in api_field_result
            }
        else:
            selected_keys = api_field_result
            user_term_map = {
                key: FIELD_METADATA.get(key, key)
                for key in selected_keys
            }

        keys_param = ",".join(selected_keys)

        # API URL
        base_url = (
            "http://13.71.23.55:8080/api/plugins/telemetry/DEVICE/"
            "fa2bef00-ed29-11ef-8baf-55643890fc3e/values/timeseries"
        )

        # Estimate high data limit for fast updates (1s/5s/etc.)
        duration_seconds = (end_dt - start_dt).total_seconds()
        dynamic_limit = min(max(int(duration_seconds * max(len(selected_keys), 1)), 1000), 100000)

        final_url = f"{base_url}?keys={keys_param}&startTs={start_ts}&endTs={end_ts}&agg=NONE&orderBy=ASC&limit={dynamic_limit}"

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

            if not dfs:
                final_df = pd.DataFrame(columns=["timestamp"] + selected_keys)
            else:
                final_df = reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='outer'), dfs)

            final_df = final_df.sort_values('timestamp')
            final_df.to_csv("output_dataframe.csv", index=False)

            if final_df.empty:
                raise Exception("API returned no data, falling back to CSV.")

            # Natural language answer generation
            natural_answer = generate_comprehensive_answer(user_query, final_df, selected_keys, user_term_map)

            if not natural_answer or 'no data found' in natural_answer.lower() or 'no relevant data' in natural_answer.lower():
                natural_answer = USER_FRIENDLY_NO_DATA_MSG

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
            # In case of complete failure
            return {
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "startTs": start_ts,
                "endTs": end_ts,
                "final_api": final_url,
                "api_response": {},
                "natural_answer": f"Error fetching data: {str(e)}"
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