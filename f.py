from typing import Dict, List, Any
from datetime import datetime
from graphs import (
    detect_graph_request,
    extract_dates_from_query,
    normalize_dates,
    to_epoch_millis,
    select_fields_from_query,
    fetch_data_from_api,
    generate_graph,
    save_graph_as_file,
    generate_comprehensive_answer,
    FIELD_LIST
)
import pandas as pd
from functools import reduce


def handle_user_query(user_query: str) -> Dict[str, Any]:
    # Filter out meaningless or irrelevant queries
    junk_keywords = [
        "your name", "who are you", "hello", "hi", "joke", "weather", "movie",
        "how are you", "news", "game", "chat", "talk", "love"
    ]
    if any(word in user_query.lower() for word in junk_keywords) or len(user_query.strip()) < 5:
        return {
            "error": "Your question does not seem related to telemetry or pump data. Please ask a valid technical question."
        }

    # Step 1: Check if graph is needed
    graph_request = detect_graph_request(user_query)
    needs_graph = graph_request['needs_graph']
    graph_type = graph_request['graph_type']

    # Step 2: Extract date range
    dates = extract_dates_from_query(user_query)
    start_dt, end_dt = normalize_dates(dates["start_date"], dates["end_date"])
    start_ts = to_epoch_millis(start_dt)
    end_ts = to_epoch_millis(end_dt)

    # Step 3: Select relevant telemetry fields
    selected_keys = select_fields_from_query(user_query, FIELD_LIST)
    if not selected_keys:
        return {"error": "Could not determine relevant telemetry fields for your query."}

    keys_param = ",".join(selected_keys)

    # Step 4: Construct API call
    base_url = (
        "http://13.71.23.55:8080/api/plugins/telemetry/DEVICE/"
        "fa2bef00-ed29-11ef-8baf-55643890fc3e/values/timeseries"
    )
    api_url = f"{base_url}?keys={keys_param}&startTs={start_ts}&endTs={end_ts}&agg=NONE&orderBy=ASC&limit=10000"

    # Step 5: Fetch data
    try:
        raw_data = fetch_data_from_api(api_url)
        dfs = []

        for key, values in raw_data.items():
            df = pd.DataFrame(values)
            if df.empty:
                continue
            df['timestamp'] = pd.to_datetime(df['ts'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
            df = df.rename(columns={'value': key})[['timestamp', key]]
            dfs.append(df)

        if not dfs:
            return {"error": "No telemetry data available for the selected time range or fields."}

        final_df = reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='outer'), dfs)
        final_df = final_df.sort_values('timestamp')

    except Exception as e:
        return {"error": f"Failed to fetch or process data: {str(e)}"}

    # Step 6: Optionally generate graph
    graph_base64 = None
    graph_file = None
    if needs_graph:
        graph_base64 = generate_graph(final_df, selected_keys, graph_type, user_query)
        if graph_base64:
            graph_file = save_graph_as_file(graph_base64)

    # Step 7: Generate natural language summary
    graph_info = {
        "generated": graph_base64 is not None,
        "filename": graph_file
    }

    response = generate_comprehensive_answer(
        user_query=user_query,
        df=final_df,
        selected_fields=selected_keys,
        graph_info=graph_info
    )

    return {
        "user_query": user_query,
        "start": start_dt.strftime('%Y-%m-%d %H:%M:%S'),
        "end": end_dt.strftime('%Y-%m-%d %H:%M:%S'),
        "api_url": api_url,
        "fields": selected_keys,
        "graph_file": graph_file,
        "summary": response
    }


if __name__ == "__main__":
    print("=== Telemetry Graph Assistant ===")
    while True:
        query = input("Enter your query (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        result = handle_user_query(query)

        if "error" in result:
            print("\n[Error]", result["error"])
        else:
            print("\n=== Summary ===")
            print(result["summary"])
            if result.get("graph_file"):
                print(f"[Chart saved as: {result['graph_file']}]")
        print("\n----------------------\n")
