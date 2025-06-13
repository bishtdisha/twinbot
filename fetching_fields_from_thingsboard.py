import requests
import datetime

# Config
base_url = "http://13.71.23.55:8080"
device_id = "fa2bef00-ed29-11ef-8baf-55643890fc3e"
username = "jwil@tenantadmin.com"
password = "cimcon@123"

# Step 1: Authenticate and get token
def get_token():
    url = f"{base_url}/api/auth/login"
    resp = requests.post(url, json={"username": username, "password": password})
    resp.raise_for_status()
    return resp.json().get("token")

# Step 2: Get telemetry keys
def get_telemetry_keys(token):
    url = f"{base_url}/api/plugins/telemetry/DEVICE/{device_id}/keys/timeseries"
    headers = {"X-Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

# Step 3: Filter keys updated in June 2025
def get_june_updated_keys(keys, token):
    headers = {"X-Authorization": f"Bearer {token}"}
    june_start = int(datetime.datetime(2025, 6, 1).timestamp() * 1000)
    june_end = int(datetime.datetime(2025, 6, 30, 23, 59, 59).timestamp() * 1000)
    
    updated_keys = {}

    for key in keys:
    # Skip placeholder/template keys
        if "_X_" in key or key.endswith("_X"):
            continue

        url = f"{base_url}/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
        params = {
            "keys": key,
            "limit": 1,
            "orderBy": "DESC"
        }
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 200 and key in resp.json():
            last_ts = resp.json()[key][0]['ts']
            if june_start <= last_ts <= june_end:
                updated_keys[key] = None

    return updated_keys

# Main
def main():
    try:
        token = get_token()
        keys = get_telemetry_keys(token)
        updated_keys_dict = get_june_updated_keys(keys, token)

        print(f"Total keys updated in June: {len(updated_keys_dict)}\n")
        print("Updated keys dictionary:")
        print(updated_keys_dict)

    except Exception as e:
        print(f"[❌] Error occurred: {e}")

if __name__ == "__main__":
    main()