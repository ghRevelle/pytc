# Data scraper for ADS-B Exchange historical data

import concurrent.futures
import requests
import json
import gzip
from io import BytesIO
from datetime import datetime, timedelta
import csv

target_icaos = [] # This will be the list of ICAOs you want to filter
OUTPUT_CSV = "filtered_aircraft.csv" # Rename as needed

fieldnames = [
    "timestamp",
    "hex",
    "type",
    "flight",
    "r",
    "t",
    "alt_baro",
    "gs",
    "track",
    "baro_rate",
    "squawk",
    "category",
    "lat",
    "lon",
    "nic",
    "rc",
    "seen_pos",
    "messages",
    "seen",
    "rssi"
]

def fetch_and_process(timestamp):
    ts = timestamp.strftime("%H%M%S")
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    url = f"https://samples.adsbexchange.com/readsb-hist/2024/07/01/{ts}Z.json.gz" # Adjust URL as needed for date and time
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            print(f"{timestamp_str}: Failed to fetch ({response.status_code})")
            return []

        raw_bytes = response.raw.read()
        if raw_bytes[:2] == b'\x1f\x8b':
            with gzip.open(BytesIO(raw_bytes), 'rt') as f:
                data = json.load(f)
        else:
            data = json.loads(raw_bytes.decode('utf-8'))

        aircraft = data.get("aircraft", [])
        filtered = [a for a in aircraft if a.get("hex", "").lower() in target_icaos]

        # Prepare rows for CSV writing
        rows = []
        for a in filtered:
            rows.append({
                "timestamp": timestamp_str,
                "hex": a.get("hex", ""),
                "type": a.get("type", ""),
                "flight": a.get("flight", "").strip(),
                "r": a.get("r", ""),
                "t": a.get("t", ""),
                "alt_baro": a.get("alt_baro", None),
                "gs": a.get("gs", None),
                "track": a.get("track", None),
                "baro_rate": a.get("baro_rate", None),
                "squawk": a.get("squawk", ""),
                "category": a.get("category", ""),
                "lat": a.get("lat", None),
                "lon": a.get("lon", None),
                "nic": a.get("nic", None),
                "rc": a.get("rc", None),
                "seen_pos": a.get("seen_pos", None),
                "messages": a.get("messages", None),
                "seen": a.get("seen", None),
                "rssi": a.get("rssi", None)
            })
        print(f"{timestamp_str}: fetched {len(filtered)} aircraft")
        return rows

    except Exception as e:
        print(f"{timestamp_str}: Error {e}")
        return []

def main():
    start_time = datetime(2024, 7, 1, 0, 0, 0)
    end_time = datetime(2024, 7, 1, 23, 59, 55)  # full day, every 5 seconds
    delta = timedelta(seconds=5)

    timestamps = []
    current = start_time
    while current <= end_time:
        timestamps.append(current)
        current += delta

    with open(OUTPUT_CSV, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            for rows in executor.map(fetch_and_process, timestamps):
                for row in rows:
                    writer.writerow(row)

if __name__ == "__main__":
    main()