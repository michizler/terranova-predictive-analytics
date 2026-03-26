import requests
import csv
import os
import time

BASE_URL = "https://www.fema.gov/api/open"

ENDPOINTS = {
    "declarations": f"{BASE_URL}/v2/DisasterDeclarationsSummaries",
    "public_assistance": f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails",
    "disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries",
}

COLUMNS = {
    "declarations": [
        "disasterNumber",
        "state",
        "incidentType",
        "declarationDate",
        "incidentBeginDate",
        "incidentEndDate",
        "declarationType",
    ],
    "public_assistance": [
        "pwNumber",
        "disasterNumber",
        "stateNumberCode",
        "incidentType",
        "totalObligated",
        "projectSize",
    ],
    "disaster_summaries": [
        "disasterNumber",
        "totalAmountIhpApproved",
        "totalAmountHaApproved",
        "totalAmountOnaApproved",
        "totalObligatedAmountPa",
        "totalObligatedAmountHmgp",
    ],
}

DATE_FILTERS = {
    "declarations": "$filter=declarationDate ge '2010-01-01T00:00:00.000z'",
    "public_assistance": "$filter=declarationDate ge '2010-01-01T00:00:00.000z'",
    "disaster_summaries": "",  # no date field — filter client-side
}

SKIP_FETCH = {"declarations", "public_assistance", "disaster_summaries"}  # already fetched

OUTPUT_DIR = "fema_data"
TOP = 1000


def load_valid_disaster_numbers():
    filepath = os.path.join(OUTPUT_DIR, "declarations.csv")
    numbers = set()
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            numbers.add(int(row["disasterNumber"]))
    print(f"Loaded {len(numbers)} disaster numbers from declarations.csv")
    return numbers


def fetch_all_records(name: str, endpoint: str, date_filter: str) -> list[dict]:
    records = []
    skip = 0
    max_retries = 5

    while True:
        if date_filter:
            url = f"{endpoint}?{date_filter}&$top={TOP}&$skip={skip}"
        else:
            url = f"{endpoint}?$top={TOP}&$skip={skip}"

        print(f"  [{name}] Fetching skip={skip} ...")

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if resp.status_code in (503, 429) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(
                        f"  [{name}] Got {resp.status_code}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                else:
                    raise

        data = resp.json()

        entity_key = next(
            (k for k in data if k not in ("metadata",) and isinstance(data[k], list)),
            None,
        )
        if entity_key is None:
            break

        batch = data[entity_key]
        if not batch:
            break

        records.extend(batch)
        skip += TOP
        time.sleep(1)

    print(f"  [{name}] Total records fetched: {len(records)}")
    return records


def write_csv(name: str, records: list[dict], columns: list[str]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{name}.csv")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({col: rec.get(col, "") for col in columns})

    print(f"  [{name}] Saved {len(records)} rows → {filepath}")
    return filepath


def main():
    valid_numbers = load_valid_disaster_numbers()

    for name in ENDPOINTS:
        if name in SKIP_FETCH:
            print(f"\n  [{name}] Already fetched — skipping.")
            continue

        print(f"\n{'='*50}")
        print(f"Processing: {name}")
        print(f"{'='*50}")

        records = fetch_all_records(name, ENDPOINTS[name], DATE_FILTERS[name])

        # Client-side filter for endpoints without a date field
        if not DATE_FILTERS[name] and records:
            before = len(records)
            records = [r for r in records if r.get("disasterNumber") in valid_numbers]
            print(f"  [{name}] Filtered {before} → {len(records)} records (matched to 2010+ declarations)")

        if records:
            write_csv(name, records, COLUMNS[name])
        else:
            print(f"  [{name}] No records returned — skipping CSV.")


if __name__ == "__main__":
    main()