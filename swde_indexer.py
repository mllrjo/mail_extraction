# File: swde_indexer.py
# Description: Smart SWDE indexer with automatic groundtruth format inference.

import os
import re
import json
import html
from tqdm import tqdm
from collections import defaultdict

# === CONFIG ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SWDE_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "swde"))
GT_BASE = os.path.join(SWDE_ROOT, "html_data", "groundtruth", "groundtruth")
SAVE_JSON = os.path.join(PROJECT_DIR, "swde_index.json")
DEBUG_SMALL_BATCH = True   # Limit initial run for safety

MAX_FILES = 20 if DEBUG_SMALL_BATCH else None

# === DYNAMIC FIELD DETECTION ===
def infer_domain_fields(gt_base):
    domain_fields = defaultdict(set)

    for domain in os.listdir(gt_base):
        domain_path = os.path.join(gt_base, domain)
        if not os.path.isdir(domain_path):
            continue

        for filename in os.listdir(domain_path):
            filepath = os.path.join(domain_path, filename)
            if not filename.endswith(".txt"):
                continue

            match = re.match(rf"{domain}-(.*?)-(.*)\.txt", filename)
            if match:
                website, field = match.group(1), match.group(2)
                domain_fields[domain].add(field)
            else:
                print(f"⚠️ Skipping unexpected groundtruth file: {filepath}")

    return domain_fields

# === SMART GROUNDTRUTH PARSER ===
def extract_groundtruth_label(filepath, target_page_id):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

            # Try to detect and skip metadata headers
            start_idx = 0
            if len(lines) >= 2:
                if not lines[0].strip()[0].isdigit() and not lines[1].strip()[0].isdigit():
                    start_idx = 2  # Likely header + stats line

            for line in lines[start_idx:]:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                page = parts[0].strip()
                if page != target_page_id:
                    continue

                # If 3+ columns, likely [pageID, indicator, text1, text2, ...]
                if len(parts) >= 3:
                    text_parts = parts[2:]  # Skip possible field indicator
                else:
                    text_parts = parts[1:]

                full_text = " ".join(text_parts)
                full_text = html.unescape(full_text).strip()

                return full_text

    except Exception as e:
        print(f"⚠️ Error reading {filepath}: {e}")

    return None

# === MAIN PARSER ===
def parse_swde_structure(swde_root, domain_fields):
    index = []
    html_base = os.path.join(swde_root, "html_data")

    for domain in os.listdir(html_base):
        domain_path = os.path.join(html_base, domain)
        if not os.path.isdir(domain_path):
            continue
        domain_subpath = os.path.join(domain_path, domain)
        if not os.path.isdir(domain_subpath):
            continue

        for website_folder in os.listdir(domain_subpath):
            website_path = os.path.join(domain_subpath, website_folder)
            if not os.path.isdir(website_path):
                continue

            match = re.match(rf"{domain}-(.*)\((\d{{4}})\)", website_folder)
            if not match:
                print(f"⚠️ Skipping unmatched folder: {website_folder}")
                continue

            website_name_raw, year = match.group(1), match.group(2)
            website_name = website_name_raw.lower().strip()

            for file in os.listdir(website_path):
                if not file.endswith(".htm"):
                    continue

                page_id = file.replace(".htm", "")

                record = {
                    "domain": domain,
                    "website": website_name,
                    "year": year,
                    "pageID": page_id,
                    "html": os.path.join("html_data", domain, domain, website_folder, file),
                    "groundtruth": {}
                }

                fields = domain_fields.get(domain, [])
                for field in fields:
                    gt_file = f"{domain}-{website_name}-{field}.txt"
                    gt_path = os.path.join("html_data", "groundtruth", "groundtruth", domain, gt_file)
                    full_gt_path = os.path.join(swde_root, gt_path)

                    if not os.path.exists(full_gt_path):
                        continue

                    label = extract_groundtruth_label(full_gt_path, page_id)
                    if label:
                        record["groundtruth"][field] = label

                index.append(record)

                if MAX_FILES and len(index) >= MAX_FILES:
                    return index

    return index

# === MAIN RUNNER ===
def main():
    print(f"📂 Indexing SWDE from {SWDE_ROOT}")

    domain_fields = infer_domain_fields(GT_BASE)
    print(f"✅ Detected fields for {len(domain_fields)} domains.")

    index = parse_swde_structure(SWDE_ROOT, domain_fields)
    print(f"✅ Indexed {len(index)} HTML pages.")

    with open(SAVE_JSON, "w") as f:
        json.dump(index, f, indent=2)

    print(f"✅ Saved index to {SAVE_JSON}")

if __name__ == "__main__":
    main()

