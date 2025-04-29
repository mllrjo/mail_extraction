# File: build_domain_fields.py
# Description: Dynamically infer DOMAIN_FIELDS dictionary based on SWDE groundtruth directory structure.

import os
import re
from collections import defaultdict

# === CONFIG ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SWDE_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "swde"))
GT_BASE = os.path.join(SWDE_ROOT, "html_data", "groundtruth", "groundtruth")

# === FUNCTION ===
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

            # Match pattern: domain-website-field.txt
            match = re.match(rf"{domain}-(.*?)-(.*)\.txt", filename)
            if match:
                website, field = match.group(1), match.group(2)
                domain_fields[domain].add(field)
            else:
                print(f"⚠️ Skipping unexpected file: {filepath}")

    return domain_fields

# === MAIN ===
def main():
    print(f"📂 Inferring domain fields from {GT_BASE}")
    domain_fields = infer_domain_fields(GT_BASE)

    print("\n✅ DOMAIN_FIELDS:")
    print("{")
    for domain, fields in sorted(domain_fields.items()):
        fields_sorted = sorted(fields)
        fields_str = ", ".join(f'"{field}"' for field in fields_sorted)
        print(f'    "{domain}": [{fields_str}],')
    print("}")

if __name__ == "__main__":
    main()

