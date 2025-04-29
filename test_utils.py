# File: test_utils.py
# Description: Runs modular tests for utils.py with isolated environment flags.

import subprocess

TEST_CASES = {
    "TEST_LABEL_MAPPING": "Label Mapping",
    "TEST_DOM_FEATURES": "DOM Feature Extraction",
    "TEST_GRAPH_MAKE": "Graph Construction",
    "TEST_TINY_GNN": "TinyGNN Model",
    "TEST_DATASET": "GNNDataset (pt only)",
    "TEST_DATASET_GZ": "GNNDataset (pt + pt.gz)",
}

def run_test(env_var):
    print(f"🧪 Running test: {TEST_CASES[env_var]} ({env_var})")
    result = subprocess.run(
        ["python", "utils.py"],
        env={env_var: "1", **dict(os.environ)},
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✅ PASS: {TEST_CASES[env_var]}")
    else:
        print(f"❌ FAIL: {TEST_CASES[env_var]}")
        print("--- stdout ---")
        print(result.stdout.strip())
        print("--- stderr ---")
        print(result.stderr.strip())

if __name__ == "__main__":
    import os
    print("🔍 Starting utils.py test suite...\n")
    for env_var in TEST_CASES:
        run_test(env_var)
        print()
    print("🧪 All tests complete.")

