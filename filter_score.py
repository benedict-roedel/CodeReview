"""
This uses libraries.io API to get scoring for packages.
Would be wise to have list of packages that are for sure trustworthy. E.g. Google Assured OSS
What should threshold be?
"""
import os
import requests

API_KEY = os.getenv("1db95a5a4e4ed23b6fd71c6ab42ad90d")
API_URL = "https://libraries.io/api/pypi/{}"

SOURCE_RANK_THRESHOLD = 15  # minimum trust threshold

def get_source_rank(package_name):
    url = API_URL.format(package_name)
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("rank", None)
    else:
        print(f"Failed to fetch {package_name}: {response.status_code}")
        return None

def parse_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                yield line.strip().split(" ==")[0]  # Only take package name without version

if __name__ == "__main__":
    print("Checking SourceRank scores...")
    for pkg in parse_requirements():
        score = get_source_rank(pkg)
        if score is None:
            print(f"{pkg}: Not found or error")
        elif score < SOURCE_RANK_THRESHOLD:
            print(f"{pkg}: Low trust (SourceRank {score})")
        else:
            print(f"{pkg}: OK (SourceRank {score})")
