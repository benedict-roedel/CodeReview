"""
This uses libraries.io API to get scoring for packages.
Would be wise to have list of packages that are for sure trustworthy. E.g. Google Assured OSS
What should threshold be?
"""
import os
import time

import requests
import logging
import audit

API_KEY = os.getenv("1db95a5a4e4ed23b6fd71c6ab42ad90d")
API_URL = "https://libraries.io/api/pypi/{}"

SOURCE_RANK_THRESHOLD = 15  # minimum trust threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_source_rank(package_name):
    url = API_URL.format(package_name)
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("rank", None)
    else:
        logger.warning(f"Failed to fetch {package_name}: {response.status_code}")
        return None


def parse_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                yield line, line.strip().split(" ==")[0]  # Only take package name without version


def check_requirements(requirements_file_path="requirements.txt"):
    logger.info("Checking SourceRank scores...")
    for pkg_version, pkg_name in parse_requirements(requirements_file_path):
        score = get_source_rank(pkg_name)
        if score is None:
            logger.warning(f"{pkg_name}: Not found or error")
            audit.download_relevant_wheels_and_unzip(pkg_version)
        elif score < SOURCE_RANK_THRESHOLD:
            logger.info(f"{pkg_name}: Low trust (SourceRank {score})")
            audit.download_relevant_wheels_and_unzip(pkg_version)
        else:
            logger.info(f"{pkg_name}: OK (SourceRank {score})")
