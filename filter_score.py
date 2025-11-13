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

def package_in_google_aoss(package_name):
    """
    Check if a package exactly matches one in google_aoss_python.txt.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is listed, False otherwise.
    """
    name = package_name.strip().lower()
    path = "google_aoss_python.txt"

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip().lower() == name:
                    return True
    except FileNotFoundError:
        logger.error(f"Package list file not found: {path}")
        raise  # re-raise the actual file error

    return False



def check_requirements(requirements_file_path="requirements.txt"):
    logger.info("Checking SourceRank scores...")
    for pkg_version, pkg_name in parse_requirements(requirements_file_path):
        # For using googles whitelist
        if package_in_google_aoss(package_name=pkg_name):
            logger.info(f"{pkg_name}: Found in Whitelist") #continue with next package
        else:
            # if not in whitelist, see if in libraries.io
            logger.info(f"{pkg_name}: Not found in Whitelist, checking source score...")
            score = get_source_rank(pkg_name)
            if score is None:
                logger.warning(f"{pkg_name}: Source score not found, downloading library")
                audit.download_relevant_wheels_and_unzip(pkg_version)
            elif score < SOURCE_RANK_THRESHOLD:
                logger.warning(f"{pkg_name}: Source score bad, downloading library")
                audit.download_relevant_wheels_and_unzip(pkg_version)
            else:
                #Source rank of file not in whitelist is fine
                logger.info(f"{pkg_name}: Source score good, no checks necessary")