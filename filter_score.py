"""
This uses libraries.io API to get scoring for packages.
Would be wise to have list of packages that are for sure trustworthy. E.g. Google Assured OSS
Also should handle yml properly
What should threshold be?
"""
import os
import time

import requests
import logging
import audit
import re

API_KEY = os.getenv("1db95a5a4e4ed23b6fd71c6ab42ad90d")
API_URL = "https://libraries.io/api/pypi/{}"

SOURCE_RANK_THRESHOLD = 15  # minimum trust threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONDA_PREFIXES = ( # cant be installed with pip
    "lib", "_lib", "llvm", "tk", "xz", "zstd", "pcre", "ncurses",
    "openssl", "sqlite", "yaml", "bzip2", "zlib",
)

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
    # allow optional leading "*" before the package name
    pattern = re.compile(r"^\s*\*?([a-zA-Z0-9_\-]+)")
    
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # skip blanks / comments
            if not line or line.startswith("#"):
                continue

            # (we don't expect "*" here, those come from pip deps)
            if line.startswith("git+"):
                pkg_name = line.split("/")[-1].split(".git")[0]
                yield line, pkg_name # Git repositories: keep full line as "pkg_version"
                continue

            # Regex: match package name at start (after optional "*")
            match = pattern.match(line)
            if match:
                pkg_name = match.group(1)
                yield line, pkg_name

def split_name_version(spec: str) -> tuple[str, str | None]:
    """
    Turn 'numpy==1.25.0' into ('numpy', '1.25.0').
    For 'tqdm' → ('tqdm', None).
    """
    spec = spec.strip()
    if spec.startswith("*"):
        spec = spec[1:].strip()

    # ignore markers, extras, etc. if they ever appear
    spec = spec.split(";")[0].strip()

    if "==" in spec:
        name, version = spec.split("==", 1)
        return name.strip(), version.strip()
    return spec.strip(), None


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
        is_conda = pkg_version.lstrip().startswith("*")

        # strip the marker for actual install spec
        clean_spec = pkg_version.lstrip()
        if is_conda:
            clean_spec = clean_spec[1:].strip()

        # Optional: still skip low-level things based on prefix
        if pkg_name.lower().startswith(CONDA_PREFIXES):
            logger.info(f"{pkg_name}: System/Conda package (prefix), skipping")
            continue

        # Whitelist
        if package_in_google_aoss(package_name=pkg_name):
            logger.info(f"{pkg_name}: Found in Whitelist")
            continue

        # Score via libraries.io (works for PyPI names, even if you later fetch via conda)
        logger.info(f"{pkg_name}: Not found in Whitelist, checking source score...")
        
        if is_conda: # No scoring for *
            score = None
        else:
            score = get_source_rank(pkg_name)

        # Decide if we need to download at all
        needs_download = (score is None) or (score < SOURCE_RANK_THRESHOLD)

        if not needs_download:
            logger.info(f"{pkg_name}: Not in Whitelist, but source score good ({score}), no checks necessary")
            continue

        # ======= download path depends on marker (*) =======
        if is_conda:
            logger.warning(f"{pkg_name}: Needs check, downloading via conda (marker '*').")
            name, version = split_name_version(clean_spec)
            if version is None:
                logger.warning(
                    f"{pkg_name}: No version in '{clean_spec}', cannot safely fetch conda package."
                )
                continue
            try:
                audit.download_conda_tarball_and_unzip(name, version)
            except Exception as e:
                logger.warning(f"{pkg_name}: Failed to fetch conda package: {e}")
        else:
            logger.warning(f"{pkg_name}: Needs check, downloading via pip.")
            audit.download_relevant_wheels_and_unzip(clean_spec)
