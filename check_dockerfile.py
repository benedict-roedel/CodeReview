#!/usr/bin/env python3
"""
check_dockerfile.py

Usage:
    python check_dockerfile.py path/to/Dockerfile

What it does:
    - Reads a Dockerfile
    - Extracts the base image from its FROM line
    - If it’s a Docker Hub image (docker.io), queries Docker Hub API for:
        * pull_count
        * star_count
        * last_updated (repo + tag)
        * is_official / is_private
        * description
    - def dockerfile_to_LLM_input() returns a JSON one can feed into the LLM (also contains the content of the file)
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DOCKERHUB_API_BASE = "https://hub.docker.com/v2"
DOCKERFILE_PATH = "Dockerfile"


def read_dockerfile(dockerfile_path: Path) -> str:
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
    return dockerfile_path.read_text(encoding="utf-8")


def extract_base_image(dockerfile_content: str) -> Optional[str]:
    """
    Returns the image reference from the FIRST non-comment FROM line.

    Examples it should handle:
        FROM python:3.10-slim
        FROM python:3.10-slim AS builder
        FROM library/python:3.10
        FROM myorg/myimage:1.2.3
    """
    for line in dockerfile_content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):  #If nothing or a comment in a line
            continue
        if stripped.upper().startswith("FROM "):
            # remove "FROM " and split by spaces to drop "AS builder" etc.
            parts = stripped.split()
            if len(parts) >= 2:
                return parts[1]  # the image ref
    return None #No FROM found

def parse_image_ref(ref: str) -> Dict[str, Optional[str]]:
    """
    Basic Docker image reference parser.

    Returns a dict with:
        registry: str or None (defaults to docker.io)
        namespace: str or None
        repo: str or None
        tag: str (defaults to 'latest')

    Examples:
        python:3.10      -> registry=docker.io, namespace=library, repo=python, tag=3.10
        library/python:3.10 -> registry=docker.io, namespace=library, repo=python, tag=3.10
        myorg/myimg      -> registry=docker.io, namespace=myorg, repo=myimg, tag=latest
        ghcr.io/org/img:1.2 -> registry=ghcr.io, namespace=org, repo=img, tag=1.2
    """
    registry = None
    remainder = ref

    # detect registry (if first component has a dot or colon, it's a registry)
    if "/" in ref:
        first, rest = ref.split("/", 1) # One split at /
        if "." in first or ":" in first or first == "localhost": # Registry if domain (e.g. .com), port (e.g. :5000) or localhost
            registry = first
            remainder = rest

    # split namespace/repo and tag
    if ":" in remainder:
        image_part, tag = remainder.rsplit(":", 1)
    else:
        image_part, tag = remainder, "latest" #Docker assumes latest, when nothing is specified.

    if "/" in image_part:
        namespace, repo = image_part.split("/", 1)
    else: # When not a registry, first component is treated as namespace
        # Official images on Docker Hub live under the 'library' namespace
        namespace, repo = "library", image_part

    if registry is None:
        registry = "docker.io"

    return {
        "registry": registry,
        "namespace": namespace,
        "repo": repo,
        "tag": tag,
    }


def get_dockerhub_metadata(namespace: str, repo: str, tag: str) -> Dict[str, Any]:
    """
    Fetch basic metadata for <namespace>/<repo>:<tag> from Docker Hub v2 API.

    Raises requests.HTTPError on HTTP failure.
    """
    # Repo-level metadata
    repo_url = f"{DOCKERHUB_API_BASE}/repositories/{namespace}/{repo}/"
    r_repo = requests.get(repo_url, timeout=10)
    r_repo.raise_for_status()
    repo_meta = r_repo.json()

    # Tag-level metadata
    tag_url = f"{DOCKERHUB_API_BASE}/repositories/{namespace}/{repo}/tags/{tag}/"
    r_tag = requests.get(tag_url, timeout=10)
    r_tag.raise_for_status()
    tag_meta = r_tag.json()

    return {
        # Some interesting informations about the image
        "image": f"{namespace}/{repo}:{tag}",
        "namespace": repo_meta.get("namespace"),
        "name": repo_meta.get("name"),
        "pull_count": repo_meta.get("pull_count"),
        "star_count": repo_meta.get("star_count"),
        "repo_last_updated": repo_meta.get("last_updated"),
        "tag_last_updated": tag_meta.get("last_updated") or tag_meta.get("tag_last_pushed"),
        "is_official": repo_meta.get("is_official"),
        "is_private": repo_meta.get("is_private"),
        "description": repo_meta.get("description"),
        #"images": tag_meta.get("images", []),
        # sometimes this points to the GitHub source repository
        "source_repo": repo_meta.get("source_repository"),
        # raw metadata could be useful for the LLM
        #"raw_repo": repo_meta,
        #"raw_tag": tag_meta,
    }

def heuristic_hints(meta: Dict[str, Any]) -> Tuple[str, str]: #Maybe useful for the LLM input?
    """
    Very rough heuristic to give a quick "feeling" for the image:

    Returns (rating, reason), where rating is one of:
        - "likely_trusted"
        - "popular_but_check_age"
        - "low_usage_or_unknown"
        - "unknown"
    """
    if not meta:
        return "unknown", "No Docker Hub metadata available."

    is_official = meta.get("is_official") is True
    pull_count = meta.get("pull_count") or 0
    star_count = meta.get("star_count") or 0

    # Arbitrary thresholds
    if is_official and pull_count > 1_000_000:
        return "likely_trusted", "Official image with high pull count."
    if pull_count > 100_000 and star_count > 10:
        return "popular_but_check_age", "Widely used community image; verify maintenance status."
    if pull_count < 10_000:
        return "low_usage_or_unknown", "Image has low pull count; may be less vetted or niche."

    return "unknown", "Metadata does not clearly indicate trust level."


def build_llm_input(
    dockerfile_content: str,
    base_image_ref: Optional[str],
    image_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Construct a JSON-serializable object with all info you might want to send to an LLM.
    """
    #hints_rating, hints_reason = heuristic_hints(image_meta or {})

    return {
        "dockerfile": dockerfile_content,
        "base_image": {
            "raw_ref": base_image_ref,
            "parsed": parse_image_ref(base_image_ref) if base_image_ref else None,
            "dockerhub_metadata": image_meta,
            #"heuristic": {
            #    "rating": hints_rating,
            #    "reason": hints_reason,
            #},
        },
    }


def dockerfile_to_LLM_input():

    dockerfile_path = Path(DOCKERFILE_PATH)

    content = read_dockerfile(dockerfile_path)

    base_image_ref = extract_base_image(content)
    if not base_image_ref:
        raise ValueError("Could not find a FROM line in the Dockerfile.")

    parsed = parse_image_ref(base_image_ref)

    logger.info("=== Dockerfile base image info ===")
    logger.info(f"  FROM:         {base_image_ref}")
    logger.info(f"  registry:     {parsed['registry']}")
    logger.info(f"  namespace:    {parsed['namespace']}")
    logger.info(f"  repo:         {parsed['repo']}")
    logger.info(f"  tag:          {parsed['tag']}")

    image_meta: Optional[Dict[str, Any]] = None

    # Only try Docker Hub API if registry is docker.io
    if parsed["registry"] == "docker.io":
        try:
            image_meta = get_dockerhub_metadata(
                namespace=parsed["namespace"],
                repo=parsed["repo"],
                tag=parsed["tag"],
            )
            logger.info("=== Docker Hub metadata (summary) ===")
            logger.info(f"  Image:            {image_meta['image']}")
            logger.info(f"  Official:         {image_meta['is_official']}")
            logger.info(f"  Private:          {image_meta['is_private']}")
            logger.info(f"  Pull count:       {image_meta['pull_count']}")
            logger.info(f"  Star count:       {image_meta['star_count']}")
            logger.info(f"  Repo updated:     {image_meta['repo_last_updated']}")
            logger.info(f"  Tag updated:      {image_meta['tag_last_updated']}")
            logger.info(f"  Description:      {image_meta['description']!r}")

        except requests.HTTPError as e:
            logger.warning("Failed to fetch Docker Hub metadata:", file=sys.stderr)
            logger.warning(f"  {e}", file=sys.stderr)
            logger.warning("Continuing without metadata.\n", file=sys.stderr)
            image_meta = None
    else:
        logger.info("Registry is not docker.io; skipping Docker Hub API lookup.\n")

    # Build object for LLM / further processing
    llm_input = build_llm_input(content, base_image_ref, image_meta)

    logger.info("=== JSON for LLM or further processing ===")
    logger.info(json.dumps(llm_input, indent=2))

    return json.dumps(llm_input, indent=2)

llm_input = dockerfile_to_LLM_input()