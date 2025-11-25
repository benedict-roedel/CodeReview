import subprocess
from pathlib import Path
import zipfile, tarfile
import shutil
import logging
import requests
import tarfile


CONDA_TARBALL_DIR = Path("conda_pkgs")
AUDIT_DIR = Path("wheels")
DEST = Path("sus_files")

class CondaPackageNotFound(Exception):
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def _channel_base_url(channel: str) -> str:
    #defaults → Anaconda main
    if channel == "defaults":
        return "https://repo.anaconda.com/pkgs/main"
    return f"https://conda.anaconda.org/{channel}"

def _load_repodata(channel: str, subdir: str) -> dict:
    base = _channel_base_url(channel)
    url = f"{base}/{subdir}/repodata.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

def _find_best_match(repodata: dict, name: str, version: str) -> tuple[str, dict] | None: #Channels are provided by priority in .yml
    candidates: list[tuple[str, dict]] = []

    for fn, meta in repodata.get("packages", {}).items():
        if meta.get("name") == name and meta.get("version") == version:
            candidates.append((fn, meta))

    for fn, meta in repodata.get("packages.conda", {}).items():
        if meta.get("name") == name and meta.get("version") == version:
            candidates.append((fn, meta))

    if not candidates:
        return None

    # Highest build_number, then build string
    candidates.sort( #sorted by place in channel listing in .yml
        key=lambda x: (x[1].get("build_number", 0), str(x[1].get("build", ""))),
        reverse=True,
    )
    return candidates[0]

def download_conda_tarball_and_unzip(
    name: str,
    version: str,                                       
    channels: tuple[str, ...] = ("defaults",),
    subdirs: tuple[str, ...] = ("linux-64", "noarch"),
):
    """
    Download a conda package tarball for name==version from the given channels,
    unpack it into DEST (same as wheels), and remove the tarball.
    """
    CONDA_TARBALL_DIR.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(exist_ok=True)

    logger.info(f"Trying to fetch conda package {name}=={version} from channels={channels}")

    for channel in channels:
        base = _channel_base_url(channel)
        for subdir in subdirs:
            try:
                repodata = _load_repodata(channel, subdir)
            except requests.HTTPError:
                continue

            found = _find_best_match(repodata, name=name, version=version)
            if not found:
                continue

            fn, meta = found
            pkg_url = f"{base}/{subdir}/{fn}"
            tarball_path = CONDA_TARBALL_DIR / fn

            if not tarball_path.exists():
                logger.info(f"Downloading {pkg_url} → {tarball_path}")
                with requests.get(pkg_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with tarball_path.open("wb") as f:
                        shutil.copyfileobj(r.raw, f)

            # Unpack into sus_files/<name>-<version> (parallel to wheels behavior)
            outdir = DEST / f"{name}-{version}"
            outdir.mkdir(exist_ok=True)

            if fn.endswith((".tar.bz2", ".tar.gz", ".tar")):
                logger.info(f"Untarring {tarball_path.name} into {outdir}")
                with tarfile.open(tarball_path, "r:*") as t:
                    t.extractall(outdir)
            else:
                logger.warning(f"Unsupported conda package format: {fn}")

            # optional: keep or clean the tarball
            # tarball_path.unlink(missing_ok=True)
            return

    raise CondaPackageNotFound(f"Could not find conda package {name}=={version}")

def download_relevant_wheels_and_unzip(package: str):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {package} into {AUDIT_DIR}/ ...")
    cmd = [
        "python", "-m", "pip", "download",
        "--no-deps",
        "--dest", str(AUDIT_DIR),
        package,
    ]

    # Don't crash if pip exits with non-zero (bug where Rust missing for bed-reader)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            f"'pip download' exited with code {result.returncode} for {package}. "
            "Continuing anyway – inspecting whatever was downloaded."
        )
        logger.debug(f"pip stdout:\n{result.stdout}")
        logger.debug(f"pip stderr:\n{result.stderr}")
    else:
        logger.info(f"Packages downloaded to {AUDIT_DIR}/")

    # Check if we actually got any files
    files = [p for p in AUDIT_DIR.iterdir() if p.is_file()]
    if not files:
        logger.error(f"No archives found in {AUDIT_DIR} after attempting to download {package}.")
        return

    logger.info(" ######## Unzipping wheels/sdists to get code #########")
    DEST.mkdir(exist_ok=True)

    for pkg in files:
        name = pkg.stem.replace(".tar", "")
        outdir = DEST / name
        outdir.mkdir(exist_ok=True)

        if pkg.suffix == ".whl":
            logger.info(f"Unzipping {pkg.name}")
            with zipfile.ZipFile(pkg) as z:
                z.extractall(outdir)
        elif pkg.suffixes[-2:] in ([ ".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]):
            logger.info(f"Untarring {pkg.name}")
            with tarfile.open(pkg, "r:*") as t:
                t.extractall(outdir)
        else:
            logger.warning(f"Skipping unsupported file: {pkg.name}")

    logger.info(f"All archives unpacked into {DEST}/")
    shutil.rmtree(AUDIT_DIR, ignore_errors=True)
