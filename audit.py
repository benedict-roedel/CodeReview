import subprocess
import time
from pathlib import Path
import zipfile, tarfile
import shutil
import logging


AUDIT_DIR = Path("wheels")
DEST = Path("sus_files")


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_relevant_wheels_and_unzip(package: str):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {package} into {AUDIT_DIR}/ ...")
    cmd = [
        "python", "-m", "pip", "download",
        "--no-deps",
        "--dest", str(AUDIT_DIR),
        package,
    ]
    subprocess.run(cmd, check=True)
    logger.info(f"Packages downloaded to {AUDIT_DIR}/")

    logger.info(" ######## Unzipping wheels to get code #########")
    DEST.mkdir(exist_ok=True)

    for pkg in AUDIT_DIR.iterdir():
        if not pkg.is_file():
            continue
        name = pkg.stem.replace(".tar", "")
        outdir = DEST / name
        outdir.mkdir(exist_ok=True)

        if pkg.suffix == ".whl":
            logger.info(f"Unzipping {pkg.name}")
            with zipfile.ZipFile(pkg) as z:
                z.extractall(outdir)
        elif pkg.suffixes[-2:] == [".tar", ".gz"] or pkg.suffixes[-2:] == [".tar", ".bz2"] or pkg.suffixes[-2:] == [".tar", ".xz"]:
            logger.info(f"Untarring {pkg.name}")
            with tarfile.open(pkg, "r:*") as t:
                t.extractall(outdir)
        else:
            logger.warning(f"Skipping unsupported file: {pkg.name}")
        logger.info(f"All archives unpacked into {DEST}/")
    shutil.rmtree(AUDIT_DIR, ignore_errors=True)

# def main():
#     download_relevant_wheels_and_unzip()
#
#
# if __name__ == "__main__":
#     main()
