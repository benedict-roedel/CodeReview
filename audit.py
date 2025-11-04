import subprocess
from pathlib import Path
import zipfile, tarfile

DOCKER_FILE = Path("Dockerfile")
REQ_FILE = Path("requirements.txt")
AUDIT_DIR = Path("audit")
DEST = Path("audit_unpacked")

def download_relevant_wheels_and_unzip():
    if not REQ_FILE.exists():
        print(f"[!] requirements.txt not found at {REQ_FILE.resolve()}")
        return

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # read all non-comment lines
    pkgs = [
        line.strip()
        for line in REQ_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    # TODO: Only for packages that are not trustworthy!!! -> look into regex
    print(pkgs)

    if not pkgs:
        print("[!] No packages found in requirements.txt")
        return

    print(f"[*] Downloading {len(pkgs)} packages into {AUDIT_DIR}/ ...")
    cmd = [
        "python", "-m", "pip", "download",
        "--no-deps",
        "--dest", str(AUDIT_DIR),
        *pkgs,
    ]
    subprocess.run(cmd, check=True)
    print(f"Packages downloaded to {AUDIT_DIR}/")

    print(" ######## Unzipping wheels to get code #########")
    DEST.mkdir(exist_ok=True)

    for pkg in AUDIT_DIR.iterdir():
        if not pkg.is_file():
            continue
        name = pkg.stem.replace(".tar", "")
        outdir = DEST / name
        outdir.mkdir(exist_ok=True)

        if pkg.suffix == ".whl":
            print(f"[*] Unzipping {pkg.name}")
            with zipfile.ZipFile(pkg) as z:
                z.extractall(outdir)

        elif pkg.suffixes[-2:] == [".tar", ".gz"] or pkg.suffixes[-2:] == [".tar", ".bz2"] or pkg.suffixes[-2:] == [".tar", ".xz"]:
            print(f"[*] Untarring {pkg.name}")
            with tarfile.open(pkg, "r:*") as t:
                t.extractall(outdir)

        else:
            print(f"[!] Skipping unsupported file: {pkg.name}")
            print(f"All archives unpacked into {DEST}/")


def main():
    download_relevant_wheels_and_unzip()


if __name__ == "__main__":
    main()
