#!/usr/bin/env python
from pathlib import Path
import yaml

SKIPPABLE_PREFIXES = ("_lib", "_openmp_mutex")

def parse_conda_dep(spec: str) -> str:
    """
    Convert a conda-style dependency like:
        'numpy=1.25.0'
    into a pip-style requirement:
        'numpy==1.25.0'

    If no version is specified (e.g. 'tqdm'), it returns 'tqdm'.
    If a build string is present (e.g. 'numpy=1.25.0=py311_0'),
    it will ignore the build part and use only the version.
    """
    spec = spec.strip()
    if not spec:
        return ""

    parts = spec.split("=")
    name = parts[0]

    if len(parts) >= 2 and parts[1]:
        version = parts[1]
        return f"{name}=={version}"
    else:
        return name


def conda_yml_to_requirements(env_path: Path, out_path: Path) -> None:
    with env_path.open() as f:
        env = yaml.safe_load(f)

    deps = env.get("dependencies", [])
    requirements = []

    for dep in deps:
        if isinstance(dep, str):
            if dep.startswith(SKIPPABLE_PREFIXES): #to skip low level libs like _lib or _openmutex -> could also be handled in the whitelist
               continue
            req = parse_conda_dep(dep)
            if req:
                requirements.append("*" + req)
        elif isinstance(dep, dict) and "pip" in dep:
            # Pip section is already in requirements.txt syntax
            for p in dep["pip"]:
                p = p.strip()
                if p:
                    requirements.append(p)

    # Write out
    wrapper = out_path.open(mode="a+")
    wrapper.write("\n" + "\n".join(requirements) + "\n")
    print(f"Wrote {len(requirements)} requirements to {out_path}")

