import os
import jsonschema
import logging
import json
import filter_score
from enum import Enum
from pydantic import BaseModel, confloat
from typing import List
from datetime import datetime

from ollama import chat, ChatResponse


class Level(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Finding(BaseModel):
    vulnerability_types: List[str]
    likelihood: confloat(ge=0.0, le=1.0)
    confidence: Level
    rationale: List[str]
    remediation: List[str]
    severity: Level
    snippet_context: str


class FileReview(BaseModel):
    file_hash: str
    file_path: str
    language: str
    findings: List[Finding]
    # static_results: StaticResults, # TODO: when implemented bandit and/or semgrep
    timestamp: str


class Review(BaseModel):
    results: List[FileReview]
    vulnerability: Level
    most_vulnerable_files: List[str]
    timestamp: str


OLLAMA_CONFIG = {
    "OLLAMA_MODEL": "codellama",
    "OLLAMA_TEMPERATURE": 0.0
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def suffix_to_language(file_path: str) -> str:
    valid_languages = {
        "py": "Python",
        "r": "R"
    }

    return valid_languages.get(file_path.split(".")[-1], "Invalid")


def built_system_prompt() -> dict:
    llm_prompt = "You are a security analysis assistant and try to find possible vulnerabilities." \
                 "The code you will get is code to analyze data in a federated learning system." \
                 "The code will only be run in a encapsulated, internet-less environment and should only output results in a privacy preserving manner. " \
                 "Vulnerabilities you should detect are only such which endanger the data privacy " \
                 "by revealing patient-specific information to the hub of the federated learning system or by accidental leakage due to debug prints, logs, etc. " \
                 "Furthermore, you should recognize attempts to break out from the container or to disguise data. "\
                 "You do not need to flag code which uses patient-specific data to generate an output, if the output does not reveal any patient information"\
                 "Given one or multiple code files and their metadata, " \
                 "return a single JSON object with fields: "\
                 "vulnerability_types (list): list the types of vulnerabilities you recognize, "\
                 "likelihood (float between 0 and 1): how likely is it, that the problem you found is a risk, "\
                 "confidence ('low'|'medium'|'high'): how confident are you that the problem is a risk, "\
                 "rationale (1-3 bullets): explain why the problem could be a risk, "\
                 "remediation (1-3 bullets): what could be improved to lower the risk, " \
                 "snippet_context (one-line): what is the user trying to achieve with the code. "\
                 "NEVER provide exploit payloads, shell commands to exploit, or secrets. "\
                 "Only non-actionable, high-level remediation guidance is allowed. " \
                 "Output only valid JSON."

    return {"role": "system", "content": llm_prompt}


def built_user_prompt(metadata, snippet) -> dict:
    data = json.dumps({"metadata": metadata, "snippet": snippet}, ensure_ascii=False)

    return {"role": "user", "content": data}


def parse_llm_to_json(llm_response: str) -> dict:
    print(llm_response)
    json_start = llm_response.find("{")
    json_end = llm_response.rfind("}")

    if json_start == -1:
        logger.warning("No JSON found in llm response (No '{' found)")
        return {}
    if json_end == -1:
        logger.warning("No JSON found in llm response (No '}' found)")
        return {}

    json_string = llm_response[json_start:json_end+1]

    try:
        json_response = json.loads(json_string)
        return json_response
    except json.JSONDecodeError as e:
        logger.warning("Exception while loading json from response string.\n"
                       "Detected json-string: %s\n"
                       "Exception: %s", json_string, e.msg)
        return {}


def llm_json_to_findings(llm_json: dict) -> list:
    findings = []
    if len(llm_json) == 0:
        return []

    try:
        vulnerability = llm_json.get("vulnerability_types", [])
        likelihood = float(llm_json.get("likelihood", -1.0))
        confidence = llm_json.get("confidence", "none")
        rationale = llm_json.get("rationale", ["No rationale provided"])
        remediation = llm_json.get("remediation", ["No remediation provided"])
        severity = "high" if likelihood > 0.8 \
            else "medium" if likelihood > 0.5 \
            else "low" if likelihood >= 0.0 \
            else "unknown"
        snippet_context = llm_json.get("snippet_context", "No context provided")

        findings.append({
            "vulnerability_types": vulnerability,
            "likelihood": likelihood,
            "confidence": confidence,
            "rationale": rationale,
            "remediation": remediation,
            "severity": severity,
            "snippet_context": snippet_context
        })
    except ValueError as e:
        logger.warning("Exception while parsing llm json: %s\n"
                       "Probably caused by wrong json string from LLM", e)

    return findings


def main():
    # 1. Environment setup
    # TODO set paths and configs
    # filter_score.check_requirements()

    # 2. Read-in files
    # 2.1 Search for file paths of all python or r scripts
    file_paths = []
    initial_path = "debug_sus_files"  # <- TODO may be adapted
    for root, dirs, files in os.walk(initial_path):
        for filename in files:
            if suffix_to_language(filename.lower()) != "Invalid":
                file_paths.append(os.path.join(root, filename))
                logger.info("File found: %s", file_paths[-1])  # prints appended file_path
            else:
                logger.warning("Only python and r files are scanned for vulnerabilities: %s is none of these",
                               os.path.join(root, filename))

    llm_messages = [built_system_prompt()]

    # 2.2
    # 2.3 load file content and sanitize
    for file_path in file_paths:
        file = open(file_path)
        file_content = file.read()  # TODO? Gesamte File mitgeben oder nur Ausschnitt?

        sanitized_file_content = file_content.replace("\n", " ")  # TODO: replace with implemented sanitization

        # 3. Prompt LLM for each file
        language = suffix_to_language(file_path)
        metadata = {"file_path": file_path, "language": language}

        llm_messages.append(built_user_prompt(metadata, sanitized_file_content))


    response: ChatResponse = chat(model=OLLAMA_CONFIG.get("OLLAMA_MODEL"),
                                  messages=llm_messages,
                                  format=Finding.model_json_schema(),
                                  options={'temperature': OLLAMA_CONFIG.get("OLLAMA_TEMPERATURE")})

    json_response = parse_llm_to_json(response.message.content)
    findings = llm_json_to_findings(json_response)

    # 4. Format output to json
    result = {
        "file_hash": str(sanitized_file_content.__hash__()),
        "file_path": file_path,
        "language": language,
        "findings": findings,
        "timestamp": datetime.now().isoformat()
    }

    # validate json schema
    if jsonschema:
        try:
            jsonschema.validate(instance=result, schema=FileReview.model_json_schema())
        except Exception as e:
            logger.warning("Report validation failed: %s", e)

    # 5. Output formatted json result
    print("output json:")
    print(result)


if __name__ == '__main__':
    main()
