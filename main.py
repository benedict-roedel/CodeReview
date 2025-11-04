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


class JSONSchema(BaseModel):
    file_hash: str
    file_path: str
    language: str
    findings: List[Finding]
    # static_results: StaticResults, # TODO: when implemented bandit and/or semgrep
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


def built_llm_prompt(metadata: dict, snippet: str) -> str:
    llm_prompt = "You are a security analysis assistant. Given a sanitized code snippet and metadata, " \
                 "return a single JSON object with fields: vulnerability_types (list), likelihood (0-1), " \
                 "confidence ('low'|'medium'|'high'), rationale (1-3 bullets), remediation (1-3 bullets), " \
                 "snippet_context (one-line). NEVER provide exploit payloads, shell commands to exploit, " \
                 "or secrets. Only non-actionable, high-level remediation guidance is allowed. " \
                 "Output only valid JSON."

    data = json.dumps({"metadata": metadata, "snippet": snippet}, ensure_ascii=False)

    full_prompt = "\n".join([llm_prompt, "", "INPUT:", data, "", "Respond only with JSON"])
    return full_prompt


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
    filter_score.check_requirements()
    # 2. Read-in files
    # 2.1 Search for file paths of all python or r scripts
    file_paths = []  # TODO? Mehrere Files möglich oder nur jeweils eine analyse-file?
    initial_path = "sus_files"  # <- TODO may be adapted
    for root, dirs, files in os.walk(initial_path):
        for filename in files:
            if filename.lower().endswith(".py") or filename.lower().endswith(".r"):
                file_paths.append(os.path.join(root, filename))
                logger.info("File found: %s", file_paths[-1])  # prints appended file_path
            else:
                logger.warning("Only python and r files are scanned for vulnerabilities: %s is none of these",
                               os.path.join(root, filename))

    # 2.2
    # 2.3 load file content and sanitize
    for file_path in file_paths:
        file = open(file_path)
        file_content = file.read()  # TODO? Gesamte File mitgeben oder nur Ausschnitt?

        sanitized_file_content = file_content.replace("\n", " ")  # TODO: replace with implemented sanitization

        # 3. Prompt LLM for each file
        language = suffix_to_language(file_path)
        metadata = {"file_path": file_path, "language": language}

        response: ChatResponse = chat(model=OLLAMA_CONFIG.get("OLLAMA_MODEL"),
                                      messages=[{
                                          'role': 'user',
                                          'content': built_llm_prompt(metadata, sanitized_file_content)}],
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
                jsonschema.validate(instance=result, schema=JSONSchema.model_json_schema())
            except Exception as e:
                logger.warning("Report validation failed: %s", e)

        # 5. Output formatted json result
        print("output json:")
        print(result)


if __name__ == '__main__':
    main()
