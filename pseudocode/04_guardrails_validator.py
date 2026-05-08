import os
import re
import json
from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.validators import PassResult, FailResult

# ─── Validator A: PII Detector ──────────────────────────────────────────────

@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    def __init__(self, on_fail=OnFailAction.FIX, **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)
        # Regex patterns
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        # US Phone: (123) 456-7890 or 123-456-7890
        self.phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        # SSN: 123-45-6789
        self.ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        # Credit Card: 16 digits
        self.cc_pattern = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

    def validate(self, value: str, metadata: dict):
        original_value = value
        
        # Redact patterns
        value = self.email_pattern.sub("[REDACTED]", value)
        value = self.phone_pattern.sub("[REDACTED]", value)
        value = self.ssn_pattern.sub("[REDACTED]", value)
        value = self.cc_pattern.sub("[REDACTED]", value)
        
        if value != original_value:
            return FailResult(
                error_message="PII detected and redacted",
                fix_value=value
            )
        return PassResult()

# ─── Validator B: JSON Formatter ─────────────────────────────────────────────

@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    def __init__(self, on_fail=OnFailAction.FIX, **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)

    def validate(self, value: str, metadata: dict):
        repaired = value.strip()
        
        # 1. Strip markdown fences
        if repaired.startswith("```json"):
            repaired = repaired[7:]
        elif repaired.startswith("```"):
            repaired = repaired[3:]
        if repaired.endswith("```"):
            repaired = repaired[:-3]
        repaired = repaired.strip()
        
        # 2. Try to parse, if fails, try to repair
        try:
            json.loads(repaired)
            return PassResult(value_override=repaired)
        except json.JSONDecodeError:
            try:
                # Fix single quotes to double quotes
                temp = repaired.replace("'", '"')
                # Remove trailing commas before closing braces/brackets
                temp = re.sub(r',\s*}', '}', temp)
                temp = re.sub(r',\s*\]', ']', temp)
                
                json.loads(temp)
                return PassResult(value_override=temp)
            except json.JSONDecodeError as e:
                # Repair failed
                fix_val = json.dumps({
                    "error": "Failed to repair JSON",
                    "raw": value
                })
                return FailResult(
                    error_message=f"Invalid JSON: {e}",
                    fix_value=fix_val
                )

# ─── Tests ──────────────────────────────────────────────────────────────────

def test_pii():
    print("\n=== Testing PII Detector ===")
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    
    test_cases = [
        "This is a clean string with no PII.",
        "Contact me at test@example.com for more info.",
        "My phone number is 123-456-7890.",
        "SSN is 123-45-6789.",
        "Card number is 1234-5678-9012-3456.",
        "Mix: Call 123-456-7890 or email test@example.com."
    ]
    
    for case in test_cases:
        print(f"\nInput:  {case}")
        result = guard.validate(case)
        print(f"Passed: {result.validation_passed}")
        print(f"Output: {result.validated_output}")

def test_json():
    print("\n=== Testing JSON Formatter ===")
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    
    test_cases = [
        '{"name": "John", "age": 30}', # Valid
        '```json\n{"name": "John", "age": 30}\n```', # Fenced
        "{'name': 'John', 'age': 30}", # Single quotes
        '{"name": "John", "age": 30,}', # Trailing comma
        '{"name": "John", broken' # Broken
    ]
    
    for case in test_cases:
        print(f"\nInput:  {case!r}")
        result = guard.validate(case)
        print(f"Passed: {result.validation_passed}")
        print(f"Output: {result.validated_output}")

if __name__ == "__main__":
    test_pii()
    test_json()
