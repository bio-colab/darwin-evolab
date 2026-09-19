import json
import urllib.request
from pathlib import Path

key_file = Path(__file__).parent / "API key for test.txt"
api_key = key_file.read_text(encoding="utf-8").strip()

buggy_code = """def parse_cli(args):
    config = {'port': 8000, 'debug': False, 'host': '127.0.0.1'}
    for arg in args:
        if arg == '--debug':
            config['debug'] = False
        elif arg.startswith('--port='):
            config['port'] = arg.split('=')[1]
        elif arg.startswith('--host='):
            config['host'] = arg.split('=')[0]
    return config"""

test_failure = "Test failed: parse_cli(['--port=9090']) returned {'port': '9090'}, expected {'port': 9090}. Value is string instead of int."

payload = {
    "state": {
        "source_code": buggy_code,
        "test_failure": test_failure
    },
    "model": "jev-latest",
    "questions": {
        "best_operator": {
            "type": "choice",
            "instructions": "Which AST mutation operator is needed to fix this specific failure?",
            "criteria": {
                "int_wrap": "Wrap subscript or expression with int(...) conversion",
                "bool_flip": "Invert boolean literal (True <-> False)",
                "index_flip": "Change indexing 0 <-> 1",
                "insert_guard": "Add defensive if condition"
            }
        },
        "suspicious_line": {
            "type": "choice",
            "instructions": "Which line of code is directly responsible for this failure?",
            "criteria": {
                "line_5": "config['debug'] = False",
                "line_7": "config['port'] = arg.split('=')[1]",
                "line_9": "config['host'] = arg.split('=')[0]"
            }
        }
    }
}

req = urllib.request.Request(
    "https://api.typesafe.ai/v1/systemone",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    data=json.dumps(payload).encode("utf-8")
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode("utf-8"))
    print("STATUS:", resp.status)
    print("ANSWERS:", json.dumps(result["answers"], indent=2))
    print("USAGE:", result["usage"])
