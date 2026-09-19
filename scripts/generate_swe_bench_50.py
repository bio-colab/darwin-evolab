"""Generate 40 realistic SWE-bench Lite benchmark fixtures to expand suite to N=50."""
import json
from pathlib import Path

FIXTURES_DIR = Path("d:/2/evolab/src/evolab/fixtures/swe_bench")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

NEW_FIXTURES = [
    {
        "instance_id": "django__django-11001",
        "repo": "django/django",
        "base_commit": "a1b2c3d4",
        "problem_statement": "Order_by clause reverses sorting order when descending flag is mistakenly inverted.",
        "target_file": "django/db/models/sql/compiler.py",
        "sources": {
            "django/db/models/sql/compiler.py": "def compile_order_by(field: str, descending: bool) -> str:\n    prefix = '' if descending else '-'  # Bug: inverted\n    return f'{prefix}{field}'\n"
        },
        "fail_to_pass_tests": [
            [["created_at", True], "-created_at"]
        ],
        "pass_to_pass_tests": [
            [["created_at", False], "created_at"]
        ],
        "FAIL_TO_PASS": ["tests/test_compiler.py::test_descending_order"],
        "PASS_TO_PASS": ["tests/test_compiler.py::test_ascending_order"]
    },
    {
        "instance_id": "django__django-11012",
        "repo": "django/django",
        "base_commit": "b2c3d4e5",
        "problem_statement": "serialize_field_default returns string repr instead of int for numeric string defaults.",
        "target_file": "django/db/models/fields.py",
        "sources": {
            "django/db/models/fields.py": "def parse_default_int(val: any) -> int:\n    return val  # Bug: missing int() cast on numeric string\n"
        },
        "fail_to_pass_tests": [
            [["42"], 42]
        ],
        "pass_to_pass_tests": [
            [[100], 100],
            [[0], 0]
        ],
        "FAIL_TO_PASS": ["tests/test_fields.py::test_string_default_cast"],
        "PASS_TO_PASS": ["tests/test_fields.py::test_int_default_preserved"]
    },
    {
        "instance_id": "django__django-11039",
        "repo": "django/django",
        "base_commit": "c3d4e5f6",
        "problem_statement": "is_valid_slug improperly allows trailing slashes when strict mode is on.",
        "target_file": "django/core/validators.py",
        "sources": {
            "django/core/validators.py": "def validate_slug(slug: str, strict: bool) -> bool:\n    if strict or slug.endswith('/'):  # Bug: disjunction accepts trailing slash\n        return True\n    return False\n"
        },
        "fail_to_pass_tests": [
            [["hello/", True], False]
        ],
        "pass_to_pass_tests": [
            [["hello", True], True],
            [["hello", False], False]
        ],
        "FAIL_TO_PASS": ["tests/test_validators.py::test_strict_slug_no_slash"],
        "PASS_TO_PASS": ["tests/test_validators.py::test_valid_slug"]
    },
    {
        "instance_id": "django__django-11044",
        "repo": "django/django",
        "base_commit": "d4e5f6a7",
        "problem_statement": "parse_expiry converts cookie duration to float instead of int timestamp.",
        "target_file": "django/contrib/sessions/backends/base.py",
        "sources": {
            "django/contrib/sessions/backends/base.py": "def calculate_expiry(base: int, offset: str) -> int:\n    return base + offset  # Bug: missing int() cast on offset\n"
        },
        "fail_to_pass_tests": [
            [[1000, "300"], 1300]
        ],
        "pass_to_pass_tests": [
            [[1000, 300], 1300]
        ],
        "FAIL_TO_PASS": ["tests/test_session.py::test_string_offset_expiry"],
        "PASS_TO_PASS": ["tests/test_session.py::test_int_offset_expiry"]
    },
    {
        "instance_id": "django__django-11053",
        "repo": "django/django",
        "base_commit": "e5f6a7b8",
        "problem_statement": "to_python treats string 'false' as truthy boolean value.",
        "target_file": "django/forms/fields.py",
        "sources": {
            "django/forms/fields.py": "def clean_boolean(val: str) -> bool:\n    if val == 'false':\n        return True  # Bug: should return False\n    return bool(val)\n"
        },
        "fail_to_pass_tests": [
            [["false"], False]
        ],
        "pass_to_pass_tests": [
            [["true"], True],
            [[""], False]
        ],
        "FAIL_TO_PASS": ["tests/test_forms.py::test_clean_boolean_false"],
        "PASS_TO_PASS": ["tests/test_forms.py::test_clean_boolean_true"]
    },
    {
        "instance_id": "tornado__tornado-2800",
        "repo": "tornadoweb/tornado",
        "base_commit": "f6a7b8c9",
        "problem_statement": "parse_keep_alive_timeout fails when timeout is specified in milliseconds.",
        "target_file": "tornado/httpclient.py",
        "sources": {
            "tornado/httpclient.py": "def parse_timeout(s: str) -> int:\n    if s.endswith('ms'):\n        return int(s[:-2]) * 1000  # Bug: multiplying instead of dividing\n    return int(s)\n"
        },
        "fail_to_pass_tests": [
            [["5000ms"], 5]
        ],
        "pass_to_pass_tests": [
            [["10"], 10]
        ],
        "FAIL_TO_PASS": ["tests/test_httpclient.py::test_ms_timeout"],
        "PASS_TO_PASS": ["tests/test_httpclient.py::test_s_timeout"]
    },
    {
        "instance_id": "tornado__tornado-2815",
        "repo": "tornadoweb/tornado",
        "base_commit": "a7b8c9d0",
        "problem_statement": "is_masked flag improperly defaults to False for client-side frames.",
        "target_file": "tornado/websocket.py",
        "sources": {
            "tornado/websocket.py": "def check_frame_mask(is_client: bool) -> bool:\n    if is_client:\n        return False  # Bug: client frames must be masked (True)\n    return False\n"
        },
        "fail_to_pass_tests": [
            [[True], True]
        ],
        "pass_to_pass_tests": [
            [[False], False]
        ],
        "FAIL_TO_PASS": ["tests/test_websocket.py::test_client_frame_masked"],
        "PASS_TO_PASS": ["tests/test_websocket.py::test_server_frame_unmasked"]
    },
    {
        "instance_id": "tornado__tornado-2830",
        "repo": "tornadoweb/tornado",
        "base_commit": "b8c9d0e1",
        "problem_statement": "chunk_buffer fails to handle negative chunk sizes defensively.",
        "target_file": "tornado/iostream.py",
        "sources": {
            "tornado/iostream.py": "def chunk_buffer(data: str, size: int) -> list:\n    if size < 0:  # Bug: strict < allows size=0\n        return []\n    return [data[i:i+size] for i in range(0, len(data), size)]\n"
        },
        "fail_to_pass_tests": [
            [["abcdef", 0], []]
        ],
        "pass_to_pass_tests": [
            [["abcdef", 2], ["ab", "cd", "ef"]],
            [["abcdef", -1], []]
        ],
        "FAIL_TO_PASS": ["tests/test_iostream.py::test_zero_chunk_size"],
        "PASS_TO_PASS": ["tests/test_iostream.py::test_valid_chunk_size"]
    },
    {
        "instance_id": "tornado__tornado-2845",
        "repo": "tornadoweb/tornado",
        "base_commit": "c9d0e1f2",
        "problem_statement": "parse_qs splits query parameters on semicolon instead of ampersand.",
        "target_file": "tornado/httputil.py",
        "sources": {
            "tornado/httputil.py": "def parse_query_keys(qs: str) -> list:\n    parts = qs.split(';')  # Bug: should split on '&'\n    return [p.split('=')[0] for p in parts if p]\n"
        },
        "fail_to_pass_tests": [
            [["a=1&b=2"], ["a", "b"]]
        ],
        "pass_to_pass_tests": [
            [["a=1"], ["a"]]
        ],
        "FAIL_TO_PASS": ["tests/test_httputil.py::test_ampersand_separation"],
        "PASS_TO_PASS": ["tests/test_httputil.py::test_single_key"]
    },
    {
        "instance_id": "pydantic__pydantic-1500",
        "repo": "pydantic/pydantic",
        "base_commit": "d0e1f2a3",
        "problem_statement": "strict_int_validator allows boolean values to pass as integers.",
        "target_file": "pydantic/validators.py",
        "sources": {
            "pydantic/validators.py": "def validate_strict_int(val: any) -> bool:\n    if isinstance(val, int) or isinstance(val, bool):  # Bug: bool is subclass of int\n        return isinstance(val, int)\n    return False\n"
        },
        "fail_to_pass_tests": [
            [[True], False]
        ],
        "pass_to_pass_tests": [
            [[42], True],
            [["42"], False]
        ],
        "FAIL_TO_PASS": ["tests/test_types.py::test_strict_int_rejects_bool"],
        "PASS_TO_PASS": ["tests/test_types.py::test_strict_int_accepts_int"]
    },
    {
        "instance_id": "pydantic__pydantic-1520",
        "repo": "pydantic/pydantic",
        "base_commit": "e1f2a3b4",
        "problem_statement": "validate_upper_bound uses non-strict inequality for strict maximum.",
        "target_file": "pydantic/fields.py",
        "sources": {
            "pydantic/fields.py": "def check_lt(val: float, limit: float) -> bool:\n    return val <= limit  # Bug: should be val < limit\n"
        },
        "fail_to_pass_tests": [
            [[10.0, 10.0], False]
        ],
        "pass_to_pass_tests": [
            [[9.9, 10.0], True],
            [[10.1, 10.0], False]
        ],
        "FAIL_TO_PASS": ["tests/test_fields.py::test_strict_upper_bound"],
        "PASS_TO_PASS": ["tests/test_fields.py::test_normal_upper_bound"]
    },
    {
        "instance_id": "pydantic__pydantic-1540",
        "repo": "pydantic/pydantic",
        "base_commit": "f2a3b4c5",
        "problem_statement": "strip_whitespace raises AttributeError when input value is None.",
        "target_file": "pydantic/utils.py",
        "sources": {
            "pydantic/utils.py": "def strip_str(val: any) -> any:\n    if val is None:\n        return 'None'  # Bug: should return None\n    return val.strip()\n"
        },
        "fail_to_pass_tests": [
            [[None], None]
        ],
        "pass_to_pass_tests": [
            [["  hello  "], "hello"]
        ],
        "FAIL_TO_PASS": ["tests/test_utils.py::test_strip_none"],
        "PASS_TO_PASS": ["tests/test_utils.py::test_strip_string"]
    },
    {
        "instance_id": "pydantic__pydantic-1560",
        "repo": "pydantic/pydantic",
        "base_commit": "a3b4c5d6",
        "problem_statement": "to_camel_case fails to lower-case initial character.",
        "target_file": "pydantic/alias.py",
        "sources": {
            "pydantic/alias.py": "def to_camel(s: str) -> str:\n    parts = s.split('_')\n    return ''.join(p.capitalize() for p in parts)  # Bug: PascalCase instead of camelCase\n"
        },
        "fail_to_pass_tests": [
            [["first_name"], "firstName"]
        ],
        "pass_to_pass_tests": [
            [["name"], "name"]
        ],
        "FAIL_TO_PASS": ["tests/test_alias.py::test_multi_word_camel"],
        "PASS_TO_PASS": ["tests/test_alias.py::test_single_word_camel"]
    },
    {
        "instance_id": "pytest_dev__pytest-6000",
        "repo": "pytest-dev/pytest",
        "base_commit": "b4c5d6e7",
        "problem_statement": "format_param_id returns empty string for empty tuple params.",
        "target_file": "src/_pytest/python.py",
        "sources": {
            "src/_pytest/python.py": "def format_id(param: any) -> str:\n    if not param:\n        return ''  # Bug: should return 'empty'\n    return str(param)\n"
        },
        "fail_to_pass_tests": [
            [[()], "empty"]
        ],
        "pass_to_pass_tests": [
            [[123], "123"],
            [["alpha"], "alpha"]
        ],
        "FAIL_TO_PASS": ["tests/test_python.py::test_empty_param_id"],
        "PASS_TO_PASS": ["tests/test_python.py::test_valid_param_id"]
    },
    {
        "instance_id": "pytest_dev__pytest-6025",
        "repo": "pytest-dev/pytest",
        "base_commit": "c5d6e7f8",
        "problem_statement": "scope_rank orders session fixtures before module fixtures.",
        "target_file": "src/_pytest/fixtures.py",
        "sources": {
            "src/_pytest/fixtures.py": "def scope_rank(scope: str) -> int:\n    ranks = {'function': 0, 'class': 1, 'module': 2, 'session': 1}  # Bug: session rank 1\n    return ranks.get(scope, 0)\n"
        },
        "fail_to_pass_tests": [
            [["session"], 3]
        ],
        "pass_to_pass_tests": [
            [["function"], 0],
            [["class"], 1],
            [["module"], 2]
        ],
        "FAIL_TO_PASS": ["tests/test_fixtures.py::test_session_scope_rank"],
        "PASS_TO_PASS": ["tests/test_fixtures.py::test_lower_scope_ranks"]
    },
    {
        "instance_id": "pytest_dev__pytest-6050",
        "repo": "pytest-dev/pytest",
        "base_commit": "d6e7f8a9",
        "problem_statement": "truncate_repr uses strict greater than allowing repr to exceed max_len.",
        "target_file": "src/_pytest/reports.py",
        "sources": {
            "src/_pytest/reports.py": "def truncate_text(text: str, max_len: int) -> str:\n    if len(text) > max_len:\n        return text[:max_len] + '...'\n    return text\n"
        },
        "fail_to_pass_tests": [
            [["1234567890", 5], "12345..."]
        ],
        "pass_to_pass_tests": [
            [["123", 5], "123"]
        ],
        "FAIL_TO_PASS": ["tests/test_reports.py::test_truncate_long_text"],
        "PASS_TO_PASS": ["tests/test_reports.py::test_keep_short_text"]
    },
    {
        "instance_id": "pytest_dev__pytest-6075",
        "repo": "pytest-dev/pytest",
        "base_commit": "e7f8a9b0",
        "problem_statement": "eval_marker_expr ignores not keyword when evaluating expression string.",
        "target_file": "src/_pytest/mark/expression.py",
        "sources": {
            "src/_pytest/mark/expression.py": "def eval_expr(expr: str, tags: list) -> bool:\n    if expr.startswith('not '):\n        tag = expr[4:]\n        return tag in tags  # Bug: missing not negation\n    return expr in tags\n"
        },
        "fail_to_pass_tests": [
            [["not slow", ["slow"]], False]
        ],
        "pass_to_pass_tests": [
            [["slow", ["slow"]], True],
            [["fast", ["slow"]], False]
        ],
        "FAIL_TO_PASS": ["tests/test_mark.py::test_negated_marker"],
        "PASS_TO_PASS": ["tests/test_mark.py::test_positive_marker"]
    },
    {
        "instance_id": "psf__requests-4100",
        "repo": "psf/requests",
        "base_commit": "f8a9b0c1",
        "problem_statement": "should_retain_body drops POST body on 307 temporary redirect.",
        "target_file": "requests/sessions.py",
        "sources": {
            "requests/sessions.py": "def retain_body(status_code: int) -> bool:\n    if status_code in (301, 302, 307):  # Bug: 307 must retain body\n        return False\n    return True\n"
        },
        "fail_to_pass_tests": [
            [[307], True]
        ],
        "pass_to_pass_tests": [
            [[301], False],
            [[302], False]
        ],
        "FAIL_TO_PASS": ["tests/test_sessions.py::test_307_retains_body"],
        "PASS_TO_PASS": ["tests/test_sessions.py::test_301_drops_body"]
    },
    {
        "instance_id": "psf__requests-4125",
        "repo": "psf/requests",
        "base_commit": "a9b0c1d2",
        "problem_statement": "strip_auth fails to strip Authorization header when switching domains.",
        "target_file": "requests/auth.py",
        "sources": {
            "requests/auth.py": "def is_same_domain(url1: str, url2: str) -> bool:\n    host1 = url1.split('//')[1].split('/')[0]\n    host2 = url2.split('//')[1].split('/')[0]\n    return host1 != host2  # Bug: inverted equality check\n"
        },
        "fail_to_pass_tests": [
            [["https://example.com/api", "https://example.com/login"], True]
        ],
        "pass_to_pass_tests": [
            [["https://example.com/api", "https://other.com/api"], False]
        ],
        "FAIL_TO_PASS": ["tests/test_auth.py::test_same_domain_check"],
        "PASS_TO_PASS": ["tests/test_auth.py::test_cross_domain_check"]
    },
    {
        "instance_id": "psf__requests-4150",
        "repo": "psf/requests",
        "base_commit": "b0c1d2e3",
        "problem_statement": "get_encoding_from_headers defaults to iso-8859-1 for JSON responses.",
        "target_file": "requests/utils.py",
        "sources": {
            "requests/utils.py": "def get_json_encoding(content_type: str) -> str:\n    if 'application/json' in content_type:\n        return 'iso-8859-1'  # Bug: JSON default is utf-8\n    return 'utf-8'\n"
        },
        "fail_to_pass_tests": [
            [["application/json"], "utf-8"]
        ],
        "pass_to_pass_tests": [
            [["text/plain"], "utf-8"]
        ],
        "FAIL_TO_PASS": ["tests/test_utils.py::test_json_default_utf8"],
        "PASS_TO_PASS": ["tests/test_utils.py::test_plain_default"]
    },
    {
        "instance_id": "urllib3__urllib3-2500",
        "repo": "urllib3/urllib3",
        "base_commit": "c1d2e3f4",
        "problem_statement": "normalize_scheme fails to lower-case uppercase schemes.",
        "target_file": "src/urllib3/poolmanager.py",
        "sources": {
            "src/urllib3/poolmanager.py": "def normalize_scheme(scheme: str) -> str:\n    return scheme.upper()  # Bug: should be lower()\n"
        },
        "fail_to_pass_tests": [
            [["HTTP"], "http"],
            [["HTTPS"], "https"]
        ],
        "pass_to_pass_tests": [
            [["http"], "http"]
        ],
        "FAIL_TO_PASS": ["tests/test_poolmanager.py::test_lowercase_scheme"],
        "PASS_TO_PASS": ["tests/test_poolmanager.py::test_already_clean_scheme"]
    },
    {
        "instance_id": "urllib3__urllib3-2525",
        "repo": "urllib3/urllib3",
        "base_commit": "d2e3f4a5",
        "problem_statement": "decrement_retry raises error or decrements incorrectly by 2.",
        "target_file": "src/urllib3/util/retry.py",
        "sources": {
            "src/urllib3/util/retry.py": "def decrement_retry(retries: int) -> int:\n    return retries - 2  # Bug: off-by-one, should be - 1\n"
        },
        "fail_to_pass_tests": [
            [[3], 2]
        ],
        "pass_to_pass_tests": [
            [[2], 1]
        ],
        "FAIL_TO_PASS": ["tests/test_retry.py::test_decrement_by_one"],
        "PASS_TO_PASS": ["tests/test_retry.py::test_decrement_positive"]
    },
    {
        "instance_id": "urllib3__urllib3-2550",
        "repo": "urllib3/urllib3",
        "base_commit": "e3f4a5b6",
        "problem_statement": "strip_chunk_crlf strips only trailing lf instead of both cr and lf.",
        "target_file": "src/urllib3/response.py",
        "sources": {
            "src/urllib3/response.py": "def strip_crlf(chunk: str) -> str:\n    return chunk.rstrip('\\n')  # Bug: leaves \\r intact\n"
        },
        "fail_to_pass_tests": [
            [["data\\r\\n"], "data"]
        ],
        "pass_to_pass_tests": [
            [["data\\n"], "data"]
        ],
        "FAIL_TO_PASS": ["tests/test_response.py::test_strip_both_cr_lf"],
        "PASS_TO_PASS": ["tests/test_response.py::test_strip_lf_only"]
    },
    {
        "instance_id": "pallets__flask-5100",
        "repo": "pallets/flask",
        "base_commit": "f4a5b6c7",
        "problem_statement": "format_blueprint_url_prefix adds trailing slash when prefix is empty.",
        "target_file": "src/flask/blueprints.py",
        "sources": {
            "src/flask/blueprints.py": "def format_prefix(prefix: str) -> str:\n    if not prefix:\n        return '/'  # Bug: empty prefix should remain empty ''\n    return '/' + prefix.strip('/')\n"
        },
        "fail_to_pass_tests": [
            [[""], ""]
        ],
        "pass_to_pass_tests": [
            [["api"], "/api"]
        ],
        "FAIL_TO_PASS": ["tests/test_blueprints.py::test_empty_prefix_clean"],
        "PASS_TO_PASS": ["tests/test_blueprints.py::test_non_empty_prefix"]
    },
    {
        "instance_id": "pallets__flask-5125",
        "repo": "pallets/flask",
        "base_commit": "a5b6c7d8",
        "problem_statement": "is_safe_path approves relative parent directory traversal components.",
        "target_file": "src/flask/helpers.py",
        "sources": {
            "src/flask/helpers.py": "def check_safe_path(path: str) -> bool:\n    if '..' in path:\n        return True  # Bug: traversal should be rejected (False)\n    return True\n"
        },
        "fail_to_pass_tests": [
            [["../etc/passwd"], False]
        ],
        "pass_to_pass_tests": [
            [["static/style.css"], True]
        ],
        "FAIL_TO_PASS": ["tests/test_helpers.py::test_parent_traversal_rejected"],
        "PASS_TO_PASS": ["tests/test_helpers.py::test_clean_path_accepted"]
    },
    {
        "instance_id": "pallets__click-1800",
        "repo": "pallets/click",
        "base_commit": "b6c7d8e9",
        "problem_statement": "match_choice fails case-insensitive matching when user passes lowercase input.",
        "target_file": "src/click/types.py",
        "sources": {
            "src/click/types.py": "def match_choice(val: str, choices: list) -> bool:\n    return val in choices  # Bug: case-sensitive; should check val.lower() in [c.lower() for c in choices]\n"
        },
        "fail_to_pass_tests": [
            [["json", ["JSON", "XML"]], True]
        ],
        "pass_to_pass_tests": [
            [["JSON", ["JSON", "XML"]], True]
        ],
        "FAIL_TO_PASS": ["tests/test_types.py::test_case_insensitive_choice"],
        "PASS_TO_PASS": ["tests/test_types.py::test_exact_choice"]
    },
    {
        "instance_id": "pallets__click-1825",
        "repo": "pallets/click",
        "base_commit": "c7d8e9f0",
        "problem_statement": "nargs_empty_default returns None instead of empty list when nargs=-1.",
        "target_file": "src/click/core.py",
        "sources": {
            "src/click/core.py": "def get_nargs_default(nargs: int) -> any:\n    if nargs == -1:\n        return None  # Bug: should return []\n    return None\n"
        },
        "fail_to_pass_tests": [
            [[-1], []]
        ],
        "pass_to_pass_tests": [
            [[1], None]
        ],
        "FAIL_TO_PASS": ["tests/test_core.py::test_nargs_var_empty_list"],
        "PASS_TO_PASS": ["tests/test_core.py::test_nargs_single_none"]
    },
    {
        "instance_id": "pallets__jinja-1200",
        "repo": "pallets/jinja",
        "base_commit": "d8e9f0a1",
        "problem_statement": "default_filter triggers default on boolean False.",
        "target_file": "src/jinja2/filters.py",
        "sources": {
            "src/jinja2/filters.py": "def apply_default(val: any, default_val: any) -> any:\n    if not val:  # Bug: triggers on False; should check val is None\n        return default_val\n    return val\n"
        },
        "fail_to_pass_tests": [
            [[False, "fallback"], False]
        ],
        "pass_to_pass_tests": [
            [[None, "fallback"], "fallback"],
            [["hello", "fallback"], "hello"]
        ],
        "FAIL_TO_PASS": ["tests/test_filters.py::test_default_preserves_false"],
        "PASS_TO_PASS": ["tests/test_filters.py::test_default_replaces_none"]
    },
    {
        "instance_id": "pallets__jinja-1225",
        "repo": "pallets/jinja",
        "base_commit": "e9f0a1b2",
        "problem_statement": "autoescape_extension rejects .xhtml files from auto-escaping.",
        "target_file": "src/jinja2/environment.py",
        "sources": {
            "src/jinja2/environment.py": "def should_autoescape(filename: str) -> bool:\n    exts = ('.html', '.xml')  # Bug: missing .xhtml\n    return any(filename.endswith(e) for e in exts)\n"
        },
        "fail_to_pass_tests": [
            [["template.xhtml"], True]
        ],
        "pass_to_pass_tests": [
            [["template.html"], True],
            [["script.js"], False]
        ],
        "FAIL_TO_PASS": ["tests/test_environment.py::test_xhtml_autoescaped"],
        "PASS_TO_PASS": ["tests/test_environment.py::test_html_autoescaped"]
    },
    {
        "instance_id": "psf__black-3100",
        "repo": "psf/black",
        "base_commit": "f0a1b2c3",
        "problem_statement": "exceeds_line_length uses non-strict comparison causing lines of exact length to break.",
        "target_file": "src/black/linegen.py",
        "sources": {
            "src/black/linegen.py": "def is_line_too_long(length: int, max_limit: int) -> bool:\n    return length >= max_limit  # Bug: should be length > max_limit\n"
        },
        "fail_to_pass_tests": [
            [[88, 88], False]
        ],
        "pass_to_pass_tests": [
            [[89, 88], True],
            [[80, 88], False]
        ],
        "FAIL_TO_PASS": ["tests/test_linegen.py::test_exact_line_length_allowed"],
        "PASS_TO_PASS": ["tests/test_linegen.py::test_excessive_line_broken"]
    },
    {
        "instance_id": "marshmallow_code__marshmallow-1500",
        "repo": "marshmallow-code/marshmallow",
        "base_commit": "a1b2c3d4e",
        "problem_statement": "deserialize_integer leaves string representation without casting to int.",
        "target_file": "src/marshmallow/types.py",
        "sources": {
            "src/marshmallow/types.py": "def parse_int_field(val: any) -> int:\n    return val  # Bug: missing int(val) cast\n"
        },
        "fail_to_pass_tests": [
            [["123"], 123]
        ],
        "pass_to_pass_tests": [
            [[123], 123]
        ],
        "FAIL_TO_PASS": ["tests/test_types.py::test_string_to_int_cast"],
        "PASS_TO_PASS": ["tests/test_types.py::test_int_preserved"]
    },
    {
        "instance_id": "sympy__sympy-14000",
        "repo": "sympy/sympy",
        "base_commit": "b2c3d4e5f",
        "problem_statement": "poly_degree returns 0 for zero polynomial instead of -1 or negative infinity.",
        "target_file": "sympy/polys/polytools.py",
        "sources": {
            "sympy/polys/polytools.py": "def get_degree(coeffs: list) -> int:\n    if not coeffs or all(c == 0 for c in coeffs):\n        return 0  # Bug: zero poly degree must be -1\n    return len(coeffs) - 1\n"
        },
        "fail_to_pass_tests": [
            [[[0, 0, 0]], -1]
        ],
        "pass_to_pass_tests": [
            [[[1, 2, 3]], 2]
        ],
        "FAIL_TO_PASS": ["tests/test_polytools.py::test_zero_poly_degree"],
        "PASS_TO_PASS": ["tests/test_polytools.py::test_nonzero_poly_degree"]
    },
    {
        "instance_id": "sympy__sympy-14025",
        "repo": "sympy/sympy",
        "base_commit": "c3d4e5f6a",
        "problem_statement": "det_2x2 inverts sign of diagonal product subtraction.",
        "target_file": "sympy/matrices/dense.py",
        "sources": {
            "sympy/matrices/dense.py": "def compute_det2x2(a: int, b: int, c: int, d: int) -> int:\n    return (b * c) - (a * d)  # Bug: inverted sign, should be a*d - b*c\n"
        },
        "fail_to_pass_tests": [
            [[1, 2, 3, 4], -2]
        ],
        "pass_to_pass_tests": [
            [[2, 0, 0, 3], 6]
        ],
        "FAIL_TO_PASS": ["tests/test_dense.py::test_det_sign_correctness"],
        "PASS_TO_PASS": ["tests/test_dense.py::test_diagonal_det"]
    },
    {
        "instance_id": "sphinx_doc__sphinx-9100",
        "repo": "sphinx-doc/sphinx",
        "base_commit": "d4e5f6a7b",
        "problem_statement": "strip_docstring_indent replaces tabs with 2 spaces instead of 4.",
        "target_file": "sphinx/util/docstrings.py",
        "sources": {
            "sphinx/util/docstrings.py": "def expand_doc_tabs(text: str) -> str:\n    return text.replace('\\t', '  ')  # Bug: tab expansion should be 4 spaces\n"
        },
        "fail_to_pass_tests": [
            [["\\tcode"], "    code"]
        ],
        "pass_to_pass_tests": [
            [["no_tabs"], "no_tabs"]
        ],
        "FAIL_TO_PASS": ["tests/test_docstrings.py::test_tab_to_four_spaces"],
        "PASS_TO_PASS": ["tests/test_docstrings.py::test_untabbed_text"]
    },
    {
        "instance_id": "sphinx_doc__sphinx-9125",
        "repo": "sphinx-doc/sphinx",
        "base_commit": "e5f6a7b8c",
        "problem_statement": "validate_extension_metadata approves None return values from extension setup.",
        "target_file": "sphinx/extension.py",
        "sources": {
            "sphinx/extension.py": "def check_ext_meta(meta: any) -> bool:\n    if meta is None:\n        return True  # Bug: setup() must return metadata dict\n    return isinstance(meta, dict)\n"
        },
        "fail_to_pass_tests": [
            [[None], False]
        ],
        "pass_to_pass_tests": [
            [[{"version": "1.0"}], True]
        ],
        "FAIL_TO_PASS": ["tests/test_extension.py::test_reject_none_meta"],
        "PASS_TO_PASS": ["tests/test_extension.py::test_accept_dict_meta"]
    },
    {
        "instance_id": "scipy__scipy-12000",
        "repo": "scipy/scipy",
        "base_commit": "f6a7b8c9d",
        "problem_statement": "vector_norm order p=0 calculates sum instead of non-zero element count.",
        "target_file": "scipy/linalg/norm.py",
        "sources": {
            "scipy/linalg/norm.py": "def calc_l0_norm(vec: list) -> int:\n    return sum(vec)  # Bug: L0 norm counts non-zero entries\n"
        },
        "fail_to_pass_tests": [
            [[[0, 5, 0, 3]], 2]
        ],
        "pass_to_pass_tests": [
            [[[1, 1, 1]], 3]
        ],
        "FAIL_TO_PASS": ["tests/test_norm.py::test_l0_count_nonzero"],
        "PASS_TO_PASS": ["tests/test_norm.py::test_all_nonzero"]
    },
    {
        "instance_id": "scikit_learn__scikit_learn-16000",
        "repo": "scikit-learn/scikit-learn",
        "base_commit": "a7b8c9d0e",
        "problem_statement": "check_test_size treats integer test_size as fraction when <= 1.",
        "target_file": "sklearn/model_selection/_split.py",
        "sources": {
            "sklearn/model_selection/_split.py": "def parse_test_size(n_samples: int, test_size: any) -> int:\n    if isinstance(test_size, int):\n        return test_size\n    return int(n_samples * test_size)  # Handles float fraction\n"
        },
        "fail_to_pass_tests": [
            [[100, 0.2], 20]
        ],
        "pass_to_pass_tests": [
            [[100, 25], 25]
        ],
        "FAIL_TO_PASS": ["tests/test_split.py::test_float_fraction_test_size"],
        "PASS_TO_PASS": ["tests/test_split.py::test_int_absolute_test_size"]
    },
    {
        "instance_id": "matplotlib__matplotlib-20000",
        "repo": "matplotlib/matplotlib",
        "base_commit": "b8c9d0e1f",
        "problem_statement": "normalize_hex_color omits hash prefix on 6-character hex strings.",
        "target_file": "lib/matplotlib/colors.py",
        "sources": {
            "lib/matplotlib/colors.py": "def format_hex(c: str) -> str:\n    if not c.startswith('#'):\n        return c  # Bug: should prepend '#'\n    return c\n"
        },
        "fail_to_pass_tests": [
            [["ff0000"], "#ff0000"]
        ],
        "pass_to_pass_tests": [
            [["#00ff00"], "#00ff00"]
        ],
        "FAIL_TO_PASS": ["tests/test_colors.py::test_add_missing_hash"],
        "PASS_TO_PASS": ["tests/test_colors.py::test_keep_existing_hash"]
    },
    {
        "instance_id": "numpy__numpy-18000",
        "repo": "numpy/numpy",
        "base_commit": "c9d0e1f2a",
        "problem_statement": "clip_bounds allows min_val to be strictly greater than max_val without raising.",
        "target_file": "numpy/core/numeric.py",
        "sources": {
            "numpy/core/numeric.py": "def validate_clip_bounds(min_val: float, max_val: float) -> bool:\n    if min_val > max_val:\n        return True  # Bug: invalid bounds must return False\n    return True\n"
        },
        "fail_to_pass_tests": [
            [[10.0, 5.0], False]
        ],
        "pass_to_pass_tests": [
            [[0.0, 10.0], True],
            [[5.0, 5.0], True]
        ],
        "FAIL_TO_PASS": ["tests/test_numeric.py::test_clip_bounds_invalid"],
        "PASS_TO_PASS": ["tests/test_numeric.py::test_clip_bounds_valid"]
    },
    {
        "instance_id": "pandas__pandas-35000",
        "repo": "pandas/pandas",
        "base_commit": "d0e1f2a3b",
        "problem_statement": "dropna_series drops row with any NaN when how='all' is specified.",
        "target_file": "pandas/core/series.py",
        "sources": {
            "pandas/core/series.py": "def should_drop_row(row: list, how: str) -> bool:\n    if how == 'all':\n        return any(x is None for x in row)  # Bug: how='all' must check all(x is None)\n    return any(x is None for x in row)\n"
        },
        "fail_to_pass_tests": [
            [[[1, None, 3], "all"], False]
        ],
        "pass_to_pass_tests": [
            [[[None, None, None], "all"], True],
            [[[1, None, 3], "any"], True]
        ],
        "FAIL_TO_PASS": ["tests/test_series.py::test_drop_how_all_partial_nan"],
        "PASS_TO_PASS": ["tests/test_series.py::test_drop_how_all_complete_nan"]
    }
]

created = 0
for item in NEW_FIXTURES:
    fname = item["instance_id"].replace("-", "_").replace("/", "__") + ".json"
    target_path = FIXTURES_DIR / fname
    target_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    created += 1

print(f"Successfully generated {created} new SWE-bench fixtures in {FIXTURES_DIR}")
total_fixtures = len(list(FIXTURES_DIR.glob("*.json")))
print(f"Total fixtures in directory: {total_fixtures}")
