from experimental.electronics.spec_schema import parse_request, spec_to_prepare_args


def test_parser_accepts_schema_rejects_prose_and_unknown():
    spec, err = parse_request('{"action":"run_scenario","scenario":"half_adder","engine":"ga"}')
    assert err == []
    assert spec is not None
    assert spec_to_prepare_args(spec)["scenario"] == "half_adder"
    _, err = parse_request("صمم لي half adder")
    assert "not_json" in err
    _, err = parse_request('{"action":"run_scenario","scenario":"cpu64","foo":1}')
    assert any("unknown_scenario" in e or "unknown_fields" in e for e in err)
