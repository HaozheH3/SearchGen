from searchgen_eval.judge_common import try_parse_judge_output


def test_parse_failure_is_none():
    assert try_parse_judge_output("not XML or JSON") is None
