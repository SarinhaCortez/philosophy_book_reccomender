from myrecsys.intent import normalize_query


def test_normalize_query_preserves_user_words():
    assert normalize_query("  modern   love  ") == "modern love"
