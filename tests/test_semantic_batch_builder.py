from scripts.build_semantic_index_batch import (
    extract_embedding_values,
    extract_generate_content_text,
    extract_profile_payload,
)


def test_extract_generate_content_text_from_candidate_parts():
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '{"summary":"abc","themes":["ethics"],"traditions":[],"reader_moods":[],"style_descriptors":[],"notable_people":[],"difficulty":"","era":""}'}
                    ]
                }
            }
        ]
    }

    text = extract_generate_content_text(response)

    assert '"summary":"abc"' in text


def test_extract_profile_payload_validates_and_returns_dict():
    response_entry = {
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"summary":"abc","themes":["ethics"],"traditions":[],"reader_moods":[],"style_descriptors":[],"notable_people":[],"difficulty":"low","era":"modern"}'
                            }
                        ]
                    }
                }
            ]
        }
    }

    payload = extract_profile_payload(response_entry)

    assert payload["summary"] == "abc"
    assert payload["themes"] == ["ethics"]
    assert payload["difficulty"] == "low"


def test_extract_embedding_values_reads_response_shape():
    response_entry = {
        "response": {
            "embedding": {
                "values": [0.1, 0.2, 0.3],
            }
        }
    }

    values = extract_embedding_values(response_entry)

    assert values == [0.1, 0.2, 0.3]
