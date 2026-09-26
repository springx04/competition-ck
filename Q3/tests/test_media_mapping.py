import json

from q3.mapping import build_mapping, unresolved_mapping


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": [101, 11, 12, 102, 0],
            "attention_mask": [1, 1, 1, 1, 0],
            "token_type_ids": [0, 0, 0, 0, 0],
            "offset_mapping": [(0, 0), (0, 2), (3, 5), (0, 0), (0, 0)],
        }

    def convert_ids_to_tokens(self, ids):
        return ["[CLS]", "go", "up", "[SEP]", "[PAD]"]


def _sample():
    return {
        "sample_id": "01",
        "raw_text": "go up",
        "raw": {
            "input_ids": [101, 11, 12, 102, 0],
            "stored_attention": [1, 1, 1, 1, 0],
            "token_type_ids": [0, 0, 0, 0, 0],
        },
    }


def _config(tmp_path):
    run = tmp_path / "special"
    (run / "alignment" / "01").mkdir(parents=True)
    (run / "alignment" / "01" / "words.jsonl").write_text(
        json.dumps({
            "sample_id": "01", "word_id": 0, "raw_word": "go",
            "start_common": 0.1, "end_common": 0.4,
            "log_score": -0.1, "algorithm_status": "accepted",
            "review_status": "pending",
        }) + "\n",
        encoding="utf-8",
    )
    review = tmp_path / "word_times.csv"
    review.write_text(
        "sample_id,word_id,raw_word,auto_start_common,auto_end_common,algorithm_status,review_status,reviewer,reviewed_at,review_start_common,review_end_common\n"
        "01,0,go,0.1,0.4,accepted,pending,,,,\n",
        encoding="utf-8",
    )
    av = tmp_path / "official_av_mapping.csv"
    av.write_text("sample_id,selection_id,modality,source_ref\n", encoding="utf-8")
    return {
        "project": {"root": str(tmp_path)},
        "output": {"special_run": str(run)},
        "alignment": {"review_file": str(review), "av_provenance_file": str(av)},
    }

def test_missing_provenance_stays_unresolved():
    result=unresolved_mapping()
    assert result["status"]=="unresolved"
    assert "provenance" in result["reason"]


def test_mapping_keeps_algorithm_ctc_and_unknown_av_separate(tmp_path):
    result = build_mapping(
        _config(tmp_path),
        _sample(),
        {"U0": [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]]},
        [
            {"selection_id": "key.T.p20.point", "path": "T", "delete_j": [3], "intervals": [[1, 2]]},
            {"selection_id": "key.A.p20.point", "path": "A", "delete_j": [4], "intervals": [[1, 2]]},
        ],
        FakeTokenizer(),
    )
    assert result["tokenizer_validation"]["exact"] is True
    assert result["evidence"]["key.T.p20.point"]["selected_text"] == "go"
    ctc = result["evidence"]["key.T.p20.point"]["word_times"][0]
    assert ctc["material_status"] == "algorithm_accepted"
    assert ctc["verified"] is False
    assert result["evidence"]["key.A.p20.point"]["material_status"] == "unresolved"
    json.dumps(result, ensure_ascii=False)


def test_mapping_uses_only_valid_human_review_for_verified_ctc(tmp_path):
    config = _config(tmp_path)
    review = tmp_path / "word_times.csv"
    review.write_text(
        "sample_id,word_id,raw_word,auto_start_common,auto_end_common,algorithm_status,review_status,reviewer,reviewed_at,review_start_common,review_end_common\n"
        "01,0,go,0.1,0.4,accepted,confirmed_match,alice,2026-09-26T12:00:00Z,0.2,0.3\n",
        encoding="utf-8",
    )
    result = build_mapping(
        config,
        _sample(),
        {"U0": [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]]},
        [{"selection_id": "key.T.p20.point", "path": "T", "delete_j": [3], "intervals": [[1, 2]]}],
        FakeTokenizer(),
    )
    ctc = result["evidence"]["key.T.p20.point"]["word_times"][0]
    assert ctc["material_status"] == "verified"
    assert ctc["verified"] is True
    assert ctc["start_common"] == 0.2
    assert ctc["end_common"] == 0.3


def test_explicit_av_source_ref_is_linked_but_unknown_is_not_verified(tmp_path):
    config = _config(tmp_path)
    av = tmp_path / "official_av_mapping.csv"
    av.write_text(
        "sample_id,selection_id,modality,source_ref\n"
        "01,key.A.p20.point,A,official-a-row-01\n",
        encoding="utf-8",
    )
    result = build_mapping(
        config,
        _sample(),
        {"U0": [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]]},
        [
            {"selection_id": "key.A.p20.point", "path": "A", "delete_j": [4], "intervals": [[1, 2]]},
            {"selection_id": "key.V.p20.point", "path": "V", "delete_j": [5], "intervals": [[1, 2]]},
        ],
        FakeTokenizer(),
    )
    audio = result["evidence"]["key.A.p20.point"]
    visual = result["evidence"]["key.V.p20.point"]
    assert audio["material_status"] == "resolved"
    assert audio["modality_status"]["A"]["source_ref"] == "official-a-row-01"
    assert audio["verified"] is False
    assert visual["material_status"] == "unresolved"
    assert visual["modality_status"]["V"]["verified"] is False


def test_tokenizer_mismatch_cannot_produce_verified_characters(tmp_path):
    sample=_sample();sample['raw']['input_ids'][1]=99
    result=build_mapping(_config(tmp_path),sample,{},[dict(selection_id='key.T',path='T',delete_j=[3],intervals=[[1,2]])],FakeTokenizer())
    assert not result['tokenizer_validation']['exact']
    assert result['evidence']['key.T']['selected_text']==''
    assert not result['evidence']['key.T']['verified']


def test_joint_and_missing_modality_do_not_invent_evidence(tmp_path):
    sels=[dict(selection_id='key.J',path='J',delete_j=[3,4],intervals=[[1,2]]),
          dict(selection_id='key.V',path='V',delete_j=[],intervals=[],status='not_applicable')]
    result=build_mapping(_config(tmp_path),_sample(),{},sels,FakeTokenizer())
    assert set(result['evidence']['key.J']['modality_status'])=={'T','A'}
    assert result['evidence']['key.J']['selected_text']=='go'
    assert result['evidence']['key.V']['status']=='not_applicable'
    assert result['evidence']['key.V']['word_times']==[]


def test_repeated_selections_do_not_multiply_verified_word_count(tmp_path):
    config=_config(tmp_path)
    (tmp_path/'word_times.csv').write_text('sample_id,word_id,raw_word,review_status,reviewer,reviewed_at\n01,0,go,confirmed_match,reviewer,2026-09-26T12:00:00Z\n',encoding='utf-8')
    sels=[dict(selection_id=f'key.T.{i}',path='T',delete_j=[3],intervals=[[1,2]]) for i in range(3)]
    result=build_mapping(config,_sample(),{},sels,FakeTokenizer())
    assert result['coverage']['ctc_word_verified']==1
    assert result['coverage']['ctc_word_denominator']==1
