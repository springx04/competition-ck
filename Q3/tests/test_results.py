import json
import pytest
from q3.results import write_json, view_id

def test_json_rejects_nan_and_view_ids_are_stable(tmp_path):
    assert view_id((5,2))==view_id((2,5))
    with pytest.raises(ValueError):write_json(tmp_path/"bad.json",{"x":float("nan")})
    write_json(tmp_path/"ok.json",{"x":None});assert json.loads((tmp_path/"ok.json").read_text())["x"] is None
