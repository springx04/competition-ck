from q3.mapping import unresolved_mapping

def test_missing_provenance_stays_unresolved():
    result=unresolved_mapping()
    assert result["status"]=="unresolved"
    assert "provenance" in result["reason"]
