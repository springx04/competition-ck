import pytest
from q3.grouping import Unit
from q3.validation import canonical_intervals

def test_control_geometry_detects_merged_runs_and_split_word():
    units=[Unit(str(i),(3*i,),i,i+1,1) for i in range(5)]
    assert canonical_intervals(units,[0,6])==[[0,1],[2,3]]
    assert canonical_intervals(units,[0,3])==[[0,2]]
    words=[Unit('first',(0,3),0,2,2),Unit('second',(6,),2,3,1)]
    with pytest.raises(ValueError,match='partial'):
        canonical_intervals(words,[0,6])

def test_natural_holes_use_official_spans_not_cost_as_length():
    units=[Unit('a',(0,),0,1,1),Unit('b',(9,),3,4,1),Unit('c',(12,),4,5,1)]
    assert canonical_intervals(units,[0,9])==[[0,4]]
    with pytest.raises(ValueError,match='outside'):
        canonical_intervals(units,[3])
