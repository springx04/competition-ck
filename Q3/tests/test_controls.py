from q3.grouping import Unit
from q3.controls import ControlSpace
def test_rank_roundtrip():
    units=[Unit('a',(0,),0,1,1),Unit('b',(1,),1,2,1),Unit('c',(2,),2,3,1)]
    space=ControlSpace(units,(0,),1);assert space.count()==2;assert space.unrank(space.rank([units[1]]) )[0].unit_id=='b'
