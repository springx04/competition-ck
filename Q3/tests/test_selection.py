from q3.grouping import Unit
from q3.selection import select_units
def test_budget_and_intervals():
    units=[Unit('a',(0,),0,1,1,1),Unit('b',(1,),1,2,1,3),Unit('c',(2,),2,3,1,2)]
    result=select_units(units,2,objective='max',exact_cost=True);assert result.atoms==(1,2);assert result.actual_cost==2
