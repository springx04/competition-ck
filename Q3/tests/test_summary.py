from q3.summary import paired_summary,task_metrics


def test_summary_does_not_drop_missing_control_denominators():
    rows=[dict(target=.4,random=.1,video_id='a',n_random=2),
          dict(target=.2,random=.1,video_id='b',n_random=50),
          dict(target=.3,random=None,video_id='c',n_random=0),
          dict(target=None,random=None,video_id='d',n_random=0)]
    result=paired_summary(rows,8,50,42)
    assert (result['n_total'],result['n_planned'],result['n_complete'],result['n_eligible'])==(8,4,3,2)
    assert result['n_without_controls']==1 and result['n_not_applicable']==1
    assert abs(result['mean_G']-.2)<1e-10


def test_task_metrics_retains_per_class_support():
    result=task_metrics([0,1,2],[0,0,2],[-1,0,1],[-1,-.2,.8])
    assert result['per_class'][1]['support']==1
    assert result['per_class'][1]['f1']==0
    assert result['accuracy']==2/3
