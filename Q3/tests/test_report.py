from q3.report import highlighted_text,top_modalities,grouped_trials,_render


def test_highlight_escapes_untrusted_text_and_preserves_exact_selection():
    row=dict(raw_text='a <b> x',mapping={'evidence':{'key.T.p20.point':{'char_spans':[[2,5]]}}})
    assert highlighted_text(row)=='a <mark>&lt;b&gt;</mark> x'


def test_zero_influence_is_not_assigned_a_primary_modality():
    assert top_modalities([0,0,0],[3,4,0],1e-6)==[]
    assert top_modalities([1e-9,0,0],[3,4,0],1e-6)==[]
    assert top_modalities([.1,.2,.9],[3,4,0],1e-6)==['A']


def test_random_deletion_and_keep_outputs_are_not_pooled():
    rows=[dict(selection_id='key.T',operation='delete',control_role='random',D=.1),
          dict(selection_id='key.T',operation='global_keep',control_role='random',D=.9)]
    grouped=grouped_trials({'experiments':rows})
    assert [r['D'] for r in grouped[('key.T','delete','random')]]==[.1]
    assert '计算尚未完整' in _render(dict(sample_id='01',status='partial',missing_files=['local.json']))
