from q2.masking import evaluation_grid, sample_train_descriptor


def test_grid_mix_covers_public_grid_and_preserves_other_curriculum_samples():
    expected = {(d["pattern"], d["position"], d["rho"]) for d in evaluation_grid()}
    observed = set()
    original_count = 0
    for i in range(2000):
        args = dict(total_epochs=20, warmup_epochs=2)
        d = sample_train_descriptor(1111, 3, i, grid_mix=True, **args)
        if d.get("position") in ("front", "middle", "rear") and d["pattern"]:
            observed.add((d["pattern"], d["position"], d["rho"]))
        else:
            original = sample_train_descriptor(1111, 3, i, **args)
            assert {k: v for k, v in d.items() if k != "rng"} == {
                k: v for k, v in original.items() if k != "rng"}
            original_count += 1
    assert observed == expected
    assert 800 < original_count < 1200
    assert sample_train_descriptor(1111, 2, 0, grid_mix=True, **args)["rho"] == 0
