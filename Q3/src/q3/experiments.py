def evaluate_experiments(selections, evaluator):
    """Evaluate already-selected atom sets without inspecting labels."""
    rows = []
    for selection in selections:
        view = evaluator(selection.atoms)
        rows.append({"selection_id": getattr(selection, "selection_id", None), "delete_j": list(selection.atoms), "view": view, "cost": selection.actual_cost})
    return rows
