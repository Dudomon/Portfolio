from ledgerflow_api.services.audit import diff_snapshots


def test_diff_reports_changed_added_and_removed_keys() -> None:
    before = {"status": "needs_review", "total_amount": "10.00", "currency": "BRL"}
    after = {"status": "approved", "total_amount": "10.00", "due_date": "2026-11-01"}
    assert diff_snapshots(before, after) == {
        "currency": {"before": "BRL", "after": None},
        "due_date": {"before": None, "after": "2026-11-01"},
        "status": {"before": "needs_review", "after": "approved"},
    }


def test_no_changes_yields_none() -> None:
    assert diff_snapshots({"a": [1, 2]}, {"a": [1, 2]}) is None
