from __future__ import annotations

import os
import types

import nalm_benchmark.experiments.op_supervisor as sup


class DummyWandb:
    def __init__(self) -> None:
        self.logged = []
        self.finished = False
        self._run = types.SimpleNamespace(summary={})

    # API used by supervisor
    def init_wandb_runpod(self):
        return types.SimpleNamespace(id="test", url="http://wandb", name="supervisor")

    @property
    def wrapper(self):
        return self

    def log(self, data, commit=True):  # noqa: ARG002
        self.logged.append(data)

    def finish(self):
        self.finished = True


def test_log_completion_and_child_failure(monkeypatch):
    d = DummyWandb()
    # Patch module wandb used inside supervisor
    monkeypatch.setattr(sup, "wandb", d)

    # Simulate init
    run = d.init_wandb_runpod()
    assert run.id == "test"

    # Emit launched + completion and failure logs
    sup.log_completion(
        "sub", launched_count=1, completed_total=0, completed_ok=0, completed_failed=0
    )
    d.log({"sub/child_failed": 1, "supervisor/last_failure": "label rc=1"})

    # Verify logs captured
    assert any("completed_total" in x for x in d.logged)
    assert any("sub/child_failed" in x for x in d.logged)
