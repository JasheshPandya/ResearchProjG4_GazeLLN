"""Configuration snapshots and durable tables for scripted experiments."""

from contextlib import contextmanager
import csv
from pathlib import Path
import tempfile
import time
import uuid

from checkpoint_provenance import (atomic_json, environment_record, fingerprint,
                                   json_value, read_training_record, utc_now)


class RunExport:
    """Write settings before work begins, including runs that fail before saving weights."""

    def __init__(self, output_dir, kind, configuration, *, sources):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.path = self.output_dir / "run_config.json"
        self.record = dict(schema_version=1, run_id=str(uuid.uuid4()), kind=kind,
                           status="running", started_at_utc=utc_now(),
                           configuration=json_value(configuration), runtime={},
                           environment=environment_record(sources))

    def update(self, **runtime):
        self.record["runtime"].update(json_value(runtime))
        atomic_json(self.path, self.record)

    @contextmanager
    def run(self):
        started = time.monotonic()
        self.update()
        try:
            yield self
        except BaseException as exc:
            self.record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            self.record["termination"] = dict(type=type(exc).__name__, message=str(exc))
            raise
        else:
            self.record["status"] = "completed"
        finally:
            self.record.update(finished_at_utc=utc_now(), duration_s=time.monotonic() - started)
            self.update()

    def write_results(self, results):
        atomic_json(self.output_dir / "results.json", dict(
            schema_version=1, run_id=self.record["run_id"], run_config=str(self.path), **results))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="table_", dir=path.parent) as temporary:
        pending = Path(temporary) / "table.csv"
        with pending.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(path)


def export_training_results(run, checkpoint, history, *, test_loss=None):
    record = read_training_record(checkpoint)
    if record:
        run.update(training_hyperparameters=record["hyperparameters"],
                   initialization=record["initialization"],
                   training_provenance=fingerprint(str(checkpoint) + ".training.json"))
    rows = record["history"] if record else [
        dict(epoch=i + 1, train_loss=train, val_loss=val)
        for i, (train, val) in enumerate(zip(history["train_loss"], history["val_loss"]))]
    fields = ["epoch", "train_loss", "val_loss", "duration_s"]
    write_csv(run.output_dir / "history.csv", [{key: row.get(key) for key in fields} for row in rows], fields)
    run.write_results(dict(checkpoint=fingerprint(checkpoint), history=history,
                           training_record=str(Path(str(checkpoint) + ".training.json")),
                           training_status=record["status"] if record else "unknown",
                           selection=record["selection"] if record else None,
                           test_loss=test_loss))
