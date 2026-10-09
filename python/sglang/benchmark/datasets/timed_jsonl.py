import json
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow

_REQUIRED_FIELDS = ("prompt", "output_len", "timestamp")


@dataclass
class TimedJsonlDataset(BaseDataset):
    """Timestamped workload: one JSON object per line with real prompt text.

    Required: prompt (str), output_len (int), timestamp (ms from trace start).
    Optional: prompt_len, rid, routing_key, extra_request_body.
    Replay the timestamps with --use-trace-timestamps.
    """

    dataset_path: str
    num_requests: int

    @classmethod
    def from_args(cls, args: Namespace) -> "TimedJsonlDataset":
        return cls(dataset_path=args.dataset_path, num_requests=args.num_prompts)

    def load(self, tokenizer: Any, model_id=None) -> List[DatasetRow]:
        records = _read_records(self.dataset_path)
        records.sort(key=lambda record: record["timestamp"])
        if len(records) > self.num_requests:
            print(
                f"timed-jsonl: keeping the first {self.num_requests} of "
                f"{len(records)} requests by timestamp (--num-prompts)."
            )
        return [
            _to_row(record=record, tokenizer=tokenizer)
            for record in records[: self.num_requests]
        ]


def _read_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in _REQUIRED_FIELDS:
                if field not in record:
                    raise ValueError(
                        f"{path} line {line_number}: missing required field {field!r}"
                    )
            records.append(record)
    return records


def _to_row(*, record: Dict[str, Any], tokenizer: Any) -> DatasetRow:
    extra_request_body = dict(record.get("extra_request_body") or {})
    if "rid" in record:
        extra_request_body["rid"] = record["rid"]
    prompt_len = record.get("prompt_len")
    if prompt_len is None:
        prompt_len = len(tokenizer.encode(record["prompt"], add_special_tokens=False))
    return DatasetRow(
        prompt=record["prompt"],
        prompt_len=int(prompt_len),
        output_len=int(record["output_len"]),
        timestamp=float(record["timestamp"]),
        routing_key=record.get("routing_key"),
        extra_request_body=extra_request_body,
    )
