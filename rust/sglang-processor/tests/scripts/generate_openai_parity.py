"""Fill tests/fixtures/openai_parity/*.json from SGLang's own OpenAI layer.

Each fixture names the engine's OpenAI `settings` and request `cases`. For every
case this records the `/generate` body Python's OpenAI handler builds, the
engine's `/generate` output for that body (recorded once from `--engine-url`),
and the OpenAI response the handler builds from that output.

Run from rust/sglang-processor in a SGLang Python environment, with `--model`
the checkpoint directory (its tokenizer and config):
    python tests/scripts/generate_openai_parity.py --model DIR \
        [--engine-url http://127.0.0.1:30000] [tests/fixtures/openai_parity/<name>.json ...]
"""

import argparse
import asyncio
import copy
import dataclasses
import json
import re
import sys
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import requests
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from starlette.requests import Request
from starlette.responses import Response

# Record this checkout's SGLang, not whichever one is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers.common import get_context_length
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/openai_parity"
HANDLERS = {
    "completions": (CompletionRequest, OpenAIServingCompletion),
    "chat": (ChatCompletionRequest, OpenAIServingChat),
}
SKIPPED_FIELDS = {"received_time"}


def generate_defaults():
    defaults = {}
    for field in dataclasses.fields(GenerateReqInput):
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            defaults[field.name] = field.default_factory()
    return defaults


def generate_body(obj, defaults):
    """The fields Python set away from their defaults, as `/generate` JSON."""
    return {
        name: getattr(obj, name)
        for name in defaults
        if name not in SKIPPED_FIELDS and getattr(obj, name) != defaults[name]
    }


def record_frames(engine_url, body):
    response = requests.post(f"{engine_url}/generate", json=body, stream=body.get("stream"))
    if not body.get("stream") or response.status_code != 200:
        return {"status": response.status_code, "body": response.json()}
    frames = []
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: ") and line != "data: [DONE]":
            frames.append(json.loads(line[len("data: "):]))
    return {"status": 200, "frames": frames}


def replay(output):
    """`generate_request` as the OpenAI handler sees the recorded `/generate` output."""

    def internal(output):
        for frame in output if isinstance(output, list) else [output]:
            reason = frame.get("meta_info", {}).get("finish_reason") or {}
            if isinstance(reason.get("status_code"), int):
                reason["status_code"] = HTTPStatus(reason["status_code"])
        return output

    async def generate_request(obj, raw_request=None):
        if "frames" not in output:
            body = output["body"]
            if output["status"] != 200:
                raise ValueError(body["error"]["message"])
            yield internal(json.loads(json.dumps(body)))
            return
        for frame in json.loads(json.dumps(output["frames"])):
            if "error" in frame:
                raise ValueError(frame["error"]["message"])
            yield internal(frame)

    return generate_request


async def openai_response(handler, request, raw_request):
    response = await handler.handle_request(request, raw_request)
    if isinstance(response, StreamingResponse):
        events = [chunk async for chunk in response.body_iterator]
        return {"events": events}
    if isinstance(response, Response):
        return {"status": response.status_code, "body": json.loads(response.body)}
    return {"status": 200, "body": jsonable_encoder(response)}


def run_case(case, settings, model, engine_url, defaults):
    request_type, handler_type = HANDLERS[case["endpoint"]]
    reset_context()
    server_args = ServerArgs(model_path=model.path, **settings)
    publish(server_args, role="tokenizer")
    recorded = {}
    tokenizer_manager = SimpleNamespace(
        tokenizer=model.tokenizer,
        server_args=server_args,
        model_path=model.path,
        model_config=SimpleNamespace(
            hf_config=model.hf_config,
            is_multimodal=False,
            get_default_sampling_params=lambda: model.sampling_defaults,
        ),
        config_value=lambda name: getattr(server_args, name),
        request_logger=SimpleNamespace(log_requests=False),
        create_abort_task=lambda obj: None,
    )
    # A checkpoint without a Jinja template, as DeepSeek-V4's.
    template_manager = SimpleNamespace(
        completion_template_name=settings.get("completion_template"),
        chat_template_name=None,
        jinja_template_content_format="string",
        jinja_template_may_reorder_tool_results=False,
        reasoning_config=None,
        force_reasoning=False,
    )
    handler = handler_type(tokenizer_manager, template_manager)
    raw_request = Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in case.get("headers", {}).items()],
        }
    )
    request = request_type(**copy.deepcopy(case["request"]))
    if handler._validate_request(request):
        raise ValueError(f"{case['name']}: SGLang rejects this request")
    obj, _ = handler._convert_to_internal_request(request, raw_request)
    recorded["generate"] = generate_body(obj, defaults)
    if case.get("lower_only"):
        return recorded
    if engine_url:
        case["engine"] = record_frames(engine_url, recorded["generate"])
    tokenizer_manager.generate_request = replay(case["engine"])
    recorded["engine"] = case["engine"]
    request = request_type(**copy.deepcopy(case["request"]))
    recorded["openai"] = asyncio.run(openai_response(handler, request, raw_request))
    return recorded


def load_model(path):
    config = json.loads(Path(path, "config.json").read_text())
    generation_file = Path(path, "generation_config.json")
    generation = json.loads(generation_file.read_text()) if generation_file.exists() else {}
    keys = ("repetition_penalty", "temperature", "top_k", "top_p", "min_p")
    return SimpleNamespace(
        path=path,
        tokenizer=get_tokenizer(path, trust_remote_code=True),
        hf_config=SimpleNamespace(**config, to_dict=lambda: config),
        text_config=SimpleNamespace(**config.get("text_config", config)),
        sampling_defaults={k: generation[k] for k in keys if generation.get(k) is not None},
    )


def generate(path, model, engine_url):
    fixture = json.loads(path.read_text())
    # The model facts the host reads, so the test runs without the checkpoint.
    fixture["chat_model"] = {
        "context_length": get_context_length(model.text_config),
        "generation_config": model.sampling_defaults,
    }
    defaults = generate_defaults()
    fixture["generate_defaults"] = {
        name: value for name, value in defaults.items() if name not in SKIPPED_FIELDS
    }
    for case in fixture["cases"]:
        if case.get("unsupported"):
            continue
        case.update(run_case(case, fixture.get("settings", {}), model, engine_url, defaults))
        print(f"{path.name}: {case['name']}", file=sys.stderr)
    path.write_text(dump(fixture))


def dump(fixture):
    """Indented JSON, with each list of numbers on one line."""
    text = json.dumps(fixture, indent=2, ensure_ascii=False)
    join = lambda m: "[" + re.sub(r",\n\s+", ", ", m.group(1)) + "]"
    return re.sub(r"\[\n\s+([^\[\]{}\"]*?)\n\s+\]", join, text) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-url")
    parser.add_argument("fixtures", nargs="*", type=Path)
    args = parser.parse_args()
    model = load_model(args.model)
    for path in args.fixtures or sorted(FIXTURES.glob("*.json")):
        generate(path, model, args.engine_url)


if __name__ == "__main__":
    main()
