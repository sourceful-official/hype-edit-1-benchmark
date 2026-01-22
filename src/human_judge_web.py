#!/usr/bin/env python3
"""Lightweight web UI for anonymous human judging of HYPE-EDIT-1 outputs."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse


SCHEMA_VERSION = 2
CANDIDATE_RANGE = range(1, 11)
start_times: Dict[int, float] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_models_arg(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_model_aliases(models: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    width = max(2, len(str(len(models))))
    aliases: Dict[str, str] = {}
    reverse: Dict[str, str] = {}
    for idx, name in enumerate(models, 1):
        alias = f"M{idx:0{width}d}"
        aliases[name] = alias
        reverse[alias] = name
    return aliases, reverse


@dataclass(frozen=True)
class JudgeItem:
    task_id: str
    candidate_num: int
    instruction: str
    input_filenames: Tuple[str, ...]
    task_types: Tuple[str, ...]
    model_name: str
    model_id: str

    @property
    def candidate_filename(self) -> str:
        return f"{self.candidate_num:03d}.webp"


class JudgeState:
    """Shared state between HTTP handler instances."""

    def __init__(
        self,
        *,
        tasks: List[Dict[str, Any]],
        tasks_dir: Path,
        outputs_dir: Path,
        dataset: str,
        results_path: Path,
        model_names: List[str],
        model_aliases: Dict[str, str],
        model_id_map: Dict[str, str],
        shuffle: bool,
    ) -> None:
        self.tasks_dir = tasks_dir.resolve()
        self.outputs_dir = outputs_dir.resolve()
        self.dataset = dataset
        self.results_path = results_path
        self.model_names = model_names
        self.model_aliases = model_aliases
        self.model_id_map = model_id_map
        self.lock = threading.Lock()

        self.items = self._build_items(tasks)
        if shuffle:
            random.shuffle(self.items)

        self.task_model_index_map: Dict[Tuple[str, str], List[int]] = {}
        for idx, item in enumerate(self.items):
            key = (item.task_id, item.model_id)
            self.task_model_index_map.setdefault(key, []).append(idx)

        self.task_type_index_map: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.items):
            for task_type in item.task_types:
                self.task_type_index_map.setdefault(task_type, []).append(idx)

        self.task_types = sorted(self.task_type_index_map.keys())

        self.results = self._load_or_init_results()

    def _build_items(self, tasks: List[Dict[str, Any]]) -> List[JudgeItem]:
        items: List[JudgeItem] = []
        for model_name in self.model_names:
            model_id = self.model_aliases[model_name]
            for task in tasks:
                task_id = task["task_id"]
                instruction = task.get("instruction", "")
                task_types = self._normalize_task_types(task.get("task_type", ""))
                input_images = self._resolve_input_filenames(
                    task_id, task.get("input_images")
                )
                for candidate_num in CANDIDATE_RANGE:
                    items.append(
                        JudgeItem(
                            task_id=task_id,
                            candidate_num=candidate_num,
                            instruction=instruction,
                            input_filenames=input_images,
                            task_types=task_types,
                            model_name=model_name,
                            model_id=model_id,
                        )
                    )
        return items

    @staticmethod
    def _normalize_task_types(value: Any) -> Tuple[str, ...]:
        if isinstance(value, list):
            values = [str(item) for item in value if item]
        elif value:
            values = [str(value)]
        else:
            values = []
        return tuple(values)

    def _resolve_input_filenames(
        self, task_id: str, declared: Optional[List[str]]
    ) -> Tuple[str, ...]:
        filenames: List[str] = list(declared or [])
        seen = set(filenames)

        task_dir = self.tasks_dir / task_id
        if task_dir.exists():
            patterns = ("*.webp", "*.png", "*.jpg", "*.jpeg")
            for pattern in patterns:
                for path in sorted(task_dir.glob(pattern)):
                    name = path.name
                    if name not in seen:
                        filenames.append(name)
                        seen.add(name)

        return tuple(filenames)

    def _load_or_init_results(self) -> Dict[str, Any]:
        if self.results_path.exists():
            data = _load_json(self.results_path)
            if isinstance(data, dict) and data.get("schema_version") == SCHEMA_VERSION:
                return data

        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "tasks_dir": str(self.tasks_dir),
            "outputs_dir": str(self.outputs_dir),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "models": {alias: name for name, alias in self.model_aliases.items()},
            "judgements": {},
        }

    def _save_results(self) -> None:
        self.results["updated_at"] = _now_iso()
        _atomic_write_json(self.results_path, self.results)

    def get_judgement(self, item: JudgeItem) -> Optional[str]:
        judgements = self.results.get("judgements", {})
        return (
            judgements.get(item.task_id, {})
            .get(item.model_id, {})
            .get(item.candidate_filename, {})
            .get("verdict")
        )

    def set_judgement(self, item: JudgeItem, verdict: str, elapsed_ms: int) -> None:
        with self.lock:
            task_map = self.results.setdefault("judgements", {}).setdefault(
                item.task_id, {}
            )
            model_map = task_map.setdefault(item.model_id, {})
            model_map[item.candidate_filename] = {
                "verdict": verdict,
                "judged_at": _now_iso(),
                "ms": elapsed_ms,
            }
            self._save_results()

    def total_items(self) -> int:
        return len(self.items)

    def done_count(self) -> int:
        return sum(1 for item in self.items if self.get_judgement(item))

    def next_index(self, idx: int) -> int:
        total = self.total_items()
        return (idx + 1) % total if total else 0

    def prev_index(self, idx: int) -> int:
        total = self.total_items()
        return (idx - 1) % total if total else 0

    def next_unjudged_index(self, start_idx: int) -> int:
        total = self.total_items()
        if total == 0:
            return 0
        for offset in range(1, total + 1):
            idx = (start_idx + offset) % total
            if self.get_judgement(self.items[idx]) is None:
                return idx
        return start_idx

    def candidate_path(self, item: JudgeItem) -> Path:
        return (
            self.outputs_dir
            / self.dataset
            / item.model_name
            / item.task_id
            / item.candidate_filename
        )

    def input_path(self, task_id: str, filename: str) -> Path:
        return self.tasks_dir / task_id / filename


class JudgeRequestHandler(BaseHTTPRequestHandler):
    state: JudgeState

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/input/"):
            self._serve_image(kind="input", path=path)
            return
        if path.startswith("/candidate/"):
            self._serve_image(kind="candidate", path=path)
            return

        if path == "/" or path == "":
            self._render_main(parse_qs(parsed.query))
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/judge":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        params = parse_qs(body.decode("utf-8"))

        try:
            idx = int(params.get("idx", ["0"])[0])
        except ValueError:
            idx = 0
        task_type = params.get("task_type", [""])[0].strip()
        verdict = params.get("verdict", [""])[0].strip().upper()
        if verdict not in {"PASS", "FAIL"}:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid verdict")
            return

        state = self.state
        filtered_indices = self._filtered_indices(task_type)
        total = len(filtered_indices)
        if total == 0:
            self._redirect(self._build_index_url(task_type=task_type))
            return
        idx = max(0, min(idx, total - 1))
        item_idx = filtered_indices[idx]
        item = state.items[item_idx]

        started_at = start_times.get(item_idx, time.time())
        elapsed_ms = int((time.time() - started_at) * 1000)
        # Reset timer for this index so repeated submissions don't accumulate.
        start_times[item_idx] = time.time()
        state.set_judgement(item, verdict, elapsed_ms)

        next_idx = self._next_unjudged_filtered_index(filtered_indices, idx)
        self._redirect(self._build_index_url(idx=next_idx, task_type=task_type))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 (match BaseHTTPRequestHandler signature)
        sys.stderr.write("[human_judge_web] " + format % args + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _filtered_indices(self, task_type: str) -> List[int]:
        if not task_type:
            return list(range(self.state.total_items()))
        return list(self.state.task_type_index_map.get(task_type, []))

    def _next_unjudged_filtered_index(
        self, filtered_indices: List[int], start_idx: int
    ) -> int:
        total = len(filtered_indices)
        if total == 0:
            return 0
        for offset in range(1, total + 1):
            idx = (start_idx + offset) % total
            item = self.state.items[filtered_indices[idx]]
            if self.state.get_judgement(item) is None:
                return idx
        return start_idx

    def _build_index_url(self, *, idx: int = 0, task_type: str = "") -> str:
        if task_type:
            return f"/?idx={idx}&task_type={quote(task_type)}"
        return f"/?idx={idx}"

    def _render_main(self, query: Dict[str, List[str]]) -> None:
        state = self.state
        task_type = query.get("task_type", [""])[0].strip()
        filtered_indices = self._filtered_indices(task_type)
        total = len(filtered_indices)
        if total == 0:
            clear_link = self._build_index_url()
            filter_msg = (
                f" for task_type '{html.escape(task_type)}'" if task_type else ""
            )
            html_body = (
                f"<h1>No tasks available{filter_msg}</h1>"
                f"<p><a href=\"{clear_link}\">Clear filter</a></p>"
            )
            self._write_html(html_body)
            return

        try:
            idx = int(query.get("idx", ["0"])[0])
        except ValueError:
            idx = 0
        idx = max(0, min(idx, total - 1))
        item_idx = filtered_indices[idx]
        start_times[item_idx] = time.time()
        item = state.items[item_idx]
        judgement = state.get_judgement(item)
        candidate_path = state.candidate_path(item)
        candidate_exists = candidate_path.exists()

        instruction_html = html.escape(item.instruction or "").replace("\n", "<br>")

        input_blocks = []
        for filename in item.input_filenames:
            path = state.input_path(item.task_id, filename)
            escaped_name = html.escape(filename)
            if path.exists():
                url = f"/input/{item.task_id}/{filename}"
                block = (
                    f"<div class=\"input-card\"><div class=\"input-label\">{escaped_name}</div>"
                    f"<a href=\"{url}\" target=\"_blank\"><img src=\"{url}\" alt=\"{escaped_name}\"></a></div>"
                )
            else:
                block = (
                    f"<div class=\"input-card missing\"><div class=\"input-label\">{escaped_name}</div>"
                    f"<div class=\"missing-note\">Missing</div></div>"
                )
            input_blocks.append(block)

        if not input_blocks:
            input_blocks.append(
                "<div class=\"input-card missing\"><div class=\"missing-note\">No input images listed</div></div>"
            )

        candidate_block = ""
        overlay_url = ""
        overlay_available = False
        for filename in item.input_filenames:
            if filename == "001.webp":
                path = state.input_path(item.task_id, filename)
                if path.exists():
                    overlay_url = f"/input/{item.task_id}/{filename}"
                    overlay_available = True
                break
        if candidate_exists:
            cand_url = (
                f"/candidate/{item.model_id}/{item.task_id}/{item.candidate_filename}"
            )
            overlay_img = ""
            if overlay_available:
                overlay_img = (
                    f"<img class=\"candidate-overlay\" src=\"{overlay_url}\" "
                    "alt=\"Original 001.webp\">"
                )
            candidate_block = (
                f"<div class=\"candidate-compare\" data-overlay=\"{str(overlay_available).lower()}\">"
                f"<a href=\"{cand_url}\" target=\"_blank\"><img class=\"candidate-base\" "
                f"src=\"{cand_url}\" alt=\"Candidate {item.candidate_filename}\"></a>"
                f"{overlay_img}</div>"
            )
        else:
            candidate_block = (
                "<div class=\"missing-note\">Candidate image not found. "
                "Ensure generation has finished.</div>"
            )

        prev_idx = (idx - 1) % total if total else 0
        next_idx = (idx + 1) % total if total else 0
        next_unjudged = self._next_unjudged_filtered_index(filtered_indices, idx)

        same_task_links = []
        task_key = (item.task_id, item.model_id)
        task_indices = state.task_model_index_map.get(task_key, [])
        filtered_positions = {global_idx: pos for pos, global_idx in enumerate(filtered_indices)}
        for task_idx in task_indices:
            candidate = state.items[task_idx]
            verdict = state.get_judgement(candidate)
            label = candidate.candidate_filename
            filtered_pos = filtered_positions.get(task_idx)
            if filtered_pos is None:
                continue
            cls = "current" if filtered_pos == idx else ""
            verdict_text = verdict if verdict else "—"
            same_task_links.append(
                f"<li class=\"{cls}\"><a href=\"{self._build_index_url(idx=filtered_pos, task_type=task_type)}\">{label}</a> — {verdict_text}</li>"
            )

        done = sum(
            1
            for global_idx in filtered_indices
            if state.get_judgement(state.items[global_idx])
        )
        progress = f"{done}/{total} judged"

        verdict_text = judgement if judgement else "—"
        task_type_options = ["<option value=\"\">All task types</option>"]
        for value in state.task_types:
            selected = " selected" if value == task_type else ""
            task_type_options.append(
                f"<option value=\"{html.escape(value)}\"{selected}>{html.escape(value)}</option>"
            )
        html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HYPE-EDIT-1 Judge ({html.escape(state.dataset)})</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 0; background: #f9f9f9; }}
    header {{ background: #222; color: #fff; padding: 16px 32px; }}
    header h1 {{ margin: 0; font-size: 1.4rem; }}
    main {{ padding: 24px 36px 60px; max-width: 1400px; margin: 0 auto; }}
    .meta {{ margin-bottom: 16px; color: #555; }}
    .instruction {{ background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; }}
    .panels {{ display: flex; gap: 24px; flex-wrap: wrap; }}
    .panel {{ flex: 1 1 320px; background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); min-width: 320px; }}
    .panel h2 {{ margin-top: 0; font-size: 1.1rem; }}
    .inputs {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .input-card {{ width: 220px; background: #fafafa; border-radius: 6px; padding: 8px; box-shadow: inset 0 0 0 1px #ddd; }}
    .input-card img {{ width: 100%; border-radius: 4px; display: block; }}
    .input-label {{ font-size: 0.85rem; margin-bottom: 6px; color: #555; }}
    .candidate img {{ width: 100%; border-radius: 8px; }}
    .candidate-compare {{ position: relative; display: inline-block; width: 100%; }}
    .candidate-overlay {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      opacity: 0;
      pointer-events: none;
      transition: opacity 120ms ease-out;
    }}
    .candidate-compare.overlay-on .candidate-overlay {{ opacity: 0.5; }}
    .candidate .missing-note {{ color: #a00; font-weight: 600; }}
    .missing .missing-note {{ color: #a00; font-weight: 600; }}
    .controls {{ margin-top: 20px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
    .controls button {{ padding: 12px 24px; font-size: 1rem; border: none; border-radius: 6px; cursor: pointer; }}
    .controls .pass {{ background: #1a8917; color: #fff; }}
    .controls .fail {{ background: #c62828; color: #fff; }}
    .controls .nav {{ background: #eee; color: #333; }}
    .verdict {{ font-size: 1.1rem; font-weight: 600; }}
    .task-candidates {{ margin-top: 12px; padding-left: 18px; }}
    .task-candidates li.current a {{ font-weight: 600; }}
    .task-candidates li {{ margin-bottom: 4px; }}
    .progress {{ font-weight: 600; }}
    .filters {{ margin: 12px 0 18px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    .filters label {{ font-weight: 600; color: #333; }}
    .filters select {{ padding: 8px 10px; border-radius: 6px; border: 1px solid #bbb; }}
    .filters button {{ padding: 8px 14px; border-radius: 6px; border: none; background: #222; color: #fff; cursor: pointer; }}
  </style>
</head>
<body>
  <header>
    <h1>HYPE-EDIT-1 Judge ({html.escape(state.dataset)})</h1>
  </header>
  <main>
    <form class="filters" method="get" action="/">
      <label for="task-type">Filter by task type</label>
      <select id="task-type" name="task_type">
        {''.join(task_type_options)}
      </select>
      <button type="submit">Apply</button>
    </form>
    <div class="meta">
      <span class="progress">{progress}</span>
      <span> | Task: {html.escape(item.task_id)} | Candidate: {item.candidate_filename}</span>
      <span> | Current verdict: {verdict_text}</span>
    </div>
    <section class="instruction">
      <h2>Instruction</h2>
      <p>{instruction_html}</p>
    </section>
    <section class="panels">
      <div class="panel">
        <h2>Input image(s)</h2>
        <div class="inputs">{''.join(input_blocks)}</div>
      </div>
      <div class="panel candidate">
        <h2>Candidate output</h2>
        {candidate_block}
      </div>
      <div class="panel">
        <h2>Candidates for this task</h2>
        <ul class="task-candidates">{''.join(same_task_links)}</ul>
      </div>
    </section>
    <section class="controls">
      <form id="judge-form" method="post" action="/judge">
        <input type="hidden" name="idx" value="{idx}">
        <input type="hidden" name="task_type" value="{html.escape(task_type)}">
        <input type="hidden" name="verdict" id="verdict-input">
      </form>
      <button class="pass" onclick="submitVerdict('PASS')">PASS (P)</button>
      <button class="fail" onclick="submitVerdict('FAIL')">FAIL (F)</button>
      <button class="nav" type="button" onclick="window.location.href='{self._build_index_url(idx=prev_idx, task_type=task_type)}'">← Prev</button>
      <button class="nav" type="button" onclick="window.location.href='{self._build_index_url(idx=next_idx, task_type=task_type)}'">Next →</button>
      <button class="nav" type="button" onclick="window.location.href='{self._build_index_url(idx=next_unjudged, task_type=task_type)}'">Next unjudged (Space)</button>
    </section>
  </main>
  <script>
    const candidateCompare = document.querySelector('.candidate-compare');
    function toggleOverlay() {{
      if (!candidateCompare || candidateCompare.dataset.overlay !== 'true') return;
      candidateCompare.classList.toggle('overlay-on');
    }}
    function submitVerdict(value) {{
      document.getElementById('verdict-input').value = value;
      document.getElementById('judge-form').submit();
    }}
    document.addEventListener('keydown', function(evt) {{
      if (evt.target.tagName === 'INPUT' || evt.target.tagName === 'TEXTAREA') return;
      if (evt.key === 'p' || evt.key === 'P') {{ evt.preventDefault(); submitVerdict('PASS'); }}
      else if (evt.key === 'f' || evt.key === 'F') {{ evt.preventDefault(); submitVerdict('FAIL'); }}
      else if (evt.key === 'o' || evt.key === 'O') {{ evt.preventDefault(); toggleOverlay(); }}
      else if (evt.key === ' ') {{ evt.preventDefault(); window.location.href = '{self._build_index_url(idx=next_unjudged, task_type=task_type)}'; }}
      else if (evt.key === 'ArrowRight') {{ window.location.href = '{self._build_index_url(idx=next_idx, task_type=task_type)}'; }}
      else if (evt.key === 'ArrowLeft') {{ window.location.href = '{self._build_index_url(idx=prev_idx, task_type=task_type)}'; }}
    }});
  </script>
</body>
</html>
"""
        self._write_html(html_body)

    def _write_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_image(self, *, kind: str, path: str) -> None:
        parts = [p for p in path.split("/") if p]
        if kind == "candidate":
            if len(parts) < 4:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            _, model_id, task_id, filename = parts[0], parts[1], parts[2], "/".join(
                parts[3:]
            )
            filename = unquote(filename)
            model_name = self.state.model_id_map.get(model_id)
            if not model_name:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            file_path = (
                self.state.outputs_dir
                / self.state.dataset
                / model_name
                / task_id
                / filename
            )
            allowed_base = self.state.outputs_dir
        else:
            if len(parts) < 3:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            _, task_id, filename = parts[0], parts[1], "/".join(parts[2:])
            filename = unquote(filename)
            file_path = self.state.input_path(task_id, filename)
            allowed_base = self.state.tasks_dir

        file_path = file_path.resolve()
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            file_path.relative_to(allowed_base)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        mime, _ = mimetypes.guess_type(str(file_path))
        with open(file_path, "rb") as f:
            data = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anonymous human judging web UI for HYPE-EDIT-1"
    )
    parser.add_argument(
        "--tasks-file",
        type=str,
        default="public.json",
        help="Tasks JSON file (default: public.json)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset name (default: tasks-file stem, e.g. public/private)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="images/tasks",
        help="Directory containing task input images",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="images/outputs",
        help="Directory containing model outputs",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="judgements",
        help="Directory to write judgement JSON files",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="anonymous.json",
        help="Results filename (default: anonymous.json)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names (default: all models under outputs-dir/dataset)",
    )
    parser.add_argument(
        "--shuffle-models",
        action="store_true",
        help="Shuffle anonymous model IDs (default: false)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle task/candidate order for judging",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind (default: 8000)"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    tasks_file = Path(args.tasks_file)
    if not tasks_file.exists():
        print(f"Error: tasks file not found: {tasks_file}")
        return 2

    try:
        tasks = _load_json(tasks_file)
    except json.JSONDecodeError as exc:
        print(f"Error: failed to parse {tasks_file}: {exc}")
        return 2
    if not isinstance(tasks, list):
        print(f"Error: expected a list of tasks in {tasks_file}")
        return 2

    tasks_dir = Path(args.tasks_dir)
    outputs_dir = Path(args.outputs_dir)
    if not tasks_dir.exists():
        print(f"Error: tasks dir not found: {tasks_dir}")
        return 2
    if not outputs_dir.exists():
        print(f"Error: outputs dir not found: {outputs_dir}")
        return 2

    dataset = args.dataset.strip() or tasks_file.stem
    outputs_dataset_dir = outputs_dir / dataset
    if not outputs_dataset_dir.exists():
        print(f"Error: outputs dataset dir not found: {outputs_dataset_dir}")
        return 2

    requested_models = _parse_models_arg(args.models)
    if requested_models:
        model_names = requested_models
    else:
        model_names = sorted(
            path.name for path in outputs_dataset_dir.iterdir() if path.is_dir()
        )

    if not model_names:
        print(f"Error: no model directories found under {outputs_dataset_dir}")
        return 2

    missing_models = [
        model for model in model_names if not (outputs_dataset_dir / model).is_dir()
    ]
    if missing_models:
        print(f"Error: missing model output directories: {', '.join(missing_models)}")
        return 2

    if args.shuffle_models:
        random.shuffle(model_names)

    model_aliases, model_id_map = _build_model_aliases(model_names)

    results_name = args.results_name
    if not results_name.endswith(".json"):
        results_name += ".json"
    results_path = Path(args.results_dir) / dataset / results_name

    state = JudgeState(
        tasks=tasks,
        tasks_dir=tasks_dir,
        outputs_dir=outputs_dir,
        dataset=dataset,
        results_path=results_path,
        model_names=model_names,
        model_aliases=model_aliases,
        model_id_map=model_id_map,
        shuffle=args.shuffle,
    )

    if state.total_items() == 0:
        print("No tasks found. Exiting.")
        return 0

    handler_class = JudgeRequestHandler
    handler_class.state = state

    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    addr = server.server_address
    print("=" * 72)
    print("Anonymous human judge web UI running")
    print(f"Dataset    : {dataset}")
    print(f"Tasks file : {tasks_file}")
    print(f"Results    : {results_path}")
    print(f"Models     : {len(model_names)} (anonymous)")
    print(f"Open       : http://{addr[0]}:{addr[1]}")
    print("Press Ctrl+C to stop")
    print("=" * 72)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

