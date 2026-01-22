"""
Gemini 3 Flash judge for HYPE-EDIT-1.

Evaluates whether an output image satisfies the instruction given the reference image(s).
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import httpx
from google import genai
from google.genai import types

ImageInput = Union[str, Path, bytes]


class Gemini3FlashJudge:
    """Judge that returns PASS/FAIL plus model reasoning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3-flash-preview",
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.model = model

    def judge(
        self,
        reference_images: Sequence[ImageInput],
        instruction: str,
        output_image: ImageInput,
    ) -> Tuple[bool, str, str]:
        """
        Compare reference images + instruction against the output image.

        Returns:
            (passed, reasoning)
        """
        if not reference_images:
            raise ValueError("At least one reference image is required")
        if not instruction.strip():
            raise ValueError("Instruction must be non-empty")

        parts = []
        prompt = (
            "You are a strict image-editing judge.\n\n"
            "Input:\n"
            "- INSTRUCTION: the user's edit request\n"
            "- REFERENCE IMAGE(S): the original image(s) before editing (may be 1 or more)\n"
            "- CANDIDATE OUTPUT: the edited result to evaluate\n\n"
            "Task:\n"
            "1) Interpret the INSTRUCTION as a checklist of REQUIRED CHANGES and REQUIRED "
            "PRESERVATIONS (anything not mentioned should remain identical to the reference).\n"
            "2) Compare the CANDIDATE OUTPUT against the REFERENCE IMAGE(S):\n"
            "   - Verify every required change is present and correct (object/region, size, "
            "position, color, text, count, identity, pose, background, lighting as applicable).\n"
            "   - Verify no unintended changes occurred anywhere else (including: "
            "cropping/aspect ratio, global color/contrast shifts, added/removed objects, "
            "altered text/logos, geometry distortion, face/identity changes, background changes, "
            "blur/sharpening, artifacts).\n"
            "3) Be conservative: mark FAIL if any required change is missing/partial/wrong, if "
            "any unrelated element changed, or if the instruction is ambiguous and cannot be "
            "confidently verified from the images.\n"
            "   - Ignore only negligible differences consistent with normal compression "
            "(minor pixel noise) that do not change content or appearance materially.\n\n"
            "Output rules:\n"
            "- Respond ONLY with valid JSON exactly in this format:\n"
            '  {"verdict":"PASS"|"FAIL","reasoning":"..."}\n'
            "- reasoning must be 1-3 full sentences.\n"
            "- If FAIL, explicitly name the mismatches (missing/incorrect required edits) and "
            "any unintended changes.\n"
            "- If PASS, briefly state that all required edits are satisfied and no unintended "
            "changes are present."
        )

        parts.append(types.Part.from_text(text=prompt))

        for idx, image in enumerate(reference_images, start=1):
            parts.append(types.Part.from_text(text=f"Reference image {idx}:"))
            parts.append(self._image_part(image))

        parts.append(types.Part.from_text(text=f"Instruction:\n{instruction}"))
        parts.append(types.Part.from_text(text="Candidate output image:"))
        parts.append(self._image_part(output_image))
        parts.append(types.Part.from_text(text=prompt))

        client = genai.Client(api_key=self.api_key)
        config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=4096,
            response_modalities=["TEXT"],
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=types.ThinkingLevel.MEDIUM,
            ),
        )
        response = client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=config,
        )

        final_text, thought_text = self._extract_response_text(response)
        verdict, reasoning = self._parse_verdict(final_text)
        return verdict == "PASS", reasoning, thought_text

    def _image_part(self, image: ImageInput) -> types.Part:
        image_bytes = self._load_image_bytes(image)
        mime_type = self._detect_mime_type(image_bytes)
        try:
            return types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
                media_resolution={"level": "media_resolution_high"},
            )
        except TypeError:
            # Older SDKs don't support media_resolution yet.
            return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    @staticmethod
    def _load_image_bytes(image: ImageInput) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, Path):
            return image.read_bytes()

        if image.startswith("data:"):
            header, data = image.split(",", 1)
            if ";base64" in header:
                return base64.b64decode(data)
            return data.encode("utf-8")

        if image.startswith("http://") or image.startswith("https://"):
            with httpx.Client(timeout=60.0) as client:
                response = client.get(image)
                response.raise_for_status()
                return response.content

        return Path(image).read_bytes()

    @staticmethod
    def _detect_mime_type(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\x89PNG"):
            return "image/png"
        if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
            return "image/webp"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        return "application/octet-stream"

    def _extract_response_text(self, response: object) -> Tuple[str, str]:
        if hasattr(response, "text") and response.text:
            return response.text.strip(), ""
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None)
                if not parts:
                    continue
                final_texts = []
                thought_texts = []
                for part in parts:
                    text = getattr(part, "text", "") or ""
                    thought = getattr(part, "thought", None)
                    if isinstance(thought, str) and thought:
                        thought_texts.append(thought)
                    if text:
                        if thought is True:
                            thought_texts.append(text)
                        else:
                            final_texts.append(text)
                final_joined = "\n".join(final_texts).strip()
                thought_joined = "\n".join(thought_texts).strip()
                if final_joined:
                    return final_joined, thought_joined
                if thought_joined:
                    debug = self._summarize_response(response, thought_joined)
                    raise RuntimeError("No final text response from Gemini. " + debug)
        debug = self._summarize_response(response, "")
        raise RuntimeError("No text response received from Gemini. " + debug)

    @staticmethod
    def _summarize_response(response: object, thought_text: str) -> str:
        parts = []
        candidates = getattr(response, "candidates", None)
        if candidates is not None:
            parts.append(f"candidates={len(candidates)}")
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            parts.append(f"prompt_feedback={prompt_feedback}")
        if thought_text:
            excerpt = thought_text[:200].replace("\n", " ")
            parts.append(f"thought_excerpt={excerpt!r}")
        return " ".join(parts).strip()

    @staticmethod
    def _parse_verdict(text: str) -> Tuple[str, str]:
        try:
            data = json.loads(text)
            verdict = str(data.get("verdict", "")).strip().upper()
            reasoning = str(data.get("reasoning", "")).strip()
            if verdict in {"PASS", "FAIL"} and reasoning:
                return verdict, reasoning
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                verdict = str(data.get("verdict", "")).strip().upper()
                reasoning = str(data.get("reasoning", "")).strip()
                if verdict in {"PASS", "FAIL"} and reasoning:
                    return verdict, reasoning
            except json.JSONDecodeError:
                pass

        verdict_match = re.search(r'"verdict"\s*:\s*"(PASS|FAIL)"', text, re.IGNORECASE)
        verdict = verdict_match.group(1).upper() if verdict_match else "FAIL"
        reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*)', text, re.DOTALL)
        if reasoning_match:
            raw_reasoning = reasoning_match.group(1)
            # Trim at the last quote if present to avoid runaway content.
            if '"' in raw_reasoning:
                raw_reasoning = raw_reasoning.split('"', 1)[0]
            reasoning = raw_reasoning.replace("\\n", "\n").replace('\\"', '"').strip()
            if reasoning:
                return verdict, reasoning

        return "FAIL", f"Unparseable model response: {text}"
