"""Replaceable translation providers for optional post-OCR translation."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_MYMEMORY_ENDPOINT = "https://api.mymemory.translated.net/get"


class TranslationError(RuntimeError):
    """A user-facing translation failure that does not invalidate OCR output."""


@dataclass(frozen=True)
class TranslationResult:
    translated_text: str
    provider: str
    source_language: str
    target_language: str


def _utf8_chunks(text: str, maximum_bytes: int = 450) -> list[str]:
    """Respect MyMemory's 500-byte segment limit without splitting code points."""
    words = text.split()
    if not words:
        return []
    chunks = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate.encode("utf-8")) <= maximum_bytes:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(word.encode("utf-8")) <= maximum_bytes:
            current = word
            continue
        fragment = ""
        for character in word:
            candidate_fragment = fragment + character
            if len(candidate_fragment.encode("utf-8")) > maximum_bytes:
                chunks.append(fragment)
                fragment = character
            else:
                fragment = candidate_fragment
        current = fragment
    if current:
        chunks.append(current)
    return chunks


class MyMemoryTranslationProvider:
    """Documented, keyless MyMemory REST integration for the project demo."""

    name = "MyMemory"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        contact_email: str | None = None,
        timeout_seconds: float = 8.0,
        opener: Callable = urlopen,
    ):
        self.endpoint = endpoint or os.environ.get(
            "ETHIOPIC_TRANSLATION_ENDPOINT", DEFAULT_MYMEMORY_ENDPOINT
        )
        self.contact_email = contact_email or os.environ.get(
            "ETHIOPIC_TRANSLATION_EMAIL"
        )
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def _translate_chunk(
        self, text: str, source_language: str, target_language: str
    ) -> str:
        parameters = {
            "q": text,
            "langpair": f"{source_language}|{target_language}",
            "mt": "1",
        }
        if self.contact_email:
            parameters["de"] = self.contact_email
        request = Request(
            f"{self.endpoint}?{urlencode(parameters)}",
            headers={"Accept": "application/json", "User-Agent": "Ethiopic-OCR/1.0"},
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise TranslationError(
                f"{self.name} translation is unavailable: {error}"
            ) from error
        translated = payload.get("responseData", {}).get("translatedText")
        if not isinstance(translated, str) or not translated.strip():
            details = payload.get("responseDetails") or "empty translation response"
            raise TranslationError(f"{self.name} could not translate the text: {details}")
        return html.unescape(translated.strip())

    def translate(
        self,
        text: str,
        *,
        source_language: str = "am",
        target_language: str = "en",
    ) -> TranslationResult:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise TranslationError("Recognized Amharic text is empty.")
        chunks = _utf8_chunks(cleaned)
        translated = " ".join(
            self._translate_chunk(chunk, source_language, target_language)
            for chunk in chunks
        )
        return TranslationResult(
            translated_text=translated,
            provider=self.name,
            source_language=source_language,
            target_language=target_language,
        )
