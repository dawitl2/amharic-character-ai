import io
import json
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translation import (  # noqa: E402
    MyMemoryTranslationProvider,
    TranslationError,
    _utf8_chunks,
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TranslationTests(unittest.TestCase):
    def test_mymemory_request_needs_no_api_key(self):
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            payload = {"responseData": {"translatedText": "Hello"}}
            return FakeResponse(json.dumps(payload).encode("utf-8"))

        provider = MyMemoryTranslationProvider(opener=opener, timeout_seconds=3.0)
        result = provider.translate("ሰላም")
        query = parse_qs(urlparse(captured["url"]).query)
        self.assertEqual(query["langpair"], ["am|en"])
        self.assertNotIn("key", query)
        self.assertEqual(captured["timeout"], 3.0)
        self.assertEqual(result.translated_text, "Hello")
        self.assertEqual(result.provider, "MyMemory")

    def test_long_text_is_split_below_provider_byte_limit(self):
        chunks = _utf8_chunks(" ".join(["ሰላም"] * 100), maximum_bytes=60)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 60 for chunk in chunks))

    def test_empty_text_fails_without_calling_provider(self):
        provider = MyMemoryTranslationProvider(
            opener=lambda *args, **kwargs: self.fail("network should not be called")
        )
        with self.assertRaisesRegex(TranslationError, "empty"):
            provider.translate("   ")


if __name__ == "__main__":
    unittest.main()
