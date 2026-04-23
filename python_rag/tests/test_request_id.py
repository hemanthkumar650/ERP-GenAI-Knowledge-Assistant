from __future__ import annotations

import re
import unittest

from request_id import assign_request_id, normalize_incoming


class RequestIdUtilTests(unittest.TestCase):
    def test_normalize_accepts_safe_token(self) -> None:
        self.assertEqual(normalize_incoming("  gw-trace_1.ok  "), "gw-trace_1.ok")

    def test_normalize_rejects_empty_and_oversize(self) -> None:
        self.assertIsNone(normalize_incoming(""))
        self.assertIsNone(normalize_incoming("   "))
        self.assertIsNone(normalize_incoming("a" * 129))

    def test_normalize_rejects_unsafe_chars(self) -> None:
        self.assertIsNone(normalize_incoming("bad id"))
        self.assertIsNone(normalize_incoming("x;y"))

    def test_assign_generates_uuid_when_missing_or_invalid(self) -> None:
        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        self.assertTrue(uuid_re.fullmatch(assign_request_id(None)))
        self.assertTrue(uuid_re.fullmatch(assign_request_id("not valid!!!")))

    def test_assign_reuses_valid_header(self) -> None:
        self.assertEqual(assign_request_id("upstream-abc"), "upstream-abc")


if __name__ == "__main__":
    unittest.main()
