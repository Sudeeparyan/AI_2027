"""Offline retry and failure classification checks for check_links.py.

Run: .venv/Scripts/python.exe tools/test_link_checks.py
"""
import json
import unittest
from unittest.mock import patch

import requests

import check_links


def response(status, data=None):
    result = requests.Response()
    result.status_code = status
    result.url = "https://test.invalid/final"
    result._content = json.dumps(data or {}).encode("utf-8")
    result._content_consumed = True
    return result


def html_response(status, text):
    result = response(status)
    result.headers["Content-Type"] = "text/html; charset=utf-8"
    result.encoding = "utf-8"
    result._content = text.encode("utf-8")
    return result


ROBOT_CHALLENGE = """<!doctype html><html><body>
JavaScript is disabled. Please verify that you&#39;re not a robot.
This verification requires JavaScript.
</body></html>"""


class LinkChecks(unittest.TestCase):
    def check(self, outcomes, url="https://test.invalid/page"):
        with patch.object(check_links.requests, "get", side_effect=outcomes) as get, \
                patch.object(check_links.time, "sleep") as sleep:
            result = check_links.check({"id": "test", "url": url})
        self.assertEqual(result["attempt_count"], get.call_count)
        self.assertEqual(len(result["attempts"]), get.call_count)
        self.assertLessEqual(get.call_count, 2)
        return result, get, sleep

    def test_transient_recovery(self):
        for status in (500, 502, 503, 504):
            with self.subTest(status=status):
                result, get, sleep = self.check([response(status), response(200)])
                self.assertTrue(result["ok"])
                self.assertEqual(result["status"], 200)
                self.assertEqual([a["status"] for a in result["attempts"]], [status, 200])
                self.assertEqual([a["attempt"] for a in result["attempts"]], [1, 2])
                self.assertEqual(get.call_count, 2)
                sleep.assert_called_once_with(0.5)

    def test_exhausted_503_remains_failed(self):
        result, get, sleep = self.check([response(503), response(503)])
        self.assertFalse(result["ok"])
        self.assertFalse(result.get("blocked", False))
        self.assertEqual(result["status"], 503)
        self.assertEqual([a["status"] for a in result["attempts"]], [503, 503])
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_404_is_not_retried(self):
        result, get, sleep = self.check([response(404)])
        self.assertFalse(result["ok"])
        self.assertFalse(result.get("blocked", False))
        self.assertEqual(result["status"], 404)
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_403_remains_blocked(self):
        result, get, sleep = self.check([response(403)])
        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["status"], 403)
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_http_200_html_robot_challenge_is_unverified(self):
        result, get, sleep = self.check([html_response(200, ROBOT_CHALLENGE)])
        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["blocked_reason"], "html_robot_challenge")
        self.assertEqual(result["status"], 200)
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_202_explicit_aws_waf_challenge_is_unverified(self):
        challenge = html_response(202, "")
        challenge.headers["x-amzn-waf-action"] = "challenge"
        result, get, sleep = self.check([challenge])
        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["blocked_reason"], "aws_waf_challenge")
        self.assertEqual(result["status"], 202)
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_ordinary_202_without_challenge_header_is_unchanged(self):
        result, _, _ = self.check([response(202)])
        self.assertTrue(result["ok"])
        self.assertFalse(result.get("blocked", False))

    def test_404_with_challenge_header_remains_failure(self):
        missing = response(404)
        missing.headers["x-amzn-waf-action"] = "challenge"
        result, _, _ = self.check([missing])
        self.assertFalse(result["ok"])
        self.assertFalse(result.get("blocked", False))

    def test_html_robot_challenge_without_content_type_is_unverified(self):
        challenge = html_response(200, ROBOT_CHALLENGE)
        challenge.headers.pop("Content-Type")
        result, _, _ = self.check([challenge])
        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])

    def test_ordinary_javascript_requirement_remains_reachable(self):
        result, _, _ = self.check([html_response(200, "<html><body>This interactive chart requires JavaScript.</body></html>")])
        self.assertTrue(result["ok"])
        self.assertFalse(result.get("blocked", False))

    def test_hidden_script_challenge_strings_are_not_page_challenge(self):
        page = "<html><body>Research paper<script>const disabled = \"JavaScript is disabled. Verify that you're not a robot\";</script></body></html>"
        result, _, _ = self.check([html_response(200, page)])
        self.assertTrue(result["ok"])
        self.assertFalse(result.get("blocked", False))

    def test_binary_pdf_is_not_challenge_html(self):
        pdf = response(200)
        pdf.headers["Content-Type"] = "application/pdf"
        pdf._content = b"%PDF-1.7 JavaScript is disabled. Please verify that you're not a robot."
        result, _, _ = self.check([pdf])
        self.assertTrue(result["ok"])
        self.assertFalse(result.get("blocked", False))

    def test_404_with_challenge_text_remains_failure(self):
        result, get, sleep = self.check([html_response(404, ROBOT_CHALLENGE)])
        self.assertFalse(result["ok"])
        self.assertFalse(result.get("blocked", False))
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_exhausted_timeout_remains_failure(self):
        result, get, sleep = self.check([requests.Timeout("timeout"), requests.Timeout("timeout")])
        self.assertFalse(result["ok"])
        self.assertFalse(result.get("blocked", False))
        self.assertIsNone(result["status"])
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_transport_recovery(self):
        for error in (requests.ConnectionError("connection reset"), requests.Timeout("timeout"),
                      requests.exceptions.ChunkedEncodingError("incomplete response")):
            with self.subTest(error=type(error).__name__):
                result, get, sleep = self.check([error, response(200)])
                self.assertTrue(result["ok"])
                self.assertEqual(result["status"], 200)
                self.assertEqual([a["status"] for a in result["attempts"]], [None, 200])
                self.assertEqual(result["attempts"][0]["error_type"], type(error).__name__)
                self.assertEqual(result["attempts"][0]["error"], str(error))
                self.assertNotIn("error", result)
                self.assertEqual(get.call_count, 2)
                sleep.assert_called_once_with(0.5)

    def test_nontransient_server_status_is_not_retried(self):
        result, get, sleep = self.check([response(501)])
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], 501)
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_nontransport_request_error_is_not_retried(self):
        result, get, sleep = self.check([requests.exceptions.InvalidURL("invalid URL")])
        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"][0]["error_type"], "InvalidURL")
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_removed_video_remains_failed(self):
        result, get, sleep = self.check([response(404)], "https://www.youtube.com/watch?v=removed")
        self.assertFalse(result["ok"])
        self.assertFalse(result.get("blocked", False))
        self.assertEqual(result["status"], 404)
        self.assertIsNone(result["title"])
        self.assertTrue(get.call_args.args[0].startswith("https://www.youtube.com/oembed?"))
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_youtube_oembed_uses_same_retry(self):
        video = {"title": "Present video", "author_name": "Test author"}
        result, get, sleep = self.check([response(503), response(200, video)],
                                        "https://www.youtube.com/watch?v=present")
        self.assertTrue(result["ok"])
        self.assertEqual(result["title"], video["title"])
        self.assertEqual(result["author"], video["author_name"])
        self.assertEqual([a["status"] for a in result["attempts"]], [503, 200])
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
