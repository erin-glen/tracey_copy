import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from utils.config_utils import normalize_langfuse_base_url
from utils.langfuse_api import create_score, fetch_traces_window


class TestLangfuseCredentialsAndCache(unittest.TestCase):
    def test_normalize_langfuse_base_url(self):
        self.assertEqual(normalize_langfuse_base_url("https://x.y/api/public"), "https://x.y")
        self.assertEqual(normalize_langfuse_base_url("https://x.y/api/public/"), "https://x.y")
        self.assertEqual(normalize_langfuse_base_url("https://x.y/"), "https://x.y")
        self.assertEqual(normalize_langfuse_base_url("  "), "")

    @patch("utils.langfuse_api.requests.post")
    def test_create_score_queue_id_does_not_crash_without_metadata(self, mock_post):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"id": "s1"}
        mock_post.return_value = response

        create_score(
            base_url="https://example.com",
            headers={"Authorization": "Basic abc"},
            trace_id="t1",
            name="score_name",
            value=1,
            metadata=None,
            queue_id="q1",
        )

        self.assertTrue(mock_post.called)
        payload = mock_post.call_args.kwargs["json"]
        self.assertIsInstance(payload.get("metadata"), dict)
        self.assertEqual(payload["metadata"].get("queue_id"), "q1")

    @patch("utils.langfuse_api.requests.post")
    def test_create_score_base_url_normalization_avoids_double_api_public(self, mock_post):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"id": "s1"}
        mock_post.return_value = response

        create_score(
            base_url="https://example.com/api/public",
            headers={"Authorization": "Basic abc"},
            trace_id="t1",
            name="score_name",
            value=1,
        )

        called_url = mock_post.call_args.args[0]
        self.assertEqual(called_url, "https://example.com/api/public/scores")

    @patch("utils.langfuse_api.requests.Session")
    def test_fetch_traces_window_does_not_write_cache_on_error(self, mock_session_cls):
        response = MagicMock()
        response.status_code = 401
        response.text = "unauthorized"
        mock_session = MagicMock()
        mock_session.get.return_value = response
        mock_session_cls.return_value = mock_session

        with tempfile.TemporaryDirectory() as tempdir:
            rows = fetch_traces_window(
                base_url="https://example.com",
                headers={"Authorization": "Basic abc"},
                from_iso="2024-01-01T00:00:00Z",
                to_iso="2024-01-02T00:00:00Z",
                envs=["production"],
                page_size=100,
                page_limit=1,
                max_traces=100,
                retry=0,
                backoff=0.1,
                use_disk_cache=True,
                cache_dir=tempdir,
            )

            self.assertEqual(rows, [])
            self.assertEqual(list(Path(tempdir).glob("traces_*.json")), [])


if __name__ == "__main__":
    unittest.main()
