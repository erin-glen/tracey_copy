import unittest

from utils.config_utils import resolve_app_password


class TestAuthPasswordResolution(unittest.TestCase):
    def test_secrets_flat_app_password_wins_over_env(self):
        resolved = resolve_app_password(
            session={},
            secrets={"APP_PASSWORD": "secrets-pw"},
            env={"APP_PASSWORD": "env-pw"},
        )

        self.assertEqual(resolved["password"], "secrets-pw")
        self.assertEqual(resolved["source"], "secrets")

    def test_secrets_nested_auth_password_works(self):
        resolved = resolve_app_password(
            session={},
            secrets={"auth": {"password": "nested-pw"}},
            env={},
        )

        self.assertEqual(resolved["password"], "nested-pw")
        self.assertEqual(resolved["source"], "secrets")

    def test_secrets_password_key_works(self):
        resolved = resolve_app_password(
            session={},
            secrets={"password": "legacy-pw"},
            env={},
        )

        self.assertEqual(resolved["password"], "legacy-pw")
        self.assertEqual(resolved["source"], "secrets")

    def test_env_works_when_secrets_missing(self):
        resolved = resolve_app_password(
            session={},
            secrets={},
            env={"APP_PASSWORD": "env-pw"},
        )

        self.assertEqual(resolved["password"], "env-pw")
        self.assertEqual(resolved["source"], "env")

    def test_missing_returns_empty_password_and_missing_source(self):
        resolved = resolve_app_password(
            session={},
            secrets={},
            env={},
        )

        self.assertEqual(resolved["password"], "")
        self.assertEqual(resolved["source"], "missing")


if __name__ == "__main__":
    unittest.main()
