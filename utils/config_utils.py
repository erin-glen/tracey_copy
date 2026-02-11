"""Configuration helpers for resolving Langfuse credentials across sources."""

from collections.abc import Mapping
from typing import Any


def normalize_langfuse_base_url(raw: str | None) -> str:
    """Normalize user-provided Langfuse base URL values."""
    if raw is None:
        return ""

    cleaned = str(raw).strip()
    if not cleaned:
        return ""

    cleaned = cleaned.rstrip("/")
    suffix = "/api/public"
    if cleaned.lower().endswith(suffix):
        cleaned = cleaned[: -len(suffix)]

    return cleaned.rstrip("/")


def get_nested(mapping: Mapping[str, Any] | None, path: tuple[str, ...]) -> Any:
    """Safely fetch a nested mapping value for a tuple path."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _clean_candidate(value: Any) -> str:
    if value is None:
        return ""
    out = str(value).strip()
    return out


def _resolve_value(candidates: list[tuple[str, Any]]) -> tuple[str, str]:
    for source, raw in candidates:
        value = _clean_candidate(raw)
        if value:
            return value, source
    return "", "missing"


def resolve_langfuse_config(
    session: Mapping[str, Any] | None,
    secrets: Mapping[str, Any] | None,
    env: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve Langfuse credentials from session, secrets, and environment sources."""
    session_map = session if isinstance(session, Mapping) else {}
    secrets_map = secrets if isinstance(secrets, Mapping) else {}
    env_map = env if isinstance(env, Mapping) else {}

    public_key, public_source = _resolve_value(
        [
            ("session", session_map.get("langfuse_public_key")),
            ("secrets", secrets_map.get("LANGFUSE_PUBLIC_KEY")),
            ("secrets", get_nested(secrets_map, ("langfuse", "public_key"))),
            ("env", env_map.get("LANGFUSE_PUBLIC_KEY")),
        ]
    )
    secret_key, secret_source = _resolve_value(
        [
            ("session", session_map.get("langfuse_secret_key")),
            ("secrets", secrets_map.get("LANGFUSE_SECRET_KEY")),
            ("secrets", get_nested(secrets_map, ("langfuse", "secret_key"))),
            ("env", env_map.get("LANGFUSE_SECRET_KEY")),
        ]
    )
    base_url_raw, base_source = _resolve_value(
        [
            ("session", session_map.get("langfuse_base_url")),
            ("secrets", secrets_map.get("LANGFUSE_BASE_URL")),
            ("secrets", get_nested(secrets_map, ("langfuse", "base_url"))),
            ("env", env_map.get("LANGFUSE_BASE_URL")),
        ]
    )

    return {
        "public_key": public_key,
        "secret_key": secret_key,
        "base_url": normalize_langfuse_base_url(base_url_raw),
        "sources": {
            "public_key": public_source,
            "secret_key": secret_source,
            "base_url": base_source,
        },
    }



def resolve_app_password(
    session: Mapping[str, Any] | None,
    secrets: Mapping[str, Any] | None,
    env: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve application password from session, secrets, and environment sources."""
    session_map = session if isinstance(session, Mapping) else {}
    secrets_map = secrets if isinstance(secrets, Mapping) else {}
    env_map = env if isinstance(env, Mapping) else {}

    password, source = _resolve_value(
        [
            ("session", session_map.get("app_password")),
            ("secrets", secrets_map.get("APP_PASSWORD")),
            ("secrets", get_nested(secrets_map, ("auth", "password"))),
            ("secrets", get_nested(secrets_map, ("auth", "APP_PASSWORD"))),
            ("secrets", secrets_map.get("password")),
            ("env", env_map.get("APP_PASSWORD")),
        ]
    )

    return {"password": password, "source": source}
