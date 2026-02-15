"""Deterministic content/structural KPI analysis over in-memory Langfuse traces."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from utils.trace_parsing import (
    current_human_prompt,
    current_turn_ai_message,
    normalize_trace_format,
    parse_trace_dt,
    slice_output_to_current_turn,
)
from utils.codeact_utils import find_codeact_parts, iter_decoded_codeact_parts

SCORED_INTENTS = {"trend_over_time", "data_lookup"}
POSITIVE_ACK_PATTERNS = [
    "thanks",
    "thank you",
    "great",
    "perfect",
    "awesome",
    "that helps",
    "sounds good",
]
NEGATIVE_ACK_PATTERNS = [
    "not what i asked",
    "doesn't answer",
    "wrong",
    "try again",
    "no",
    "huh",
]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    if isinstance(content, list):
        bits: list[str] = []
        for part in content:
            txt = _content_to_text(part)
            if txt:
                bits.append(txt)
        return "\n".join(bits).strip()
    return ""


def _message_role(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    return str(msg.get("type") or msg.get("role") or "").strip().lower()


def _is_assistant(msg: Any) -> bool:
    return _message_role(msg) in {"assistant", "ai"}


def _is_user(msg: Any) -> bool:
    return _message_role(msg) in {"user", "human"}


def _find_first_key(obj: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] not in (None, "", []):
                return obj[key]
        for value in obj.values():
            found = _find_first_key(value, keys)
            if found not in (None, "", []):
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_key(item, keys)
            if found not in (None, "", []):
                return found
    return None


def _extract_struct_flags(output_obj: dict[str, Any] | None) -> dict[str, Any]:
    output_obj = output_obj or {}

    # Selected AOI: prefer explicit top-level keys; avoid treating candidate AOIs (e.g. under
    # `aoi_options`) as if they were selected.
    aoi = None
    if isinstance(output_obj, dict):
        for k in ("selected_aoi", "selectedAOI", "aoi"):
            if k in output_obj and output_obj[k] not in (None, "", []):
                aoi = output_obj[k]
                break
        if aoi is None:
            # common wrappers
            for wrapper in ("result", "payload", "data"):
                w = output_obj.get(wrapper)
                if isinstance(w, dict):
                    for k in ("selected_aoi", "selectedAOI", "aoi"):
                        if k in w and w[k] not in (None, "", []):
                            aoi = w[k]
                            break
                if aoi is not None:
                    break
    aoi_selected = False
    if isinstance(aoi, dict):
        aoi_selected = bool(
            str(aoi.get("name") or "").strip()
            or str(aoi.get("id") or "").strip()
            or str(aoi.get("src_id") or "").strip()
            or str(aoi.get("gadm_id") or "").strip()
        )
    elif isinstance(aoi, str):
        aoi_selected = bool(aoi.strip())

    aoi_options = _find_first_key(
        output_obj, ("aoi_options", "aoiOptions", "aoi_candidates", "aoiCandidates", "candidates", "aois")
    )
    aoi_options_count = int(len(aoi_options)) if isinstance(aoi_options, list) else 0

    def _aoi_opt_id(opt: Any) -> str:
        if not isinstance(opt, dict):
            return ""
        inner = opt.get("aoi")
        if isinstance(inner, dict):
            for k in ("gadm_id", "src_id", "id", "name"):
                v = inner.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        for k in ("gadm_id", "src_id", "id", "name"):
            v = opt.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    aoi_options_unique_count = 0
    if isinstance(aoi_options, list) and aoi_options:
        ids = {i for i in (_aoi_opt_id(o) for o in aoi_options) if i}
        aoi_options_unique_count = len(ids) if ids else len(aoi_options)

    aoi_candidates = bool(aoi_options_unique_count > 1)

    # Time keys are explicitly named in GNW outputs. Avoid generic keys like "to" which
    # may appear in unrelated nested structures (e.g., raw_data payloads).
    time_start = _find_first_key(output_obj, ("start_date", "startDate", "time_start", "timeStart"))
    time_end = _find_first_key(output_obj, ("end_date", "endDate", "time_end", "timeEnd"))
    time_range_struct = bool(time_start and time_end)

    dataset_name = _extract_dataset_name(output_obj)
    dataset_struct = bool(dataset_name)

    citations = _find_first_key(
        output_obj,
        (
            "citation",
            "citations",
            "sources",
            "sourceUrls",
            "source_urls",
            "source_url",
            "references",
        ),
    )
    citations_struct = bool(citations)

    dataset_obj = _find_first_key(output_obj, ("dataset", "dataset_info", "datasetInfo", "layer", "collection"))
    dataset_has_citation = False
    if isinstance(dataset_obj, dict):
        ds_cit = dataset_obj.get("citation") or dataset_obj.get("citations")
        dataset_has_citation = bool(isinstance(ds_cit, str) and ds_cit.strip())

    aoi_name = ""
    aoi_type = ""
    if isinstance(aoi, dict):
        aoi_name = str(aoi.get("name") or aoi.get("title") or "").strip()
        # GNW AOI objects typically carry subtype/source rather than a generic "type".
        aoi_type = str(aoi.get("subtype") or aoi.get("type") or aoi.get("source") or "").strip()

    return {
        "aoi_selected_struct": aoi_selected,
        "aoi_candidates_struct": aoi_candidates,
        "aoi_options_count": aoi_options_count,
        "aoi_options_unique_count": aoi_options_unique_count,
        "time_range_struct": time_range_struct,
        "dataset_struct": dataset_struct,
        "citations_struct": citations_struct,
        "dataset_has_citation": dataset_has_citation,
        "dataset_name": dataset_name,
        "aoi_name": aoi_name,
        "aoi_type": aoi_type,
        "time_start": str(time_start or "").strip(),
        "time_end": str(time_end or "").strip(),
    }


def _extract_dataset_name(output_obj: dict[str, Any]) -> str:
    """Best-effort extraction of a dataset/layer name from heterogeneous output shapes.

    Common GNW shapes include:
    - output["dataset"] as a dict containing dataset_name
    - output["result"]["dataset"] as a dict
    - output["dataset_name"] as a top-level string

    Important: avoid generic `name` matches outside a dataset/layer container (e.g., AOI name).
    """

    if not isinstance(output_obj, dict):
        return ""

    container_keys = ("dataset", "datasets", "layer", "layers", "collection")
    name_keys = ("dataset_name", "datasetName", "layer_name", "layerName", "name", "title")
    explicit_name_keys = ("dataset_name", "datasetName", "layer_name", "layerName")

    def _extract_from_dataset_container(container: Any) -> str:
        if isinstance(container, str) and container.strip():
            return container.strip()
        if isinstance(container, dict):
            for k in name_keys:
                v = container.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            nested = _find_first_key(container, explicit_name_keys)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
            return ""
        if isinstance(container, list):
            for item in container:
                name = _extract_from_dataset_container(item)
                if name:
                    return name
        return ""

    for direct_block in (
        _find_first_key(output_obj, container_keys),
        _find_first_key(output_obj, ("dataset_info", "datasetInfo")),
    ):
        name = _extract_from_dataset_container(direct_block)
        if name:
            return name

    wrapper = _find_first_key(output_obj, ("result", "data", "payload"))
    if isinstance(wrapper, dict):
        for ck in container_keys:
            name = _extract_from_dataset_container(_find_first_key(wrapper, (ck,)))
            if name:
                return name
        nested = _find_first_key(wrapper, explicit_name_keys)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()

    found = _find_first_key(output_obj, explicit_name_keys)
    if isinstance(found, str) and found.strip():
        return found.strip()

    return ""


def _text_flags(prompt: str, response: str) -> dict[str, bool]:
    p = (prompt or "").lower()
    r = (response or "").lower()
    combined = f"{p}\n{r}"
    return {
        "aoi_text": any(
            k in combined
            for k in [
                "aoi",
                "area",
                "location",
                "region",
                "country",
                "state",
                "province",
                "district",
                "city",
                # ES/PT (best-effort)
                "ubicación",
                "ubicacion",
                "región",
                "region",
                "país",
                "pais",
                "estado",
                "provincia",
                "município",
                "municipio",
            ]
        ),
        # Avoid the bare token "to" which appears in many generic phrases ("would you like to...")
        "time_text": bool(
            re.search(r"\b(19|20)\d{2}\b", combined)
            or any(
                k in combined
                for k in [
                    "year",
                    "month",
                    "time",
                    "time range",
                    "date range",
                    "between",
                    "since",
                    "from",
                    "over time",
                    "trend",
                    # ES/PT/ID (best-effort)
                    "entre",
                    "desde",
                    "a partir",
                    "últimos",
                    "ultimos",
                    "anos",
                    "años",
                    "anos",
                    "sejak",
                    "selama",
                ]
            )
        ),
        "dataset_text": any(
            k in combined
            for k in [
                "dataset",
                "data set",
                "layer",
                "collection",
                "tree cover",
                "land cover",
                "alerts",
                "mangrove",
                "peatland",
                "grassland",
                # ES/PT (best-effort)
                "capa",
                "camada",
                "conjunto de datos",
            ]
        ),
        "citations_text": any(k in combined for k in ["source", "sources", "citation", "doi", "http://", "https://"]),
    }


# --- Intent helpers ---------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _looks_like_parameter_refinement(prompt: str) -> bool:
    """Heuristic for short follow-up turns.

    Examples:
    - "Cambodia"
    - "entre 2020 y 2024"
    - "2021"

    We keep this intentionally broad but exclude common question starts.
    """

    t = (prompt or "").strip()
    if not t:
        return False

    low = t.lower()

    # Greetings / language-selection messages are not parameter refinements.
    if re.fullmatch(
        r"(hi|hello|hey|hola|halo|hallo|bonjour|salut|ciao|ola|olá|namaste|good (morning|afternoon|evening)|buenos d[ií]as|buenas (tardes|noches)|selamat (pagi|siang|sore|malam))[\s!?.]*",
        low,
    ):
        return False

    if low.strip(" !?.") in {
        "english",
        "spanish",
        "español",
        "espanol",
        "portuguese",
        "português",
        "bahasa",
        "bahasa indonesia",
        "indonesian",
        "français",
        "french",
        "deutsch",
        "german",
    }:
        return False

    # If it looks like a fresh data request (not a follow-up value), don't treat it as a refinement.
    if _looks_like_trend_request(t) or _looks_like_data_lookup_request(t):
        return False

    # Pure year or ISO date / year range
    if re.fullmatch(r"(19|20)\d{2}", low):
        return True
    if re.fullmatch(r"(19|20)\d{2}[-/](19|20)\d{2}", low):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", low):
        return True

    # Short time-range phrases
    if _YEAR_RE.search(low) and len(low.split()) <= 6 and any(
        k in low
        for k in [
            "between",
            "since",
            "from",
            "over",
            "past",
            "last",
            "entre",
            "desde",
            "a partir",
            "últim",
            "ultim",
            "anos",
            "años",
            "até",
            "hasta",
            "sejak",
            "selama",
        ]
    ):
        return True

    # Very short location/entity responses
    tokens = re.findall(r"[\w']+", low)
    if len(tokens) <= 3 and "?" not in low and not re.search(
        r"\b(what|where|when|why|which|how|cu[aá]l|d[oó]nde|qu[eé]|c[oó]mo|apa|mengapa|berapa)\b",
        low,
    ):
        return True

    return False


def _looks_like_conceptual_or_capability(prompt: str) -> bool:
    """Detect capability/explanatory/metadata/validation questions.

    These should generally *not* be scored as data_lookup/trend_over_time.
    """

    t = (prompt or "").strip()
    if not t:
        return False
    p = t.lower()

    # Capability/product questions
    if re.search(
        r"\b(what can you do|what do you do|capabilit(?:y|ies)|limitations?|supported|help me with|how does (?:this|it) work|how (?:do|can) (?:i|we) use|how can (?:i|we) use|what is this tool|what is this for|what's this for|qu[eé] puedes hacer|o que voc[eê] pode fazer|apa yang bisa (?:kamu|anda) lakukan|apa yang dapat (?:kamu|anda) lakukan)\b",
        p,
    ):
        return True

    # Data export / access questions (often scored incorrectly as data intents)
    if re.search(
        r"\b(download|export|csv|geojson|shapefile|shape\s*file|kml|kmz|api|endpoint)\b",
        p,
    ):
        return True

    # Data provenance / metadata questions
    if re.search(
        r"\b(where does .* data come from|data sources?|sources? of (?:the )?data|sources? of (?:the )?dataset|dataset sources?|citations?|cite|references?|fuentes? de datos|de d[oó]nde vienen los datos|origem dos dados|sumber data)\b",
        p,
    ):
        return True

    # Validation / QA / methodology questions about the analysis.
    #
    # NOTE: Avoid misclassifying alert "confidence" filters (e.g., "high confidence alerts")
    # as conceptual. "Confidence" by itself is too overloaded; only treat it as conceptual
    # when the user is asking about confidence *in the result/analysis*.
    if re.search(
        r"\b(can this be wrong|could this be wrong|is this wrong|is that wrong|"
        r"how accurate|accuracy|uncertain(?:ty)?|margin of error|error bars?|"
        r"is this a general analysis|general analysis|did you analy[sz]e|"
        r"how did you analy[sz]e|how was this analy[sz]ed|assumptions?|methodolog(?:y|ies)|methods?)\b",
        p,
    ):
        return True

    if "confidence" in p:
        # "Confidence alerts" / "high confidence GLAD alerts" should stay as a data intent.
        is_alert_conf_filter = bool(re.search(r"\b(alert|alerts|dist[- ]?alert|glad|radd|viirs)\b", p))
        if not is_alert_conf_filter and re.search(
            r"\b(how confident|confidence (?:in|about|that)|confidence interval|confidence band|"
            r"confidence level(?: of)? (?:this|the) (?:analysis|result|estimate))\b",
            p,
        ):
            return True

    # Imagery requests are usually capability questions in GNW (not direct data intents)
    if re.search(r"\b(satellite images?|imagery|raw satellite)\b", p):
        return True

    # Definitional "what is" questions without an obvious AOI/time anchoring.
    #
    # Heuristic: allow short definitional queries (<= 8 words) to be conceptual, but avoid
    # misclassifying long metric prompts like "What are the deforestation emissions for Brazil...".
    if re.match(r"^(define|definition of)\b", p):
        return True

    if re.match(r"^(what is|what are|qu[eé] es|o que [eé]|apa itu)\b", p):
        word_count = len(re.findall(r"\b\w+\b", p))
        has_geo_anchor = bool(re.search(r"\b(in|en|em|within|near|around|between|since|from|entre|desde)\b", p))
        has_year = bool(_YEAR_RE.search(p))

        # "for <place>" in English is extremely common in metric-style prompts.
        has_for_anchor = False
        if re.search(r"\bfor\s+(?:the\s+)?[A-Z]", t):
            has_for_anchor = True
        elif re.search(r"\bfor\s+(?:the\s+)?\w{3,}", p) and re.search(
            r"\b(emissions?|flux|extent|area|loss|gain|change|alerts?|hectares?|km2|percent)\b", p
        ):
            has_for_anchor = True

        if not (has_geo_anchor or has_year or has_for_anchor) and word_count <= 8:
            return True

    return False

def _looks_like_trend_request(p_lower: str) -> bool:
    p = p_lower
    if not p:
        return False

    # Explicit year range
    if re.search(r"\b(19|20)\d{2}\s*(?:-|–|—|to|a|hasta|até)\s*(19|20)\d{2}\b", p):
        return True

    # English
    if any(
        k in p
        for k in [
            "over time",
            "trend",
            "time series",
            "year by year",
            "month by month",
            "yearly",
            "annual",
            "annually",
            "monthly",
            "changes over",
            "change over",
            "evolution",
            "increase",
            "decrease",
            "between",
            "since",
            "over the last",
            "over the past",
            "last ",
            "past ",
        ]
    ) and (_YEAR_RE.search(p) or re.search(r"\b(between|since|from|over|last|past)\b", p)):
        return True

    # Spanish / Portuguese / Indonesian (minimal coverage)
    if any(
        k in p
        for k in [
            "tendencia",
            "a lo largo del tiempo",
            "a trav",
            "cambio",
            "cambios",
            "evoluci",
            "entre",
            "desde",
            "últim",
            "ultim",
            "ao longo do tempo",
            "tendência",
            "mudan",
            "evoluç",
            "sejak",
            "selama",
            "perubahan",
            "tren",
        ]
    ) and (_YEAR_RE.search(p) or re.search(r"\b(entre|desde|sejak|selama|últim|ultim)\b", p)):
        return True

    return False


def _looks_like_data_lookup_request(prompt: str) -> bool:
    t = (prompt or "").strip().lower()
    if not t:
        return False

    # Direct lookup verbs / question forms
    if re.search(
        r"\b(show|map|display|give me|list|find|lookup|look up|calculate|estimate|compute|derive|quantif(?:y|ies)|summari[sz]e|generate|produce)\b",
        t,
    ):
        return True

    if re.search(r"\b(how much|how many|what is the|what are the|where is|which areas)\b", t):
        return True

    # Explicit comparison requests are still data intents (even if not time-series)
    if re.search(r"\b(compare|comparison|versus|vs\.?|difference between|higher than|lower than|more than|less than)\b", t):
        return True

    # Common GNW dataset / metric keywords
    domain_keywords = [
        # Forest / land
        "tree cover",
        "tree cover loss",
        "tree cover gain",
        "forest loss",
        "forest gain",
        "deforestation",
        "land cover",
        "land use",
        "natural lands",
        "grassland",
        "mangrove",
        "peatland",
        "restoration",
        "regrowth",
        # Alerts / fires
        "alert",
        "alerts",
        "dist-alert",
        "dist alert",
        "glad",
        "radd",
        "viirs",
        "fire",
        "fires",
        "burn",
        "burned area",
        "burnt area",
        # Carbon / climate
        "carbon",
        "co2",
        "emission",
        "emissions",
        "greenhouse",
        "greenhouse gas",
        "ghg",
        "flux",
        "net flux",
        "biomass",
        "aboveground",
        "belowground",
        "soil carbon",
        "soil organic carbon",
        # Biodiversity / habitat
        "biodiversity",
        "habitat",
        "species",
        "occurrence",
        "range map",
        # ES/PT (light)
        "emisiones",
        "emissão",
        "emissoes",
        "carbono",
        "biomasa",
        "biodiversidad",
        "biodiversidade",
        "hábitat",
        "habitat",
        "incendios",
        "queimadas",
    ]
    if any(k in t for k in domain_keywords):
        return True

    # Units / quantities often signal a data lookup even without verbs.
    if re.search(r"\b(ha|hectares?|km2|sq\s?km|percent|%)\b", t):
        return True

    # Bare year + metric keyword (e.g., "forest loss 2020 brazil")
    if _YEAR_RE.search(t) and any(k in t for k in ["loss", "gain", "alert", "emission", "flux", "carbon", "land cover"]):
        return True

    return False


def _classify_intent(prompt: str) -> tuple[str, str]:
    raw = (prompt or "").strip()
    p = raw.lower()

    # --- Parameter refinement / follow-up turns ---
    # Many multi-turn interactions contain a short user follow-up like "Cambodia" or
    # "entre 2020 y 2024". These are not primary intents on their own, but we still
    # want deterministic handling downstream.
    if _looks_like_parameter_refinement(raw):
        return "parameter_refinement", ""

    # --- Conceptual / capability / metadata questions ---
    if _looks_like_conceptual_or_capability(raw):
        return "conceptual_or_capability", ""

    # --- Trend over time ---
    if _looks_like_trend_request(p):
        return "trend_over_time", ""

    # --- Data lookup (default for most analytical queries) ---
    if _looks_like_data_lookup_request(p):
        return "data_lookup", ""

    return "other", ""


def _infer_requires(intent: str, prompt: str) -> dict[str, bool]:
    """Infer which structured fields are required for an interaction.

    This is intentionally conservative: it's used to score structural completeness and should
    avoid spurious requirements (which inflate `needs_user_input` and `incomplete_answer`).
    """
    raw = prompt or ""
    p = raw.lower()

    if intent in {"conceptual_or_capability", "parameter_refinement"}:
        return {
            "requires_data": False,
            "requires_aoi": False,
            "requires_time_range": False,
            "requires_dataset": False,
        }

    requires_data = intent in SCORED_INTENTS
    global_scope = bool(
        re.search(
            r"\b(global|worldwide|around the world|across the world|in the world|all countries|by country|per country|top\s+\d+\s+(?:countries|regions)|ranking)\b",
            p,
        )
    )

    # Place detection: keep it lightweight, but cover the multilingual
    # prompts present in GNW exports.
    place_tokens = [
        "aoi", "location", "place", "area", "region", "country", "state", "province", "district", "county", "city", "municipality",
        "país", "pais", "estado", "provincia", "província", "distrito", "región", "region", "ciudad", "municipio", "ubicación", "ubicacion", "lugar", "zona",
        "negara", "provinsi", "kota", "kabupaten", "wilayah", "lokasi", "daerah",
    ]
    mentions_place = bool(
        re.search(
            r"\b(in|within|near|around|across|en|dentro|cerca|alrededor|em|perto|pr[oó]ximo|di|dalam|sekitar)\b",
            p,
        )
    ) or any(t in p for t in place_tokens)

    if not mentions_place and re.search(r"\bfor\s+(?:the\s+)?[A-Z]", raw):
        mentions_place = True

    if intent == "trend_over_time":
        requires_aoi = not global_scope
    elif intent == "data_lookup":
        requires_aoi = mentions_place and not global_scope
    else:
        requires_aoi = False

    requires_time = bool(
        intent == "trend_over_time"
        or re.search(r"\b(19\d{2}|20\d{2})\b", p)
        or any(
            k in p
            for k in [
                "over time", "trend", "between", "since", "from", "during", "last", "past", "decade",
                "entre", "desde", "a partir", "durante", "últim", "ultim", "año", "años", "ano", "anos", "década", "decada",
                "ao longo", "tendência", "tendencia", "mudança", "mudanca", "mensal", "mensual", "anual",
                "sejak", "selama", "antara", "tahun", "bulanan", "tahunan",
            ]
        )
    )

    requires_dataset = bool(
        requires_data
        or any(
            k in p
            for k in [
                "dataset", "data set", "layer", "capa", "camada", "conjunto de datos", "conjunto de dados",
                "tree cover", "land cover", "grassland", "mangrove", "peat", "alert", "carbon", "emission",
            ]
        )
    )
    return {
        "requires_data": requires_data,
        "requires_aoi": requires_aoi,
        "requires_time_range": requires_time,
        "requires_dataset": requires_dataset,
    }


def _answer_type(
    response: str, response_missing: bool, output_json_ok: bool, level: str | None = None, error_count: Any = None
) -> str:
    """Coarse answer type classifier.

    Priority order:
    1) Missing/empty outputs
    2) Trace-level errors (Langfuse) / strong system failure markers
    3) No-data / unsupported coverage messages
    4) Normal answers

    NOTE: Avoid treating generic words like "error" in explanatory contexts as model errors.
    """
    if response_missing:
        return "missing_output"
    if not output_json_ok and (level or "").upper() == "ERROR":
        return "model_error"
    if (level or "").upper() == "ERROR":
        return "model_error"
    try:
        if error_count is not None and float(error_count) > 0:
            return "model_error"
    except Exception:
        pass

    t = (response or "").strip().lower()
    if not t or len(t) < 20:
        return "empty_or_short"

    # Strong system / processing failure markers (text-only fallback)
    if re.search(
        r"\b(traceback|exception|internal error|unexpected error|something went wrong|technical (?:issue|problem)|service unavailable|timed out|timeout)\b",
        t,
    ):
        return "model_error"

    # No data / unsupported scope markers.
    # Include explicit global/continental scope limitation language which is common in GNW.
    if re.search(
        r"\b(?:no data|no results|not (?:currently )?available|not found|no matching data|"
        r"could not (?:find|locate)|couldn't (?:find|locate)|"
        r"cannot (?:find|locate)|can't (?:find|locate)|unable to (?:find|locate)|"
        r"(?:do not|don't|cannot|can't)\s+(?:currently\s+)?support|"
        r"(?:do not|don't|cannot|can't)\s+have\s+access|"
        r"not supported|unsupported|outside (?:our|the) coverage|outside coverage|"
        r"unable to (?:process|handle).{0,40}\b(?:global|world(?:wide)?|entire world|whole world|continent(?:al)?)\b|"
        r"\b(?:global|worldwide|continental|entire world|whole world)\b.{0,60}\b(?:not supported|unsupported|too large|can't|cannot|unable)\b)\b",
        t,
    ):
        return "no_data"

    # Chatty meta disclaimers about being an AI (not necessarily an error, but not a data answer either)
    if re.search(r"\b(as an ai|as a language model|i do not have access to real time|i can't access real time)\b", t):
        return "fallback"

    return "answer"


def _needs_user_input(response: str, requires: dict[str, bool], struct: dict[str, Any]) -> tuple[bool, str]:
    """Detect when the assistant is blocked on missing user-provided parameters.

    This is intentionally conservative:
    - We only emit needs_user_input when the assistant is explicitly asking the user
      to clarify/provide/select something OR when there's strong AOI disambiguation language.
    - We avoid triggering on polite follow-up offers ("Would you like...?") that appear
      at the end of complete answers.

    IMPORTANT: We do *not* rely solely on `requires_*` gating because prompt-level intent
    inference can be wrong on short prompts. If the assistant is clearly asking for a missing
    AOI/time/dataset, we mark needs_user_input even when the prompt-level requires flags were false.
    """
    r = (response or "").strip()
    if not r:
        return False, ""

    rl = r.lower()

    ask_verbs = re.compile(
        r"\b(please|pls|could you|can you|select|choose|provide|specify|tell me|clarify|confirm|"
        r"por favor|puedes|podr[ií]a(?:s)?|selecciona|elige|indica|especifica|necesito|"
        r"pode|voc[eê] pode|voce pode|selecione|escolha|informe|especifique|preciso|"
        r"tolong|mohon|bisa(?:kah)?|dapatkah|pilih|tentukan|perlu)\b"
    )
    aoi_terms = re.compile(
        r"\b(aoi|"
        r"area(?:s)?|location(?:s)?|place(?:s)?|region(?:s)?|polygon(?:s)?|"
        r"countr(?:y|ies)|state(?:s)?|province(?:s)?|district(?:s)?|count(?:y|ies)|municipalit(?:y|ies)|"
        r"[áa]rea(?:s)?|zona(?:s)?|ubicaci[oó]n(?:es)?|lugar(?:es)?|regi[oó]n(?:es)?|"
        r"pa[ií]s(?:es)?|estado(?:s)?|provincia(?:s)?|municipio(?:s)?|"
        r"local(?:es)?|regi[aã]o|regi[oõ]es|prov[ií]ncia(?:s)?|munic[ií]pio(?:s)?|"
        r"wilayah|daerah|lokasi|kawasan|provinsi|kabupaten|kota)\b"
    )
    time_terms = re.compile(
        r"\b(time range|date range|timeframe|period|years?|months?|dates?|start|end|"
        r"rango de tiempo|rango de fechas|periodo|per[ií]odo|a[nñ]os?|fechas?|inicio|fin|"
        r"intervalo de tempo|anos?|datas|data(?:s)?\s+de|in[ií]cio|fim|"
        r"rentang waktu|periode|tahun|tanggal|mulai|akhir)\b"
    )
    dataset_terms = re.compile(
        r"\b(dataset(?:s)?|data set(?:s)?|layer(?:s)?|map layer(?:s)?|"
        r"conjunto de datos|capa(?:s)?|"
        r"conjunto de dados|camada(?:s)?|"
        r"lapisan)\b"
    )

    # Disambiguation: multiple location matches / "did you mean" / "which one"
    disambig = re.compile(
        r"\b(found|there (?:are|were)|i see)\b.{0,120}\b"
        r"(multiple|several|two|2|three|3|four|4|five|5|\d{2,4})\b.{0,120}\b"
        r"(location(?:s)?|place(?:s)?|match(?:es)?|option(?:s)?|result(?:s)?|candidate(?:s)?|"
        r"protected area(?:s)?|territor(?:y|ies)|admin(?:istrative)? units?|district(?:s)?|"
        r"province(?:s)?|state(?:s)?|country(?:ies)?|region(?:s)?|area(?:s)?)\b"
    )
    did_you_mean = re.compile(r"\b(did you mean|do you mean|if you meant|if you mean|if you intended)\b")
    which_one = re.compile(r"\b(which one|which of (?:these|those)|choose one|select one|pick one)\b")
    too_many = re.compile(
        r"\b(too many|hundreds of|dozens of|more than\s+\d+)\b.{0,80}\b(location(?:s)?|place(?:s)?|match(?:es)?|option(?:s)?|result(?:s)?|area(?:s)?|protected area(?:s)?|territor(?:y|ies)|admin(?:istrative)? units?|district(?:s)?|province(?:s)?|state(?:s)?|country(?:ies)?|region(?:s)?)\b"
    )
    system_limit = re.compile(
        r"\b(system limitation(?:s)?|limitations with processing|system limitations with processing|too (?:many|large) to (?:process|analy[sz]e)|cannot (?:process|handle)|can't (?:process|handle))\b"
    )

    q_ask = re.compile(r"\b(which|what|cu[aá]l|qu[eé]|qual)\b[^?.]{0,120}\?")

    # --- Structured missingness (independent of requires) -----------------
    aoi_missing_struct = not bool(struct.get("aoi_selected_struct"))
    time_missing_struct = not bool(struct.get("time_range_struct"))
    dataset_missing_struct = not bool(struct.get("dataset_struct"))

    # --- Response-driven asked field detection ----------------------------
    asked_fields: list[str] = []

    # AOI: explicit ask, or disambiguation cues
    aoi_disambig = bool(
        disambig.search(rl)
        or did_you_mean.search(rl)
        or too_many.search(rl)
        or system_limit.search(rl)
        or (
            which_one.search(rl)
            and (
                struct.get("aoi_candidates_struct")
                or struct.get("aoi_options_unique_count", 0) > 1
            )
        )
    )
    if aoi_disambig:
        # If there are multiple candidates OR we simply have no AOI selected, treat as blocking.
        if bool(requires.get("requires_aoi")) or aoi_missing_struct or bool(struct.get("aoi_candidates_struct")) or int(struct.get("aoi_options_unique_count") or 0) > 1:
            asked_fields.append("missing_aoi")

    if (ask_verbs.search(rl) or q_ask.search(rl)) and aoi_terms.search(rl):
        if aoi_missing_struct:
            asked_fields.append("missing_aoi")

    # Time
    if (ask_verbs.search(rl) or q_ask.search(rl)) and time_terms.search(rl):
        if time_missing_struct:
            asked_fields.append("missing_time")

    # Dataset
    if (ask_verbs.search(rl) or q_ask.search(rl)) and dataset_terms.search(rl):
        if dataset_missing_struct:
            asked_fields.append("missing_dataset")

    asked_fields = sorted(set(asked_fields))
    if asked_fields:
        reason = "multiple_missing" if len(asked_fields) > 1 else asked_fields[0]
        return True, reason

    # --- Requires-gated fallback (legacy behavior) -------------------------
    missing: list[str] = []
    if requires.get("requires_aoi") and aoi_missing_struct:
        missing.append("missing_aoi")
    if requires.get("requires_time_range") and time_missing_struct:
        missing.append("missing_time")
    if requires.get("requires_dataset") and dataset_missing_struct:
        missing.append("missing_dataset")

    if not missing:
        return False, ""

    # Generic clarification phrasing, gated on required missingness.
    generic_clarify = bool(
        re.search(r"\b(need|requires|required|necesito|preciso|perlu)\b", rl)
        and re.search(r"\b(specify|provide|select|choose|clarify|especifica|indica|informe|tentukan|pilih)\b", rl)
    )

    if generic_clarify:
        reason = "multiple_missing" if len(missing) > 1 else missing[0]
        return True, reason

    return False, ""


def _metric_sanity_fail(response: str) -> bool:
    """Flag obviously broken percentage outputs.

    This is intentionally conservative: it should *not* fire on phrases like
    "I'm 100% sure". It focuses on numerically implausible percentages and
    share/portion contexts that exceed 100%.
    """
    text = response or ""
    if not text:
        return False

    lower = text.lower()
    pct_re = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
    share_ctx = re.compile(r"\b(share|percentage|percent of|portion|proportion|of total)\b", re.I)

    for m in pct_re.finditer(text):
        try:
            val = float(m.group(1))
        except Exception:
            continue

        # Hard bounds: clearly broken.
        if val < -200 or val > 2000:
            return True

        # Softer bounds if the local context indicates a share/portion.
        ctx = lower[max(0, m.start() - 50) : m.start()]
        if share_ctx.search(ctx) and (val < 0 or val > 100):
            return True

    return False


def _classify_dataset_family(dataset_name: str) -> str | None:
    """Bucket dataset names into a small, stable taxonomy."""
    d = (dataset_name or "").lower()
    if not d:
        return None

    if "tree cover loss" in d or "deforestation" in d or "forest loss" in d:
        if "driver" in d:
            return "tree_cover_loss_drivers"
        return "tree_cover_loss"
    if "dist-alert" in d or "dist alert" in d or "alert" in d:
        return "alerts"
    if "land cover" in d or "land use" in d:
        return "land_cover"
    if "grassland" in d:
        return "grassland"
    if "natural lands" in d or "sbtn" in d:
        return "natural_lands"
    if "greenhouse" in d or "ghg" in d or "carbon" in d or "flux" in d:
        return "ghg_carbon"

    if "tree" in d or "forest" in d:
        return "forest"
    if "climate" in d or "temperature" in d:
        return "climate"
    if "population" in d or "demograph" in d:
        return "population"
    return "other"


def _parse_time_window_days(start: str, end: str) -> float | None:
    if not start or not end:
        return None
    try:
        s = pd.to_datetime(start, utc=True, errors="coerce")
        e = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.isna(s) or pd.isna(e):
            return None
        return float((e - s).days)
    except Exception:
        return None


def _extract_codeact(output_obj: Any) -> dict[str, Any]:
    # Use canonical CodeAct discovery to avoid mismatches between nested output shapes.
    parts = find_codeact_parts(output_obj)
    if not parts:
        return {
            "codeact_present": False,
            "codeact_parts_count": 0,
            "codeact_code_blocks_count": 0,
            "codeact_exec_outputs_count": 0,
            "codeact_uses_analytics_api": False,
            "codeact_decoded_chars_total": 0,
        }

    decoded_parts = list(iter_decoded_codeact_parts(output_obj))
    # Fallback to raw parts count if decoding yields nothing.
    parts_count = len(decoded_parts) or len(parts)

    code_blocks = 0
    exec_outputs = 0
    uses_analytics_api = False
    decoded_chars_total = 0
    analytics_host_re = re.compile(r"\banalytics\.globalnaturewatch\.org\b", re.IGNORECASE)

    for part in decoded_parts:
        ptype = str(part.get("type") or "").lower().strip()
        decoded = str(part.get("decoded") or "")

        if ptype in {"code_block", "code"} or "code_block" in ptype:
            code_blocks += 1
        if ptype in {"execution_output", "exec_output", "execution", "exec"} or "execution" in ptype:
            exec_outputs += 1

        decoded_chars_total += len(decoded)
        if analytics_host_re.search(decoded) or "/v1/query" in decoded.lower():
            uses_analytics_api = True

    return {
        "codeact_present": True,
        "codeact_parts_count": parts_count,
        "codeact_code_blocks_count": code_blocks,
        "codeact_exec_outputs_count": exec_outputs,
        "codeact_uses_analytics_api": uses_analytics_api,
        "codeact_decoded_chars_total": decoded_chars_total,
    }


def _count_raw_data_records(raw_data: Any) -> int:
    """Best-effort count of raw records.

    In current GNW outputs, `raw_data` is commonly a dict keyed by query/run IDs
    with nested dicts keyed by row index. Counting list/dict lengths at one level
    provides a stable proxy for record volume.
    """

    if isinstance(raw_data, list):
        return len(raw_data)
    if isinstance(raw_data, dict):
        total = 0
        for v in raw_data.values():
            if isinstance(v, list):
                total += len(v)
            elif isinstance(v, dict):
                total += len(v)
        return total if total > 0 else len(raw_data)
    return 0


def _completion_state(
    intent: str,
    answer_type: str,
    needs_user_input: bool,
    struct: dict[str, Any],
    requires: dict[str, bool],
    metric_sanity_fail: bool,
    has_citations: bool,
) -> tuple[str, bool, bool, str]:
    reasons: list[str] = []

    if intent in SCORED_INTENTS:
        if requires["requires_aoi"] and not struct.get("aoi_selected_struct"):
            reasons.append("missing_aoi")
        if requires["requires_time_range"] and not struct.get("time_range_struct"):
            reasons.append("missing_time")
        if requires["requires_dataset"] and not struct.get("dataset_struct"):
            reasons.append("missing_dataset")

        if (
            intent == "trend_over_time"
            and requires.get("requires_data")
            and not has_citations
            and not needs_user_input
        ):
            reasons.append("no_citation")

        if metric_sanity_fail:
            reasons.append("metric_sanity")

    if answer_type in {"missing_output", "model_error", "empty_or_short"}:
        state = "error"
        # For hard error states, the specific answer_type is the most actionable failure token.
        reasons = [answer_type]
    elif needs_user_input:
        state = "needs_user_input"
    elif answer_type == "no_data":
        state = "no_data"
    elif intent in SCORED_INTENTS and reasons:
        state = "incomplete_answer"
    elif answer_type in {"answer", "text_only"}:
        state = "complete_answer"
    else:
        state = "other"

    struct_good_trend = bool(intent == "trend_over_time" and state == "complete_answer")
    struct_good_lookup = bool(intent == "data_lookup" and state == "complete_answer")

    return state, struct_good_trend, struct_good_lookup, "|".join(reasons)




def _similarity(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())
def infer_conversation_outcomes(derived: pd.DataFrame) -> pd.Series:
    if derived.empty:
        return pd.Series(dtype="string")

    df = derived.copy()
    group_keys = []
    for row in df.itertuples(index=False):
        if getattr(row, "thread_id", ""):
            group_keys.append(f"thread:{row.thread_id}")
        elif getattr(row, "sessionId", ""):
            group_keys.append(f"session:{row.sessionId}")
        else:
            group_keys.append(f"trace:{row.trace_id}")
    df["_group"] = group_keys
    df = df.sort_values("timestamp", na_position="last")

    outcomes = pd.Series(["unknown"] * len(df), index=df.index, dtype="string")
    for _, g in df.groupby("_group", sort=False):
        idxs = list(g.index)
        prompts_raw = g["prompt"].fillna("").astype(str).tolist()
        prompts_lc = [p.lower() for p in prompts_raw]

        for i, idx in enumerate(idxs):
            if i >= len(idxs) - 1:
                outcomes.loc[idx] = "unknown"
                continue

            nxt = prompts_lc[i + 1]
            if any(k in nxt for k in POSITIVE_ACK_PATTERNS):
                outcomes.loc[idx] = "success"
                continue
            if any(k in nxt for k in NEGATIVE_ACK_PATTERNS):
                outcomes.loc[idx] = "clarification_needed"
                continue
            if _similarity(prompts_lc[i], nxt) >= 0.8:
                outcomes.loc[idx] = "repeat_question"
                continue

            tokens = re.findall(r"[a-z0-9]+", nxt)
            if 0 < len(tokens) <= 4:
                outcomes.loc[idx] = "clarification_needed"
                continue

            outcomes.loc[idx] = "unknown"
    return outcomes.sort_index()


def compute_derived_interactions(traces: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in traces:
        trace = normalize_trace_format(raw)
        prompt = current_human_prompt(trace)
        if not prompt.strip():
            continue

        out_msgs = slice_output_to_current_turn(trace)
        response = current_turn_ai_message(trace)

        response_missing = True
        if isinstance(out_msgs, list):
            for m in reversed(out_msgs):
                if _is_assistant(m) and _content_to_text(m.get("content")):
                    response_missing = False
                    break
        if response_missing:
            response = ""

        output_json_ok = isinstance(trace.get("output"), dict)
        output_obj = trace.get("output") if isinstance(trace.get("output"), dict) else {}

        struct = _extract_struct_flags(output_obj)

        # --- Analysis payload (raw/charts) ---------------------------------
        raw_data = _find_first_key(output_obj, ("raw_data", "rawData"))
        raw_data_len = _count_raw_data_records(raw_data)
        has_raw_data = raw_data_len > 0

        charts_data = _find_first_key(output_obj, ("charts_data", "chartsData", "charts"))
        has_charts_data = isinstance(charts_data, list) and len(charts_data) > 0
        charts_count = len(charts_data) if isinstance(charts_data, list) else 0

        analysis_executed = bool(has_raw_data or has_charts_data)

        # --- Time window ----------------------------------------------------
        # (Used both for QA and for intent rescue.)
        time_window_days = _parse_time_window_days(
            str(struct.get("time_start") or ""),
            str(struct.get("time_end") or ""),
        )

        text_flags = _text_flags(prompt, response)

        intent_primary, intent_secondary = _classify_intent(prompt)

        # If a prompt is short/ambiguous (common in threaded clarifications) but the
        # system clearly executed an analysis and returned a structured dataset, treat
        # it as a scored data interaction.
        if (
            intent_primary in {"other", "parameter_refinement"}
            and analysis_executed
            and bool(struct.get("dataset_struct"))
        ):
            p_low = (prompt or "").lower()
            if _looks_like_trend_request(p_low):
                intent_primary = "trend_over_time"
            else:
                intent_primary = "data_lookup"

        requires = _infer_requires(intent_primary, prompt)
        level = str(trace.get("level") or "")
        error_count = trace.get("errorCount")

        response_lower = (response or "").lower()
        mentions_no_data_language = any(
            m in response_lower
            for m in [
                "no data available",
                "data is not available",
                "not available for",
                "couldn't find data",
                "could not find data",
                "unable to find",
                "no results found",
                "no matching data",
                "does not exist in the dataset",
            ]
        )

        answer_type = _answer_type(response, response_missing, output_json_ok, level=level, error_count=error_count)

        # If the agent still executed analysis and returned structured output, "no_data"
        # is misleading. Keep the language as a separate flag for QA filtering.
        no_data_with_analysis = False
        if answer_type == "no_data" and analysis_executed:
            no_data_with_analysis = True
            answer_type = "answer"
        needs_ui, needs_reason = _needs_user_input(response, requires, struct)
        metric_fail = _metric_sanity_fail(response)
        citations_any = bool(
            struct.get("citations_struct") or text_flags.get("citations_text") or struct.get("dataset_has_citation")
        )
        completion_state, struct_good_trend, struct_good_lookup, struct_fail_reason = _completion_state(
            intent_primary,
            answer_type,
            needs_ui,
            struct,
            requires,
            bool(metric_fail),
            citations_any,
        )
        dataset_name = struct.get("dataset_name", "")
        dataset_family = _classify_dataset_family(dataset_name)
        codeact = _extract_codeact(output_obj)

        metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
        ts = parse_trace_dt(trace)
        timestamp = ts.astimezone(timezone.utc) if isinstance(ts, datetime) else None

        rows.append(
            {
                "trace_id": str(trace.get("id") or ""),
                "timestamp": timestamp,
                "sessionId": str(trace.get("sessionId") or ""),
                "thread_id": str(metadata.get("thread_id") or metadata.get("threadId") or ""),
                "userId": str(trace.get("userId") or ""),
                "level": level,
                "errorCount": error_count,
                "latency": trace.get("latency"),
                "input_tokens": trace.get("inputTokens"),
                "output_tokens": trace.get("outputTokens"),
                "total_tokens": trace.get("totalTokens"),
                "prompt": prompt,
                "response": response,
                "response_missing": bool(response_missing),
                "output_json_ok": bool(output_json_ok),
                "intent_primary": intent_primary,
                "complexity_bucket": "simple" if len(prompt.split()) < 12 else "complex",
                **requires,
                **{
                    k: struct.get(k)
                    for k in [
                        "aoi_selected_struct",
                        "aoi_candidates_struct",
                        "aoi_options_count",
                        "aoi_options_unique_count",
                        "time_range_struct",
                        "dataset_struct",
                        "citations_struct",
                        "dataset_has_citation",
                    ]
                },
                **text_flags,
                "dataset_name": dataset_name,
                "dataset_family": dataset_family,
                "dataset_identifiable": bool(dataset_name),
                "analysis_executed": analysis_executed,
                "has_raw_data": has_raw_data,
                "raw_data_len": raw_data_len,
                "has_charts_data": has_charts_data,
                "charts_count": charts_count,
                "aoi_type": struct.get("aoi_type", ""),
                "aoi_name": struct.get("aoi_name", ""),
                "time_start": struct.get("time_start", ""),
                "time_end": struct.get("time_end", ""),
                "time_window_days": time_window_days,
                "answer_type": answer_type,
                "mentions_no_data_language": bool(mentions_no_data_language),
                "no_data_with_analysis": bool(no_data_with_analysis),
                "metric_sanity_fail": bool(metric_fail),
                "needs_user_input": bool(needs_ui),
                "needs_user_input_reason": needs_reason,
                "completion_state": completion_state,
                "struct_good_trend": bool(struct_good_trend),
                "struct_good_lookup": bool(struct_good_lookup),
                "struct_fail_reason": struct_fail_reason,
                **codeact,
            }
        )

    derived = pd.DataFrame(rows)
    if derived.empty:
        return derived
    return derived


def _pct(n: float, d: float) -> float:
    if not d:
        return 0.0
    return float(n) / float(d)


def compute_thread_key(df: pd.DataFrame) -> pd.Series:
    """Build deterministic thread grouping keys with fallback priority.

    Priority: thread_id -> sessionId -> trace_id -> synthetic row key.
    """
    if df.empty:
        return pd.Series(dtype="string")

    idx_series = pd.Series(df.index, index=df.index)

    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].fillna("").astype(str).str.strip()
        return pd.Series("", index=df.index, dtype="string")

    thread_id = _col("thread_id")
    session_id = _col("sessionId")
    trace_id = _col("trace_id")

    thread_key = thread_id.where(thread_id != "", session_id)
    thread_key = thread_key.where(thread_key != "", trace_id)
    return thread_key.where(thread_key != "", idx_series.map(lambda x: f"row:{x}")).astype("string")


def _unique_in_order(values: pd.Series, cap: int) -> tuple[int, str]:
    seen: list[str] = []
    for raw in values.fillna("").astype(str):
        v = raw.strip()
        if not v or v in seen:
            continue
        seen.append(v)

    total = len(seen)
    if total <= cap:
        return total, ",".join(seen)
    return total, ",".join(seen[:cap] + [f"+{total - cap} more"])


def build_thread_summary(
    derived: pd.DataFrame,
    timestamp_col: str = "timestamp",
    max_datasets: int = 5,
    max_families: int = 5,
) -> pd.DataFrame:
    if derived.empty:
        return pd.DataFrame(
            columns=[
                "thread_key",
                "thread_id",
                "sessionId",
                "start_utc",
                "end_utc",
                "n_turns",
                "first_intent_primary",
                "last_intent_primary",
                "ever_complete_answer",
                "ever_needs_user_input",
                "ever_error",
                "ended_after_needs_user_input",
                "ended_after_error",
                "n_complete_answer",
                "n_needs_user_input",
                "n_error",
                "datasets_seen_count",
                "datasets_seen",
                "dataset_families_seen_count",
                "dataset_families_seen",
                "needs_user_input_reasons",
                "last_completion_state",
                "last_needs_user_input_reason",
            ]
        )

    df = derived.copy()
    df["thread_key"] = compute_thread_key(df)
    df["_ts"] = pd.to_datetime(df.get(timestamp_col), utc=True, errors="coerce")
    df = df.sort_values(["thread_key", "_ts"], na_position="last")

    summaries: list[dict[str, Any]] = []
    for key, g in df.groupby("thread_key", sort=False):
        group = g.sort_values("_ts", na_position="last")
        last = group.iloc[-1]
        first = group.iloc[0]

        ts = group["_ts"].dropna()
        start_utc = ts.min().isoformat() if len(ts) else ""
        end_utc = ts.max().isoformat() if len(ts) else ""

        datasets_seen_count, datasets_seen = _unique_in_order(group.get("dataset_name", pd.Series(dtype=str)), max_datasets)
        families_seen_count, families_seen = _unique_in_order(group.get("dataset_family", pd.Series(dtype=str)), max_families)
        nui_reasons_count, nui_reasons = _unique_in_order(
            group.loc[group.get("completion_state", "") == "needs_user_input", "needs_user_input_reason"]
            if "needs_user_input_reason" in group.columns
            else pd.Series(dtype=str),
            cap=9999,
        )
        del nui_reasons_count

        last_completion_state = str(last.get("completion_state") or "")
        last_nui_reason = str(last.get("needs_user_input_reason") or "") if last_completion_state == "needs_user_input" else ""

        summaries.append(
            {
                "thread_key": str(key),
                "thread_id": str(first.get("thread_id") or ""),
                "sessionId": str(first.get("sessionId") or ""),
                "start_utc": start_utc,
                "end_utc": end_utc,
                "n_turns": int(len(group)),
                "first_intent_primary": str(first.get("intent_primary") or ""),
                "last_intent_primary": str(last.get("intent_primary") or ""),
                "ever_complete_answer": bool((group.get("completion_state") == "complete_answer").any()),
                "ever_needs_user_input": bool((group.get("completion_state") == "needs_user_input").any()),
                "ever_error": bool((group.get("completion_state") == "error").any()),
                "ended_after_needs_user_input": bool(last_completion_state == "needs_user_input"),
                "ended_after_error": bool(last_completion_state == "error"),
                "n_complete_answer": int((group.get("completion_state") == "complete_answer").sum()),
                "n_needs_user_input": int((group.get("completion_state") == "needs_user_input").sum()),
                "n_error": int((group.get("completion_state") == "error").sum()),
                "datasets_seen_count": int(datasets_seen_count),
                "datasets_seen": datasets_seen,
                "dataset_families_seen_count": int(families_seen_count),
                "dataset_families_seen": families_seen,
                "needs_user_input_reasons": nui_reasons,
                "last_completion_state": last_completion_state,
                "last_needs_user_input_reason": last_nui_reason,
            }
        )

    out = pd.DataFrame(summaries)
    out["_end_dt"] = pd.to_datetime(out["end_utc"], utc=True, errors="coerce")
    out = out.sort_values("_end_dt", ascending=False, na_position="last").drop(columns=["_end_dt"])
    return out.reset_index(drop=True)


def summarize_content(derived: pd.DataFrame, timestamp_col: str = "timestamp") -> dict[str, Any]:
    rows = int(len(derived))
    unique_users = int(derived["userId"].replace("", pd.NA).dropna().nunique()) if "userId" in derived.columns else 0

    window = {"start": None, "end": None}
    if timestamp_col in derived.columns and rows:
        ts = pd.to_datetime(derived[timestamp_col], utc=True, errors="coerce").dropna()
        if len(ts):
            window = {"start": ts.min().isoformat(), "end": ts.max().isoformat()}

    answer_counts = derived["answer_type"].value_counts(dropna=False).to_dict() if rows else {}
    completion_counts = derived["completion_state"].value_counts(dropna=False).to_dict() if rows else {}
    reason_counts = derived["needs_user_input_reason"].replace("", pd.NA).dropna().value_counts().to_dict() if rows else {}
    citations_shown = derived["citations_text"].fillna(False) if rows else pd.Series(dtype=bool)
    citation_metadata_present = (
        derived["citations_struct"].fillna(False) | derived["dataset_has_citation"].fillna(False)
    ) if rows else pd.Series(dtype=bool)
    citations_any = (citations_shown | citation_metadata_present) if rows else pd.Series(dtype=bool)

    scored = derived[derived["intent_primary"].isin(SCORED_INTENTS)] if rows else derived
    data_intents = derived[derived["requires_data"] == True] if rows else derived

    if rows:
        thread_key = compute_thread_key(derived)
        tmp = derived.assign(_thread=thread_key)
        tmp = tmp.sort_values("timestamp", na_position="last")
        ended_nui = 0
        ended_err = 0
        total_threads = 0
        for _, g in tmp.groupby("_thread"):
            total_threads += 1
            last = g.iloc[-1]
            if str(last.get("completion_state")) == "needs_user_input":
                ended_nui += 1
            if str(last.get("completion_state")) == "error":
                ended_err += 1
    else:
        ended_nui = ended_err = total_threads = 0

    intent_summary: dict[str, dict[str, Any]] = {}
    for intent in sorted(SCORED_INTENTS):
        subset = derived[derived["intent_primary"] == intent] if rows else derived
        c = len(subset)
        intent_summary[intent] = {
            "count": int(c),
            "share_of_total": _pct(c, rows),
            "complete_answer_rate": _pct((subset["completion_state"] == "complete_answer").sum(), c),
            "needs_user_input_rate": _pct((subset["completion_state"] == "needs_user_input").sum(), c),
            "error_rate": _pct((subset["completion_state"] == "error").sum(), c),
            "structural_complete_rate": _pct(
                subset["struct_good_trend"].sum() if intent == "trend_over_time" else subset["struct_good_lookup"].sum(),
                c,
            ),
        }

    struct_subset = scored
    fail_reasons = Counter()
    if len(struct_subset):
        for val in struct_subset["struct_fail_reason"].fillna(""):
            for token in [x for x in str(val).split("|") if x.strip()]:
                fail_reasons[token] += 1

    dataset_family_summary: dict[str, dict[str, Any]] = {}
    for family, g in data_intents.groupby("dataset_family") if len(data_intents) else []:
        c = len(g)
        dataset_family_summary[str(family)] = {
            "count_data_intents": int(c),
            "complete_answer_rate_data_intents": _pct((g["completion_state"] == "complete_answer").sum(), c),
            "needs_user_input_rate_data_intents": _pct((g["completion_state"] == "needs_user_input").sum(), c),
            "error_rate_data_intents": _pct((g["completion_state"] == "error").sum(), c),
            "codeact_present_rate_data_intents": _pct(g["codeact_present"].fillna(False).sum(), c),
        }

    global_quality = {
        "answer_type_counts": answer_counts,
        "answer_type_percentages": {k: _pct(v, rows) for k, v in answer_counts.items()},
        "completion_state_counts": completion_counts,
        "completion_state_rates": {k: _pct(v, rows) for k, v in completion_counts.items()},
        "needs_user_input_reason_counts": reason_counts,
        "metric_sanity_fail_rate": _pct(derived["metric_sanity_fail"].fillna(False).sum(), rows),
        "citations_shown_rate": _pct(citations_shown.sum(), rows),
        "citation_metadata_present_rate": _pct(citation_metadata_present.sum(), rows),
        "citations_any_rate": _pct(citations_any.sum(), rows),
        "dataset_identifiable_rate_scored_intents": _pct(scored["dataset_identifiable"].fillna(False).sum(), len(scored)),
        "codeact_present_rate": _pct(derived["codeact_present"].fillna(False).sum(), rows),
        "codeact_present_rate_scored_intents": _pct(scored["codeact_present"].fillna(False).sum(), len(scored)),
        "threads_ended_after_needs_user_input_rate": _pct(ended_nui, total_threads),
        "threads_ended_after_error_rate": _pct(ended_err, total_threads),
    }

    kpis = {
        "complete_answer_rate_scored_intents": _pct((scored["completion_state"] == "complete_answer").sum(), len(scored)),
        "needs_user_input_rate_scored_intents": _pct((scored["completion_state"] == "needs_user_input").sum(), len(scored)),
        "error_rate_scored_intents": _pct((scored["completion_state"] == "error").sum(), len(scored)),
        "global_dataset_identifiable_rate_scored_intents": global_quality["dataset_identifiable_rate_scored_intents"],
        "citations_shown_rate_scored_intents": _pct(scored["citations_text"].fillna(False).sum(), len(scored)),
        "citation_metadata_present_rate_scored_intents": _pct(
            (scored["citations_struct"].fillna(False) | scored["dataset_has_citation"].fillna(False)).sum(),
            len(scored),
        ),
        "threads_ended_after_needs_user_input_rate": global_quality["threads_ended_after_needs_user_input_rate"],
    }

    return {
        "rows": rows,
        "unique_users": unique_users,
        "window_utc": window,
        "kpis": kpis,
        "global_quality": global_quality,
        "intent_summary": intent_summary,
        "struct_outcome_summary": {
            "total": int(len(struct_subset)),
            "complete": int((struct_subset["completion_state"] == "complete_answer").sum()) if len(struct_subset) else 0,
            "needs_user_input": int((struct_subset["completion_state"] == "needs_user_input").sum()) if len(struct_subset) else 0,
            "errors": int((struct_subset["completion_state"] == "error").sum()) if len(struct_subset) else 0,
            "no_data": int((struct_subset["completion_state"] == "no_data").sum()) if len(struct_subset) else 0,
            "failures_excluding_needs_user_input": int(
                (~struct_subset["completion_state"].isin(["complete_answer", "needs_user_input"])).sum()
            ) if len(struct_subset) else 0,
            "needs_user_input_reasons": reason_counts,
            "failure_reasons": dict(fail_reasons),
        },
        "dataset_family_summary": dataset_family_summary,
    }


def build_content_slices(derived: pd.DataFrame) -> pd.DataFrame:
    if derived.empty:
        return pd.DataFrame(
            columns=[
                "intent_primary",
                "complexity_bucket",
                "count",
                "complete_answer_rate",
                "needs_user_input_rate",
                "error_rate",
                "metric_sanity_fail_rate",
                "citations_shown_rate",
                "citation_metadata_present_rate",
            ]
        )

    df = derived.copy()
    df["citations_shown"] = df["citations_text"].fillna(False)
    df["citation_metadata_present"] = df["citations_struct"].fillna(False) | df["dataset_has_citation"].fillna(False)

    grouped = (
        df.groupby(["intent_primary", "complexity_bucket"], dropna=False)
        .agg(
            count=("trace_id", "count"),
            complete_answer_rate=("completion_state", lambda s: _pct((s == "complete_answer").sum(), len(s))),
            needs_user_input_rate=("completion_state", lambda s: _pct((s == "needs_user_input").sum(), len(s))),
            error_rate=("completion_state", lambda s: _pct((s == "error").sum(), len(s))),
            metric_sanity_fail_rate=("metric_sanity_fail", lambda s: _pct(s.fillna(False).sum(), len(s))),
            citations_shown_rate=("citations_shown", lambda s: _pct(s.fillna(False).sum(), len(s))),
            citation_metadata_present_rate=(
                "citation_metadata_present",
                lambda s: _pct(s.fillna(False).sum(), len(s)),
            ),
        )
        .reset_index()
    )
    return grouped.sort_values(["intent_primary", "complexity_bucket"]).reset_index(drop=True)
