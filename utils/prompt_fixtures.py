DEFAULT_EVIDENCE_PROMPT = """You are a relevance scorer. Given a hypothesis or criteria and a trace, score how relevant the trace is to the hypothesis.

HYPOTHESIS/CRITERIA: {hypothesis}

TRACE:
{search_text}

Return JSON: {{"relevant": true/false, "score": 0-100, "reason": "brief explanation"}}"""

DEFAULT_TAGGING_PROMPT = """You are a trace tagger. Given a user prompt and assistant output, tag the trace according to these criteria:

{criteria_desc}

USER PROMPT:
{user_prompt}

ASSISTANT OUTPUT:
{assistant_output}

Return JSON with keys: {criteria_keys}. Values should be short labels or categories."""

DEFAULT_GAP_ANALYSIS_PROMPT = """You are a product analyst. Analyze these conversation traces between users and a AI assistant with access to GIS information and tools.

{traces_summary}

Generate a comprehensive product gap analysis report with these sections:

## User Jobs (JTBD)
List the main jobs/tasks (When I <context>, I want to <motivation>, so that <outcome>) users are trying to accomplish, with rough frequency (high/medium/low).

## Coverage Assessment
For each user job, assess how well the assistant handles it:
- Well covered (consistently good responses)
- Partially covered (sometimes good, sometimes lacking)
- Underserved (poor or no capability)

## Success Patterns
What types of queries does the assistant handle best? What patterns lead to good outcomes?

## Gap Identification
What are the biggest gaps or areas for improvement?
- Missing capabilities users are asking for
- Common failure patterns
- Quality issues

## Recommendations
Top 3-5 product recommendations based on this analysis.

Format as clean markdown."""

DEFAULT_ENRICH_PROMPT = (
    "Given a user prompt, extract structured metadata as JSON with keys: "
    "datasets (list of dataset names if mentioned), topics (list of short topics), "
    "query_flavour (one of: data_request, interpretation, comparison, how_to, troubleshooting, general_chat, other). "
    "If unknown, use null or empty lists. Return JSON only.\n\n"
    "USER_PROMPT:\n{prompt}"
)

DEFAULT_ENRICH_BATCH_PROMPT_TEMPLATE = """Analyze each user prompt and extract structured metadata as JSON.
For each prompt, return: datasets (list of dataset names if mentioned), topics (list of short topics), query_flavour (one of: data_request, interpretation, comparison, how_to, troubleshooting, general_chat, other).
If unknown, use null or empty lists.

Return a JSON array with one object per prompt, in the same order as the input.
Return ONLY the JSON array, no other text.

PROMPTS:
{prompts_json}"""
