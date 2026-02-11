from utils.codeact_explorer_features import (
    classify_codeact_chart_prep_mode,
    classify_codeact_retrieval_mode,
    compute_codeact_flags,
)


def test_retrieval_mode_analytics_api() -> None:
    decoded_code_blocks = ["requests.get('https://analytics.globalnaturewatch.org/v1/analytics/query')"]
    output_obj = {"raw_data": {}}
    assert classify_codeact_retrieval_mode(decoded_code_blocks, output_obj) == "analytics_api"


def test_retrieval_mode_prefetched_only() -> None:
    decoded_code_blocks = ["df = df.groupby('x').sum()"]
    output_obj = {
        "raw_data": {
            "nested": {
                "source_url": "https://example.com/source.csv",
                "dataset_name": "dataset-a",
            }
        }
    }
    assert classify_codeact_retrieval_mode(decoded_code_blocks, output_obj) == "prefetched_only"


def test_chart_prep_mode_hardcoded() -> None:
    decoded_code_blocks = ["import pandas as pd\ndf = pd.DataFrame({'a':[1,2], 'b':[3,4]})"]
    assert classify_codeact_chart_prep_mode(decoded_code_blocks) == "hardcoded_chart_data"


def test_flag_pie_percent_sum_off() -> None:
    decoded_code_blocks = ["print('noop')"]
    output_obj = {
        "charts_data": [
            {
                "type": "pie",
                "data": [
                    {"label": "A", "value": 70},
                    {"label": "B", "value": 60},
                ],
            }
        ]
    }
    flags = compute_codeact_flags(decoded_code_blocks, output_obj)
    assert flags["flag_pie_percent_sum_off"] is True
