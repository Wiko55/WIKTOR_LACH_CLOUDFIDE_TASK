import pandas as pd

from pipeline import clean_sales_data, run_pipeline


def test_clean_sales_data_calculates_revenue():
    raw_df = pd.DataFrame(
        {
            "order_id": ["SO-00001", "SO-00002"],
            "order_date": ["2025-01-01", "2025-01-02"],
            "region": [" north ", "NORTH"],
            "product": ["Widget", "Widget"],
            "quantity": ["2", "3"],
            "unit_price": ["10.0", "10.0"],
            "discount": ["1.0", float("nan")],
        }
    )

    cleaned_df = clean_sales_data(raw_df)

    assert cleaned_df["discount"].tolist() == [1.0, 0.0]
    assert cleaned_df["revenue"].tolist() == [19.0, 30.0]
    assert cleaned_df["region"].tolist() == ["North", "North"]


def test_run_pipeline_uses_given_paths_and_returns_dataframes(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "order_id": ["SO-00001", "SO-00002"],
            "order_date": ["2025-01-01", "2025-01-02"],
            "region": ["North", "South"],
            "product": ["Widget", "Widget"],
            "quantity": ["2", "3"],
            "unit_price": ["10.0", "10.0"],
            "discount": ["1.0", ""],
        }
    ).to_csv(input_path, index=False)

    cleaned_df, summary_df = run_pipeline(input_path, output_path)

    assert output_path.exists()
    assert cleaned_df["revenue"].tolist() == [19.0, 30.0]
    assert summary_df.to_dict("records") == [
        {"region": "North", "revenue": 19.0},
        {"region": "South", "revenue": 30.0},
    ]
