from pathlib import Path

import pandas as pd
import pytest

from pipeline import clean_sales_data, run_pipeline


SOURCE_COLUMNS = [
    "Order ID",
    "Region",
    "Product",
    "Quantity ",
    "Unit Price",
    "Discount",
    "Order Date",
]


def _sales_row(
    order_id="SO-00001",
    region="North",
    product="Widget",
    quantity="2",
    unit_price="10.0",
    discount="1.0",
    order_date="2025-01-01",
):
    return {
        "Order ID": order_id,
        "Region": region,
        "Product": product,
        "Quantity ": quantity,
        "Unit Price": unit_price,
        "Discount": discount,
        "Order Date": order_date,
    }


def _sales_frame(*rows):
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def test_clean_sales_data_normalizes_headers_and_removes_structural_and_duplicate_rows():
    first_row = _sales_row()
    repeated_header = dict(zip(SOURCE_COLUMNS, SOURCE_COLUMNS))
    same_order_different_line = _sales_row(
        quantity="3",
        discount="0.0",
        order_date="2025-01-02",
    )
    total_row = _sales_row(
        order_id="TOTAL",
        region="",
        product="",
        quantity="5",
        unit_price="",
        discount="1.0",
        order_date="",
    )

    cleaned_df = clean_sales_data(
        _sales_frame(
            first_row,
            repeated_header,
            first_row.copy(),
            same_order_different_line,
            total_row,
        )
    )

    assert list(cleaned_df.columns) == [
        "order_id",
        "region",
        "product",
        "quantity",
        "unit_price",
        "discount",
        "order_date",
        "revenue",
    ]
    assert str(cleaned_df["quantity"].dtype) == "Int64"
    assert len(cleaned_df) == 2
    assert cleaned_df["order_id"].tolist() == ["SO-00001", "SO-00001"]
    assert cleaned_df["quantity"].tolist() == [2, 3]


def test_clean_sales_data_removes_internal_quantity_whitespace_and_decimal_commas():
    cleaned_df = clean_sales_data(
        _sales_frame(_sales_row(quantity="1 762", unit_price="39,90", discount="0"))
    )

    assert cleaned_df.loc[0, "quantity"] == 1762
    assert str(cleaned_df["quantity"].dtype) == "Int64"
    assert cleaned_df.loc[0, "unit_price"] == pytest.approx(39.9)
    assert cleaned_df.loc[0, "revenue"] == pytest.approx(1762 * 39.9)


def test_clean_sales_data_preserves_negative_quantity_and_calculates_negative_revenue():
    cleaned_df = clean_sales_data(
        _sales_frame(
            _sales_row(quantity="-2", unit_price="10.0", discount="1.0")
        )
    )

    assert cleaned_df.loc[0, "quantity"] == -2
    assert cleaned_df.loc[0, "revenue"] == -21.0


def test_clean_sales_data_imputes_missing_price_with_product_median():
    cleaned_df = clean_sales_data(
        _sales_frame(
            _sales_row(order_id="SO-00001", unit_price="10.0"),
            _sales_row(order_id="SO-00002", unit_price="10.0"),
            _sales_row(order_id="SO-00003", unit_price=""),
        )
    )

    assert cleaned_df["unit_price"].tolist() == [10.0, 10.0, 10.0]
    assert cleaned_df.loc[2, "revenue"] == 19.0


def test_clean_sales_data_rejects_inconsistent_product_prices():
    raw_df = _sales_frame(
        _sales_row(order_id="SO-00001", unit_price="10.0"),
        _sales_row(order_id="SO-00002", unit_price="11.0"),
        _sales_row(order_id="SO-00003", unit_price=""),
    )

    with pytest.raises(ValueError, match="not consistent"):
        clean_sales_data(raw_df)


def test_clean_sales_data_rejects_product_without_any_observed_price():
    raw_df = _sales_frame(
        _sales_row(order_id="SO-00001", unit_price=""),
        _sales_row(order_id="SO-00002", unit_price=""),
    )

    with pytest.raises(ValueError, match="product-specific median"):
        clean_sales_data(raw_df)


def test_clean_sales_data_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        clean_sales_data([])


def test_clean_sales_data_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="at least one sales row"):
        clean_sales_data(pd.DataFrame(columns=SOURCE_COLUMNS))


def test_clean_sales_data_rejects_input_with_no_detail_rows():
    repeated_header = dict(zip(SOURCE_COLUMNS, SOURCE_COLUMNS))
    total_row = _sales_row(order_id="TOTAL")

    with pytest.raises(ValueError, match="No sales detail rows remain"):
        clean_sales_data(_sales_frame(repeated_header, total_row))


def test_clean_sales_data_rejects_missing_required_columns():
    raw_df = _sales_frame(_sales_row()).drop(columns=["Order Date"])

    with pytest.raises(ValueError, match="Missing required columns"):
        clean_sales_data(raw_df)


@pytest.mark.parametrize("column", ["Region", "Product"])
def test_clean_sales_data_rejects_empty_grouping_categories(column):
    row = _sales_row()
    row[column] = "   "

    with pytest.raises(ValueError, match="region or product is missing"):
        clean_sales_data(_sales_frame(row))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Quantity ", "not-a-number", "Quantity contains"),
        ("Unit Price", "not-a-price", "Unit Price contains"),
        ("Discount", "not-a-discount", "Discount contains"),
        ("Order Date", "not-a-date", "Order Date contains"),
    ],
)
def test_clean_sales_data_rejects_unparseable_values(column, value, message):
    row = _sales_row()
    row[column] = value

    with pytest.raises(ValueError, match=message):
        clean_sales_data(_sales_frame(row))


def test_run_pipeline_uses_given_paths_and_returns_dataframes(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    _sales_frame(
        _sales_row(order_id="SO-00001", discount="1.0"),
        _sales_row(
            order_id="SO-00002",
            region="South",
            quantity="3",
            discount="",
        ),
    ).to_csv(input_path, index=False)

    cleaned_df, summary_df = run_pipeline(input_path, output_path)

    assert output_path.exists()
    assert cleaned_df["revenue"].tolist() == [19.0, 30.0]
    assert summary_df.to_dict("records") == [
        {"region": "North", "revenue": 19.0},
        {"region": "South", "revenue": 30.0},
    ]


def test_run_pipeline_on_sales_csv_matches_checked_in_regional_summary(tmp_path):
    input_path = Path(__file__).with_name("sales.csv")
    output_path = tmp_path / "revenue_by_region.csv"

    cleaned_df, summary_df = run_pipeline(input_path, output_path)

    assert len(cleaned_df) == 1500
    assert cleaned_df["revenue"].notna().all()
    assert (cleaned_df["quantity"] != 0).all()
    assert (
        cleaned_df["discount"]
        <= (cleaned_df["quantity"] * cleaned_df["unit_price"]).abs()
    ).all()

    checked_in_summary = pd.read_csv(
        Path(__file__).with_name("revenue_by_region.csv")
    )
    written_summary = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(
        summary_df.reset_index(drop=True),
        checked_in_summary,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        written_summary,
        checked_in_summary,
        check_dtype=False,
    )
