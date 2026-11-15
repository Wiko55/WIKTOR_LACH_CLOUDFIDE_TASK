"""Clean sales data and write revenue totals by region.

``clean_sales_data`` standardizes a raw sales DataFrame, removes repeated
headers, the ``TOTAL`` summary row, and exact duplicates, then converts the
calculation fields to numeric types. Blank discounts are treated as zero.
Missing unit prices are filled with the median for the same product after
checking that observed prices are consistent within each product. Revenue is
calculated with ``add_virtual_column`` as
``quantity * unit_price - discount``.

``run_pipeline`` reads the input CSV, cleans its data, sums revenue by region,
writes the summary CSV, and returns both the cleaned data and summary. Relative
paths are resolved from the directory containing this script.
"""

from pathlib import Path

import pandas as pd

from solution import add_virtual_column


def clean_sales_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw sales data and add a revenue column."""
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame.")
    if raw_df.empty:
        raise ValueError("raw_df must contain at least one sales row.")

    sales = raw_df.copy()
    sales.columns = (
        pd.Index(sales.columns.map(str))
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    sales = sales.astype("string")

    required_columns = {
        "order_id",
        "order_date",
        "region",
        "product",
        "quantity",
        "unit_price",
        "discount",
    }
    missing_columns = required_columns.difference(sales.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Remove repeated header rows and the summary row before converting values.
    header_rows = sales.apply(
        lambda column: column.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .eq(column.name)
        .fillna(False)
    ).all(axis=1)
    sales = sales.loc[~header_rows].copy()

    total_rows = (
        sales["order_id"]
        .str.strip()
        .str.upper()
        .eq("TOTAL")
        .fillna(False)
    )
    sales = sales.loc[~total_rows].copy()
    if sales.empty:
        raise ValueError("No sales detail rows remain after removing summary rows.")

    # Standardize categories and reject rows that cannot be grouped meaningfully.
    for column in ["region", "product"]:
        values = sales[column].str.strip().replace("", pd.NA)
        sales[column] = values.str.capitalize().astype("string")
    sales["order_id"] = sales["order_id"].str.strip().astype("string")

    if sales[["region", "product"]].isna().any().any():
        raise ValueError("A region or product is missing after removing header rows.")

    # Normalize and validate quantity values before converting them to integers.
    quantity_text = sales["quantity"].str.strip().str.replace(
        r"\s+", "", regex=True
    )
    quantity_values = pd.to_numeric(
        quantity_text.replace({"<NA>": pd.NA, "": pd.NA}),
        errors="coerce",
    )
    invalid_quantity = quantity_text.ne("") & quantity_values.isna()
    if invalid_quantity.any() or quantity_values.isna().any():
        raise ValueError("Quantity contains missing or unparseable values.")
    sales["quantity"] = quantity_values.astype("Int64")

    # Normalize decimal commas and preserve missing prices for product-level imputation.
    unit_price_text = sales["unit_price"].str.strip().str.replace(
        ",", ".", regex=False
    )
    unit_price_values = pd.to_numeric(
        unit_price_text.replace({"<NA>": pd.NA, "": pd.NA}),
        errors="coerce",
    )
    invalid_unit_price = unit_price_text.ne("") & unit_price_values.isna()
    if invalid_unit_price.any():
        raise ValueError("Unit Price contains unparseable values.")
    sales["unit_price"] = unit_price_values.astype("Float64")

    # Convert blank discounts and the literal marker to missing values first.
    discount_text = sales["discount"].str.strip()
    discount_text = discount_text.replace({"<NA>": pd.NA, "": pd.NA})
    discount_values = pd.to_numeric(discount_text, errors="coerce")
    invalid_discount = discount_text.notna() & discount_values.isna()
    if invalid_discount.any():
        raise ValueError("Discount contains unparseable values.")
    sales["discount"] = discount_values.astype("Float64").fillna(0.0)

    # Parse dates and reject non-empty values that cannot be interpreted.
    order_date_text = sales["order_date"].str.strip().replace("", pd.NA)
    order_dates = pd.to_datetime(order_date_text, errors="coerce")
    invalid_dates = order_date_text.notna() & order_dates.isna()
    if invalid_dates.any():
        raise ValueError("Order Date contains unparseable values.")
    sales["order_date"] = order_dates

    # Remove full duplicate rows while retaining distinct order lines.
    sales = sales.drop_duplicates().copy()

    # Fill missing prices only when each product has one consistent observed price.
    product_price_counts = sales.groupby("product")["unit_price"].nunique(
        dropna=True
    )
    if product_price_counts.gt(1).any():
        raise ValueError("Unit Price is not consistent within every product.")

    product_medians = sales.groupby("product")["unit_price"].transform("median")
    missing_prices = sales["unit_price"].isna()
    if product_medians.loc[missing_prices].isna().any():
        raise ValueError("Cannot determine a product-specific median Unit Price.")
    sales.loc[missing_prices, "unit_price"] = product_medians.loc[missing_prices]

    # Calculate revenue with the Part 1 helper function.
    cleaned_df = add_virtual_column(
        sales,
        "quantity * unit_price - discount",
        "revenue",
    )
    if "revenue" not in cleaned_df.columns:
        raise ValueError("Could not calculate the revenue column.")

    return cleaned_df


def run_pipeline(
    input_path: str | Path = "sales.csv",
    output_path: str | Path = "revenue_by_region.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read, clean, summarize, and save sales data; return both DataFrames."""
    script_directory = Path(__file__).resolve().parent
    resolved_input_path = Path(input_path)
    resolved_output_path = Path(output_path)
    if not resolved_input_path.is_absolute():
        resolved_input_path = script_directory / resolved_input_path
    if not resolved_output_path.is_absolute():
        resolved_output_path = script_directory / resolved_output_path

    # Keep raw values as strings so clean_sales_data controls all conversions.
    raw_df = pd.read_csv(
        resolved_input_path,
        dtype=str,
        keep_default_na=False,
    )
    cleaned_df = clean_sales_data(raw_df)

    # Aggregate calculated line revenue by region and write the requested CSV.
    summary_df = (
        cleaned_df.groupby("region", as_index=False)["revenue"]
        .sum()
        .sort_values("region")
    )
    summary_df.to_csv(resolved_output_path, index=False)
    print(f"Wrote regional revenue totals to {resolved_output_path}.")

    return cleaned_df, summary_df


if __name__ == "__main__":
    run_pipeline()
