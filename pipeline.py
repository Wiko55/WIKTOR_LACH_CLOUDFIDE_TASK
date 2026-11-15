"""Build regional revenue totals from the raw sales export.

The pipeline reads ``sales.csv`` as text so source values can be normalized
consistently, standardizes the column names and categorical values, and removes
the repeated header row, the ``TOTAL`` summary row, and fully duplicated rows.
It converts quantity, unit price, discount, and order date to usable types.
Blank discounts are treated as zero because the source defines them as orders
with no discount. Missing unit prices are filled with the median price for the
same product after verifying that each product has a consistent observed price.
The ``add_virtual_column`` function calculates revenue as
``quantity * unit_price - discount``. Finally, the pipeline sums revenue by
region and writes the result to ``revenue_by_region.csv`` beside this script.
"""

from pathlib import Path

import pandas as pd

from solution import add_virtual_column


def main() -> None:
    data_path = Path(__file__).with_name("sales.csv")
    output_path = Path(__file__).with_name("revenue_by_region.csv")

    # Read as text to preserve source formatting until each field is normalized.
    sales = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    sales.columns = (
        sales.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Remove the repeated header and the summary row before converting values.
    header_rows = sales.apply(
        lambda column: column.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .eq(column.name)
    ).all(axis=1)
    sales = sales.loc[~header_rows].copy()

    total_rows = sales["order_id"].str.strip().str.upper().eq("TOTAL")
    if int(total_rows.sum()) != 1:
        raise ValueError(
            f"Expected exactly one TOTAL summary row, found: {int(total_rows.sum())}"
        )
    sales = sales.loc[~total_rows].copy()

    # Standardize categorical fields and keep the order ID as text.
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
        quantity_text.replace("", pd.NA),
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
        unit_price_text.replace("", pd.NA),
        errors="coerce",
    )
    invalid_unit_price = unit_price_text.ne("") & unit_price_values.isna()
    if invalid_unit_price.any():
        raise ValueError("Unit Price contains unparseable values.")
    sales["unit_price"] = unit_price_values.astype("Float64")

    # Convert blank discounts and the literal marker to missing values first.
    discount_text = sales["discount"].astype("string").str.strip()
    discount_text = discount_text.replace({"<NA>": pd.NA, "": pd.NA})
    discount_values = pd.to_numeric(discount_text, errors="coerce")
    invalid_discount = discount_text.notna() & discount_values.isna()
    if invalid_discount.any():
        raise ValueError("Discount contains unparseable values.")
    sales["discount"] = discount_values.astype("Float64").fillna(0.0)

    # Parse dates and reject non-empty values that cannot be interpreted.
    order_date_text = sales["order_date"].astype("string").str.strip().replace(
        "", pd.NA
    )
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

    # Use the Part 1 function to calculate line revenue.
    sales_with_revenue = add_virtual_column(
        sales,
        "quantity * unit_price - discount",
        "revenue",
    )
    if "revenue" not in sales_with_revenue.columns:
        raise ValueError("Could not calculate the revenue column.")

    # Aggregate line revenue by region and write the requested output file.
    revenue_by_region = (
        sales_with_revenue.groupby("region", as_index=False)["revenue"]
        .sum()
        .sort_values("region")
    )
    revenue_by_region.to_csv(output_path, index=False)
    print(f"Wrote regional revenue totals to {output_path}.")


if __name__ == "__main__":
    main()
