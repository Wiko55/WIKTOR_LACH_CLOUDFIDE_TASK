import pandas as pd
import pytest
from solution import add_virtual_column


def test_operator_precedence_multiplication_before_addition():
    df = pd.DataFrame({"a": [2, 3], "b": [4, 5], "c": [10, 2]})
    expected = pd.DataFrame(
        {"a": [2, 3], "b": [4, 5], "c": [10, 2], "result": [42, 13]}
    )
    result = add_virtual_column(df, "a + b * c", "result")
    assert result.equals(expected)


def test_operator_precedence_revenue_formula():
    df = pd.DataFrame(
        {
            "quantity": [10, 5],
            "unit_price": [20.0, 12.0],
            "discount": [15.0, 0.0],
        }
    )
    expected = df.copy()
    expected["revenue"] = [185.0, 60.0]

    result = add_virtual_column(df, "quantity * unit_price - discount", "revenue")
    assert result.equals(expected)


def test_left_associativity_for_subtraction():
    df = pd.DataFrame({"a": [10, 20], "b": [3, 5], "c": [2, 4]})
    # Must evaluate as (10 - 3) - 2 = 5, not 10 - (3 - 2) = 9
    expected = pd.DataFrame(
        {"a": [10, 20], "b": [3, 5], "c": [2, 4], "result": [5, 11]}
    )
    result = add_virtual_column(df, "a - b - c", "result")
    assert result.equals(expected)


def test_overwrite_existing_column():
    df = pd.DataFrame({"a": [2, 4], "b": [3, 5]})
    expected = pd.DataFrame({"a": [5, 9], "b": [3, 5]})
    result = add_virtual_column(df, "a + b", "a")
    assert result.equals(expected)


def test_input_dataframe_is_not_mutated():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    original = df.copy()
    _ = add_virtual_column(df, "a + b", "c")
    assert df.equals(original)


@pytest.mark.parametrize(
    "invalid_new_col",
    ["col3", "new-col", "new col", "", "   ", "col!", 123, None],
)
def test_invalid_new_column_name_returns_empty_df(invalid_new_col):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = add_virtual_column(df, "a + b", invalid_new_col)
    assert result.empty


@pytest.mark.parametrize("invalid_df_col", ["col 1", "col2", "col-a", "price$"])
def test_invalid_existing_dataframe_column_returns_empty_df(invalid_df_col):
    df = pd.DataFrame({invalid_df_col: [1, 2], "valid_col": [3, 4]})
    result = add_virtual_column(df, "valid_col + valid_col", "result")
    assert result.empty


@pytest.mark.parametrize(
    "invalid_role",
    [
        "",
        "   ",
        "a",
        "-a",
        "+ a",
        "a +",
        "a -",
        "a * ",
        "a + * b",
        "a b",
        "a / b",
        "a ** b",
        "a % b",
        "(a + b)",
        "a + 10",
        "a & b",
        None,
    ],
)
def test_invalid_role_syntax_returns_empty_df(invalid_role):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = add_virtual_column(df, invalid_role, "result")
    assert result.empty


def test_missing_column_in_role_returns_empty_df():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = add_virtual_column(df, "a + non_existent_col", "result")
    assert result.empty


def test_non_numeric_or_boolean_columns_return_empty_df():
    df_str = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
    assert add_virtual_column(df_str, "a + b", "result").empty

    df_bool = pd.DataFrame({"a": [True, False], "b": [False, True]})
    assert add_virtual_column(df_bool, "a + b", "result").empty


def test_empty_input_dataframe_returns_empty_df():
    df = pd.DataFrame(columns=["a", "b"])
    assert add_virtual_column(df, "a + b", "result").empty
