import re

import pandas as pd

# Pattern allowing only uppercase/lowercase letters and underscores (_)
VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z_]+$")

# Operator precedence: multiplication runs before addition and subtraction
OPERATOR_PRIORITY = {
    "+": 1,
    "-": 1,
    "*": 2,
}


def _is_valid_column_name(name: object) -> bool:
    """Check whether a column label is a string containing only letters and underscores."""
    if not isinstance(name, str):
        return False
    return bool(VALID_NAME_PATTERN.match(name))


def _split_expression(role: str) -> list[str] | None:
    """
    Convert an expression string into tokens and validate its syntax.
    Example: 'quantity*unit_price - discount' -> ['quantity', '*', 'unit_price', '-', 'discount']
    """
    if not isinstance(role, str):
        return None

    # Surround operators with spaces so .split() cleanly separates them
    spaced_role = role
    for op in OPERATOR_PRIORITY:
        spaced_role = spaced_role.replace(op, f" {op} ")

    tokens = spaced_role.split()

    # A valid expression needs at least 3 tokens (e.g., col_a + col_b)
    if len(tokens) < 3:
        return None

    # Enforce alternating pattern: column_name -> operator -> column_name
    expecting_column = True
    for token in tokens:
        if expecting_column:
            if not _is_valid_column_name(token):
                return None
        else:
            if token not in OPERATOR_PRIORITY:
                return None
        expecting_column = not expecting_column

    # Expression cannot end with an operator (expecting_column would still be True)
    if expecting_column:
        return None

    return tokens


def _to_rpn(tokens: list[str]) -> list[str]:
    """
    Convert standard infix notation into Reverse Polish Notation (RPN)
    using the Shunting-yard algorithm to preserve operator precedence.
    """
    output: list[str] = []
    operators_stack: list[str] = []

    for token in tokens:
        if token in OPERATOR_PRIORITY:
            # Pop operators with higher or equal precedence before pushing the new one
            while (
                operators_stack
                and OPERATOR_PRIORITY[operators_stack[-1]] >= OPERATOR_PRIORITY[token]
            ):
                output.append(operators_stack.pop())
            operators_stack.append(token)
        else:
            # Column names go directly to the output queue
            output.append(token)

    # Pop any remaining operators onto the output queue
    while operators_stack:
        output.append(operators_stack.pop())

    return output


def _evaluate_rpn(df: pd.DataFrame, rpn_tokens: list[str]) -> pd.Series:
    """Evaluate a list of RPN tokens against the columns of `df`."""
    stack: list[pd.Series] = []

    for token in rpn_tokens:
        if token not in OPERATOR_PRIORITY:
            # Push the entire pandas Series onto the stack
            stack.append(df[token])
        else:
            # Pop the top two Series (order matters for subtraction)
            right_col = stack.pop()
            left_col = stack.pop()

            if token == "+":
                stack.append(left_col + right_col)
            elif token == "-":
                stack.append(left_col - right_col)
            elif token == "*":
                stack.append(left_col * right_col)

    return stack[0]


def add_virtual_column(
    df: pd.DataFrame,
    role: str,
    new_column: str,
) -> pd.DataFrame:
    """Return a copy of `df` with `new_column` computed from the `role` expression."""
    # 1. Validate the input DataFrame and column names
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    if not _is_valid_column_name(new_column):
        return pd.DataFrame()

    if not all(_is_valid_column_name(col) for col in df.columns):
        return pd.DataFrame()

    # 2. Tokenize and validate the expression syntax
    tokens = _split_expression(role)
    if tokens is None:
        return pd.DataFrame()

    # 3. Verify that all referenced columns exist in `df` and contain numeric data
    used_columns = [token for token in tokens if token not in OPERATOR_PRIORITY]
    for col in used_columns:
        if col not in df.columns:
            return pd.DataFrame()
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
            df[col]
        ):
            return pd.DataFrame()

    # 4. Convert to RPN, compute the values, and attach the new column to a copy of `df`
    rpn_tokens = _to_rpn(tokens)
    result = df.copy()
    result[new_column] = _evaluate_rpn(result, rpn_tokens)

    return result
