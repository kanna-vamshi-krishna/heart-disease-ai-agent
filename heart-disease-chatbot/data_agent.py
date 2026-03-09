"""
Data Agent — generates and executes Pandas/SQL-style queries on heart.csv
in a sandboxed environment. Returns structured results back to the chatbot.
"""

import pandas as pd
import numpy as np
import traceback
import io
import contextlib


# ── Load dataset once ─────────────────────────────────────────────────────────
def load_dataset(path: str = "heart.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        return None


# ── Safe code executor ────────────────────────────────────────────────────────
def safe_exec(code: str, df: pd.DataFrame) -> dict:
    """
    Execute AI-generated pandas code in a restricted namespace.
    Returns {"success": True, "output": ..., "error": None} or
            {"success": False, "output": None, "error": "..."}
    """
    # Allowed builtins only
    safe_builtins = {
        "len": len, "range": range, "print": print,
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "round": round, "abs": abs, "min": min, "max": max, "sum": sum,
        "sorted": sorted, "enumerate": enumerate, "zip": zip,
        "isinstance": isinstance, "type": type,
    }
    namespace = {
        "__builtins__": safe_builtins,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "result": None,
    }

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, namespace)
        printed = stdout_capture.getvalue().strip()
        result = namespace.get("result")
        return {
            "success": True,
            "result": result,
            "printed": printed,
            "error": None,
        }
    except Exception:
        return {
            "success": False,
            "result": None,
            "printed": "",
            "error": traceback.format_exc(limit=3),
        }


# ── Format result for display ─────────────────────────────────────────────────
def format_result(exec_result: dict) -> str:
    if not exec_result["success"]:
        return f"⚠️ Query error:\n```\n{exec_result['error']}\n```"

    parts = []
    if exec_result["printed"]:
        parts.append(exec_result["printed"])

    r = exec_result["result"]
    if r is None and not exec_result["printed"]:
        return "✅ Query ran but returned no result. Make sure the code assigns to `result`."

    if r is not None:
        if isinstance(r, pd.DataFrame):
            if r.empty:
                parts.append("*Empty result — no rows matched.*")
            else:
                # Limit to 30 rows for display
                display_df = r.head(30)
                parts.append(display_df.to_markdown(index=False))
                if len(r) > 30:
                    parts.append(f"*... showing 30 of {len(r)} rows*")
        elif isinstance(r, pd.Series):
            parts.append(r.to_string())
        else:
            parts.append(str(r))

    return "\n\n".join(parts)


# ── Dataset summary for context ───────────────────────────────────────────────
def get_dataset_summary(df: pd.DataFrame) -> str:
    if df is None:
        return "Dataset not loaded."
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    buf.write(f"Columns: {list(df.columns)}\n")
    buf.write(f"Dtypes:\n{df.dtypes.to_string()}\n")
    buf.write(f"\nFirst 3 rows:\n{df.head(3).to_string()}\n")
    buf.write(f"\nBasic stats:\n{df.describe().round(2).to_string()}\n")
    return buf.getvalue()
