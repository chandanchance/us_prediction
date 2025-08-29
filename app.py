from __future__ import annotations

import os
from typing import Any, Dict
import json

from flask import Flask, jsonify, render_template, request, url_for, redirect
import pandas as pd
import pdb

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_ROOT, "test.csv")


def load_dataframe() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        # Give a clear error to the user so they know to generate the CSV
        raise FileNotFoundError(
            f"CSV not found at {CSV_PATH}. Run generate_csv.py to create a sample file."
        )
    df = pd.read_csv(CSV_PATH)
    return df


def build_prediction_output(record_id: int) -> Dict[str, Any]:
    df = load_dataframe()

    if "id" not in df.columns:
        raise ValueError("CSV must contain an 'id' column")

    # Locate the row by id
    row_match = df[df["id"] == record_id]
    if row_match.empty:
        raise ValueError(f"No record found for id {record_id}")

    row_dict: Dict[str, Any] = row_match.iloc[0].to_dict()
    # Convert any numpy/pandas scalar types to built-in Python types
    row_clean: Dict[str, Any] = json.loads(pd.Series(row_dict).to_json())

    # Ensure is_fit is present; default to 0 if missing
    is_fit_value = int(row_dict.get("is_fit", 0))

    # Build output blocks according to spec
    output: Dict[str, Any] = {
        "block_ribbon": {
            "id": int(row_dict.get("id", record_id)),
            "retrieved": row_clean,
        },
        "block_middle_upper": {
            "is_fit": is_fit_value,
            "key_factors": "I think the prediction was fit because of training frequency, cardio score, and diet consistency.",
        },
        "block_middle_lower": {
            # Use to_json/to_dict round-trip to coerce numpy types → Python built-ins
            "table_rows": json.loads(df.head(10).to_json(orient="records")),
            "columns": list(df.columns),
        },
        "block_nearest_neighbors": {"Nearest neighbors": "Nearest neighbors"},
        "block_right": {
            "narrative": f"For the id {record_id}, it is predicted as the person is "
            + ("fit" if is_fit_value == 1 else "not fit"),
        },
    }

    return output


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        df = load_dataframe()
        records = df.to_dict(orient="records")
        columns = list(df.columns)
        # Expect an 'id' column present
        return render_template("index.html", records=records, columns=columns)

    @app.route("/on_predict")
    def on_predict():
        # Takes id as input and returns JSON per spec
        id_param = request.args.get("id")
        if id_param is None:
            return jsonify({"error": "Missing required query parameter 'id'"}), 400
        try:
            record_id = int(id_param)
        except ValueError:
            return jsonify({"error": "'id' must be an integer"}), 400

        try:
            output = build_prediction_output(record_id)
        except Exception as exc:  # provide error context in development-friendly way
            return jsonify({"error": str(exc)}), 400

        # Per requirements, return JSON
        return jsonify({"output": output})

    @app.route("/result")
    def result_page():
        # Load and render page 2 using the same generation logic (server-side rendering)
        id_param = request.args.get("id")
        if id_param is None:
            # If user navigates here directly, go back home
            return redirect(url_for("index"))
        try:
            record_id = int(id_param)
        except ValueError:
            return redirect(url_for("index"))

        try:
            output = build_prediction_output(record_id)
        except Exception:
            return redirect(url_for("index"))

        return render_template("result.html", output=output)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=True)


