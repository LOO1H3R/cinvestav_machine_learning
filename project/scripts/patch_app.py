import re

with open("project/app.py", "r") as f:
    app_code = f.read()

new_route = """
@app.get("/dataset")
def dataset_page():
    import pandas as pd
    data_file = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not data_file.exists():
        return HTMLResponse("<h1>Dataset not found.</h1>")
    
    df = pd.read_csv(data_file)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    n_rows, n_cols = df.shape
    missing_vals = df.isnull().sum().sum()
    churn_counts = df["Churn"].value_counts().to_dict() if "Churn" in df.columns else {}
    
    head_html = df.head().to_html(classes="data-table", index=False)
    desc_html = df.describe().round(2).to_html(classes="data-table")
    dtypes_html = pd.DataFrame({"Type": df.dtypes.astype(str), "Missings": df.isnull().sum()}).to_html(classes="data-table")
    
    html = f\"\"\"<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Dataset Overview</title>
    <style>
        :root {{
            --bg-1: #f6f7fb; --bg-2: #e8efff; --surface: #ffffff;
            --text: #1f2937; --muted: #6b7280; --border: #d1d5db;
            --shadow: 0 12px 30px rgba(15,23,42,0.12); --radius: 14px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0; font-family: "Poppins", "Segoe UI", sans-serif;
            color: var(--text); padding: 24px;
            background: linear-gradient(160deg, var(--bg-1), var(--bg-2));
            min-height: 100vh;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; display: grid; gap: 18px; }}
        .hero {{
            background: linear-gradient(135deg, #8b5cf6, #ec4899);
            color: #fff; border-radius: var(--radius); padding: 24px; box-shadow: var(--shadow);
        }}
        .card {{
            background: var(--surface); border: 1px solid rgba(255,255,255,0.35);
            border-radius: var(--radius); box-shadow: var(--shadow); padding: 20px; overflow-x: auto;
        }}
        .data-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }}
        .data-table th, .data-table td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
        .data-table th {{ background: #f8fafc; font-weight: 600; text-transform: capitalize; }}
        .actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 5px; }}
        .btn {{
            display: inline-block; text-decoration: none; border-radius: 999px;
            padding: 10px 16px; border: 1px solid var(--border); color: var(--text);
            background: #fff; font-weight: 600; font-size: 0.95rem;
        }}
        .btn:hover {{ background: #f1f5f9; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }}
        .stat-box {{
            padding: 15px; background: #f8fafc; border-radius: 12px;
            text-align: center; border: 1px solid var(--border);
        }}
        .stat-value {{ font-size: 1.8em; font-weight: 700; color: #8b5cf6; margin-top: 8px; }}
        .stat-label {{ color: var(--muted); font-size: 0.9rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }}
    </style>
</head>
<body>
    <main class='container'>
        <section class='hero'>
            <h1 style='margin:0 0 8px 0;'>Telco Customer Dataset Overview</h1>
            <p style='margin:0; opacity:.95;'>Essential distributions and data preview of the source CSV.</p>
        </section>

        <section class='card'>
            <div class='stat-grid'>
                <div class='stat-box'><div class='stat-label'>Total Rows</div><div class='stat-value'>{n_rows}</div></div>
                <div class='stat-box'><div class='stat-label'>Total Columns</div><div class='stat-value'>{n_cols}</div></div>
                <div class='stat-box'><div class='stat-label'>Missing Values</div><div class='stat-value'>{missing_vals}</div></div>
                <div class='stat-box'><div class='stat-label'>Churn (Yes)</div><div class='stat-value'>{churn_counts.get('Yes', 0)}</div></div>
                <div class='stat-box'><div class='stat-label'>Churn (No)</div><div class='stat-value'>{churn_counts.get('No', 0)}</div></div>
            </div>
        </section>

        <section class='card'>
            <h2 style='margin-top:0;'>Numeric Statistics</h2>
            {desc_html}
        </section>

        <section class='card'>
            <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px;">
                <div>
                    <h2 style='margin-top:0;'>Columns & Types</h2>
                    {dtypes_html}
                </div>
                <div style="overflow-x: auto;">
                    <h2 style='margin-top:0;'>Data Preview (First 5 Rows)</h2>
                    {head_html}
                </div>
            </div>
        </section>

        <section class='actions'>
            <a class='btn' href='/'>Back to Predictor</a>
            <a class='btn' href='/performance'>Performance Comparison</a>
            <a class='btn' href='/tracking'>MLflow / Metaflow Runs</a>
        </section>
    </main>
</body>
</html>\"\"\"

    return HTMLResponse(html)

if __name__ == "__main__":
"""

app_code = app_code.replace('if __name__ == "__main__":', new_route)

with open("project/app.py", "w") as f:
    f.write(app_code)
