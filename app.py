"""
Gradio UI for the candidate ranker.

Wraps rank.py in a small web interface: upload a candidates file, get back
a ranked table and a downloadable CSV. Runs locally (`python app.py`) or as
a HuggingFace Space - the YAML block at the top of README.md configures it
as one.

The heavy lifting is all in rank.py; this file just shells out to it and
displays the result, so the UI is always running the exact same code that
produces the real submission.
"""

import os
import re
import subprocess
import sys
import tempfile
import time

import gradio as gr
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "sample_candidates.jsonl")
RANK_SCRIPT = os.path.join(HERE, "rank.py")

SUMMARY_RE = re.compile(
    r"([\d,]+) total candidates, (\d+) honeypots excluded, ([\d,]+) remain"
)
SOURCE_RE = re.compile(r"^Source: (.+)$", re.MULTILINE)


def run_ranker(uploaded_file=None):
    input_path = uploaded_file if uploaded_file else SAMPLE_PATH

    out_dir = tempfile.mkdtemp(prefix="ranker_")
    out_path = os.path.join(out_dir, "ranked_candidates.csv")

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, RANK_SCRIPT,
             "--candidates", input_path,
             "--out", out_path,
             "--top-k", "100"],
            capture_output=True, text=True, timeout=600,
            cwd=HERE,
        )
    except subprocess.TimeoutExpired:
        return pd.DataFrame(), "Timed out after 10 minutes.", None
    elapsed = time.time() - t0

    if result.returncode != 0:
        return pd.DataFrame(), f"rank.py failed:\n\n{result.stderr[-2000:]}", None

    df = pd.read_csv(out_path, encoding="utf-8")

    source_match = SOURCE_RE.search(result.stderr)
    source = source_match.group(1) if source_match else os.path.basename(input_path)

    match = SUMMARY_RE.search(result.stderr)
    if match:
        total, honeypots, kept = match.groups()
        summary = (
            f"Source file:            {source}\n"
            f"Candidates processed:   {total}\n"
            f"Excluded as honeypots:  {honeypots}\n"
            f"Eligible for ranking:   {kept}\n"
            f"Returned (top {len(df)}):       {len(df)} rows\n"
            f"Runtime:                {elapsed:.1f}s"
        )
    else:
        summary = (
            f"Source file: {source}\n"
            f"Returned {len(df)} rows in {elapsed:.1f}s."
        )

    return df, summary, out_path


with gr.Blocks(title="Candidate Ranker", theme=gr.themes.Soft(), analytics_enabled=False) as demo:
    gr.Markdown(
        "# Candidate Ranker\n"
        "Upload a candidates file and run the ranking pipeline against it. "
        "Accepts `.jsonl`, gzipped `.jsonl.gz`, or the original challenge "
        "`.zip`. Works on any size, from a handful of records up to the "
        "full 100,000-candidate pool (around a minute). Leave the upload "
        "empty to try it on a small sample."
    )

    with gr.Row():
        file_in = gr.File(
            label="candidates file",
            file_types=[".jsonl", ".gz", ".zip"],
            type="filepath",
            scale=3,
        )
        run_btn = gr.Button("Run ranker", variant="primary", scale=1)

    summary_out = gr.Textbox(label="Summary", lines=6, interactive=False)
    table_out = gr.Dataframe(label="Ranked candidates", wrap=True)
    download_out = gr.File(label="Download CSV")

    run_btn.click(run_ranker, inputs=file_in, outputs=[table_out, summary_out, download_out])
    demo.load(run_ranker, inputs=None, outputs=[table_out, summary_out, download_out])


if __name__ == "__main__":
    demo.launch()