"""
Gradio 5.x-compatible UI for the candidate ranker.
Uses gradio 5.35.0 (stable HF Spaces version) to avoid version-conflict issues.
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


def _resolve_path(uploaded_file) -> str:
    """
    Gradio 3.x returns a dict {'name': '/tmp/...', 'size': ..., 'data': ...}.
    Gradio 4.x with type='filepath' returns a plain string.
    This helper handles both.
    """
    if uploaded_file is None:
        return SAMPLE_PATH
    if isinstance(uploaded_file, dict):
        path = uploaded_file.get("name") or uploaded_file.get("path") or ""
        return path if path else SAMPLE_PATH
    return str(uploaded_file)


def run_ranker(uploaded_file=None):
    input_path = _resolve_path(uploaded_file)

    out_dir = tempfile.mkdtemp(prefix="ranker_")
    out_path = os.path.join(out_dir, "ranked_candidates.csv")

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, RANK_SCRIPT,
             "--candidates", input_path,
             "--out", out_path,
             "--top-k", "100"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=HERE,
        )
    except subprocess.TimeoutExpired:
        return pd.DataFrame(), "Timed out after 10 minutes.", None

    elapsed = time.time() - t0

    if result.returncode != 0:
        err = result.stderr[-2000:] if result.stderr else "(no stderr)"
        return pd.DataFrame(), f"rank.py failed:\n\n{err}", None

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


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Candidate Ranker") as demo:
    gr.Markdown(
        "# 🏆 Candidate Ranker\n"
        "Upload a candidates file (`.jsonl`, `.jsonl.gz`, or `.zip`) and run "
        "the ranking pipeline against it. Leave the upload empty to run on the "
        "built-in sample. The full 100 k-candidate pool takes ~1 minute."
    )

    with gr.Row():
        file_in = gr.File(
            label="Candidates file  (.jsonl / .jsonl.gz / .zip)",
            file_types=[".jsonl", ".gz", ".zip"],
            # No type="filepath" - avoid gradio_client JSON schema bugs in 4.x
            scale=3,
        )
        run_btn = gr.Button("▶  Run ranker", variant="primary", scale=1)

    summary_out = gr.Textbox(label="Run summary", lines=7, interactive=False)
    table_out   = gr.Dataframe(label="Ranked candidates (top 100)", wrap=True)
    download_out = gr.File(label="⬇  Download full CSV")

    run_btn.click(
        fn=run_ranker,
        inputs=file_in,
        outputs=[table_out, summary_out, download_out],
        api_name=False,
    )
    demo.load(
        fn=run_ranker,
        inputs=None,
        outputs=[table_out, summary_out, download_out],
        api_name=False,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)