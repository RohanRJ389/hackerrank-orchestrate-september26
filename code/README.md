# Buy or Wait? solution

From the repository root:

```bash
python3 -m pip install -r code/requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY and ANTHROPIC_WORKSPACE_ID
python3 code/main.py
```

This reads `dataset/requests.csv`, writes repo-root `output.csv`, and writes
`code/evaluation/usage_report.md` for the run.

Optional environment variables: `NORMALIZER_CONCURRENCY` (default 5),
`NORMALIZER_REQUEST_TIMEOUT_S` (default 90), `NORMALIZER_MODEL`.
