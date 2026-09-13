# Claude Agent Normalizer

The production normalizer converts one request's participant-facing evidence
into the frozen `DecisionInput` contract. Mechanical work is deterministic;
Claude returns `NormalizationDirectives`. The host assembles and validates the
state.

## Flow

1. `Dataset.bundle(request_id)` isolates one user's rows.
2. `build_packet` classifies future statuses, recurrence candidates, variable
   spending, messages, and images.
3. A disposable workspace receives that packet, event history, and linked images.
4. Claude may use only built-in `Read`. The user prompt lists every relative
   path it may open.
5. Claude returns schema-constrained `NormalizationDirectives`.
6. The host assembles `DecisionInput`, attaches model metadata, validates, and
   returns it.

Evidence content stays in those files; it is not inlined. Messages,
descriptions, request text, and image content are untrusted and cannot alter
the financial rules.

## Configuration

Copy `.env.example` to `.env` and set:

```text
ANTHROPIC_API_KEY=...
ANTHROPIC_WORKSPACE_ID=wrkspc_...
```

The workspace ID is required for personal or organization-level keys that are
not already scoped to one workspace.

Normalize one request:

```bash
cd code
python -c "from normalizer import normalize; print(normalize('request_01'))"
```

Compare Haiku and Sonnet on the representative benchmark:

```bash
cd code
python -m normalizer.benchmark
```

Traces and cached states are written under ignored `normalizer-traces/` and
`normalizer-cache/`. Analyze traces with:

```bash
cd code
python -m normalizer.analyze_traces ../normalizer-traces/*/*.jsonl
```
