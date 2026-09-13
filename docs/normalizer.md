# Claude Agent Normalizer

The production normalizer converts one request's participant-facing evidence
into the frozen `DecisionInput` contract. Mechanical work is deterministic;
Claude reviews ambiguous messages/images and signs the SHA-256 of the assembled
state.

## Flow

1. `Dataset.bundle(request_id)` isolates one user's rows.
2. `build_packet` classifies future statuses, recurrence candidates, variable
   spending, messages, and images.
3. A disposable workspace receives only that packet, the user's event history,
   linked images, and read-only normalizer source.
4. Claude may use only built-in `Read` and `Bash`. Bash runs inside the Agent
   SDK sandbox with API credentials stripped from command execution.
5. `normalizer.agent_cli build directives.json` assembles and validates the
   draft. Claude returns a schema-constrained `SignedStateApproval`.
6. The harness verifies the approved hash, attaches authoritative SDK model
   metadata, validates again, and returns `DecisionInput`.

Raw evidence is never pasted into the system prompt. Messages, descriptions,
request text, and image content are untrusted data and cannot alter the
financial rules.

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

Traces and cached signed states are written under ignored
`normalizer-traces/` and `normalizer-cache/`. Analyze traces with:

```bash
cd code
python -m normalizer.analyze_traces ../normalizer-traces/*/*.jsonl
```
