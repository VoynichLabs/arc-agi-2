# Confluence Labs ARC-AGI-2 Solver — Architecture Analysis

**Authors:** Larry (Claude Sonnet 4.6), Egon (EgonBot)  
**Date:** 24 February 2026  
**Source repo:** https://github.com/confluence-labs/arc-agi-2  
**Our fork:** https://github.com/VoynichLabs/arc-agi-2  
**Purpose:** Understand the architecture deeply enough to adapt the best ideas for VoynichLabs' own agent orchestration work, without the cloud compute budget they used.

---

## Result Context

Confluence Labs achieved **97.9% on ARC-AGI-2 public eval** at **$11.77/task average**. As recently as a year ago, best models scored in the single digits. This is a significant result.

Full run config that produced it:
- 12 agents per test input
- 132 simultaneous E2B sandboxes
- Gemini 3.1 Pro as the model
- Max 10 iterations per agent
- 12-hour wall-clock limit

---

## System Overview

Three Python files do all the work. They have clean separation of concerns:

```
run.sh                  → top-level coordinator (env, watchdog, SIGTERM handling)
  orchestrator.py       → dispatches agents to E2B sandboxes, aggregates results
    agent_runner.py     → runs inside each sandbox; the actual LLM loop
  submission.py         → majority vote, scoring, cost breakdown, security scan
```

---

## Layer 1: run.sh — The Coordinator

A bash script wrapping the whole pipeline. Interesting patterns:

**Deadline watchdog:**  
A background subprocess sleeps for `(WALL_CLOCK_LIMIT - DEADLINE_BUFFER)` seconds, then kills the solver. This guarantees `submission.py` always runs even if the solver is still churning — you get partial results instead of nothing.

```bash
SLEEP_SECS=$(( WALL_CLOCK_LIMIT - DEADLINE_BUFFER ))
(sleep "$SLEEP_SECS"; kill "$PID_SOLVER") &
PID_WATCHDOG=$!
```

**SIGTERM trap:**  
If the outer process is killed (e.g., Kaggle infrastructure kill), the trap fires, kills the solver gracefully, waits 5 seconds, force-kills if needed, then runs `submission.py` on whatever partial results exist.

**Key takeaway:** Never trust that a long-running job will finish cleanly. Always have a SIGTERM handler that produces output from partial results.

---

## Layer 2: orchestrator.py — The Dispatcher

Manages N async tasks × M agents each, with concurrency control and incremental writes.

### Concurrency via asyncio Queue

```python
backend_queue = asyncio.Queue()
for _ in range(args.concurrency):
    backend_queue.put_nowait("token")

# Each agent acquires a token, runs, then releases it:
token = await backend_queue.get()
try:
    result = await run_agent(...)
finally:
    backend_queue.put_nowait(token)
```

This is a clean, provider-agnostic pattern. The queue acts as a semaphore — N tokens in the queue = N concurrent agents max. Works for any async I/O, not just E2B.

### Incremental Atomic Writes

After each agent finishes, results are written immediately using tmp-then-rename:

```python
tmp_file.write_text(json.dumps(data, indent=2))
os.rename(str(tmp_file), str(task_file))
```

This guarantees:
- No partial JSON on disk (rename is atomic on POSIX)
- If the process dies mid-run, all completed agent results survive
- Resume logic can pick up from where it left off

### Resume Logic

On startup, the orchestrator scans `task_results/*.json` and skips any task that already has a file. Reruns are safe and cheap — only incomplete tasks pay compute costs.

### Retry with Exponential Backoff + Jitter

10 retries, catches a broad list of transient errors (429s, timeouts, connection resets, 502/503, rate limits):

```python
backoff = min(INITIAL_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
jitter = random.uniform(0, backoff * 0.5)
wait = backoff + jitter
```

Jitter prevents thundering herd when many agents hit rate limits simultaneously.

### E2B Sandbox — What It Actually Costs

Each agent runs in a Firecracker microVM (2 vCPUs, 12h timeout, network restricted to `generativelanguage.googleapis.com` only via SNI filtering):

```
E2B cost = (duration_seconds / 3600) * 2 vCPUs * $0.05/vCPU-hour
```

A 12-hour full run per agent = $1.20 in E2B infrastructure alone, before any Gemini API costs. With 12 agents × 400 tasks × multiple test inputs, the E2B costs add up fast. This is the primary reason we cannot replicate their setup directly.

The network isolation is smart but overkill for our use case — it prevents the agent from exfiltrating its API key. We can achieve the same with simpler subprocess sandboxing or just by not passing secrets into agent subprocess environments.

---

## Layer 3: agent_runner.py — The Core Loop

**This is where 97.9% actually comes from.**

The orchestrator is infrastructure. The agent_runner is the intelligence. Here's the full loop:

### Setup

A `/workspace` directory is created with three files:
- `task.json` — the ARC puzzle (training examples + one test input, no answer)
- `GEMINI.md` — instructions telling the agent what to do and how
- `.gemini/settings.json` — Gemini CLI config: 500 max turns, no loop detection, 6-hour agent timeout

The system prompt (`GEMINI.md`) is remarkably short:

> Write `transform.py` with a Python function `transform(grid: np.ndarray) -> np.ndarray`.  
> Test against ALL training pairs. Iterate until correct.

That's essentially it. No chain-of-thought scaffolding, no elaborate prompting. The agent figures out the rest.

### The Iteration Loop

```
for iteration in range(max_iterations):  # default: 10

    if iteration == 0:
        gemini -p "Read GEMINI.md, then solve the ARC puzzle in task.json."
    else:
        gemini --resume latest   # continues the same conversation
        # feedback injected via stdin

    if transform.py doesn't exist:
        feedback = "You haven't written transform.py yet..."
        continue

    # Validate transform against ALL training examples (not just one)
    all_pass, feedback, fn = test_transform(transform_path, raw_task["train"])

    if not all_pass:
        feedback = "Your transform function doesn't pass the training examples. Try a fundamentally different approach."
        continue

    # Run transform on test input, record answer
    grid = fn(test_input).tolist()
    attempts.append(grid)
    break  # done!
```

### Why This Works

1. **Precise executable feedback loop.** The agent doesn't just get "you're wrong." It gets exact pass/fail against the actual training data. No ambiguity.

2. **The agent sees its own failures.** By running `transform.py` and injecting the result back via `--resume`, the conversation continues with full context. The agent knows what it tried and why it failed.

3. **Program synthesis over pattern matching.** Writing a `transform()` function forces the agent to produce a generalizable rule, not just a grid that looks right. The function is then mechanically verified.

4. **Fallback grid extraction.** If the transform loop produces nothing (e.g., transform.py was never written or always failed), the system scans the raw Gemini CLI output for any valid 2D integer grid in tool_result events. The agent's stdout often contains the grid printed by Python scripts it ran.

5. **Config file is deleted before agent starts.** Prevents the LLM from reading orchestrator configuration (API keys, internal metadata) even if it tries.

### What the Agent Actually Runs

Gemini CLI is a coding agent — it has file read/write/edit tools and shell execution. The typical agent session looks like:

1. Read `task.json`, visualize the training examples
2. Hypothesize a transformation rule
3. Write `transform.py` implementing the rule
4. Execute a test script: `python3 test_transform.py`
5. See the output (pass/fail per training pair)
6. If fail → iterate, refine hypothesis, rewrite transform.py
7. If pass → exit (orchestrator validates and records the answer)

The agent has 200 turns max and 6 hours max per session. In practice, most tasks resolve in 2-5 iterations.

---

## Layer 4: submission.py — Voting and Scoring

### Majority Voting

Each task has N agents, each producing one answer grid. All grids are pooled and ranked by frequency:

```python
counts = Counter(canonicalize_grid(g) for g in grids)
top_2 = [lookup[key] for key, _ in counts.most_common(2)]
# attempt_1 = most common, attempt_2 = second most common
```

The ARC-AGI scoring format allows 2 attempts per test. With 12 agents, you often get 8+ votes for the correct answer and 2-4 votes for variants. Majority voting handles noise from imperfect transforms.

### Security Scan

`submission.py` scans all transcript logs for suspicious patterns:
- API key env var names (`GEMINI_API_KEY`, `GOOGLE_API_KEY`)
- Env inspection commands (`printenv`, `os.environ`, `/proc/self/environ`)

They're actively checking whether their LLM tried to steal its own API key. This is a real concern with code-executing agents given API keys.

---

## Summary: Key Patterns Worth Adopting

### Immediately Applicable (no E2B required)

| Pattern | Where | Benefit |
|---------|-------|---------|
| asyncio Queue semaphore | orchestrator.py | Provider-agnostic concurrency limiting |
| tmp-then-rename atomic writes | orchestrator.py | Crash-safe incremental results |
| Resume from existing results | orchestrator.py | Reruns only pay for incomplete work |
| Retry with exponential backoff + jitter | orchestrator.py | Handles rate limits gracefully |
| Deadline watchdog subprocess | run.sh | Always produce output even on timeout |
| SIGTERM trap → partial submission | run.sh | Infrastructure kill doesn't lose results |
| Delete config before agent starts | agent_runner.py | Prevents LLM reading secrets |
| Executable feedback loop | agent_runner.py | Precise, verifiable iteration |
| Fallback grid extraction from stdout | agent_runner.py | Resilient answer recovery |
| Transcript security scan | submission.py | Detect agent key exfiltration attempts |
| Majority vote across agent pool | submission.py | Noise reduction without ground truth |

### Requires Adaptation

| Pattern | Their approach | Our approach |
|---------|---------------|--------------|
| Sandbox isolation | E2B Firecracker ($$$) | subprocess + no-secret env, or Docker |
| Provider | Gemini only, hardcoded | OpenRouter, model-agnostic |
| Concurrency scale | 132 simultaneous agents | 4-8 (budget constraint) |
| Ensemble size | 12 agents/task | 2-3 agents/task |

---

## Gaps and Blind Spots

**Agents never talk to each other.** Every agent starts from zero, with only the task prompt and GEMINI.md. There is no mechanism for agents to share discovered partial solutions, useful subproblems, or failed hypotheses. They run in perfect isolation. With 12 agents this is fine — you cover the hypothesis space by brute force. With 2-3 agents this is a significant limitation.

**Hypothesis: inter-agent communication is our potential edge.** If agents can share "I tried X and it failed for training pair 3" before another agent starts, the second agent can skip that branch. This is more like how human teams solve hard problems — you don't repeat each other's failed experiments.

**No adaptive resource allocation.** All tasks get the same 12 agents and 10 iterations, whether the task is trivially easy or genuinely hard. A smarter system would detect easy tasks early, stop after 2 agents, and redirect budget to harder tasks. The current system has no signal for task difficulty before running agents.

**Gemini CLI dependency.** The entire agent architecture is built around Gemini CLI as the execution layer. Swapping to a different LLM requires replacing the CLI, the `--resume` mechanism, the JSONL stream format, and the tool name mappings. It's more locked in than it appears.

**Cost at scale.** $11.77/task × 400 tasks = ~$4,700 per full evaluation run. This is accessible for a funded lab but not for us.

---

## Ideas for VoynichLabs

1. **Build a provider-agnostic orchestrator** using the asyncio Queue pattern, targeting OpenRouter. The orchestrator itself is clean and reusable — it's just E2B and Gemini that are expensive.

2. **Implement agent communication.** After each iteration, the orchestrator could write a shared `failed_approaches.md` that all agents in the ensemble can read. "Don't try recoloring — tried it, fails on pair 2."

3. **Adaptive agent budget.** Use a cheap first pass (1 agent, 3 iterations). If it solves the task, done. If not, escalate to more agents/iterations. Only hard tasks pay full price.

4. **LODA as prior knowledge.** The agent currently has no domain knowledge beyond what's in GEMINI.md. LODA's 150,420 programs are a corpus of transformation primitives. Could be injected as a retrieval tool: "Here are 10 LODA programs that produce similar output patterns — consider these as building blocks."

5. **Simpler sandbox.** For non-secret environments, subprocess isolation is enough. No E2B required. The key E2B feature we'd need to replicate is: agent subprocess cannot read parent environment variables (no `os.environ` leakage). Python `subprocess.Popen(env={...})` with a clean minimal env solves this for free.

---

*Last updated: 24 February 2026*
