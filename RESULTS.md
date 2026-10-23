# SuperTaco sweep results

**Date:** 2026-10-23 - **judge backend:** nebius_job - **runs recorded:** 10 - **successful:** 10

**How to verify:** every row cites its `run_id` - `grep -c '"run_id": "<run_id>"' logs/llm_calls.jsonl` counts that run's audit lines (plan-phase calls share the id, so the count is >= the artifact's `llm_calls` length and > 0), and `logs/runs/<run_id>.json` holds its full artifact (schema v1).

| run_id | goal | model | attempts | healed | base avg | ft avg | delta | judge | tokens | wall-clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `sw6ca8` | Answer like a pirate chef: warm, nautical, precise about cooking times and temperatures | Qwen/Qwen2.5-1.5B-Instruct | 1 | no | 1.60 | 2.20 | +0.60 | nebius_job · base:real / ft:real ... | judge:3.9k | 662s |
| `swd6ac` | Act as a UK-law study assistant: explain legal principles in plain English for non-lawyers, with a reminder that this is not formal legal advice | Qwen/Qwen2.5-1.5B-Instruct | 1 | no | 3.40 | 3.80 | +0.40 | nebius_job · base:real / ft:real ... | judge:4.7k | 684s |
| `swba18` | SQL tutor: show the query first, then explain each clause step by step | Qwen/Qwen2.5-1.5B-Instruct | 1 | no | 4.40 | 9.20 | +4.80 | nebius_job · base:real / ft:real ... | judge:3.7k | 635s |
| `sw9b1c` | Concise technical writer: short, direct answers with no filler and no restating the question | Qwen/Qwen2.5-0.5B-Instruct | 1 | no | 5.20 | 8.00 | +2.80 | nebius_job · base:real / ft:real ... | judge:2.1k | 432s |
| `sw61bf` | Therapist tone: warm, reflective, validates feelings, never clinical or diagnostic | Qwen/Qwen2.5-1.5B-Instruct | 1 | no | 5.40 | 7.40 | +2.00 | nebius_job · base:real / ft:real ... | judge:3.0k | 570s |
| `swbcd5` | JSON-only extractor: return exactly one valid JSON object for each request, no prose before or after | Qwen/Qwen2.5-0.5B-Instruct | 1 | no | 4.60 | 8.40 | +3.80 | nebius_job · base:real / ft:real ... | judge:2.9k | 486s |
| `sw0443` | Friendly travel agent: organized day-by-day itineraries; train on the Hugging Face dataset yahma/alpaca-cleaned | Qwen/Qwen2.5-1.5B-Instruct | 2 | yes | 2.00 | 2.40 | +0.40 | nebius_job · base:real / ft:real ... | classify:82.6k · judge:7.6k · patch:828 · tavily:0 | 1158s |
| `swe7a2` | Practical cooking-substitution helper: common swaps and what changes in the result; train on the Hugging Face dataset tatsu-lab/alpaca | Qwen/Qwen2.5-1.5B-Instruct | 1 | no | 4.60 | 7.20 | +2.60 | nebius_job · base:real / ft:real ... | judge:3.6k | 495s |
| `sw3942` | Motivational coach for first-time marathon trainees: encouraging, evidence-aware, week-by-week advice | Qwen/Qwen2.5-3B-Instruct | 1 | no | 3.20 | 7.20 | +4.00 | nebius_job · base:real / ft:real ... | judge:3.6k | 723s |
| `swb986` | Kids' science explainer: simple analogies, curious tone, never talks down | Qwen/Qwen2.5-1.5B-Instruct | 1 | no | 4.20 | 4.20 | +0.00 | nebius_job · base:real / ft:real ... | judge:3.6k | 522s |
| **totals** | 10 runs | | 11 | 10 ok | | | | | classify:82.6k · judge:38.7k · patch:828 · tavily:0 | 6367s |
