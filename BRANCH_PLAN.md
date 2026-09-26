# Proposed branch

`docs/community-launch`

## Goal

Turn the repository from an implementation-first project into a public experiment that is easy to understand, reproduce, challenge, and benchmark without overstating results that have not yet been measured.

## Files in this branch

- `README.md` — new public-facing project positioning and honest project status
- `README.zh.md` — Chinese mirror
- `LICENSE` — MIT
- `CONTRIBUTING.md` — contribution and architecture rules
- `docs/ARCHITECTURE.md` — moves the detailed loop design out of the README front page
- `docs/BENCHMARK_PLAN.md` — preregisters the intended JEV-vs-LLM comparison before results exist
- `.github/ISSUE_TEMPLATE/task_failure.md` — invites people to break the agent
- `.github/ISSUE_TEMPLATE/benchmark_result.md` — standardizes community benchmark reports
- `.github/PULL_REQUEST_TEMPLATE.md` — protects the sensor-grounded architecture boundary

## Deliberately not included yet

- demo GIF/video placeholders that do not exist,
- fabricated latency/cost/success numbers,
- SOTA claims,
- physical-robot claims,
- code changes to the current control loop.
