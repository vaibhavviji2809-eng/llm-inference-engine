# Profiling Dashboard

Implemented in a lightweight first-pass form:

- `GET /dashboard` serves a live HTML dashboard
- `GET /metrics/summary` exposes recent request, benchmark, and batch-run metrics

Current panels:

- request count
- benchmark count
- batch-run count
- average tokens/sec
- recent request table
- recent batching table

Next improvements:

- add benchmark charts
- add GPU metrics when running on a CUDA-enabled machine
- persist metrics over longer sessions
