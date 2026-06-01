# Continuous Batching Plan

Target behavior:

- merge independent decode streams into a single GPU pass
- maintain per-request state machines
- maximize throughput without starving short requests

First benchmark:

- serial requests: `A`, `B`, `C`, `D`
- merged decode step across active requests
