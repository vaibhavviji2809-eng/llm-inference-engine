from __future__ import annotations

from fastapi.responses import HTMLResponse


def render_dashboard() -> HTMLResponse:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Inference Dashboard</title>
  <style>
    :root {
      --bg: #f4efe7;
      --panel: #fffdf8;
      --ink: #1d1a17;
      --muted: #6f665d;
      --accent: #be5a2c;
      --border: #e6dccf;
    }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(190,90,44,0.10), transparent 30%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    h1 {
      font-size: 2.6rem;
      margin: 0 0 8px;
      letter-spacing: -0.03em;
    }
    p.lead {
      color: var(--muted);
      max-width: 760px;
      margin: 0 0 28px;
      line-height: 1.5;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(58, 43, 29, 0.05);
    }
    .label {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .value {
      font-size: 2rem;
      font-weight: 700;
    }
    .row {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: 16px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }
    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 600;
    }
    code {
      background: rgba(190,90,44,0.08);
      padding: 2px 6px;
      border-radius: 999px;
    }
    @media (max-width: 900px) {
      .row { grid-template-columns: 1fr; }
      h1 { font-size: 2.1rem; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Inference Engine Dashboard</h1>
    <p class="lead">Live runtime summary for requests, batching, and benchmarks from the current CPU build.</p>
    <div class="grid">
      <div class="panel"><div class="label">Requests</div><div class="value" id="requestCount">-</div></div>
      <div class="panel"><div class="label">Benchmarks</div><div class="value" id="benchmarkCount">-</div></div>
      <div class="panel"><div class="label">Batch Runs</div><div class="value" id="batchRunCount">-</div></div>
      <div class="panel"><div class="label">Avg Tokens/Sec</div><div class="value" id="avgTps">-</div></div>
      <div class="panel"><div class="label">Avg Batch Size</div><div class="value" id="avgBatchSize">-</div></div>
      <div class="panel"><div class="label">Batch Speedup</div><div class="value" id="avgBatchSpeedup">-</div></div>
      <div class="panel"><div class="label">Cache Hit Rate</div><div class="value" id="avgCacheHitRate">-</div></div>
      <div class="panel"><div class="label">VRAM (MB)</div><div class="value" id="avgVram">-</div></div>
    </div>
    <div class="row">
      <div class="panel">
        <div class="label">Recent Requests</div>
        <table id="requestsTable">
          <thead><tr><th>Prompt</th><th>Latency</th><th>TPS</th><th>Cache</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="panel">
        <div class="label">Recent Batch Runs</div>
        <table id="batchTable">
          <thead><tr><th>Requests</th><th>Steps</th><th>TPS</th><th>Batch Size</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="panel">
        <div class="label">Recent Benchmarks</div>
        <table id="benchmarksTable">
          <thead><tr><th>Prompt</th><th>Speedup</th><th>TPS</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>
  <script>
    async function load() {
      const res = await fetch('/metrics/summary');
      const data = await res.json();
      document.getElementById('requestCount').textContent = data.request_count;
      document.getElementById('benchmarkCount').textContent = data.benchmark_count;
      document.getElementById('batchRunCount').textContent = data.batch_run_count;
      document.getElementById('avgTps').textContent = data.avg_tokens_per_second ? data.avg_tokens_per_second.toFixed(1) : '-';
      document.getElementById('avgBatchSize').textContent = data.avg_batch_size ? data.avg_batch_size.toFixed(2) : '-';
      document.getElementById('avgBatchSpeedup').textContent = data.avg_batch_speedup ? data.avg_batch_speedup.toFixed(2) + 'x' : '-';
      document.getElementById('avgCacheHitRate').textContent = data.avg_cache_hit_rate ? (data.avg_cache_hit_rate * 100).toFixed(1) + '%' : '-';
      document.getElementById('avgVram').textContent = data.avg_vram_allocated_mb ? data.avg_vram_allocated_mb.toFixed(1) : '-';

      const requestsBody = document.querySelector('#requestsTable tbody');
      requestsBody.innerHTML = '';
      for (const row of data.recent_requests.slice().reverse()) {
        requestsBody.insertAdjacentHTML('beforeend',
          `<tr><td><code>${(row.prompt || '').slice(0, 28)}</code></td><td>${row.latency_seconds?.toFixed?.(4) ?? '-'}</td><td>${row.tokens_per_second?.toFixed?.(1) ?? '-'}</td><td>${row.use_kv_cache ?? '-'}</td></tr>`
        );
      }

      const batchBody = document.querySelector('#batchTable tbody');
      batchBody.innerHTML = '';
      for (const row of data.recent_batch_runs.slice().reverse()) {
        batchBody.insertAdjacentHTML('beforeend',
          `<tr><td>${row.request_count}</td><td>${row.steps}</td><td>${row.tokens_per_second?.toFixed?.(1) ?? '-'}</td><td>${row.avg_batch_size?.toFixed?.(2) ?? '-'}</td></tr>`
        );
      }

      const benchmarksBody = document.querySelector('#benchmarksTable tbody');
      benchmarksBody.innerHTML = '';
      for (const row of data.recent_benchmarks.slice().reverse()) {
        const batched = row.results?.[1];
        benchmarksBody.insertAdjacentHTML('beforeend',
          `<tr><td><code>${(row.prompt || '').slice(0, 20)}</code></td><td>${batched?.speedup_vs_serial ? batched.speedup_vs_serial.toFixed(2) + 'x' : '-'}</td><td>${batched?.tokens_per_second?.toFixed?.(1) ?? '-'}</td></tr>`
        );
      }
    }
    load();
    setInterval(load, 3000);
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
