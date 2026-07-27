// Compare GPU stock + on-demand price across RunPod datacenters.
// Network volumes are datacenter-locked, so this decides which DC we commit to.
// Usage: node scripts/gpu-availability.mjs [DC ...]

import { readFileSync } from 'node:fs';

const env = Object.fromEntries(
  readFileSync(new URL('../.env', import.meta.url), 'utf8')
    .split('\n')
    .filter((l) => l.trim() && !l.trimStart().startsWith('#'))
    .map((l) => {
      const i = l.indexOf('=');
      return [l.slice(0, i).trim(), l.slice(i + 1).trim().replace(/^["']|["']$/g, '')];
    })
);

const KEY = env.RUNPOD_API_KEY;
if (!KEY) throw new Error('RUNPOD_API_KEY missing from .env');

const DCS = process.argv.slice(2).length ? process.argv.slice(2) : ['EUR-IS-1', 'EU-RO-1'];

async function fetchDc(dc) {
  const query = `query { gpuTypes {
    id displayName memoryInGb
    lowestPrice(input:{gpuCount:1, dataCenterId:"${dc}"}) { uninterruptablePrice minimumBidPrice stockStatus }
  } }`;
  const res = await fetch(`https://api.runpod.io/graphql?api_key=${KEY}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  const json = await res.json();
  if (json.errors) throw new Error(JSON.stringify(json.errors));
  return json.data.gpuTypes;
}

const byDc = {};
for (const dc of DCS) byDc[dc] = await fetchDc(dc);

// Union of GPUs that have stock anywhere, sorted by VRAM then price.
const gpus = new Map();
for (const dc of DCS) {
  for (const g of byDc[dc]) {
    if (!g.lowestPrice?.stockStatus) continue;
    if (!gpus.has(g.id)) gpus.set(g.id, { name: g.displayName, vram: g.memoryInGb, dc: {} });
    gpus.get(g.id).dc[dc] = g.lowestPrice;
  }
}

const rows = [...gpus.entries()]
  .map(([id, g]) => ({ id, ...g }))
  .sort((a, b) => a.vram - b.vram || a.name.localeCompare(b.name));

const cell = (p) => (p ? `${p.stockStatus.padEnd(6)} $${String(p.uninterruptablePrice ?? '-').padEnd(5)}` : '-'.padEnd(13));
const w = Math.max(...rows.map((r) => r.name.length), 14);

console.log('GPU'.padEnd(w) + ' VRAM  ' + DCS.map((d) => d.padEnd(13)).join(' '));
console.log('-'.repeat(w + 7 + DCS.length * 14));
for (const r of rows) {
  console.log(
    r.name.padEnd(w) + String(r.vram).padStart(4) + 'GB ' + DCS.map((d) => cell(r.dc[d])).join(' ')
  );
}
console.log('\nStock: High > Medium > Low. Price is on-demand $/hr for 1 GPU, secure cloud.');
console.log(`GPUs with stock in at least one DC: ${rows.length}`);
for (const dc of DCS) {
  console.log(`  ${dc}: ${rows.filter((r) => r.dc[dc]).length} available`);
}
