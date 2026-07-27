// Show current pods and their live SSH connection details.
// The direct-TCP IP and port change every time a pod is stopped and restarted,
// so re-run this after any restart and update the `kimodo` block in ~/.ssh/config.
// Usage: node scripts/pod-status.mjs

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

const rest = async (path) => {
  const r = await fetch(`https://rest.runpod.io/v1/${path}`, {
    headers: { Authorization: `Bearer ${KEY}` },
  });
  return r.json();
};

const gql = async (query) => {
  const r = await fetch(`https://api.runpod.io/graphql?api_key=${KEY}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  const j = await r.json();
  if (j.errors) throw new Error(JSON.stringify(j.errors));
  return j.data;
};

const vols = await rest('networkvolumes');
console.log('=== network volumes ===');
if (!vols.length) console.log('  none');
for (const v of vols) console.log(`  ${v.name}  ${v.id}  ${v.size}GB  ${v.dataCenterId}`);

const pods = await rest('pods');
console.log('\n=== pods ===');
if (!pods.length) {
  console.log('  none running — nothing billing except volume storage');
  process.exit(0);
}

let hourly = 0;
for (const p of pods) {
  hourly += p.costPerHr ?? 0;
  // REST returns machine:{} — only the create response populates it. GPU details and
  // port mappings both have to come from GraphQL.
  const d = await gql(
    `query { pod(input:{podId:"${p.id}"}) {
       machine { gpuDisplayName location }
       runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
     } }`
  );
  console.log(`\n  ${p.name}  (${p.id})`);
  console.log(`    status : ${p.desiredStatus}`);
  console.log(
    `    gpu    : ${d.pod?.machine?.gpuDisplayName ?? '?'} x${p.gpuCount}  @ ${d.pod?.machine?.location ?? '?'}`
  );
  console.log(`    cost   : $${p.costPerHr}/hr`);
  const rt = d.pod?.runtime;
  if (!rt) {
    console.log('    ssh    : not ready yet — pod still starting');
    continue;
  }
  console.log(`    uptime : ${Math.floor((rt.uptimeInSeconds ?? 0) / 60)} min`);
  const ssh = (rt.ports || []).find((x) => x.privatePort === 22 && x.isIpPublic);
  if (ssh) {
    console.log(`    ssh    : ssh -i ~/.ssh/id_ed25519_runpod -p ${ssh.publicPort} root@${ssh.ip}`);
    console.log(`    config : update Host kimodo -> HostName ${ssh.ip} / Port ${ssh.publicPort}`);
  }
  const http = (rt.ports || []).filter((x) => x.type === 'http');
  for (const h of http) {
    console.log(`    http   : port ${h.privatePort} -> https://${p.id}-${h.privatePort}.proxy.runpod.net`);
  }
}

console.log(`\ntotal compute: $${hourly.toFixed(2)}/hr  (~$${(hourly * 24).toFixed(2)}/day if left running)`);
console.log('Stop the pod when idle — GPU billing continues while it runs.');
