// Boot the standard Kimodo dev pod: RTX 3090 in EU-CZ-1 with the project volume.
//
// The spec is encoded here rather than described in prose because two constraints are
// easy to get wrong and expensive to get wrong: the network volume is locked to EU-CZ-1
// (so the pod must run there), and the GPU must be one the datacenter actually stocks.
//
// Usage:
//   node scripts/pod-create.mjs                 # RTX 3090, the default
//   node scripts/pod-create.mjs "NVIDIA GeForce RTX 4090"
//   node scripts/pod-create.mjs --dry-run       # check stock without spending anything

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

const DATACENTER = 'EU-CZ-1';
const VOLUME_NAME = 'kimodo-motion-cz';
const IMAGE = 'runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204';
const CONTAINER_DISK_GB = 50;

const argv = process.argv.slice(2);
const dryRun = argv.includes('--dry-run');
const GPU = argv.find((a) => !a.startsWith('--')) ?? 'NVIDIA GeForce RTX 3090';

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

const rest = async (path, init = {}) => {
  const r = await fetch(`https://rest.runpod.io/v1/${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${KEY}`,
      'Content-Type': 'application/json',
      ...(init.headers ?? {}),
    },
  });
  const text = await r.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  if (!r.ok) throw new Error(`${path} -> ${r.status}: ${JSON.stringify(body).slice(0, 400)}`);
  return body;
};

// 1. Refuse to strand the pod away from its volume.
const volumes = await rest('networkvolumes');
const volume = volumes.find((v) => v.name === VOLUME_NAME);
if (!volume) {
  console.error(`No network volume named "${VOLUME_NAME}" on this account.`);
  console.error('Volumes found:', volumes.map((v) => `${v.name} (${v.dataCenterId})`).join(', ') || 'none');
  process.exit(1);
}
if (volume.dataCenterId !== DATACENTER) {
  console.error(`Volume "${VOLUME_NAME}" lives in ${volume.dataCenterId}, not ${DATACENTER}.`);
  process.exit(1);
}
console.log(`volume : ${volume.name} (${volume.id}) ${volume.size}GB @ ${volume.dataCenterId}`);

// 2. Confirm the GPU is actually stocked. secureCloud:true matters — without it the API
//    reports community pricing, which for some cards is a rate you can never rent at.
const { gpuTypes } = await gql(`query { gpuTypes {
  id displayName securePrice
  lowestPrice(input:{gpuCount:1, dataCenterId:"${DATACENTER}", secureCloud:true}) { stockStatus }
} }`);
const gpu = gpuTypes.find((g) => g.id === GPU);
if (!gpu) {
  console.error(`Unknown GPU id "${GPU}".`);
  console.error('In-stock ids for this datacenter:');
  gpuTypes
    .filter((g) => g.lowestPrice?.stockStatus)
    .forEach((g) => console.error(`  ${JSON.stringify(g.id)}  $${g.securePrice}/hr`));
  process.exit(1);
}
const stock = gpu.lowestPrice?.stockStatus;
console.log(`gpu    : ${gpu.displayName}  $${gpu.securePrice}/hr  stock=${stock ?? 'NONE'}`);
if (!stock) {
  console.error(`\n${gpu.displayName} is out of stock in ${DATACENTER}. Retry, or pass another GPU id.`);
  process.exit(1);
}

if (dryRun) {
  console.log('\n--dry-run: stock confirmed, nothing created.');
  process.exit(0);
}

// 3. Create it.
const pod = await rest('pods', {
  method: 'POST',
  body: JSON.stringify({
    name: 'kimodo-dev',
    imageName: IMAGE,
    gpuTypeIds: [GPU],
    gpuCount: 1,
    cloudType: 'SECURE',
    dataCenterIds: [DATACENTER],
    networkVolumeId: volume.id,
    containerDiskInGb: CONTAINER_DISK_GB,
    volumeMountPath: '/workspace',
    ports: ['22/tcp', '8888/http'],
    interruptible: false,
    supportPublicIp: true,
  }),
});

console.log(`\ncreated: ${pod.name} (${pod.id}) at $${pod.costPerHr}/hr — billing has started`);

// 4. Wait for SSH. Port mappings only appear on the GraphQL runtime object; REST
//    returns portMappings:null even once the pod is up.
process.stdout.write('waiting for ssh');
for (let i = 0; i < 40; i++) {
  const d = await gql(
    `query { pod(input:{podId:"${pod.id}"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }`
  );
  const ssh = (d.pod?.runtime?.ports ?? []).find((p) => p.privatePort === 22 && p.isIpPublic);
  if (ssh) {
    console.log(`\n\nssh -i ~/.ssh/<your-runpod-key> -p ${ssh.publicPort} root@${ssh.ip}`);
    console.log(`viewer (once the server is running): https://${pod.id}-8888.proxy.runpod.net`);
    console.log('\nUpdate the Host kimodo block in ~/.ssh/config with the address above.');
    console.log('Start the motion server with: ssh kimodo /workspace/run_server.sh');
    process.exit(0);
  }
  process.stdout.write('.');
  await new Promise((r) => setTimeout(r, 5000));
}
console.log('\n\nPod created but SSH did not come up in ~3 minutes. Check: node scripts/pod-status.mjs');
