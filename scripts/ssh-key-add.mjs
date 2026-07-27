// Append an SSH public key to the RunPod account's authorised keys.
//
// RunPod stores every authorised key in ONE account-wide field, newline-separated, and
// the mutation that writes it replaces the whole thing. Editing that field by hand — or
// naively setting it — silently removes everyone else's access. This script reads the
// current value, appends, and writes back, refusing to act if the key is already present.
//
// Usage:
//   node scripts/ssh-key-add.mjs ~/path/to/their_key.pub
//   node scripts/ssh-key-add.mjs --list
//   node scripts/ssh-key-add.mjs --remove "comment-substring"

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

const describe = (line) => {
  const [type, blob, ...rest] = line.trim().split(/\s+/);
  return `${type} ...${(blob ?? '').slice(-12)} ${rest.join(' ') || '(no comment)'}`;
};

const { myself } = await gql('query { myself { pubKey } }');
const existing = (myself.pubKey ?? '').split('\n').map((l) => l.trim()).filter(Boolean);

const argv = process.argv.slice(2);

if (argv[0] === '--list' || argv.length === 0) {
  console.log(`${existing.length} key(s) registered on the account:`);
  existing.forEach((k, i) => console.log(`  [${i}] ${describe(k)}`));
  if (argv.length === 0) console.log('\nUsage: node scripts/ssh-key-add.mjs <path-to-.pub>');
  process.exit(0);
}

if (argv[0] === '--remove') {
  const needle = argv[1];
  if (!needle) throw new Error('--remove needs a substring matching the key to drop');
  const keep = existing.filter((k) => !k.includes(needle));
  if (keep.length === existing.length) {
    console.error(`No registered key matches "${needle}".`);
    process.exit(1);
  }
  if (keep.length === 0) {
    console.error('Refusing to remove the last key — that would lock everyone out of every new pod.');
    process.exit(1);
  }
  existing.filter((k) => k.includes(needle)).forEach((k) => console.log(`removing: ${describe(k)}`));
  await gql(
    `mutation { updateUserSettings(input:{pubKey:${JSON.stringify(keep.join('\n'))}}) { id } }`
  );
  console.log(`\nDone — ${keep.length} key(s) remain.`);
  console.log('Note: running pods keep the keys they were created with. Recreate to apply.');
  process.exit(0);
}

const incoming = readFileSync(argv[0], 'utf8').trim();
if (!/^(ssh-ed25519|ssh-rsa|ecdsa-)/.test(incoming)) {
  throw new Error(`${argv[0]} does not look like an SSH public key. Did you pass the private key by mistake?`);
}
if (incoming.split('\n').length > 1) {
  throw new Error('Expected a single public key in that file.');
}

const blob = incoming.split(/\s+/)[1];
if (existing.some((k) => k.split(/\s+/)[1] === blob)) {
  console.log('That key is already registered — nothing to do.');
  console.log(`  ${describe(incoming)}`);
  process.exit(0);
}

const merged = [...existing, incoming].join('\n');
await gql(`mutation { updateUserSettings(input:{pubKey:${JSON.stringify(merged)}}) { id } }`);

console.log(`Added: ${describe(incoming)}`);
console.log(`Account now has ${existing.length + 1} key(s):`);
[...existing, incoming].forEach((k, i) => console.log(`  [${i}] ${describe(k)}`));
console.log(
  '\nKeys are injected when a pod is CREATED. Any pod already running will not accept\n' +
  'this key — append it to /root/.ssh/authorized_keys there, or create a new pod.'
);
