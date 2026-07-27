# Linel.uk SaaS — Roadmap (external plan, saved for reference)

> **Provenance:** this document was produced externally (ChatGPT) and is saved verbatim
> below as the working roadmap. Direction may change. Corrections and measured findings
> from this project are recorded in `docs/saas-hosting-decision.md` — **read that
> alongside this**, because at least one factual premise below (the licensing section)
> does not match what this project actually runs.

---

Yes — **this now looks like a plausible low-cost SaaS**, provided the CPU benchmark works.

The decisive metric is no longer model size. It is:

> **How many CPU-seconds and how much peak process memory does one useful animation consume?**

A 300–500 MB checkpoint can still create a 2–3 GB Python process or take several minutes per generation. The original Kimodo checkpoint is approximately 1.13 GB in F32 form, even before its separate text machinery is considered, so we should measure your extracted service rather than infer capacity from the weight file.

## The product architecture

Your revised system has a strong shape:

**User instruction → improved external LLM interpretation → owned motion service → Skelator anatomical skeleton → browser preview/export**

Moving the language interpretation to API calls is particularly useful here. The hosting server does not need to run a local LLM; it only needs to run your smaller motion-generation component and browser service.

I would deploy it as:

```text
Browser
  ↓ HTTPS
Nginx
  ↓
Web application
  ├── Google login
  ├── account and credit management
  ├── browser-based 3D interface
  └── job submission
         ↓
    Private job queue
         ↓
    One CPU inference worker
         ↓
    Output storage
```

The important detail is that generation should be a **queued job**, not a normal synchronous website request. The browser submits the animation, receives a job ID and polls or uses a WebSocket for progress.

That lets a modest VPS run one generation at a time without taking the rest of your websites down with it.

## Google authentication

Use the current **Google Identity Services** integration, not the older deprecated Google Sign-In library.

Google provides the browser identity token, but your backend must verify that token and then create its own secure application session. Google specifically recommends server-side verification of the signature, audience, issuer and expiry; accounts should be keyed by Google's stable `sub` identifier rather than the email address.

The flow should be:

1. User clicks **Sign in with Google**.
2. Google returns an ID token.
3. Your backend verifies it.
4. Your backend creates or finds the user.
5. Your application issues an HTTP-only secure session cookie.
6. Every generation is associated with that internal user ID.

Google authentication only establishes **identity**. Your database still controls:

* whether the account is active;
* whether it has paid;
* how many credits remain;
* which jobs belong to it;
* whether it has been suspended;
* whether you have manually granted free credits.

Do not expose the raw model endpoint publicly. Only the authenticated application backend should be able to call it.

## The £1 plan

The idea works, but **£1 collected monthly is the wrong payment structure**.

Stripe currently charges 1.5% plus 20p for a standard UK card, while Stripe Billing adds another 0.7% of subscription volume. A £1 payment therefore leaves approximately **77.8p before hosting, API calls, tax, failed payments or support**.

Use this instead:

### Founder plan

**£1 per month, billed as £12 annually**

The advertised price remains £1/month, but one £12 annual transaction leaves approximately £11.54 before other costs.

That is dramatically healthier than processing twelve £1 payments.

The plan must not be unlimited. It should include:

* a small monthly generation allowance;
* one generation running at a time;
* maximum animation duration;
* maximum samples per job;
* slower queue priority for free accounts;
* additional credit packs later.

The exact allowance should come from the server benchmark. Until we know that one generation costs, say, 10 seconds or 180 seconds of CPU time, any credit number is fiction.

Your free-credit approach is good. Give users a small initial allowance, then let them request more with a brief explanation of what they are building. That does three useful things simultaneously:

* prevents casual abuse;
* gives serious developers room to experiment;
* produces actual use cases and potential customer conversations.

## External LLM costs

This is the second figure that must be measured.

At £1/month, the language-improvement API should ideally:

* make one compact call per generation;
* return strict structured JSON;
* use a low-cost model for ordinary prompts;
* cache identical or near-identical interpretations;
* avoid sending skeleton documentation repeatedly;
* use a more capable model only when requested.

Every generation record should store:

```text
user
job
input prompt
interpreted motion instructions
LLM provider/model
input and output tokens
LLM cost
inference duration
output size
status
```

That gives you actual product economics rather than vibes wearing a calculator hat.

## GitHub fame without giving away the product

An **open-core structure** fits this very well.

Publish:

* browser interface;
* API schema;
* client SDK;
* Docker deployment;
* Google-auth integration;
* model-adapter interface;
* example lightweight backend;
* Blender and Unity export examples;
* screenshots and demonstration animations.

Keep private:

* your production weights;
* proprietary training data;
* the strongest prompt-to-motion interpretation logic;
* Skelator production assets;
* production abuse controls;
* hosted-service operational code where appropriate.

That gives developers something genuinely useful to star, fork and run while the hosted version offers the thing most people actually want: **no installation, no GPU wrestling and immediate results**.

The public repository can support alternative motion engines through adapters. Your hosted service remains the best implementation.

## Ownership wording

> ⚠️ **See `docs/saas-hosting-decision.md` — this section's premise is factually wrong
> for this project. We do not use any NVIDIA model.**

The NVIDIA licence is permissive: it allows commercial use and derivative models, and says that you own your derivative subject to NVIDIA's underlying ownership rights. NVIDIA does not claim ownership of generated outputs.

Therefore, the safest accurate wording is:

> **Our proprietary motion model, anatomical system and hosted service build upon research and open components including NVIDIA Kimodo.**

Avoid saying you "own Kimodo". You own:

* Skelator;
* your original code;
* your training data;
* your independently trained components;
* your modifications and derivative model rights;
* the service, workflow and product.

If your new model was trained from scratch without NVIDIA weights, the ownership position becomes stronger. Keep a written provenance record now: starting checkpoint, datasets, code sources, licences and exactly what was retrained.

## The server test

Do not install billing or public authentication first.

Deploy the inference process privately on the hosting VPS, bound only to localhost, and run a fixed benchmark pack:

| Test                         | What matters                                     |
| ---------------------------- | ------------------------------------------------ |
| Cold start                   | Time and peak RAM while loading                  |
| Typical animation            | Median generation time                           |
| Long animation               | Worst normal workload                            |
| Twenty consecutive jobs      | Memory leaks and stability                       |
| Two simultaneous submissions | Correct queuing, no accidental parallel overload |
| Existing website load        | Whether other hosted services remain responsive  |
| Failed generation            | Worker recovery and cleanup                      |
| Output lifecycle             | Storage consumption and deletion                 |

A practical interpretation:

* **Under 30 seconds per ordinary generation:** excellent shared SaaS candidate.
* **30–120 seconds:** completely usable with a queue.
* **Two–five minutes:** viable for specialist users, but credits need tighter limits.
* **Over five minutes:** probably unsuitable for £1 hosting on the existing VPS without optimisation or separate compute.

Peak memory should leave enough headroom for Nginx, databases and every existing hosted service. I would not let the worker consume more than roughly half the server's usable RAM during the first public beta.

## My assessment

This is no longer merely a hosted demo. It has the ingredients of a genuine product:

* a clear specialist function;
* a browser-native interface;
* low apparent compute requirements;
* differentiated language interpretation;
* an owned anatomical asset;
* exports useful to Blender, Unity and game development;
* an open-source story capable of attracting attention;
* a hosted version cheap enough to become an impulse purchase.

The correct immediate move is to create a **production benchmark build** of the current local service and deploy it privately to the VPS. Once it survives twenty realistic jobs without harming the existing sites, Google authentication and the £12 annual founder plan become straightforward additions.

### Source links cited in the original document

- `https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1`
- `https://developers.google.com/identity/gsi/web/guides/verify-google-id-token`
- `https://stripe.com/gb/billing/pricing`
- `https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1/blob/main/LICENSE`
