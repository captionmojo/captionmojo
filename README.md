# CaptionMojo

AI-powered social media caption generator. Trained on your brand once, generates on-brand captions forever.

## Stack

- **Frontend:** Single-file HTML + vanilla JS (`public/index.html`)
- **Backend:** Cloudflare Worker (`worker/index.js`)
- **AI:** Anthropic Claude API (Sonnet 4.6)
- **Hosting:** Cloudflare Workers with Static Assets
- **Storage (beta):** Browser localStorage (no accounts yet)

## Folder structure

```
captionmojo/
├── wrangler.jsonc            # Workers deployment config
├── public/                   # Static assets (served directly)
│   ├── index.html            # The full app
│   ├── _headers              # Cloudflare security headers
│   └── .assetsignore         # Files to exclude from static serving
└── worker/
    └── index.js              # Backend: API routes + asset fallback
```

## How routing works

- Request hits Cloudflare's edge
- If path starts with `/api/`, the Worker code runs
  - `POST /api/generate` → AI caption generation
  - `POST /api/segments` → AI audience segment generation
  - Anything else under `/api/` → 404
- Otherwise, the file is served from `public/`

This is configured by the `run_worker_first: ["/api/*"]` rule in `wrangler.jsonc`.

## Environment variables (set in Cloudflare dashboard)

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (starts with `sk-ant-...`). Stored as a **Secret** (encrypted) in the Worker. |

## Security architecture

- API key is **never** in client code. Lives only as an encrypted Worker secret.
- All AI calls go through the Worker, never directly to Anthropic from the browser.
- Per-IP rate limits (12 generations/min, 6 segments/min) inside the Worker.
- Per-browser daily cap (30 generations/day) tracked in localStorage as friction for casual abuse.
- Input size clipping prevents token-cost amplification attacks.
- Anthropic Console has a hard monthly spending cap of $25 as the final safety net.

## Deployment

This repo deploys automatically to Cloudflare Workers on every push to `main`. The Workers project is configured to use the `wrangler.jsonc` file in the root.

## Roadmap

- [ ] User accounts (Supabase Auth)
- [ ] Server-side data storage (Supabase Postgres)
- [ ] Multi-device sync
- [ ] Stripe subscriptions
- [ ] Team workspaces
- [ ] Document upload + parsing (currently the upload UI accepts files but they aren't sent to the backend)
- [ ] Email waitlist for paid plan launch

## Local development

```bash
npx wrangler dev
```

Set `ANTHROPIC_API_KEY` in a local `.dev.vars` file (gitignored).

## Notes

- The "free beta" cap is enforced both client-side (localStorage friction) and server-side (IP rate limit).
- Daily browser cap resets at midnight UTC.
- Hardcoded fallback segments trigger if the segment API fails — keeps the UX from getting stuck.
