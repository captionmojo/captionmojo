// worker/index.js
// Cloudflare Worker entry point for CaptionMojo.
// Handles API routes (/api/generate, /api/segments) and delegates everything else
// to Cloudflare's static asset handler (which serves the index.html and other files
// from the /public directory).

// ====== Shared constants ======
const MODEL = 'claude-sonnet-4-6';
const MAX_INPUT_CHARS = 12000;
const MAX_QTY = 20;
const MAX_OUTPUT_TOKENS_GEN = 2048;
const MAX_OUTPUT_TOKENS_SEG = 6000;
const MAX_PROFILE_CHARS = 8000;

// Document limits (per request)
const MAX_DOCS = 5;
const MAX_PASTED_CHARS = 80000;          // ~20k tokens of pasted text
const MAX_TEXT_DOC_CHARS = 80000;        // per text doc cap
const MAX_PDF_B64_BYTES = 8 * 1024 * 1024;   // ~8MB base64 per PDF
const MAX_IMG_B64_BYTES = 8 * 1024 * 1024;   // ~8MB base64 per image
const MAX_TOTAL_DOC_BYTES = 25 * 1024 * 1024; // 25MB total per request — safety cap

// ====== Rate limiting (in-memory, per Worker instance) ======
const ipBucketsGen = new Map();
const ipBucketsSeg = new Map();
const ipBucketsBrain = new Map();
const ipBucketsRefine = new Map();
const RATE_LIMIT_GEN = { max: 12, windowMs: 60_000 };
const RATE_LIMIT_SEG = { max: 6, windowMs: 60_000 };
const RATE_LIMIT_BRAIN = { max: 3, windowMs: 60_000 };
const RATE_LIMIT_REFINE = { max: 30, windowMs: 60_000 }; // higher — quick single-caption edits

function rateLimitOk(bucket, ip, limit) {
  const now = Date.now();
  const arr = bucket.get(ip) || [];
  const fresh = arr.filter(t => now - t < limit.windowMs);
  if (fresh.length >= limit.max) {
    bucket.set(ip, fresh);
    return false;
  }
  fresh.push(now);
  bucket.set(ip, fresh);
  if (bucket.size > 5000) {
    for (const [k, v] of bucket) {
      if (v.every(t => now - t > limit.windowMs)) bucket.delete(k);
    }
  }
  return true;
}

// ====== Response helpers ======
function corsHeaders() {
  return {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  };
}
function jsonResp(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: corsHeaders() });
}
function clipString(s, max) {
  if (!s) return '';
  s = String(s);
  return s.length > max ? s.slice(0, max) : s;
}

// ====== Error classification ======
// Maps an HTTP status code or fetch failure to a category. The frontend uses
// these categories to show specific, helpful messages instead of "AI generation
// failed" generic errors. Also drives retry timing — overloads recover fast,
// outages last minutes, payload errors never recover.
//
// Categories:
//   anthropic_overload       — 529, 503. Capacity issue, usually <30s.
//   anthropic_outage         — 502, 504. Provider problem, may last minutes.
//   anthropic_rate_limit     — 429. Our request rate too high.
//   anthropic_payload_too_large — 413. Docs too big, won't fix itself.
//   anthropic_auth_error     — 401, 403. Our API key issue.
//   anthropic_bad_request    — 400. Malformed request, our bug.
//   worker_error             — 500 or anything unexpected.
//   network_error            — fetch threw (no HTTP response at all).
function classifyError(statusOrThrown) {
  if (statusOrThrown === 'network') return 'network_error';
  const s = Number(statusOrThrown);
  if (s === 529 || s === 503) return 'anthropic_overload';
  if (s === 502 || s === 504) return 'anthropic_outage';
  if (s === 429) return 'anthropic_rate_limit';
  if (s === 413) return 'anthropic_payload_too_large';
  if (s === 401 || s === 403) return 'anthropic_auth_error';
  if (s === 400) return 'anthropic_bad_request';
  if (s >= 500) return 'anthropic_outage'; // unknown 5xx — treat as outage
  return 'worker_error';
}

// ====== Anthropic API call with category-aware retry ======
// Calls the Anthropic /v1/messages endpoint with automatic retry. Retry behavior
// depends on the error category:
//   - anthropic_overload (529, 503): retry 3 times with 2s/5s/10s backoff (~17s total)
//     Capacity issues usually recover within seconds.
//   - anthropic_outage (502, 504, 5xx): retry 3 times with 5s/10s/20s backoff (~35s total)
//     Outages last longer; give them more time to resolve.
//   - everything else: bail out immediately. Rate limits, bad requests, auth errors,
//     and payload-too-large will not fix themselves with retries.
//
// Returns: { response, retriesAttempted, finalStatus, category }
//   - response: the fetch Response object (success or final failure)
//   - retriesAttempted: how many retries fired (0 = success on first try)
//   - finalStatus: HTTP status of the final response (e.g. 200, 502, 529)
//   - category: error category string (see classifyError) — 'success' if response.ok
//
// Throws only if all attempts had network errors (no HTTP response received at all).
async function callAnthropicWithRetry(env, requestBody, label = 'unknown') {
  // Category-specific retry delays in milliseconds.
  // Overloads recover fast → tight retries. Outages last longer → wider spacing.
  const RETRY_DELAYS_OVERLOAD = [2000, 5000, 10000];   // total wait ~17s
  const RETRY_DELAYS_OUTAGE = [5000, 10000, 20000];     // total wait ~35s
  const RETRY_DELAYS_NETWORK = [3000, 6000, 12000];     // total wait ~21s, treat as outage-ish

  // Categories that should trigger retries. Everything else bails immediately.
  const RETRYABLE_CATEGORIES = new Set(['anthropic_overload', 'anthropic_outage', 'network_error']);

  let lastResponse = null;
  let lastCategory = null;
  let lastError = null;
  let retriesAttempted = 0;

  // We allow up to 4 attempts total (initial + 3 retries) regardless of category.
  // The category determines how LONG we wait between attempts.
  const MAX_ATTEMPTS = 4;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    // Wait between retries (skip wait on first attempt).
    // Delay duration depends on the last error category.
    if (attempt > 0) {
      let delays;
      if (lastCategory === 'anthropic_overload') delays = RETRY_DELAYS_OVERLOAD;
      else if (lastCategory === 'anthropic_outage') delays = RETRY_DELAYS_OUTAGE;
      else if (lastCategory === 'network_error') delays = RETRY_DELAYS_NETWORK;
      else delays = RETRY_DELAYS_OUTAGE; // safe fallback

      const delay = delays[attempt - 1] || delays[delays.length - 1];
      console.log(`[${label}] Retry ${attempt}/3 after ${delay}ms (category: ${lastCategory})`);
      await new Promise(resolve => setTimeout(resolve, delay));
      retriesAttempted = attempt;
    }

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': env.ANTHROPIC_API_KEY,
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify(requestBody)
      });

      lastResponse = response;

      // Success — return immediately
      if (response.ok) {
        if (attempt > 0) {
          console.log(`[${label}] Recovered on retry ${attempt} (final status: ${response.status})`);
        }
        return { response, retriesAttempted, finalStatus: response.status, category: 'success' };
      }

      // Classify the error to decide whether to retry
      lastCategory = classifyError(response.status);

      // Not retryable — bail out immediately
      if (!RETRYABLE_CATEGORIES.has(lastCategory)) {
        console.warn(`[${label}] Got status ${response.status} (${lastCategory}) — not retrying, returning to caller`);
        return { response, retriesAttempted, finalStatus: response.status, category: lastCategory };
      }

      console.warn(`[${label}] Attempt ${attempt + 1}/${MAX_ATTEMPTS} got status ${response.status} (${lastCategory}), will retry if attempts remain`);

    } catch (e) {
      // Network error — no HTTP response at all
      lastError = e;
      lastCategory = 'network_error';
      console.warn(`[${label}] Attempt ${attempt + 1}/${MAX_ATTEMPTS} threw network error: ${e.message}`);
    }
  }

  // All attempts exhausted.
  if (lastResponse) {
    console.error(`[${label}] All ${MAX_ATTEMPTS} attempts exhausted, final status: ${lastResponse.status}, category: ${lastCategory}`);
    return { response: lastResponse, retriesAttempted, finalStatus: lastResponse.status, category: lastCategory };
  }
  console.error(`[${label}] All ${MAX_ATTEMPTS} attempts exhausted with network errors`);
  throw lastError || new Error('All Anthropic API attempts failed');
}

// ====== Platform link conventions ======
// Each platform has its own conventions for how external links are handled.
// Instagram & TikTok DON'T allow clickable URLs in captions — "link in bio" is native.
// Twitter/X allows direct URLs and they get unfurled into rich cards — direct link is better.
// Facebook & LinkedIn also allow direct URLs.
//
// When a caption is generated for MULTIPLE platforms simultaneously, we default to
// the most restrictive convention (link in bio) so the caption never breaks on any
// chosen platform. This is a safety default — it means a caption tagged for both
// Instagram and Twitter uses "link in bio" instead of a direct URL.
const PLATFORM_LINK_RULES = {
  'Instagram':      { allowsLinks: false, native: 'link in bio' },
  'TikTok':         { allowsLinks: false, native: 'link in bio' },
  'Twitter / X':    { allowsLinks: true,  native: 'direct link works well' },
  'Facebook':       { allowsLinks: true,  native: 'direct link works well' },
  'LinkedIn':       { allowsLinks: true,  native: 'direct link works well' }
};

// Given the array of platforms the user selected for this batch, return the
// strongest link instruction. If ANY selected platform doesn't allow links,
// the whole batch must use "link in bio" — most-restrictive wins.
//
// "All" means generate for every platform → most restrictive applies.
// A single platform like ["Twitter / X"] → direct link allowed.
// Mixed like ["Instagram", "Twitter / X"] → most restrictive (link in bio).
function platformLinkInstruction(platforms) {
  const list = Array.isArray(platforms) ? platforms : [];
  // Normalize: "All" means every platform — treat as most-restrictive
  if (!list.length || list.includes('All')) {
    return {
      mode: 'restrictive',
      instruction: 'CTA / LINK RULE: This batch targets multiple platforms including Instagram and TikTok, which do NOT allow clickable links in captions. Always use "link in bio" as the CTA. Never write a direct URL like "yoursite.com" or "listen.to/anything". This rule overrides any brand examples that show direct URLs — those examples may have been for a different platform context.'
    };
  }
  // Are there any restrictive platforms in the selection?
  const restrictive = list.filter(p => PLATFORM_LINK_RULES[p] && !PLATFORM_LINK_RULES[p].allowsLinks);
  const permissive = list.filter(p => PLATFORM_LINK_RULES[p] && PLATFORM_LINK_RULES[p].allowsLinks);
  if (restrictive.length > 0) {
    // Mixed or all-restrictive → most-restrictive wins
    const platformList = restrictive.join(' and ');
    const mixed = permissive.length > 0;
    return {
      mode: 'restrictive',
      instruction: mixed
        ? `CTA / LINK RULE: This batch targets ${platformList} (which don't allow links in captions) AS WELL AS ${permissive.join(' and ')}. To make every caption work on every selected platform, use "link in bio" as the CTA. Never write a direct URL — even if your brand's smart-link format (listen.to/something, yoursite.com, etc.) appears in the brand docs. Those formats are for Twitter/Facebook/LinkedIn only; they break the Instagram and TikTok native experience.`
        : `CTA / LINK RULE: This batch targets ${platformList}, which do NOT allow clickable links in captions. Always use "link in bio" as the CTA. Never write a direct URL like "yoursite.com", "listen.to/anything", or any other link format — Instagram and TikTok hide URLs in captions, so they're useless and look broken.`
    };
  }
  // All permissive — direct links are fine
  return {
    mode: 'permissive',
    instruction: `CTA / LINK RULE: This batch targets ${list.join(' and ')}, which support direct URLs in posts. A direct link to your site or smart-link format is appropriate — readers can click. "Link in bio" is fine too, but it's not required on these platforms.`
  };
}

// ====== Document normalization ======
// Accepts what the frontend sends — an array of { name, kind, content, media_type, size }.
// Returns { docBlocks, textParts, totalBytes } where:
//   - docBlocks: array of Anthropic content blocks for PDFs and images
//   - textParts: array of strings to inline into the user message (DOCX/TXT/MD extracted text)
//   - totalBytes: rough byte total for capping
function normalizeDocs(rawDocs) {
  const safe = Array.isArray(rawDocs) ? rawDocs.slice(0, MAX_DOCS) : [];
  const docBlocks = [];
  const textParts = [];
  let totalBytes = 0;

  for (const d of safe) {
    if (!d || typeof d !== 'object') continue;
    const name = clipString(d.name, 200) || 'document';
    const kind = String(d.kind || '').toLowerCase();
    const content = typeof d.content === 'string' ? d.content : '';
    if (!content) continue;

    if (kind === 'pdf') {
      const bytes = content.length;
      if (bytes > MAX_PDF_B64_BYTES) continue;
      if (totalBytes + bytes > MAX_TOTAL_DOC_BYTES) continue;
      totalBytes += bytes;
      docBlocks.push({
        type: 'document',
        source: { type: 'base64', media_type: 'application/pdf', data: content },
        title: name
      });
    } else if (kind === 'image') {
      const bytes = content.length;
      if (bytes > MAX_IMG_B64_BYTES) continue;
      if (totalBytes + bytes > MAX_TOTAL_DOC_BYTES) continue;
      totalBytes += bytes;
      const mt = (d.media_type && String(d.media_type).startsWith('image/')) ? d.media_type : 'image/jpeg';
      docBlocks.push({
        type: 'image',
        source: { type: 'base64', media_type: mt, data: content }
      });
    } else if (kind === 'text') {
      const clipped = clipString(content, MAX_TEXT_DOC_CHARS);
      if (totalBytes + clipped.length > MAX_TOTAL_DOC_BYTES) continue;
      totalBytes += clipped.length;
      textParts.push(`--- DOC: ${name} ---\n${clipped}\n--- END DOC ---`);
    }
    // unknown kinds: silently skip
  }

  return { docBlocks, textParts, totalBytes };
}

// ====== Caption generation prompt builder ======
// When a Brain is present, we use it as the primary source of truth (tight, distilled).
// When no Brain, we fall back to the raw brand profile fields.
// The `platforms` array (passed from the per-request payload) is used to determine
// which link-format rules apply to this specific batch — see PLATFORM_LINK_RULES.
function buildSystemPrompt(bp, brain, platforms, selectedSegments) {
  const parts = [];

  // ============= OWNER NOTES — HIGHEST PRIORITY =============
  // Notes come directly from the brand owner during onboarding.
  // These are EXPLICIT INSTRUCTIONS that override everything else, including
  // the Brain's voice rules. Always include them first, regardless of Brain status.
  if (bp.notes && bp.notes.trim().length > 0) {
    parts.push('=== BRAND OWNER INSTRUCTIONS — HIGHEST PRIORITY, MUST FOLLOW EXACTLY ===');
    parts.push('These rules come directly from the brand owner. They override every other instruction in this prompt. Read them carefully and follow them on every single caption you write.');
    parts.push('');
    parts.push(bp.notes.trim());
    parts.push('');
  }

  if (brain && (brain.identity || brain.industry)) {
    // ============= BRAIN MODE — distilled brand context =============
    parts.push(`You are a senior social media copywriter for ${bp.name || 'this business'}.`);
    parts.push('');
    parts.push('=== BRAND IDENTITY ===');
    if (brain.identity) parts.push(brain.identity);
    if (brain.industry) parts.push(`Industry: ${brain.industry}`);
    parts.push('');

    if (brain.voiceRules && brain.voiceRules.length) {
      parts.push('=== VOICE RULES — MUST FOLLOW ON EVERY CAPTION ===');
      parts.push('These rules OVERRIDE default capitalization, punctuation, and tone conventions. They come directly from the brand owner. NEVER violate them. If a rule says "lowercase" — every single character must be lowercase. If a rule says "link in bio" — never use a URL.');
      brain.voiceRules.forEach(r => parts.push(`- ${r}`));
      parts.push('');
    }
    if (brain.toneTags && brain.toneTags.length) {
      parts.push(`Tone: ${brain.toneTags.join(', ')}`);
      parts.push('');
    }
    if (brain.neverDo && brain.neverDo.length) {
      parts.push('=== NEVER DO — HARD PROHIBITIONS ===');
      brain.neverDo.forEach(n => parts.push(`- ${n}`));
      parts.push('');
    }
    if (brain.proofPoints && brain.proofPoints.length) {
      parts.push('=== PROOF POINTS — specific, authentic details you can drop into captions ===');
      brain.proofPoints.forEach(p => {
        parts.push(`- ${p.name}: ${p.what || ''}`);
      });
      parts.push('Use these as concrete anchors. Real details beat marketing speak.');
      parts.push('');
    }
    if (brain.culturalReferences && brain.culturalReferences.length) {
      parts.push(`Cultural references this brand authentically uses: ${brain.culturalReferences.join(', ')}`);
      parts.push('');
    }
    if (brain.ctas && brain.ctas.length) {
      parts.push('=== CTAs — use the brand\'s actual CTAs, not generic ones ===');
      brain.ctas.forEach(c => parts.push(`- ${c}`));
      parts.push('');
    }
    if (brain.customerPortraits && brain.customerPortraits.length) {
      parts.push('=== WHO THIS BRAND WRITES FOR ===');
      brain.customerPortraits.forEach(c => parts.push(`- ${c}`));
      parts.push('');
    }
    if (brain.exampleCaptions && brain.exampleCaptions.length) {
      parts.push('=== EXAMPLE CAPTIONS — match this voice and energy ===');
      brain.exampleCaptions.forEach((c, i) => parts.push(`[${i+1}] ${c}`));
      parts.push('');
    }
    if (brain.industryStyleNote) {
      parts.push('=== INDUSTRY STYLE LAYER ===');
      parts.push(brain.industryStyleNote);
      parts.push('');
    }
  } else {
    // ============= FALLBACK MODE — raw profile fields =============
    const tones = (bp.tones || []).join(', ') || 'professional and approachable';
    parts.push(`You are a senior social media copywriter for ${bp.name || 'this business'}${bp.industry ? ` — a ${bp.industry} business` : ''}.`);
    parts.push('');
    parts.push(`WHAT THEY OFFER: ${bp.what || 'products and services'}`);
    if (bp.customer) parts.push(`THEIR CUSTOMER: ${bp.customer}`);
    if (bp.usp) parts.push(`UNIQUE SELLING POINT: ${bp.usp}`);
    if (bp.lookalike) parts.push(`LOOKALIKE AUDIENCES: ${bp.lookalike}`);
    parts.push('');
    parts.push('BRAND VOICE:');
    parts.push(`- Tone: ${tones}`);
    if (bp.voice) parts.push(`- Voice description: ${bp.voice}`);
    if (bp.avoid) parts.push(`- NEVER use these words or phrases: ${bp.avoid}`);
    if (bp.goodEx) parts.push(`- Example captions they love (match this voice and energy):\n${bp.goodEx}`);
    if (bp.badEx) parts.push(`- Captions they hate (avoid this style entirely):\n${bp.badEx}`);
    if (bp.platRules) parts.push(`- Platform rules: ${bp.platRules}`);
    parts.push('');
  }

  // ============= UNIVERSAL COPYWRITING RULES (every brand) =============
  parts.push('=== UNIVERSAL COPYWRITING RULES ===');
  UNIVERSAL_RULES.forEach(r => parts.push(`- ${r}`));
  parts.push('');

  // ============= DYNAMIC FINAL ENFORCEMENT =============
  // Scan voice rules for the most commonly-ignored rules and explicitly remind the AI at the end.
  // System prompts have a "recency bias" — instructions at the end weigh heavily.
  const allRulesText = [
    ...(brain && brain.voiceRules || []),
    ...(brain && brain.neverDo || []),
    bp.voice || '',
    bp.avoid || '',
    bp.notes || ''  // CRITICAL: owner notes scanned for enforcement triggers even when Brain failed
  ].join(' ').toLowerCase();

  const finalChecks = [];

  // === PLATFORM-AWARE LINK RULE (always fires, based on the platforms array) ===
  // This is independent of the Brain's extracted rules — it's a hard, platform-driven
  // rule that always applies. Even if the Brain extracted a wrong rule (e.g., "smart link
  // format is allowed"), this block enforces correct platform behavior.
  const linkRule = platformLinkInstruction(platforms);
  finalChecks.push(linkRule.instruction);

  // === DIMENSIONAL COHERENCE CHECK (always fires when segments are selected) ===
  // Prevents tag drift: the AI sees dozens of proof points in the Brain and is
  // tempted to use any of them. This explicit check at the end of the prompt
  // (where instructions have most weight due to recency bias) forces it to
  // sanity-check that the proof points it chose actually fit the dimensions
  // for THIS specific batch.
  if (Array.isArray(selectedSegments) && selectedSegments.length > 0) {
    const segNames = selectedSegments.map(s => `"${s.name}"`).join(' × ');
    finalChecks.push(`DIMENSIONAL COHERENCE CHECK: This batch targets the dimensions: ${segNames}. Before finalizing each caption, look at every proof point, artist name, location, product, or reference you included. Ask: "Does this specifically fit ${segNames}?" If a reference belongs to a different genre, mood, audience, or use-case within the brand — even if it's a real brand asset — REPLACE IT or REMOVE IT. The brand has many assets. Only use the ones that align with this exact combination. A caption that references the wrong artist/playlist/venue feels off-brand even when every word is technically "on brand."`);
  }

  if (/lowercase|never capitalize|all lower|no capital/.test(allRulesText)) {
    finalChecks.push('LOWERCASE CHECK: Every character in every caption MUST be lowercase. No capital letters anywhere — not at sentence starts, not in proper nouns, not in brand names. This rule comes from the brand owner and overrides standard English capitalization. Before finalizing each caption, scan for any uppercase letter and convert it to lowercase.');
  }
  if (/no emoji|never.*emoji|no emojis/.test(allRulesText)) {
    finalChecks.push('NO EMOJIS: Do not use any emoji characters anywhere in any caption.');
  }
  if (/no exclamation|never.*exclamation|no !/.test(allRulesText)) {
    finalChecks.push('NO EXCLAMATION MARKS: Do not use the "!" character anywhere in any caption.');
  }
  if (/no hashtag|never.*hashtag|no #/.test(allRulesText)) {
    finalChecks.push('NO HASHTAGS: Do not use the "#" character anywhere in any caption.');
  }

  // Final enforcement block ALWAYS present now (because the link rule always fires)
  parts.push('=== FINAL ENFORCEMENT — CHECK BEFORE RETURNING EACH CAPTION ===');
  finalChecks.forEach(c => parts.push(c));
  parts.push('');

  // ============= OUTPUT FORMAT =============
  parts.push('Return ONLY a valid JSON array. No preamble, no markdown, no code fences.');
  parts.push('Format: [{"text":"caption text here\\nwith a CTA","platforms":["Instagram"]}]');
  parts.push('For platforms use: "Instagram", "TikTok", "Twitter / X", "Facebook", "LinkedIn", or ["All"].');
  return parts.join('\n');
}

// ====== URL stripping post-processor ======
// Deterministic safety net: AFTER the model returns captions, scan each one
// for direct URL patterns. If the platform mix is restrictive (Instagram/TikTok),
// strip the URL and replace it with "link in bio". Prompts alone can't enforce
// hard rules 100% of the time — a regex check guarantees it.
//
// Patterns matched:
//   - listen.to/anything, music.apple.com/anything, etc. (smart links)
//   - example.com, mysite.org (bare domains with TLD)
//   - http:// and https:// links
//   - www. prefixed bare domains
//
// Returns { captions, urlsStripped, captionsModified } so we can log activity
// and potentially surface to the frontend later.
function stripUrlsForRestrictivePlatforms(captions, platforms) {
  // Decide whether stripping applies for this batch.
  // Restrictive = at least one Instagram/TikTok in the selection, OR "All".
  const list = Array.isArray(platforms) ? platforms : [];
  const restrictivePlatforms = ['Instagram', 'TikTok'];
  const hasRestrictive = !list.length
    || list.includes('All')
    || list.some(p => restrictivePlatforms.includes(p));

  if (!hasRestrictive) {
    // Permissive batch (Twitter/Facebook/LinkedIn only) — direct URLs are fine
    return { captions, urlsStripped: 0, captionsModified: 0 };
  }

  // Detect URL patterns. We're aggressive on detection — false positives just
  // mean "link in bio" replaces a non-URL, which is rare and harmless.
  // Order matters: match http(s):// first since they often contain the others.
  // Each regex uses global+case-insensitive flags so we catch every instance.
  const urlPatterns = [
    /https?:\/\/[^\s]+/gi,                              // http://... https://...
    /\bwww\.[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+(?:\/[^\s]*)?/gi,  // www.example.com/path
    /\blisten\.to\/[a-z0-9_-]+/gi,                      // listen.to/artistname (very common in music)
    /\bmusic\.apple\.com\/[^\s]+/gi,                    // music.apple.com/...
    /\bopen\.spotify\.com\/[^\s]+/gi,                   // open.spotify.com/...
    /\bspotify\.link\/[a-z0-9]+/gi,                     // spotify.link/...
    /\byoutu\.be\/[a-z0-9_-]+/gi,                       // youtu.be/...
    /\byoutube\.com\/[^\s]+/gi,                         // youtube.com/...
    /\blinktr\.ee\/[a-z0-9_-]+/gi,                      // linktr.ee/...
    /\b[a-z0-9][a-z0-9-]*\.(?:com|org|net|io|co|app|ai|me|to|link|ly|fm)(?:\/[^\s]*)?/gi  // generic bare domains
  ];

  let urlsStripped = 0;
  let captionsModified = 0;

  const cleaned = captions.map(c => {
    let text = c.text;
    let modified = false;

    // Apply each pattern in order, replacing matches with "link in bio"
    for (const pattern of urlPatterns) {
      const matches = text.match(pattern);
      if (matches && matches.length > 0) {
        urlsStripped += matches.length;
        // Replace the URL with "link in bio" — but if "link in bio" is already
        // adjacent in the caption, just remove the URL to avoid duplication
        text = text.replace(pattern, (match) => {
          // Check ~30 chars before/after the match for existing "link in bio"
          const idx = text.indexOf(match);
          const before = text.slice(Math.max(0, idx - 30), idx).toLowerCase();
          const after = text.slice(idx + match.length, Math.min(text.length, idx + match.length + 30)).toLowerCase();
          if (before.includes('link in bio') || after.includes('link in bio')) {
            return ''; // already has the right CTA, just remove the URL
          }
          return 'link in bio';
        });
        modified = true;
      }
    }

    // Clean up artifacts from removal: double spaces, orphan dashes/colons,
    // trailing punctuation followed by nothing
    if (modified) {
      text = text
        .replace(/[ \t]+/g, ' ')                  // collapse multi-space
        .replace(/\s+\n/g, '\n')                  // trim line ends
        .replace(/\n\s+/g, '\n')                  // trim line starts
        .replace(/[—–-]\s*$/gm, '')               // trailing dashes
        .replace(/[:.]\s*$/gm, '')                // trailing colons/periods on empty lines
        .replace(/^\s+|\s+$/g, '')                // trim full string
        .replace(/\n{3,}/g, '\n\n');              // max 2 consecutive newlines
      captionsModified++;
    }

    return { ...c, text };
  });

  return { captions: cleaned, urlsStripped, captionsModified };
}

// ====== Model output parsers ======
function parseCaptionsJson(text) {
  if (!text) return [];
  text = text.replace(/```json\s*/gi, '').replace(/```\s*$/g, '').trim();
  const start = text.indexOf('[');
  const end = text.lastIndexOf(']');
  if (start === -1 || end === -1) return [];
  try {
    const arr = JSON.parse(text.slice(start, end + 1));
    if (!Array.isArray(arr)) return [];
    return arr
      .filter(x => x && typeof x.text === 'string' && x.text.trim().length > 0)
      .map(x => ({
        text: String(x.text).slice(0, 2000),
        platforms: Array.isArray(x.platforms) && x.platforms.length ? x.platforms.slice(0, 5).map(String) : ['All']
      }));
  } catch (e) {
    return [];
  }
}

function parseSegmentsJson(text) {
  if (!text) return [];
  text = text.replace(/```json\s*/gi, '').replace(/```\s*$/g, '').trim();
  const start = text.indexOf('[');
  const end = text.lastIndexOf(']');
  if (start === -1 || end === -1) return [];
  try {
    const arr = JSON.parse(text.slice(start, end + 1));
    if (!Array.isArray(arr)) return [];
    return arr
      .filter(x => x && typeof x.name === 'string' && typeof x.desc === 'string')
      .map(x => ({
        name: String(x.name).slice(0, 80),
        desc: String(x.desc).slice(0, 240),
        category: typeof x.category === 'string' && x.category.trim() ? String(x.category).slice(0, 40) : 'General'
      }))
      .slice(0, 200);
  } catch (e) {
    return [];
  }
}

// ====== Build the cacheable user message ======
// Pattern: brand-context blocks (docs + text + pasted) go FIRST in the user message
// with cache_control on the LAST one. The variable per-request content (segment,
// quantity, direction) comes after and is NOT cached. This lets us reuse the
// expensive brand-context portion across consecutive requests for ~10% of the cost.
function buildUserContent({ docBlocks, textParts, pastedDocs, variablePart }) {
  const blocks = [];
  // 1. Document blocks (PDFs, images) — most expensive, go first
  for (const b of docBlocks) blocks.push(b);
  // 2. Inline text from DOCX/TXT/MD docs + pasted content
  const textBundle = [];
  if (textParts.length) textBundle.push(textParts.join('\n\n'));
  if (pastedDocs) textBundle.push(`--- PASTED BRAND CONTENT ---\n${pastedDocs}\n--- END PASTED ---`);
  if (textBundle.length) {
    blocks.push({ type: 'text', text: textBundle.join('\n\n') });
  }
  // 3. Mark the last brand-context block as the cache breakpoint
  if (blocks.length > 0) {
    const lastIdx = blocks.length - 1;
    blocks[lastIdx] = { ...blocks[lastIdx], cache_control: { type: 'ephemeral' } };
  }
  // 4. Variable per-request part — NOT cached
  blocks.push({ type: 'text', text: variablePart });
  return blocks;
}

// ====== Handler: POST /api/generate ======
async function handleGenerate(request, env) {
  const ip = request.headers.get('CF-Connecting-IP') || request.headers.get('x-forwarded-for') || 'unknown';
  if (!rateLimitOk(ipBucketsGen, ip, RATE_LIMIT_GEN)) {
    return jsonResp({ error: 'Rate limit exceeded. Try again in a minute.' }, 429);
  }

  let body;
  try {
    body = await request.json();
  } catch (e) {
    return jsonResp({ error: 'Invalid JSON body' }, 400);
  }

  const bp = body.brandProfile || {};
  // segments: new multi-select array of {category, name, desc} OR strings; segment: legacy single string
  let selectedSegments = [];
  if (Array.isArray(body.segments)) {
    selectedSegments = body.segments.slice(0, 8).map(s => {
      if (typeof s === 'string') return { name: clipString(s, 80), category: 'General' };
      if (s && typeof s === 'object') {
        return {
          name: clipString(s.name, 80),
          category: clipString(s.category, 40) || 'General',
          desc: clipString(s.desc, 240)
        };
      }
      return null;
    }).filter(Boolean);
  } else if (body.segment) {
    selectedSegments = [{ name: clipString(body.segment, 200) || 'General', category: 'General' }];
  }
  if (!selectedSegments.length) {
    selectedSegments = [{ name: 'General', category: 'General' }];
  }
  const platforms = Array.isArray(body.platforms) && body.platforms.length ? body.platforms.slice(0, 5).map(p => clipString(p, 50)) : ['All'];
  const quantity = Math.min(MAX_QTY, Math.max(1, parseInt(body.quantity) || 10));
  const direction = clipString(body.direction, 500);
  const pastedDocs = clipString(body.pastedDocs, MAX_PASTED_CHARS);

  const safeBp = {
    name: clipString(bp.name, 200),
    what: clipString(bp.what, 1500),
    industry: clipString(bp.industry, 100),
    customer: clipString(bp.customer, 2000),
    tones: Array.isArray(bp.tones) ? bp.tones.slice(0, 8).map(t => clipString(t, 50)) : [],
    avoid: clipString(bp.avoid, 500),
    usp: clipString(bp.usp, 1000),
    voice: clipString(bp.voice, 1000),
    lookalike: clipString(bp.lookalike, 500),
    platRules: clipString(bp.platRules, 1500),
    goodEx: clipString(bp.goodEx, 2500),
    badEx: clipString(bp.badEx, 1500),
    // Owner notes from onboarding — these are EXPLICIT RULES from the brand owner.
    // They MUST reach the generation prompt regardless of Brain success/failure.
    notes: clipString(bp.notes, 2500)
  };

  const segmentNamesStr = selectedSegments.map(s => s.name).join(', ');
  const totalInputSize = JSON.stringify(safeBp).length + direction.length + segmentNamesStr.length;
  if (totalInputSize > MAX_INPUT_CHARS) {
    return jsonResp({ error: 'Brand profile too long. Please shorten.' }, 400);
  }

  if (!env.ANTHROPIC_API_KEY) {
    return jsonResp({ error: 'Server not configured (missing API key)' }, 500);
  }

  // Brain takes priority. When present, we don't send raw docs (cheaper, sharper).
  // Raw docs only fall back when brain is absent.
  const brain = (body.brain && typeof body.brain === 'object') ? body.brain : null;
  const { docBlocks, textParts } = brain
    ? { docBlocks: [], textParts: [] }
    : normalizeDocs(body.docs);

  // Pass platforms into the prompt builder so it can apply platform-aware link rules.
  const systemPrompt = buildSystemPrompt(safeBp, brain, platforms, selectedSegments);

  // Build the segment-combination instruction. Multiple segments = mix them all simultaneously.
  // CRITICAL: proof points and references MUST align with selected dimensions, not just the
  // brand at large. If user picks "Chill Lofi" as genre, don't pull an Afrobeats artist as
  // proof point. This prevents tag drift where the AI uses any brand asset regardless of fit.
  let segmentLine;
  if (selectedSegments.length === 1) {
    const s = selectedSegments[0];
    segmentLine = `Target audience dimension: "${s.name}"${s.desc ? ` (${s.desc})` : ''}.

IMPORTANT: Every proof point, artist name, location, product, and reference you drop into a caption MUST fit this specific dimension. The brand has many assets — only use the ones that match "${s.name}". A proof point that belongs to a different genre, mood, or audience would feel wrong here, even if it's part of the same brand.`;
  } else {
    // Group by category so the AI sees the dimensional structure
    const byCategory = {};
    for (const s of selectedSegments) {
      const cat = s.category || 'General';
      if (!byCategory[cat]) byCategory[cat] = [];
      byCategory[cat].push(s);
    }
    const groupedLines = Object.entries(byCategory).map(([cat, items]) =>
      `- ${cat}: ${items.map(i => i.desc ? `"${i.name}" (${i.desc})` : `"${i.name}"`).join(', ')}`
    );
    const dimensionList = selectedSegments.map(s => `"${s.name}"`).join(' × ');
    segmentLine = `Target multiple audience dimensions SIMULTANEOUSLY. Each caption must hit ALL of these at once:
${groupedLines.join('\n')}

Every caption should feel right for someone matching this entire combination (${dimensionList}).

CRITICAL — DIMENSIONAL COHERENCE: Every proof point, artist name, location, product, and reference you drop into a caption MUST fit ALL of the selected dimensions simultaneously. The brand has many assets — only use the ones that align with this exact combination. For example, if the dimensions are "Chill Lofi" + "Late-Night" + "Crate-Diggers", don't pull a daytime-afrobeat artist or a morning-coffee playlist as proof point — even if they're part of the same brand. Pick proof points that actually fit Chill Lofi AND Late-Night AND Crate-Diggers, all at once.`;
  }

  // Parse feedback (likes + dislikes) sent from frontend.
  // Frontend has already applied tier-based capping (free=10+10, paid=unlimited).
  // We still cap at 50+50 here as a server-side safety net to prevent abuse.
  function safeFeedbackArr(arr, max){
    if(!Array.isArray(arr)) return [];
    return arr.slice(0, max).map(f => {
      if(!f || typeof f !== 'object') return null;
      const text = clipString(f.text, 600);
      if(!text) return null;
      const segs = Array.isArray(f.segments) ? f.segments.slice(0,8).map(s => clipString(s, 80)).filter(Boolean) : [];
      return { text, segments: segs };
    }).filter(Boolean);
  }
  const feedback = (body.feedback && typeof body.feedback === 'object') ? body.feedback : {};
  const fbLikes = safeFeedbackArr(feedback.likes, 50);
  const fbDislikes = safeFeedbackArr(feedback.dislikes, 50);

  // Build feedback section ONLY if there's actual feedback to send.
  // Goes in variablePart (per-request, doesn't pollute the cached system prompt).
  let feedbackSection = '';
  if(fbLikes.length || fbDislikes.length){
    const formatFb = (f) => `"${f.text}"${f.segments.length ? ` [tags: ${f.segments.join(', ')}]` : ''}`;
    const parts = ['', 'USER FEEDBACK FROM PAST GENERATIONS:'];
    if(fbLikes.length){
      parts.push('');
      parts.push(`✓ Captions the user LIKED (write more captions like these — match the voice, length, structure, tone):`);
      fbLikes.forEach(f => parts.push(`- ${formatFb(f)}`));
    }
    if(fbDislikes.length){
      parts.push('');
      parts.push(`✗ Captions the user DISLIKED (DO NOT write captions like these — avoid this voice, this structure, this tone):`);
      fbDislikes.forEach(f => parts.push(`- ${formatFb(f)}`));
    }
    parts.push('');
    parts.push('This feedback overrides general voice rules. Lean hard into what they liked, away from what they disliked.');
    feedbackSection = parts.join('\n');
  }

  const variablePart = `Generate ${quantity} caption${quantity > 1 ? 's' : ''}.

${segmentLine}

Target platform${platforms.length > 1 ? 's' : ''}: ${platforms.join(', ')}.
${direction ? `Specific direction: ${direction}` : ''}
${feedbackSection}

Return only the JSON array.`;

  // When brain is present, skip pasted docs too (brain has already digested them)
  const userContent = buildUserContent({
    docBlocks,
    textParts,
    pastedDocs: brain ? '' : pastedDocs,
    variablePart
  });

  try {
    const { response: apiResp, retriesAttempted, category } = await callAnthropicWithRetry(env, {
      model: MODEL,
      max_tokens: MAX_OUTPUT_TOKENS_GEN,
      system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
      messages: [{ role: 'user', content: userContent }]
    }, 'gen');

    if (!apiResp.ok) {
      const errText = await apiResp.text();
      console.error(`Anthropic API error after ${retriesAttempted} retries:`, apiResp.status, category, errText);
      // Return the error category so frontend can show specific helpful messages
      // instead of generic "AI generation failed." Includes retriesAttempted so
      // frontend can tell user how hard we tried before giving up.
      return jsonResp({
        error: 'AI generation failed',
        category: category || 'worker_error',
        upstream_status: apiResp.status,
        retries_attempted: retriesAttempted
      }, apiResp.status >= 500 ? 502 : apiResp.status);
    }

    const data = await apiResp.json();
    const text = (data.content || []).find(b => b.type === 'text')?.text || '';
    let captions = parseCaptionsJson(text);

    // ============= POST-PROCESS: STRIP URLS FOR RESTRICTIVE PLATFORMS =============
    // Deterministic safety net for the link rule. Even if the model includes a
    // direct URL despite our prompt instructions, this catches and replaces it.
    // Only fires when Instagram/TikTok (or "All") are in the platform selection.
    const stripResult = stripUrlsForRestrictivePlatforms(captions, platforms);
    captions = stripResult.captions;

    // Log token usage so we can verify caching is working. Cloudflare logs only.
    if (data.usage) {
      const u = data.usage;
      console.log(`[gen] in=${u.input_tokens||0} out=${u.output_tokens||0} cache_create=${u.cache_creation_input_tokens||0} cache_read=${u.cache_read_input_tokens||0} fb_likes=${fbLikes.length} fb_dislikes=${fbDislikes.length} urls_stripped=${stripResult.urlsStripped} captions_modified=${stripResult.captionsModified}`);
    }

    if (!captions.length) {
      return jsonResp({ error: 'AI returned an unparseable response. Try again.' }, 502);
    }

    return jsonResp({ captions });
  } catch (e) {
    console.error('Generate exception:', e);
    return jsonResp({ error: 'Server error. Try again in a moment.' }, 500);
  }
}

// ====== Handler: POST /api/refine ======
// One-click edit endpoint. Takes a single caption + an action chip + brand context,
// returns a refined caption. Reuses Brain + Notes so refinements stay on-brand.
//
// Action chips supported (frontend sends one of these as `action`):
//   shorter       — tighten to 1-2 sentences
//   punchier      — sharper hook, more attitude, keep length
//   addCta        — add a call to action that fits the caption
//   moreLikeThis  — generate a sibling caption in same flavor
//   rewrite       — full rewrite of the same idea, different words
//   changeTone    — pick the next tone in the brand's tone list
//   lessSalesy    — strip marketing-y language, more authentic
//   tryAgain      — regenerate fresh with same context
async function handleRefine(request, env) {
  const ip = request.headers.get('CF-Connecting-IP') || request.headers.get('x-forwarded-for') || 'unknown';
  if (!rateLimitOk(ipBucketsRefine, ip, RATE_LIMIT_REFINE)) {
    return jsonResp({ error: 'Rate limit exceeded. Try again in a minute.' }, 429);
  }

  let body;
  try {
    body = await request.json();
  } catch (e) {
    return jsonResp({ error: 'Invalid JSON body' }, 400);
  }

  // Required: original caption + action
  const originalCaption = clipString(body.caption, 1500);
  const action = clipString(body.action, 40);
  if (!originalCaption) {
    return jsonResp({ error: 'Missing caption to refine' }, 400);
  }
  const allowedActions = ['shorter','punchier','addCta','moreLikeThis','rewrite','changeTone','lessSalesy','tryAgain'];
  if (!allowedActions.includes(action)) {
    return jsonResp({ error: 'Invalid action' }, 400);
  }

  // Brand context — same shape as /api/generate but slimmer use
  const bp = body.brandProfile || {};
  const safeBp = {
    name: clipString(bp.name, 200),
    what: clipString(bp.what, 1500),
    industry: clipString(bp.industry, 100),
    audience: clipString(bp.audience, 1500),
    tones: Array.isArray(bp.tones) ? bp.tones.slice(0, 8).map(t => clipString(t, 60)).filter(Boolean) : [],
    lookalike: clipString(bp.lookalike, 500),
    platRules: clipString(bp.platRules, 1500),
    goodEx: clipString(bp.goodEx, 2500),
    badEx: clipString(bp.badEx, 1500),
    notes: clipString(bp.notes, 2500)
  };
  const brain = (body.brain && typeof body.brain === 'object') ? body.brain : null;

  // Optional context: platform + segment (so refinements stay aligned)
  const platforms = Array.isArray(body.platforms) && body.platforms.length
    ? body.platforms.slice(0, 5).map(p => clipString(p, 50))
    : ['All'];
  const segmentName = clipString(body.segmentName, 200);

  if (!env.ANTHROPIC_API_KEY) {
    return jsonResp({ error: 'Server not configured (missing API key)' }, 500);
  }

  // Build action-specific instruction
  let actionInstruction;
  switch (action) {
    case 'shorter':
      actionInstruction = 'Tighten this caption to ONE or TWO sentences maximum. Cut every word that isn\'t essential. Keep the core idea and voice. Preserve any required hashtags or @mentions if present, but the body must be substantially shorter than the original.';
      break;
    case 'punchier':
      actionInstruction = 'Rewrite with a sharper hook and more attitude. Keep roughly the same length, but make every sentence hit harder. Start with a strong opening line. Cut hedging and softening language. The voice stays on-brand — just more confident.';
      break;
    case 'addCta':
      actionInstruction = 'Add a call to action at the end that fits the caption naturally. The CTA should match the brand voice — not generic "shop now" or "click the link" unless that fits. For a song release: "out now," "stream it." For a restaurant: "swing by tonight," "book a table." For a product: "grab one." Match the energy of the rest of the caption.';
      break;
    case 'moreLikeThis':
      actionInstruction = 'Write a NEW caption in the same flavor, voice, and structure as the original — but with different specific content. Same vibe, same energy, fresh words. This is a sibling caption, not a rewrite. Different opening, different specifics, but unmistakably from the same brand and same tonal universe.';
      break;
    case 'rewrite':
      actionInstruction = 'Rewrite this caption completely. Same core idea, same approximate length, but use entirely different words and sentence structure. The voice must stay on-brand. Treat the original as inspiration, not a template.';
      break;
    case 'changeTone':
      actionInstruction = `Rewrite this caption with a different tonal angle while keeping it on-brand. If the original was confident, try playful. If it was warm, try direct. If it was poetic, try grounded. Pick from the brand's approved tone list${safeBp.tones.length ? ` (${safeBp.tones.join(', ')})` : ''} — but pick one that's DIFFERENT from the tone the original caption is using. Same idea, fresh tonal register.`;
      break;
    case 'lessSalesy':
      actionInstruction = 'Strip out any marketing-y or sales-pitch language. No "discover," "unlock," "elevate," "transform," "experience," "indulge," "exclusive," "limited time." Replace with plain, honest, human language. The caption should feel like a friend telling you about something cool, not a brand selling at you. Keep the same core information.';
      break;
    case 'tryAgain':
      actionInstruction = 'Regenerate this caption from scratch with the same brief and context, but produce a meaningfully different result. Don\'t mirror the original\'s structure or word choices. Fresh angle, fresh language.';
      break;
  }

  // Reuse the brand voice system prompt — same builder as /api/generate.
  // Segment context optional — single segment if provided, fallback to general.
  const selectedSegments = segmentName
    ? [{ name: segmentName, category: 'General' }]
    : [{ name: 'General', category: 'General' }];
  const systemPrompt = buildSystemPrompt(safeBp, brain, platforms, selectedSegments);

  // User message: original caption + action
  const userMessage = `Here is an existing on-brand caption:

"""
${originalCaption}
"""

Refinement task: ${actionInstruction}

Target platform${platforms.length > 1 ? 's' : ''}: ${platforms.join(', ')}.

Return ONLY the refined caption as plain text. No JSON, no quotation marks around it, no labels, no preamble — just the new caption text and nothing else.`;

  try {
    const { response: apiResp, retriesAttempted, category } = await callAnthropicWithRetry(env, {
      model: MODEL,
      max_tokens: 800, // single caption, much smaller than batch generation
      system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
      messages: [{ role: 'user', content: userMessage }]
    }, 'refine');

    if (!apiResp.ok) {
      const errText = await apiResp.text();
      console.error(`Refine API error after ${retriesAttempted} retries:`, apiResp.status, category, errText);
      return jsonResp({
        error: 'AI refinement failed',
        category: category || 'worker_error',
        upstream_status: apiResp.status,
        retries_attempted: retriesAttempted
      }, apiResp.status >= 500 ? 502 : apiResp.status);
    }

    const data = await apiResp.json();
    let refined = (data.content || []).find(b => b.type === 'text')?.text || '';
    refined = refined.trim();

    // Strip wrapping quotes if the AI added them despite instructions
    if ((refined.startsWith('"') && refined.endsWith('"')) ||
        (refined.startsWith("'") && refined.endsWith("'"))) {
      refined = refined.slice(1, -1).trim();
    }

    if (!refined) {
      return jsonResp({ error: 'AI returned an empty response. Try again.' }, 502);
    }

    // Strip URLs for restrictive platforms (same as /api/generate).
    // The stripper expects {text:...} objects, so wrap then unwrap.
    const stripResult = stripUrlsForRestrictivePlatforms([{ text: refined }], platforms);
    refined = (stripResult.captions[0] && stripResult.captions[0].text) || refined;

    if (data.usage) {
      const u = data.usage;
      console.log(`[refine action=${action}] in=${u.input_tokens||0} out=${u.output_tokens||0} cache_read=${u.cache_read_input_tokens||0}`);
    }

    return jsonResp({ caption: refined, action });
  } catch (e) {
    console.error('Refine exception:', e);
    return jsonResp({ error: 'Server error. Try again in a moment.' }, 500);
  }
}

// ====== Handler: POST /api/segments ======
async function handleSegments(request, env) {
  const ip = request.headers.get('CF-Connecting-IP') || request.headers.get('x-forwarded-for') || 'unknown';
  if (!rateLimitOk(ipBucketsSeg, ip, RATE_LIMIT_SEG)) {
    return jsonResp({ error: 'Rate limit exceeded.' }, 429);
  }

  let body;
  try {
    body = await request.json();
  } catch (e) {
    return jsonResp({ error: 'Invalid JSON body' }, 400);
  }

  const profile = String(body.profile || '').slice(0, MAX_PROFILE_CHARS);
  const brain = (body.brain && typeof body.brain === 'object') ? body.brain : null;

  // Need either profile OR brain to generate segments
  if (!profile.trim() && !brain) {
    return jsonResp({ error: 'Profile or brain required' }, 400);
  }

  if (!env.ANTHROPIC_API_KEY) {
    return jsonResp({ error: 'Server not configured' }, 500);
  }

  // If brain present, skip docs (already digested into the brain). Otherwise process docs.
  const pastedDocs = brain ? '' : clipString(body.pastedDocs, MAX_PASTED_CHARS);
  const { docBlocks, textParts } = brain
    ? { docBlocks: [], textParts: [] }
    : normalizeDocs(body.docs);

  const systemPrompt = `You are a senior marketing strategist. Given a brand profile (and any brand documents the business has provided), generate a MULTI-DIMENSIONAL tag set this business can mix-and-match when generating social posts.

THE KEY INSIGHT: real social media managers don't think "I'm posting for one audience." They think in COMBINATIONS — e.g., "Lo-Fi track + mental health angle + educated women" for a record label, or "weekend brunch + new diners + plant-based menu" for a restaurant. You must produce orthogonal tag dimensions that can be COMBINED.

YOUR JOB:
1. First, infer 3-6 RELEVANT CATEGORIES for this specific business. Categories are the DIMENSIONS of marketing — different axes that compose together. Pick categories that make sense for THIS industry.
   - Record labels typically use: Genre, Mood/Use-Case, Audience Segment, Content Pillar
   - Restaurants typically use: Cuisine/Menu, Day-Part, Occasion, Customer Type
   - Fitness studios typically use: Discipline, Skill Level, Goal, Member Lifecycle
   - E-commerce typically use: Product Category, Customer Lifecycle, Use-Case, Demographic
   - Creators typically use: Content Pillar, Audience Segment, Mood, Platform Context
   - Adjust based on the actual brand. Don't force-fit.

2. Then, for EACH category, generate AS MANY tags as the brand's actual catalog/audience/offerings justify. Be COMPREHENSIVE, NOT consolidated. This is the most important rule.

   - If a record label has 17 playlists across genres, return 17 separate genre tags — do NOT consolidate "Chill Lofi" + "Chill Jazz" + "Chill Ambient" into one "Lo-Fi & Ambient" bucket. Keep them separate so the user can mark a specific drop as the specific genre.
   - If a restaurant has 8 distinct menu sections, return 8 menu tags.
   - If a fitness studio offers 12 class types, return 12 discipline tags.
   - If a brand serves 5 distinct customer types in their docs, return 5 audience tags.

   Target ranges per category:
   - Minimum 3 tags per category
   - Typical 5-15 tags per category
   - For catalog-heavy industries (record labels with many playlists, restaurants with deep menus, fitness studios with many classes, e-commerce with many product lines): 15-30+ tags per category is correct and expected.
   - HARD CAP: 40 tags per category. Beyond that, consolidate similar concepts.

   Tags should be:
   - SPECIFIC, drawn directly from the brand documents (artist names, playlist names, menu items, class types, audience descriptions the brand actually uses)
   - REUSABLE across many posts (not hyper-narrow micro-personas pulled from one doc reference)
   - MIXABLE with tags from other categories
   - DISTINCT from each other — don't return near-duplicates ("Boom Bap" and "Boom Bap Beats" should be one tag)

GOOD EXAMPLES (Record label — comprehensive):
- Genre: "Chill Lofi", "Chill Jazz", "Boom Bap", "Nu-Bap", "Chill Trip-Hop", "Bossa Nova", "Chill Reggae Dub", "Chill Jazz House", "Chill Afro Beats", "Afro House", "Neo-Psychedelic", "Chill Ambient", "Chill Sleep", "Progressive House", "Deep House", "Chill House", "Chill Garage" (all 17 distinct playlists from the docs)
- Mood/Use-Case: "Deep Focus", "Mental Health & Decompression", "Late-Night Creative", "Workout Energy", "Sleep & Wind-Down", "Morning Routine"
- Audience: "Educated Black Women", "HBCU Students", "Hip-Hop Heads", "Jazz Heads", "Beatmakers & Producers", "Remote Workers 25-40", "Global Crate-Diggers"
- Content Pillar: "The Roost (Cigar Bar)", "Native Dialects (Animated Band)", "True Dialect Radio", "The Producer's Label", "Archival Citation"

BAD EXAMPLES (over-consolidated or one-off):
- "Lo-Fi & Ambient" as one tag when the label has separate Lofi, Jazz, Ambient, Sleep playlists (CONSOLIDATION ERROR — split into 4)
- "Animated Universe Fans" as a standalone segment (HYPER-NARROW — fold into a content pillar instead)
- "Adults 25-50" (TOO GENERIC — be specific about lifestyle/psychographic)

Each tag must have:
- name: short and memorable, 2-5 words
- desc: one sentence on who they are / what it means
- category: the category this tag belongs to

Return ONLY a valid JSON array (no preamble, no markdown, no code fences). Return EVERY tag the brand documents justify — comprehensive coverage, not summary.
Format: [{"category":"Genre","name":"Chill Lofi","desc":"Warm, mellow instrumental lo-fi beats."}, {"category":"Mood","name":"Mental Health & Decompression","desc":"Listeners using music to unwind, regulate, and reset."}]`;

  const variablePart = `Brand profile:\n${profile}\n\nGenerate comprehensive multi-dimensional tag categories for this business. Return EVERY distinct tag the brand documents justify — do NOT consolidate or summarize. If the docs reference 17 playlists, return 17 separate genre tags. If 8 audience types, return 8 audience tags. Return only the JSON array.`;

  const userContent = buildUserContent({ docBlocks, textParts, pastedDocs, variablePart });

  try {
    const { response: apiResp, retriesAttempted, category } = await callAnthropicWithRetry(env, {
      model: MODEL,
      max_tokens: MAX_OUTPUT_TOKENS_SEG,
      system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
      messages: [{ role: 'user', content: userContent }]
    }, 'seg');

    if (!apiResp.ok) {
      const errText = await apiResp.text();
      console.error(`Anthropic API error after ${retriesAttempted} retries:`, apiResp.status, category, errText);
      return jsonResp({
        error: 'AI segment generation failed',
        category: category || 'worker_error',
        upstream_status: apiResp.status,
        retries_attempted: retriesAttempted
      }, apiResp.status >= 500 ? 502 : apiResp.status);
    }

    const data = await apiResp.json();
    const text = (data.content || []).find(b => b.type === 'text')?.text || '';
    const segments = parseSegmentsJson(text);

    if (data.usage) {
      const u = data.usage;
      console.log(`[seg] in=${u.input_tokens||0} out=${u.output_tokens||0} cache_create=${u.cache_creation_input_tokens||0} cache_read=${u.cache_read_input_tokens||0}`);
    }

    if (!segments.length) {
      return jsonResp({ error: 'AI returned no parseable segments.' }, 502);
    }

    return jsonResp({ segments });
  } catch (e) {
    console.error('Segments exception:', e);
    return jsonResp({ error: 'Server error.' }, 500);
  }
}

// ====== Brand Brain extraction ======
// Reads all docs + onboarding form + paste, returns a structured "Brand Brain":
// a tight, authoritative distillation of what the brand IS, how it sounds,
// proof points, CTAs, etc. This Brain is then sent on every generation call
// INSTEAD of all the raw docs — cheaper, faster, more consistent output.
function parseBrainJson(text) {
  if (!text) return null;
  // Strip code fences anywhere they appear, not just at start/end
  text = text.replace(/```(?:json)?\s*/gi, '').replace(/```/g, '').trim();
  // Try to find the outermost JSON object
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1) return null;
  let jsonStr = text.slice(start, end + 1);

  // Resilient parse: try direct parse first, then with common-fix preprocessing.
  let obj;
  try {
    obj = JSON.parse(jsonStr);
  } catch (e) {
    // Fallback 1: remove trailing commas before } or ]
    try {
      const fixed = jsonStr.replace(/,(\s*[}\]])/g, '$1');
      obj = JSON.parse(fixed);
    } catch (e2) {
      console.warn('Brain parse failed even after trailing-comma fix:', e2.message);
      return null;
    }
  }
  if (!obj || typeof obj !== 'object') return null;
  try {
    // Sanitize and clip everything
    const clipArr = (a, n, max) => Array.isArray(a) ? a.slice(0, n).map(x => clipString(typeof x === 'string' ? x : JSON.stringify(x), max)).filter(Boolean) : [];
    const clipObjArr = (a, n, fields) => Array.isArray(a) ? a.slice(0, n).map(x => {
      if (!x || typeof x !== 'object') return null;
      const out = {};
      for (const [k, max] of Object.entries(fields)) {
        if (x[k]) out[k] = clipString(String(x[k]), max);
      }
      return Object.keys(out).length ? out : null;
    }).filter(Boolean) : [];

    return {
      identity: clipString(obj.identity, 600),
      industry: clipString(obj.industry, 60),
      industryStyleNote: clipString(obj.industryStyleNote, 600),
      voiceRules: clipArr(obj.voiceRules, 25, 200),
      toneTags: clipArr(obj.toneTags, 15, 50),
      neverDo: clipArr(obj.neverDo, 25, 200),
      proofPoints: clipObjArr(obj.proofPoints, 80, { name: 100, what: 300 }),
      ctas: clipArr(obj.ctas, 50, 150),
      culturalReferences: clipArr(obj.culturalReferences, 60, 150),
      customerPortraits: clipArr(obj.customerPortraits, 30, 300),
      exampleCaptions: clipArr(obj.exampleCaptions, 15, 600)
    };
  } catch (e) {
    return null;
  }
}

async function handleBrain(request, env) {
  const ip = request.headers.get('CF-Connecting-IP') || request.headers.get('x-forwarded-for') || 'unknown';
  if (!rateLimitOk(ipBucketsBrain, ip, RATE_LIMIT_BRAIN)) {
    return jsonResp({ error: 'Rate limit exceeded.' }, 429);
  }

  let body;
  try {
    body = await request.json();
  } catch (e) {
    return jsonResp({ error: 'Invalid JSON body' }, 400);
  }

  const profile = String(body.profile || '').slice(0, MAX_PROFILE_CHARS);
  if (!profile.trim() && !(Array.isArray(body.docs) && body.docs.length) && !body.pastedDocs) {
    return jsonResp({ error: 'Need a profile or docs to extract a Brain.' }, 400);
  }

  if (!env.ANTHROPIC_API_KEY) {
    return jsonResp({ error: 'Server not configured' }, 500);
  }

  const pastedDocs = clipString(body.pastedDocs, MAX_PASTED_CHARS);

  // === BRAIN INPUT CAP ===
  // Cloudflare Workers can time out on very large doc sets. Hard-cap input to
  // ~100k tokens (≈400k chars). If the user uploaded more, drop trailing docs
  // (NOT summarize — that destroys detail). Frontend can show a transparent warning.
  const BRAIN_MAX_INPUT_CHARS = 400000;
  let braininputBudget = BRAIN_MAX_INPUT_CHARS - (profile.length + pastedDocs.length);
  let { docBlocks, textParts } = normalizeDocs(body.docs);
  let docsDropped = 0;
  if (braininputBudget > 0) {
    // Keep docs while we have budget, in original order. textParts are text docs; docBlocks are PDFs/images (base64).
    const keptText = [];
    for (const tp of textParts) {
      const len = (tp && tp.text) ? tp.text.length : 0;
      if (len <= braininputBudget) {
        keptText.push(tp);
        braininputBudget -= len;
      } else {
        docsDropped++;
      }
    }
    textParts = keptText;
    // Base64 docs (PDFs/images): rough estimate of 0.75 chars per byte of base64 in cost
    const keptDoc = [];
    for (const db of docBlocks) {
      // db.source.data is base64
      const approxChars = db?.source?.data ? db.source.data.length : 0;
      if (approxChars <= braininputBudget) {
        keptDoc.push(db);
        braininputBudget -= approxChars;
      } else {
        docsDropped++;
      }
    }
    docBlocks = keptDoc;
  } else {
    // Even profile+pasted exceeds budget (very rare). Drop all docs and rely on text alone.
    docsDropped = (docBlocks?.length || 0) + (textParts?.length || 0);
    docBlocks = [];
    textParts = [];
  }
  if (docsDropped > 0) {
    console.warn(`[brain] Doc cap triggered: dropped ${docsDropped} docs over ~100k token budget`);
  }

  const systemPrompt = `You are a senior brand strategist + copywriter. Your job is to read everything the user has provided about their business — onboarding form, uploaded documents (PDFs, images, text), pasted content — and produce a tight, AUTHORITATIVE "Brand Brain" that will be used to write social captions for them on every future call.

THE BRAND BRAIN MUST CAPTURE:

1. identity — One paragraph distilling what this brand IS, its lineage, what genuinely makes it different. Specific. Not "we make great products." Examples:
   - "Independent record label, archivist-leaning. Roots in jazz, boom bap, lofi, afrobeats, dub, jazz house. Listener-first, never explains the music."
   - "Brooklyn pizzeria — third-generation family-owned, late-night oriented. Beloved by locals who treat the slice as a ritual, not just food."

2. industry — One of: music_label, restaurant, fitness_studio, wellness, creator, ecommerce, beauty, fashion, hospitality, professional_services, education, nonprofit, real_estate, technology, other

3. industryStyleNote — Industry-specific style layer. The kind of language and conventions native to this industry. Specific.

4. voiceRules — Concrete writing rules. This is THE MOST IMPORTANT FIELD. Future captions will be generated using ONLY these rules. If you miss a rule here, the AI will violate it.

   EXTRACTION PROCESS — follow all three steps:

   STEP A: Scan all docs for EXPLICIT rule statements. Examples of what to look for:
   - "We always write in lowercase" / "Our captions are always lowercase" → extract as "Always lowercase. Never capitalize anything."
   - "We never use exclamation marks" / "no !" → extract as "Never use exclamation marks"
   - "Our CTA is always 'link in bio'" → extract as "Use 'link in bio' as the CTA. Never use direct URLs."
   - "Avoid corporate language" → extract as "Never use corporate or salesy language"
   - "We use minimal punctuation" → extract as "Minimal punctuation — no semicolons, no dashes between phrases"
   When you find an explicit rule, store it VERBATIM. Do not paraphrase. Do not soften ("always lowercase" should NEVER become "casual tone").

   STEP B: Scan the example captions in the docs and REVERSE-ENGINEER the rules. Look at every example caption mechanically:
   - Is every example lowercase? → add "Always lowercase. Never capitalize anything."
   - Do they all end with "link in bio" (or similar)? → add "Use 'link in bio' as the CTA. Never use direct URLs."
   - Do any contain exclamation marks? If NO examples use them → add "Never use exclamation marks"
   - Do any contain emojis? If NO examples use them → add "Never use emojis"
   - Do any contain hashtags? If NO examples use them → add "Never use hashtags"
   - Are they all under 3 lines? → add "Terse — max 2-3 lines before the CTA"
   - Do they speak TO the listener/customer (you/your) rather than ABOUT the brand (we/our)? → add "The listener is the hero. Speak to them, not about the brand."
   - Do they avoid explaining the product? → add "Never explain the product — name the feeling it produces"

   STEP C: Add any user-provided notes from the onboarding form. These override all other extracted rules.

   THE GOLDEN RULE: If a brand's example captions are all lowercase, you MUST add "Always lowercase" as a rule — even if no doc explicitly says so. The examples ARE the rule.

   Format each rule as a single imperative sentence. Be specific about both the rule AND what NOT to do.

5. toneTags — Short tone descriptors (3-6 tags)

6. neverDo — Hard prohibitions. Things this brand never says, never does. Mirror the voiceRules — for every "Always X" rule, also add a "Never Y" rule here. Pull from their "words to avoid" if given. Examples:
   - "Never use capital letters anywhere (always lowercase)"
   - "Never use exclamation marks"
   - "Never use emojis"
   - "Never use hashtags"
   - "Never write direct URLs (use 'link in bio' instead)"
   - "Never 'unlock your potential', 'hustle', 'limited time', 'game-changer'"
   - "Never explain the product — name the feeling instead"

7. proofPoints — Specific, authentic details the AI can drop into captions as proof the brand is real. Names of people, products, places, history, references. Each is {name, what}. EXAMPLE for a record label:
   - {name: "lonnie.", what: "their artist — warm jazz lofi, blue note energy"}
   - {name: "the roost", what: "their cigar bar at 2am, named after charlie parker's first record label"}
   - {name: "modu", what: "their artist — boom bap, named after photographer chi modu who shot source magazine covers"}
   EXTRACT EVERY PROOF POINT THE DOCS JUSTIFY. Don't cap at 5. A record label with 7 artists + 17 playlists + venues = 25+ proof points.

8. ctas — The brand's actual CTAs. Pull from example captions if given. Infer if not. Don't artificially limit. Examples for a record label might be:
   - "link in bio"
   - "curated sounds — link in bio"
   - "stream [artist] — link in bio"
   - "true dialect radio — always on"

9. culturalReferences — Names, places, eras the brand authentically references. Used to ground captions. Examples for a record label: "tribe", "lauryn", "minton's", "blue note", "stones throw", "1992". For a restaurant: neighborhood names, family origin, dishes that have a story.

10. customerPortraits — Specific customer types. Used later as seeds for segments. Each is one sentence. Get specific. Examples:
    - "Women in their 20s-30s using music to process anxiety"
    - "Crate-digging hip-hop heads who care about lineage"
    - "Remote workers and students looking for 8-hour focus tracks"

11. exampleCaptions — 3-5 captions you'd write that BEST exemplify the brand voice given everything provided. These should be aspirational and tight — what you'd want every future caption to feel like.

CRITICAL RULES FOR EXTRACTION:
- Be SPECIFIC, never generic. "Authentic" is not specific. "Named after charlie parker's first record label" is specific.
- If the docs contain dozens of proof points, return dozens. Comprehensiveness > brevity in this artifact.
- If the brand has clearly stated voice rules ("always lowercase", "no exclamation"), follow them exactly.
- Extract example captions from the docs verbatim if any exist that are obviously gold.
- If something isn't in the docs, infer it from the industry and brand description — but flag your inference is interpretive.

Return ONLY a single valid JSON object. No preamble, no markdown, no code fences. Use the keys above exactly.`;

  const variablePart = `Onboarding form data:\n${profile}\n\nRead all attached documents and pasted content. Build the Brand Brain as a single JSON object. Be comprehensive on proof points, CTAs, and cultural references — extract every specific detail the docs justify.`;

  const userContent = buildUserContent({ docBlocks, textParts, pastedDocs, variablePart });

  try {
    const { response: apiResp, retriesAttempted, category } = await callAnthropicWithRetry(env, {
      model: MODEL,
      max_tokens: 8000,
      system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
      messages: [{ role: 'user', content: userContent }]
    }, 'brain');

    if (!apiResp.ok) {
      const errText = await apiResp.text();
      console.error(`Anthropic API error after ${retriesAttempted} retries:`, apiResp.status, category, errText);
      return jsonResp({
        error: 'AI Brain extraction failed',
        category: category || 'worker_error',
        upstream_status: apiResp.status,
        retries_attempted: retriesAttempted
      }, apiResp.status >= 500 ? 502 : apiResp.status);
    }

    const data = await apiResp.json();
    const text = (data.content || []).find(b => b.type === 'text')?.text || '';
    const brain = parseBrainJson(text);

    if (data.usage) {
      const u = data.usage;
      console.log(`[brain] in=${u.input_tokens||0} out=${u.output_tokens||0} cache_create=${u.cache_creation_input_tokens||0} cache_read=${u.cache_read_input_tokens||0}`);
    }

    if (!brain) {
      return jsonResp({ error: 'AI returned unparseable Brain. Try again.' }, 502);
    }

    return jsonResp({ brain, docsDropped });
  } catch (e) {
    console.error('Brain exception:', e);
    return jsonResp({ error: 'Server error.' }, 500);
  }
}

// Universal copywriting rules — applied to EVERY generation regardless of industry.
// These are the timeless rules that show up in every great brand's copy.
const UNIVERSAL_RULES = [
  "Never explain the product — name the feeling it produces or the moment it fits",
  "The customer is the hero of every caption, never the business",
  "No hollow filler ('In a world where...', 'Let's be real...', 'Game-changer', 'Unlock your potential')",
  "Specific > generic. Real details beat marketing speak",
  "Vary the captions — different angles, different openings, different lengths",
  "End with a natural CTA appropriate to the platform"
];

// ====== Main Worker entry ======
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    if (url.pathname === '/api/generate' && request.method === 'POST') {
      return handleGenerate(request, env);
    }
    if (url.pathname === '/api/refine' && request.method === 'POST') {
      return handleRefine(request, env);
    }
    if (url.pathname === '/api/segments' && request.method === 'POST') {
      return handleSegments(request, env);
    }
    if (url.pathname === '/api/brain' && request.method === 'POST') {
      return handleBrain(request, env);
    }
    if (url.pathname.startsWith('/api/')) {
      return jsonResp({ error: 'Not found' }, 404);
    }

    return env.ASSETS.fetch(request);
  }
};
