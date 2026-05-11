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
const MAX_OUTPUT_TOKENS_SEG = 1200;
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
const RATE_LIMIT_GEN = { max: 12, windowMs: 60_000 };
const RATE_LIMIT_SEG = { max: 6, windowMs: 60_000 };
const RATE_LIMIT_BRAIN = { max: 3, windowMs: 60_000 };

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
function buildSystemPrompt(bp, brain) {
  const parts = [];

  if (brain && (brain.identity || brain.industry)) {
    // ============= BRAIN MODE — distilled brand context =============
    parts.push(`You are a senior social media copywriter for ${bp.name || 'this business'}.`);
    parts.push('');
    parts.push('=== BRAND IDENTITY ===');
    if (brain.identity) parts.push(brain.identity);
    if (brain.industry) parts.push(`Industry: ${brain.industry}`);
    parts.push('');

    if (brain.voiceRules && brain.voiceRules.length) {
      parts.push('=== VOICE RULES (non-negotiable) ===');
      brain.voiceRules.forEach(r => parts.push(`- ${r}`));
      parts.push('');
    }
    if (brain.toneTags && brain.toneTags.length) {
      parts.push(`Tone: ${brain.toneTags.join(', ')}`);
      parts.push('');
    }
    if (brain.neverDo && brain.neverDo.length) {
      parts.push('=== NEVER DO ===');
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

  // ============= OUTPUT FORMAT =============
  parts.push('Return ONLY a valid JSON array. No preamble, no markdown, no code fences.');
  parts.push('Format: [{"text":"caption text here\\nwith a CTA","platforms":["Instagram"]}]');
  parts.push('For platforms use: "Instagram", "TikTok", "Twitter / X", "Facebook", "LinkedIn", or ["All"].');
  return parts.join('\n');
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
      .slice(0, 24);
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
    badEx: clipString(bp.badEx, 1500)
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

  const systemPrompt = buildSystemPrompt(safeBp, brain);

  // Build the segment-combination instruction. Multiple segments = mix them all simultaneously.
  let segmentLine;
  if (selectedSegments.length === 1) {
    const s = selectedSegments[0];
    segmentLine = `Target audience dimension: "${s.name}"${s.desc ? ` (${s.desc})` : ''}.`;
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
    segmentLine = `Target multiple audience dimensions SIMULTANEOUSLY. Each caption must hit ALL of these at once:\n${groupedLines.join('\n')}\n\nDo not treat these as alternatives — every caption should feel right for someone matching this entire combination.`;
  }

  const variablePart = `Generate ${quantity} caption${quantity > 1 ? 's' : ''}.

${segmentLine}

Target platform${platforms.length > 1 ? 's' : ''}: ${platforms.join(', ')}.
${direction ? `Specific direction: ${direction}` : ''}

Return only the JSON array.`;

  // When brain is present, skip pasted docs too (brain has already digested them)
  const userContent = buildUserContent({
    docBlocks,
    textParts,
    pastedDocs: brain ? '' : pastedDocs,
    variablePart
  });

  try {
    const apiResp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: MAX_OUTPUT_TOKENS_GEN,
        system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
        messages: [{ role: 'user', content: userContent }]
      })
    });

    if (!apiResp.ok) {
      const errText = await apiResp.text();
      console.error('Anthropic API error:', apiResp.status, errText);
      if (apiResp.status === 429) return jsonResp({ error: 'AI service is busy. Try again shortly.' }, 429);
      if (apiResp.status === 401 || apiResp.status === 403) return jsonResp({ error: 'Server auth error. Admin has been notified.' }, 500);
      if (apiResp.status === 413) return jsonResp({ error: 'Brand docs too large. Remove a file and try again.' }, 413);
      return jsonResp({ error: 'AI generation failed. Try again in a moment.' }, 502);
    }

    const data = await apiResp.json();
    const text = (data.content || []).find(b => b.type === 'text')?.text || '';
    const captions = parseCaptionsJson(text);

    // Log token usage so we can verify caching is working. Cloudflare logs only.
    if (data.usage) {
      const u = data.usage;
      console.log(`[gen] in=${u.input_tokens||0} out=${u.output_tokens||0} cache_create=${u.cache_creation_input_tokens||0} cache_read=${u.cache_read_input_tokens||0}`);
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
1. First, infer 3-4 RELEVANT CATEGORIES for this specific business. Categories are the DIMENSIONS of marketing — different axes that compose together. Pick categories that make sense for THIS industry.
   - Record labels typically use: Genre, Mood/Use-Case, Demographic, Activity/Context
   - Restaurants typically use: Cuisine/Menu, Day-Part, Occasion, Customer Type
   - Fitness studios typically use: Discipline, Skill Level, Goal, Member Lifecycle
   - E-commerce typically use: Product Category, Customer Lifecycle, Use-Case, Demographic
   - Creators typically use: Content Pillar, Audience Segment, Mood, Platform Context
   - Adjust based on the actual brand. Don't force-fit.

2. Then, for EACH category, generate 3-5 specific tags. Tags should be:
   - SPECIFIC and useful (not generic buckets like "Adults" or "Customers")
   - REUSABLE across many posts (not hyper-narrow micro-personas pulled from one doc reference)
   - MIXABLE with tags from other categories
   - Drawn from the brand documents where possible

GOOD EXAMPLES (Record label):
- Genre: "Afrobeat", "Lo-Fi Hip-Hop", "Ambient Jazz", "Neo-Soul"
- Mood/Use-Case: "Deep Focus", "Mental Health & Decompression", "Late-Night Creative", "Workout Energy"
- Demographic: "Educated Black Women", "Gen Z Creatives", "Remote Workers 25-40"
- Activity: "Studying/Reading", "Working From Home", "Commuting", "Morning Routine"

BAD EXAMPLES (too narrow, hyper-extracted):
- "Animated Universe Fans" (one-off, not reusable)
- "Late Night Beatmakers" (mixes too many dimensions into one tag)
- "Content Creators Needing Clearance" (super-specific business edge case, not a marketing dimension)

Each tag must have:
- name: short and memorable, 2-5 words
- desc: one sentence on who they are / what it means
- category: the category this tag belongs to

Return ONLY a valid JSON array (no preamble, no markdown, no code fences). 10-20 tags total across 3-4 categories.
Format: [{"category":"Genre","name":"Afrobeat","desc":"Modern Afrobeat-influenced sound."}, {"category":"Mood","name":"Mental Health & Decompression","desc":"Listeners using music to unwind, regulate, and reset."}]`;

  const variablePart = `Brand profile:\n${profile}\n\nGenerate multi-dimensional tag categories for this business. Return only the JSON array.`;

  const userContent = buildUserContent({ docBlocks, textParts, pastedDocs, variablePart });

  try {
    const apiResp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: MAX_OUTPUT_TOKENS_SEG,
        system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
        messages: [{ role: 'user', content: userContent }]
      })
    });

    if (!apiResp.ok) {
      const errText = await apiResp.text();
      console.error('Anthropic API error:', apiResp.status, errText);
      if (apiResp.status === 429) return jsonResp({ error: 'AI service is busy. Try again.' }, 429);
      if (apiResp.status === 413) return jsonResp({ error: 'Brand docs too large. Remove a file and try again.' }, 413);
      return jsonResp({ error: 'AI generation failed.' }, 502);
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
  text = text.replace(/```json\s*/gi, '').replace(/```\s*$/g, '').trim();
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1) return null;
  try {
    const obj = JSON.parse(text.slice(start, end + 1));
    if (!obj || typeof obj !== 'object') return null;
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
      voiceRules: clipArr(obj.voiceRules, 15, 200),
      toneTags: clipArr(obj.toneTags, 10, 50),
      neverDo: clipArr(obj.neverDo, 15, 200),
      proofPoints: clipObjArr(obj.proofPoints, 30, { name: 100, what: 300 }),
      ctas: clipArr(obj.ctas, 25, 150),
      culturalReferences: clipArr(obj.culturalReferences, 30, 150),
      customerPortraits: clipArr(obj.customerPortraits, 15, 300),
      exampleCaptions: clipArr(obj.exampleCaptions, 8, 600)
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
  const { docBlocks, textParts } = normalizeDocs(body.docs);

  const systemPrompt = `You are a senior brand strategist + copywriter. Your job is to read everything the user has provided about their business — onboarding form, uploaded documents (PDFs, images, text), pasted content — and produce a tight, AUTHORITATIVE "Brand Brain" that will be used to write social captions for them on every future call.

THE BRAND BRAIN MUST CAPTURE:

1. identity — One paragraph distilling what this brand IS, its lineage, what genuinely makes it different. Specific. Not "we make great products." Examples:
   - "Independent record label, archivist-leaning. Roots in jazz, boom bap, lofi, afrobeats, dub, jazz house. Listener-first, never explains the music."
   - "Brooklyn pizzeria — third-generation family-owned, late-night oriented. Beloved by locals who treat the slice as a ritual, not just food."

2. industry — One of: music_label, restaurant, fitness_studio, wellness, creator, ecommerce, beauty, fashion, hospitality, professional_services, education, nonprofit, real_estate, technology, other

3. industryStyleNote — Industry-specific style layer. The kind of language and conventions native to this industry. Specific.

4. voiceRules — Concrete writing rules from the brand's actual voice. Pull from how they describe themselves AND from any example captions they provided. Examples:
   - "Always lowercase"
   - "Terse — max 2 lines before the CTA"
   - "Never explain the product — name the feeling it produces"
   - "The customer is the hero, not the business"

5. toneTags — Short tone descriptors (3-6 tags)

6. neverDo — Hard prohibitions. Things this brand never says, never does. Pull from their "words to avoid" if given, and from inferring opposite of their voice. Examples:
   - "No exclamation marks"
   - "No emojis"
   - "No hashtags"
   - "Never 'unlock your potential', 'hustle', 'limited time'"

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
    const apiResp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 4096,
        system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
        messages: [{ role: 'user', content: userContent }]
      })
    });

    if (!apiResp.ok) {
      const errText = await apiResp.text();
      console.error('Anthropic API error:', apiResp.status, errText);
      if (apiResp.status === 429) return jsonResp({ error: 'AI service is busy. Try again.' }, 429);
      if (apiResp.status === 413) return jsonResp({ error: 'Brand docs too large. Remove a file and try again.' }, 413);
      return jsonResp({ error: 'AI Brain extraction failed.' }, 502);
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

    return jsonResp({ brain });
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
