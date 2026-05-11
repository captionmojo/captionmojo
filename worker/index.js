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
const RATE_LIMIT_GEN = { max: 12, windowMs: 60_000 };
const RATE_LIMIT_SEG = { max: 6, windowMs: 60_000 };

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
function buildSystemPrompt(bp) {
  const tones = (bp.tones || []).join(', ') || 'professional and approachable';
  const parts = [];
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
  parts.push('CAPTION RULES:');
  parts.push("- Write in the brand's voice exactly — not a generic AI approximation");
  parts.push('- Concise — typically 2-3 lines before the CTA, unless platform rules say otherwise');
  parts.push('- No hollow filler phrases or clichés ("In a world where...", "Let\'s be real...", etc.)');
  parts.push('- End with a natural CTA relevant to that platform');
  parts.push('- The audience segment should shape the angle and tone');
  parts.push('- Vary the captions — different angles, openings, lengths');
  parts.push('- Use brand documents (provided as attached files and inline text) as authoritative context about voice, product, customer, and example phrasing.');
  parts.push('');
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
        desc: String(x.desc).slice(0, 240)
      }))
      .slice(0, 10);
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
  const segment = clipString(body.segment, 200) || 'General';
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

  const totalInputSize = JSON.stringify(safeBp).length + direction.length + segment.length;
  if (totalInputSize > MAX_INPUT_CHARS) {
    return jsonResp({ error: 'Brand profile too long. Please shorten.' }, 400);
  }

  if (!env.ANTHROPIC_API_KEY) {
    return jsonResp({ error: 'Server not configured (missing API key)' }, 500);
  }

  const { docBlocks, textParts } = normalizeDocs(body.docs);

  const systemPrompt = buildSystemPrompt(safeBp);
  const variablePart = `Generate ${quantity} caption${quantity > 1 ? 's' : ''} for this audience segment: "${segment}".
Target platform${platforms.length > 1 ? 's' : ''}: ${platforms.join(', ')}.
${direction ? `Specific direction: ${direction}` : ''}

Return only the JSON array.`;

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
  if (!profile.trim()) {
    return jsonResp({ error: 'Profile required' }, 400);
  }

  if (!env.ANTHROPIC_API_KEY) {
    return jsonResp({ error: 'Server not configured' }, 500);
  }

  const pastedDocs = clipString(body.pastedDocs, MAX_PASTED_CHARS);
  const { docBlocks, textParts } = normalizeDocs(body.docs);

  const systemPrompt = `You are a senior marketing strategist. Given a brand profile (and any brand documents the business has provided), generate 5-8 specific audience segments this business should create social media content for.

These segments should be REAL, SPECIFIC slices of their audience — not generic buckets like "Young Adults" or "Customers." Think like a marketing director who knows the business. A restaurant might have "weekend brunch crowd," "private events inquiries," "weekday regulars." A fitness studio might have "new members," "competitive athletes," "over-40 wellness crowd."

Use the brand documents (PDFs, images, text) as authoritative context. If the docs reveal specific audience details — actual customer types, product lines, service categories — let those shape the segments.

Each segment must have:
- name: short and memorable, 3-5 words max
- desc: one sentence explaining who they are and what they care about

Return ONLY a valid JSON array. No preamble, no markdown, no code fences.
Format: [{"name":"Segment Name","desc":"One sentence describing this audience."}]`;

  const variablePart = `Brand profile:\n${profile}\n\nGenerate 5-8 specific audience segments for this business. Return only the JSON array.`;

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
    if (url.pathname.startsWith('/api/')) {
      return jsonResp({ error: 'Not found' }, 404);
    }

    return env.ASSETS.fetch(request);
  }
};
