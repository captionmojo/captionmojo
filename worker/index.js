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

// ====== Rate limiting (in-memory, per Worker instance) ======
// Two separate buckets — generate is heavier so it gets stricter limits
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

  const systemPrompt = buildSystemPrompt(safeBp);
  const userPrompt = `Generate ${quantity} caption${quantity > 1 ? 's' : ''} for this audience segment: "${segment}".
Target platform${platforms.length > 1 ? 's' : ''}: ${platforms.join(', ')}.
${direction ? `Specific direction: ${direction}` : ''}

Return only the JSON array.`;

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
        system: systemPrompt,
        messages: [{ role: 'user', content: userPrompt }]
      })
    });

    if (!apiResp.ok) {
      const errText = await apiResp.text();
      console.error('Anthropic API error:', apiResp.status, errText);
      if (apiResp.status === 429) return jsonResp({ error: 'AI service is busy. Try again shortly.' }, 429);
      if (apiResp.status === 401 || apiResp.status === 403) return jsonResp({ error: 'Server auth error. Admin has been notified.' }, 500);
      return jsonResp({ error: 'AI generation failed. Try again in a moment.' }, 502);
    }

    const data = await apiResp.json();
    const text = (data.content || []).find(b => b.type === 'text')?.text || '';
    const captions = parseCaptionsJson(text);

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

  const systemPrompt = `You are a senior marketing strategist. Given a brand profile, generate 5-8 specific audience segments this business should create social media content for.

These segments should be REAL, SPECIFIC slices of their audience — not generic buckets like "Young Adults" or "Customers." Think like a marketing director who knows the business. A restaurant might have "weekend brunch crowd," "private events inquiries," "weekday regulars." A fitness studio might have "new members," "competitive athletes," "over-40 wellness crowd."

Each segment must have:
- name: short and memorable, 3-5 words max
- desc: one sentence explaining who they are and what they care about

Return ONLY a valid JSON array. No preamble, no markdown, no code fences.
Format: [{"name":"Segment Name","desc":"One sentence describing this audience."}]`;

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
        system: systemPrompt,
        messages: [{
          role: 'user',
          content: `Brand profile:\n${profile}\n\nGenerate 5-8 specific audience segments for this business. Return only the JSON array.`
        }]
      })
    });

    if (!apiResp.ok) {
      const errText = await apiResp.text();
      console.error('Anthropic API error:', apiResp.status, errText);
      if (apiResp.status === 429) return jsonResp({ error: 'AI service is busy. Try again.' }, 429);
      return jsonResp({ error: 'AI generation failed.' }, 502);
    }

    const data = await apiResp.json();
    const text = (data.content || []).find(b => b.type === 'text')?.text || '';
    const segments = parseSegmentsJson(text);

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

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    // API routing
    if (url.pathname === '/api/generate' && request.method === 'POST') {
      return handleGenerate(request, env);
    }
    if (url.pathname === '/api/segments' && request.method === 'POST') {
      return handleSegments(request, env);
    }
    if (url.pathname.startsWith('/api/')) {
      return jsonResp({ error: 'Not found' }, 404);
    }

    // Anything else falls through to static asset handler (configured in wrangler.jsonc)
    return env.ASSETS.fetch(request);
  }
};
