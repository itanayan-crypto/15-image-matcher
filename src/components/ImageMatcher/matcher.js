// Image matching engine: pure browser code, koi library nahi chahiye.
// Pipeline: resize -> grayscale -> 4 alag similarity measures -> weighted score (0 to 1)

export const METRICS = [
  { id: "phash", label: "Perceptual hash (pHash)", weight: 0.3, hint: "Low-frequency structure. Survives resize, compression and small colour shifts." },
  { id: "dhash", label: "Difference hash (dHash)", weight: 0.2, hint: "Left-to-right brightness gradients across the picture." },
  { id: "color", label: "Colour histogram", weight: 0.2, hint: "Overall colour mix (64 bins), compared by histogram intersection." },
  { id: "ssim", label: "Structural similarity (SSIM)", weight: 0.3, hint: "Block-wise luminance, contrast and structure on a 32x32 grey copy." },
];

export const verdictOf = (score) =>
  score >= 0.85 ? { id: "strong", label: "Strong match" }
  : score >= 0.65 ? { id: "probable", label: "Probable match" }
  : score >= 0.4 ? { id: "weak", label: "Weak similarity" }
  : { id: "none", label: "No match" };

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
const sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Could not load image"));
    im.src = src;
  });
}

// Image ko w x h canvas par stretch karke draw karta hai (white background, taaki transparent PNG bhi theek chale)
function draw(src, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const x = c.getContext("2d", { willReadFrequently: true });
  x.imageSmoothingEnabled = true;
  x.imageSmoothingQuality = "high";
  x.fillStyle = "#fff";
  x.fillRect(0, 0, w, h);
  x.drawImage(src, 0, 0, w, h);
  return c;
}
const pixels = (c) => c.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, c.width, c.height).data;
const gray = (rgba) => {
  const g = new Float32Array(rgba.length / 4);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
  return g;
};

// dHash: 9x8 grey image, har pixel ko apne right wale se compare karo -> 64 bits
function dHash(g) {
  const bits = new Uint8Array(64);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) bits[y * 8 + x] = g[y * 9 + x] > g[y * 9 + x + 1] ? 1 : 0;
  return bits;
}

// pHash: 32x32 grey image ka DCT, sirf top-left 8x8 low frequencies, median se upar = 1 (DC chhod ke -> 63 bits)
const COS = Array.from({ length: 8 }, (_, u) => Float32Array.from({ length: 32 }, (_, x) => Math.cos(((2 * x + 1) * u * Math.PI) / 64)));
function pHash(g) {
  const tmp = new Float32Array(32 * 8);
  for (let y = 0; y < 32; y++) for (let u = 0; u < 8; u++) {
    let s = 0;
    for (let x = 0; x < 32; x++) s += g[y * 32 + x] * COS[u][x];
    tmp[y * 8 + u] = s;
  }
  const coef = [];
  for (let v = 0; v < 8; v++) for (let u = 0; u < 8; u++) {
    let s = 0;
    for (let y = 0; y < 32; y++) s += tmp[y * 8 + u] * COS[v][y];
    coef.push(s);
  }
  const ac = coef.slice(1);
  const med = [...ac].sort((a, b) => a - b)[Math.floor(ac.length / 2)];
  return Uint8Array.from(ac, (c) => (c > med ? 1 : 0));
}

// 4 bins per channel = 64 bin colour histogram (normalised)
function histogram(rgba) {
  const h = new Float32Array(64);
  const n = rgba.length / 4;
  for (let i = 0; i < n; i++) h[(rgba[i * 4] >> 6) * 16 + (rgba[i * 4 + 1] >> 6) * 4 + (rgba[i * 4 + 2] >> 6)]++;
  for (let i = 0; i < 64; i++) h[i] /= n;
  return h;
}

const hamming = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++; return d; };

// SSIM: 32x32 grey ko 8x8 blocks mein baanto, har block ka SSIM nikalo, average lo
function ssim(a, b) {
  const C1 = 6.5025, C2 = 58.5225; // (0.01*255)^2, (0.03*255)^2
  let sum = 0, n = 0;
  for (let by = 0; by < 32; by += 8) for (let bx = 0; bx < 32; bx += 8) {
    const A = [], B = [];
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { const i = (by + y) * 32 + bx + x; A.push(a[i]); B.push(b[i]); }
    const ma = mean(A), mb = mean(B);
    let va = 0, vb = 0, cov = 0;
    for (let i = 0; i < 64; i++) { va += (A[i] - ma) ** 2; vb += (B[i] - mb) ** 2; cov += (A[i] - ma) * (B[i] - mb); }
    va /= 63; vb /= 63; cov /= 63;
    sum += ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2));
    n++;
  }
  return sum / n;
}

export function extractFeatures(img) {
  const base = draw(img, 128, 128); // pehle 128 px par laao taaki bade photo ka aliasing na ho
  const g32 = gray(pixels(draw(base, 32, 32)));
  const g98 = gray(pixels(draw(base, 9, 8)));
  const rgb = pixels(draw(base, 64, 64));
  return {
    w: img.naturalWidth,
    h: img.naturalHeight,
    g32,
    dhash: dHash(g98),
    phash: pHash(g32),
    hist: histogram(rgb),
    brightness: mean(g32),
    contrast: sd(g32),
  };
}

// Random hash ka similarity ~50% hota hai, isliye 50% ko 0 aur 100% ko 1 maan ke normalise kiya hai
const hashScore = (dist, bits) => clamp01((1 - dist / bits - 0.5) * 2);

export function compare(q, d) {
  const hp = hamming(q.phash, d.phash);
  const hd = hamming(q.dhash, d.dhash);
  let inter = 0;
  for (let i = 0; i < 64; i++) inter += Math.min(q.hist[i], d.hist[i]);
  const rawSsim = ssim(q.g32, d.g32);
  const parts = { phash: hashScore(hp, q.phash.length), dhash: hashScore(hd, q.dhash.length), color: clamp01(inter), ssim: clamp01(rawSsim) };
  const score = METRICS.reduce((s, m) => s + parts[m.id] * m.weight, 0);
  return { parts, score, pDist: hp, dDist: hd, pBits: q.phash.length, dBits: q.dhash.length, rawSsim };
}

// Poore DB ke scores ka statistics + confidence
export function summarise(sorted) {
  const scores = sorted.map((r) => r.score);
  const m = mean(scores);
  const s = sd(scores);
  const best = scores[0] ?? 0;
  const second = scores[1] ?? 0;
  const margin = best - second;
  const others = scores.slice(1);
  const z = others.length > 1 && sd(others) > 0.0001 ? (best - mean(others)) / sd(others) : null;
  const med = [...scores].sort((a, b) => a - b)[Math.floor(scores.length / 2)] ?? 0;
  let confidence = "Low";
  if (best >= 0.85 && margin >= 0.1) confidence = "High";
  else if (best >= 0.65 && margin >= 0.05) confidence = "Medium";
  return { mean: m, sd: s, median: med, best, second, margin, z, confidence };
}
