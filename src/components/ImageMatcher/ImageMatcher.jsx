import { useCallback, useEffect, useRef, useState } from "react";
import "./ImageMatcher.css";

// "Database" = every image in src/assets/image-db (picked up automatically at build time)
const DB_FILES = import.meta.glob("../../assets/image-db/*.{jpg,jpeg,png,webp}", { eager: true, query: "?url", import: "default" });
const DB = Object.entries(DB_FILES).map(([path, url]) => ({ url, name: path.split("/").pop().replace(/\.[^.]+$/, "").replace(/[-_]/g, " ") }));

// Weights of each algorithm in the final score
const ALGOS = [
  { id: "phash", name: "Perceptual hash (DCT)", w: 0.3, hint: "Overall structure, survives resizing and compression" },
  { id: "dhash", name: "Difference hash", w: 0.2, hint: "Edge / gradient pattern" },
  { id: "ncc", name: "Pixel correlation", w: 0.25, hint: "Normalised cross-correlation of a 16×16 grayscale copy" },
  { id: "hist", name: "Colour histogram", w: 0.25, hint: "Overlap of the colour distribution (4×4×4 RGB bins)" },
];

// ---------- feature extraction ----------
const loadImg = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
const pixels = (img, w, h, flip) => {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d", { willReadFrequently: true });
  if (flip) { g.translate(w, 0); g.scale(-1, 1); }
  g.drawImage(img, 0, 0, w, h);
  return g.getImageData(0, 0, w, h).data;
};
const gray = (d, n) => { const o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]; return o; };

const COS = Array.from({ length: 8 }, (_, u) => Array.from({ length: 32 }, (_, x) => Math.cos(((2 * x + 1) * u * Math.PI) / 64)));
function phash(g32) { // 2-D DCT, keep the 8×8 low frequencies, threshold at the median
  const tmp = new Float32Array(8 * 32), co = [];
  for (let u = 0; u < 8; u++) for (let y = 0; y < 32; y++) { let s = 0; for (let x = 0; x < 32; x++) s += g32[y * 32 + x] * COS[u][x]; tmp[u * 32 + y] = s; }
  for (let v = 0; v < 8; v++) for (let u = 0; u < 8; u++) { let s = 0; for (let y = 0; y < 32; y++) s += tmp[u * 32 + y] * COS[v][y]; co.push(s); }
  const sorted = co.slice(1).sort((a, b) => a - b), med = sorted[Math.floor(sorted.length / 2)];
  return co.map((x) => (x > med ? 1 : 0));
}
function dhash(g) { // 9×8 grid: is the left pixel brighter than its right neighbour?
  const o = []; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) o.push(g[y * 9 + x] > g[y * 9 + x + 1] ? 1 : 0); return o;
}
function hist(d, n) { const h = new Float32Array(64); for (let i = 0; i < n; i++) h[(d[i * 4] >> 6) * 16 + (d[i * 4 + 1] >> 6) * 4 + (d[i * 4 + 2] >> 6)]++; return h.map((v) => v / n); }
const norm = (g) => { const m = g.reduce((s, x) => s + x, 0) / g.length; const s = Math.sqrt(g.reduce((a, x) => a + (x - m) ** 2, 0) / g.length) || 1; return g.map((x) => (x - m) / s); };

async function features(src, flip = false) {
  const img = await loadImg(src);
  const d32 = pixels(img, 32, 32, flip), d9 = pixels(img, 9, 8, flip), d16 = pixels(img, 16, 16, flip), d64 = pixels(img, 64, 64, flip);
  return { w: img.naturalWidth, h: img.naturalHeight, phash: phash(gray(d32, 1024)), dhash: dhash(gray(d9, 72)), ncc: norm(gray(d16, 256)), hist: hist(d64, 4096) };
}
const hamSim = (a, b) => 1 - a.reduce((s, x, i) => s + (x !== b[i] ? 1 : 0), 0) / a.length;
const compare = (q, f) => {
  const ncc = q.ncc.reduce((s, x, i) => s + x * f.ncc[i], 0) / q.ncc.length;
  const s = {
    phash: Math.max(0, (hamSim(q.phash, f.phash) - 0.5) * 2), dhash: Math.max(0, (hamSim(q.dhash, f.dhash) - 0.5) * 2), // 50% bits equal = unrelated
    ncc: Math.max(0, ncc), hist: q.hist.reduce((a, x, i) => a + Math.min(x, f.hist[i]), 0),
  };
  const total = ALGOS.reduce((a, x) => a + s[x.id] * x.w, 0);
  return { s, total };
};
const verdict = (t) => (t >= 0.85 ? ["Match found", "ok"] : t >= 0.6 ? ["Similar image", "mid"] : ["No match in database", "no"]);

// ---------- charts ----------
function Hist({ a, b }) {
  const bins = 16, group = (h) => Array.from({ length: bins }, (_, i) => h.slice(i * 4, i * 4 + 4).reduce((s, x) => s + x, 0));
  const A = group(a), B = group(b), max = Math.max(...A, ...B) || 1;
  return (
    <svg viewBox="0 0 320 90" className="im-hist" role="img" aria-label="Colour distribution of the uploaded and matched images">
      {A.map((v, i) => <rect key={"a" + i} x={i * 20 + 1} width="8" y={86 - (v / max) * 80} height={(v / max) * 80} fill="#4d8bff" />)}
      {B.map((v, i) => <rect key={"b" + i} x={i * 20 + 10} width="8" y={86 - (v / max) * 80} height={(v / max) * 80} fill="#ffd23f" />)}
    </svg>
  );
}

export default function ImageMatcher() {
  const [dbFeat, setDbFeat] = useState(null); // [{...db, f}]
  const [dbErr, setDbErr] = useState(0);
  const [q, setQ] = useState(null); // { url, name, f }
  const [res, setRes] = useState(null); // { ranked, ms }
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [err, setErr] = useState("");
  const fileRef = useRef(null);

  // Load + index the whole folder once
  useEffect(() => {
    let dead = false;
    (async () => {
      const out = []; let bad = 0;
      for (const d of DB) { try { out.push({ ...d, f: await features(d.url) }); } catch { bad++; } }
      if (!dead) { setDbFeat(out); setDbErr(bad); }
    })();
    return () => { dead = true; };
  }, []);

  const run = useCallback(async (url, name) => {
    if (!dbFeat?.length) return;
    setBusy(true); setErr("");
    try {
      const t0 = performance.now();
      const f = await features(url), ff = await features(url, true); // also try the mirrored upload
      const ranked = dbFeat.map((d) => {
        const a = compare(f, d.f), b = compare(ff, d.f);
        return b.total > a.total + 0.02 ? { ...d, ...b, mirrored: true } : { ...d, ...a };
      }).sort((a, b) => b.total - a.total);
      setQ({ url, name, f }); setRes({ ranked, ms: performance.now() - t0 });
    } catch { setErr("That file could not be read as an image."); }
    setBusy(false);
  }, [dbFeat]);

  const onFile = (file) => {
    if (!file) return;
    if (!/^image\//.test(file.type)) { setErr("Please choose an image file (jpg, png, webp)."); return; }
    const url = URL.createObjectURL(file);
    run(url, file.name);
  };

  // Demo: a brightened, mirrored, downsized copy of a random database image (shows robustness of the matcher)
  const sample = async () => {
    if (!dbFeat?.length) return;
    const src = dbFeat[Math.floor(Math.random() * dbFeat.length)];
    const img = await loadImg(src.url), c = document.createElement("canvas"), w = 200, h = Math.round((200 * img.naturalHeight) / img.naturalWidth);
    c.width = w; c.height = h; const g = c.getContext("2d"); g.filter = "brightness(1.15) contrast(.95)"; g.translate(w, 0); g.scale(-1, 1); g.drawImage(img, 0, 0, w, h);
    run(c.toDataURL("image/jpeg", 0.8), `sample of ${src.name} (mirrored, brighter)`);
  };

  const top = res?.ranked[0], second = res?.ranked[1];
  const [vText, vCls] = top ? verdict(top.total) : [];
  const pct = (x) => `${Math.round(x * 100)}%`;
  const [open, setOpen] = useState(false);

  return (
    <div className="im">
      <section className="im-card">
        <div className="im-head">
          <div><h2 className="im-h">Image database</h2>
            <p className="im-note">{dbFeat ? `${dbFeat.length} image${dbFeat.length === 1 ? "" : "s"} indexed from the folder src/assets/image-db/${dbErr ? ` (${dbErr} unreadable)` : ""}` : `Indexing ${DB.length} images…`}</p></div>
          <button type="button" className="im-ghost" onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Show"} database</button>
        </div>
        {open && <div className="im-db">{DB.map((d) => <figure key={d.url}><img src={d.url} alt={d.name} loading="lazy" /><figcaption>{d.name}</figcaption></figure>)}</div>}
        {DB.length === 0 && <p className="im-err">The folder is empty. Add images to src/assets/image-db/ and restart.</p>}
      </section>

      <section className="im-card">
        <h2 className="im-h">Upload the latest image</h2>
        <div className={"im-drop" + (drag ? " is-drag" : "")} onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files[0]); }}>
          {q ? <img src={q.url} alt="Uploaded" /> : <span>Drop an image here</span>}
          <div className="im-actions">
            <button type="button" className="im-go" disabled={!dbFeat || busy} onClick={() => fileRef.current.click()}>{busy ? "Matching…" : q ? "Upload another" : "Choose image"}</button>
            <button type="button" className="im-ghost" disabled={!dbFeat || busy} onClick={sample}>Try a modified DB image</button>
          </div>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => { onFile(e.target.files[0]); e.target.value = ""; }} />
        </div>
        {err && <p className="im-err" role="alert">{err}</p>}
      </section>

      {res && top && (
        <>
          <section className="im-card">
            <div className={"im-verdict is-" + vCls}>
              <b>{vText}</b><span>{pct(top.total)} overall similarity with “{top.name}”{top.mirrored ? " (the upload is a mirrored copy)" : ""}</span>
            </div>
            <div className="im-side">
              <figure><img src={q.url} alt="Uploaded" /><figcaption>Uploaded · {q.f.w}×{q.f.h}</figcaption></figure>
              <div className="im-vs">vs</div>
              <figure><img src={top.url} alt={top.name} /><figcaption>Best match · {top.name}<br />{top.f.w}×{top.f.h}</figcaption></figure>
            </div>
          </section>

          <section className="im-card">
            <h2 className="im-h">Analytical report</h2>
            <div className="im-algos">
              {ALGOS.map((a) => (
                <div key={a.id} className="im-algo" title={a.hint}>
                  <span>{a.name}<small>weight {Math.round(a.w * 100)}%</small></span>
                  <i><u style={{ width: pct(top.s[a.id]) }} /></i><b>{pct(top.s[a.id])}</b>
                </div>
              ))}
              <div className="im-algo im-total"><span>Weighted score</span><i><u style={{ width: pct(top.total) }} /></i><b>{pct(top.total)}</b></div>
            </div>
            <div className="im-facts">
              <div><small>Confidence margin</small><b>{second ? `${Math.round((top.total - second.total) * 100)} pts` : "–"}</b><span>{second ? `ahead of #2 “${second.name}”` : "only one image in database"}</span></div>
              <div><small>Images compared</small><b>{res.ranked.length}</b><span>{res.ms.toFixed(0)} ms</span></div>
              <div><small>Aspect ratio</small><b>{(q.f.w / q.f.h).toFixed(2)} / {(top.f.w / top.f.h).toFixed(2)}</b><span>uploaded / matched</span></div>
            </div>
            <h3 className="im-h2">Colour distribution <span><i className="b" /> uploaded <i className="y" /> matched</span></h3>
            <Hist a={q.f.hist} b={top.f.hist} />
          </section>

          <section className="im-card">
            <h2 className="im-h">Ranking</h2>
            <div className="im-scroll"><table className="im-table">
              <thead><tr><th>#</th><th>Image</th><th>Score</th>{ALGOS.map((a) => <th key={a.id}>{a.id}</th>)}</tr></thead>
              <tbody>{res.ranked.slice(0, 6).map((r, i) => (
                <tr key={r.url} className={i === 0 ? "is-top" : ""}>
                  <td>{i + 1}</td><td><span className="im-row"><img src={r.url} alt="" />{r.name}{r.mirrored ? " ⇋" : ""}</span></td><td><b>{pct(r.total)}</b></td>
                  {ALGOS.map((a) => <td key={a.id}>{pct(r.s[a.id])}</td>)}
                </tr>))}</tbody>
            </table></div>
            <p className="im-note">Score = {ALGOS.map((a) => `${Math.round(a.w * 100)}% ${a.id}`).join(" + ")}. 85%+ is reported as a match, 60–85% as similar. Hash scores are rescaled so unrelated images sit near 0%.</p>
          </section>
        </>
      )}
    </div>
  );
}
