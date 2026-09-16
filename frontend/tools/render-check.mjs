/**
 * Headless shader verification.
 *
 * Compiles schwarzschild.frag.glsl in a real WebGL context and renders a
 * set of representative states to PNG so the physics can be inspected
 * rather than assumed. Run: node tools/render-check.mjs
 */
import { chromium } from "playwright";
import { readFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const frag = readFileSync(join(root, "src/sim/schwarzschild.frag.glsl"), "utf8");
const vert = readFileSync(join(root, "src/sim/schwarzschild.vert.glsl"), "utf8");

const W = 960;
const H = 540;

// Camera math mirrored from src/sim/camera.ts.
function camera(distance, inclination, azimuth) {
  const p = [
    distance * Math.cos(inclination) * Math.cos(azimuth),
    distance * Math.sin(inclination),
    distance * Math.cos(inclination) * Math.sin(azimuth),
  ];
  const len = Math.hypot(...p) || 1;
  const f = p.map((v) => v / len);
  let up = Math.abs(f[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
  let r = [
    up[1] * f[2] - up[2] * f[1],
    up[2] * f[0] - up[0] * f[2],
    up[0] * f[1] - up[1] * f[0],
  ];
  const rl = Math.hypot(...r) || 1;
  r = r.map((v) => v / rl);
  const t = [
    f[1] * r[2] - f[2] * r[1],
    f[2] * r[0] - f[0] * r[2],
    f[0] * r[1] - f[1] * r[0],
  ];
  return { pos: p, basis: [...r, ...t, ...f] };
}

const CASES = [
  {
    name: "01-poster-strong",
    note: "Hero framing, dense index, strong grounding, four citations",
    cam: camera(24, 0.29, 0),
    u: {
      uRs: 1.4, uDiskOuter: 13, uDiskLuminosity: 0.9, uDiskIntegrity: 1,
      uCollapse: 1, uGrounding: 1, uAbstained: 0, uTraceProgress: 1,
      uHotspotCount: 4,
      uHotspots: [0.1, 0.9, 0, 0.35, 0.7, 0, 0.6, 0.55, 0, 0.85, 0.42, 0],
      uRedshift: 0, uSteps: 460, uDiskSamples: 3, uParticleSeed: 12.5,
    },
  },
  {
    name: "02-edge-on",
    note: "Edge-on preset — tests that the secondary lensed image appears",
    cam: camera(26, 0.045, 0.6),
    u: {
      uRs: 1.4, uDiskOuter: 13, uDiskLuminosity: 0.85, uDiskIntegrity: 1,
      uCollapse: 1, uGrounding: 0.66, uAbstained: 0, uTraceProgress: 0,
      uHotspotCount: 0, uHotspots: new Array(24).fill(0),
      uRedshift: 0, uSteps: 460, uDiskSamples: 3, uParticleSeed: 3.1,
    },
  },
  {
    name: "03-no-document",
    note: "Empty state: no index, so no disk. Lensed starfield + shadow only",
    cam: camera(24, 0.29, 0),
    u: {
      uRs: 1.0, uDiskOuter: 6, uDiskLuminosity: 0, uDiskIntegrity: 1,
      uCollapse: 0, uGrounding: 0, uAbstained: 0, uTraceProgress: 0,
      uHotspotCount: 0, uHotspots: new Array(24).fill(0),
      uRedshift: 0, uSteps: 460, uDiskSamples: 2, uParticleSeed: 0,
    },
  },
  {
    name: "04-collapse-midway",
    note: "Ingestion at 45% — disk ignites from the inside out",
    cam: camera(24, 0.29, 0),
    u: {
      uRs: 1.4, uDiskOuter: 13, uDiskLuminosity: 0.7, uDiskIntegrity: 1,
      uCollapse: 0.45, uGrounding: 0.33, uAbstained: 0, uTraceProgress: 0,
      uHotspotCount: 0, uHotspots: new Array(24).fill(0),
      uRedshift: 0, uSteps: 460, uDiskSamples: 2, uParticleSeed: 7.7,
    },
  },
  {
    name: "05-abstained",
    note: "SIGNAL LOST — abstained. Disk goes dark; geometry remains",
    cam: camera(24, 0.29, 0),
    u: {
      uRs: 1.4, uDiskOuter: 13, uDiskLuminosity: 0.9, uDiskIntegrity: 1,
      uCollapse: 1, uGrounding: 0, uAbstained: 1, uTraceProgress: 0,
      uHotspotCount: 0, uHotspots: new Array(24).fill(0),
      uRedshift: 0, uSteps: 460, uDiskSamples: 2, uParticleSeed: 7.7,
    },
  },
  {
    name: "06-low-integrity",
    note: "40% of pages low quality — disk should be visibly holed",
    cam: camera(24, 0.29, 0),
    u: {
      uRs: 1.4, uDiskOuter: 13, uDiskLuminosity: 0.9, uDiskIntegrity: 0.6,
      uCollapse: 1, uGrounding: 0.66, uAbstained: 0, uTraceProgress: 0,
      uHotspotCount: 0, uHotspots: new Array(24).fill(0),
      uRedshift: 0, uSteps: 460, uDiskSamples: 2, uParticleSeed: 21.3,
    },
  },
  {
    name: "07-redshift",
    note: "Frequency shift applied (translate panel)",
    cam: camera(24, 0.29, 0),
    u: {
      uRs: 1.4, uDiskOuter: 13, uDiskLuminosity: 0.9, uDiskIntegrity: 1,
      uCollapse: 1, uGrounding: 1, uAbstained: 0, uTraceProgress: 0,
      uHotspotCount: 0, uHotspots: new Array(24).fill(0),
      uRedshift: -0.8, uSteps: 460, uDiskSamples: 2, uParticleSeed: 12.5,
    },
  },
  {
    name: "08-minimal-quality",
    note: "Lowest tier, 110 steps — must still show a coherent ring",
    cam: camera(24, 0.29, 0),
    u: {
      uRs: 1.4, uDiskOuter: 13, uDiskLuminosity: 0.9, uDiskIntegrity: 1,
      uCollapse: 1, uGrounding: 1, uAbstained: 0, uTraceProgress: 0,
      uHotspotCount: 0, uHotspots: new Array(24).fill(0),
      uRedshift: 0, uSteps: 110, uDiskSamples: 1, uParticleSeed: 12.5,
    },
  },
];

const page_html = `<!doctype html><html><body style="margin:0;background:#000">
<canvas id="c" width="${W}" height="${H}"></canvas></body></html>`;

const browser = await chromium.launch({
  executablePath: "/opt/pw-browsers/chromium",
  args: [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
  ],
});
const page = await browser.newPage({ viewport: { width: W, height: H } });
await page.setContent(page_html);

const results = await page.evaluate(
  async ({ vert, frag, cases, W, H }) => {
    const canvas = document.getElementById("c");
    const gl = canvas.getContext("webgl", { antialias: false, alpha: false, preserveDrawingBuffer: true });
    if (!gl) return { error: "no webgl context" };

    function sh(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        return { err: gl.getShaderInfoLog(s) };
      }
      return { s };
    }
    const v = sh(gl.VERTEX_SHADER, vert);
    if (v.err) return { error: "vertex: " + v.err };
    const f = sh(gl.FRAGMENT_SHADER, frag);
    if (f.err) return { error: "fragment: " + f.err };

    const prog = gl.createProgram();
    gl.attachShader(prog, v.s);
    gl.attachShader(prog, f.s);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      return { error: "link: " + gl.getProgramInfoLog(prog) };
    }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "aPosition");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.viewport(0, 0, W, H);

    const U = (n) => gl.getUniformLocation(prog, n);
    const out = [];

    for (const c of cases) {
      gl.uniform2f(U("uResolution"), W, H);
      gl.uniform1f(U("uTime"), 3.0);
      gl.uniform3f(U("uCamPos"), ...c.cam.pos);
      gl.uniformMatrix3fv(U("uCamBasis"), false, new Float32Array(c.cam.basis));
      gl.uniform1f(U("uFov"), 0.9);
      gl.uniform1i(U("uSteps"), c.u.uSteps);
      gl.uniform1i(U("uDiskSamples"), c.u.uDiskSamples);
      gl.uniform1f(U("uRs"), c.u.uRs);
      gl.uniform1f(U("uDiskInner"), c.u.uRs * 3.0);
      gl.uniform1f(U("uDiskOuter"), c.u.uDiskOuter);
      gl.uniform1f(U("uDiskLuminosity"), c.u.uDiskLuminosity);
      gl.uniform1f(U("uDiskIntegrity"), c.u.uDiskIntegrity);
      gl.uniform1f(U("uParticleSeed"), c.u.uParticleSeed);
      gl.uniform1f(U("uCollapse"), c.u.uCollapse);
      gl.uniform1f(U("uGrounding"), c.u.uGrounding);
      gl.uniform1f(U("uAbstained"), c.u.uAbstained);
      gl.uniform1f(U("uTraceProgress"), c.u.uTraceProgress);
      gl.uniform1i(U("uHotspotCount"), c.u.uHotspotCount);
      gl.uniform3fv(U("uHotspots[0]"), new Float32Array(c.u.uHotspots));
      gl.uniform1f(U("uRedshift"), c.u.uRedshift);

      const t0 = performance.now();
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.finish();
      const ms = performance.now() - t0;

      // Measure the image so the check is quantitative, not just a picture.
      const px = new Uint8Array(W * H * 4);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
      let lit = 0;
      let hot = 0;
      let sum = 0;
      for (let i = 0; i < px.length; i += 4) {
        const l = (px[i] + px[i + 1] + px[i + 2]) / 3;
        sum += l;
        if (l > 12) lit++;
        if (l > 200) hot++;
      }
      const n = W * H;
      out.push({
        name: c.name,
        note: c.note,
        ms: Math.round(ms),
        litPct: +((lit / n) * 100).toFixed(2),
        hotPct: +((hot / n) * 100).toFixed(3),
        meanLuma: +(sum / n).toFixed(2),
        dataUrl: canvas.toDataURL("image/png"),
      });
    }
    return { out };
  },
  { vert, frag, cases: CASES, W, H }
);

await browser.close();

if (results.error) {
  console.error("SHADER FAILED:", results.error);
  process.exit(1);
}

mkdirSync(join(root, "tools/render-check"), { recursive: true });
console.log("case                    ms    lit%   hot%   meanLuma  note");
for (const r of results.out) {
  const b64 = r.dataUrl.split(",")[1];
  const { writeFileSync } = await import("node:fs");
  writeFileSync(join(root, "tools/render-check", `${r.name}.png`), Buffer.from(b64, "base64"));
  console.log(
    `${r.name.padEnd(22)} ${String(r.ms).padStart(5)} ${String(r.litPct).padStart(6)} ${String(r.hotPct).padStart(6)} ${String(r.meanLuma).padStart(9)}  ${r.note}`
  );
}
