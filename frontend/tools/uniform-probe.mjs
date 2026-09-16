import { chromium } from "playwright";
const b = await chromium.launch({executablePath:"/opt/pw-browsers/chromium",
  args:["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"]});
const p = await b.newPage({viewport:{width:1200,height:700}});
await p.addInitScript(() => {
  window.__u = {};
  const names = new WeakMap();
  const P = WebGLRenderingContext.prototype;
  const gul = P.getUniformLocation;
  P.getUniformLocation = function(prog, name) {
    const loc = gul.call(this, prog, name);
    if (loc) names.set(loc, name);
    return loc;
  };
  for (const fn of ["uniform1f","uniform1i","uniform2f","uniform3f","uniform3fv"]) {
    const orig = P[fn];
    P[fn] = function(loc, ...args) {
      const n = names.get(loc);
      if (n) window.__u[n] = args.length === 1 ? (ArrayBuffer.isView(args[0]) ? Array.from(args[0]).slice(0,12) : args[0]) : args;
      return orig.call(this, loc, ...args);
    };
  }
});
await p.goto("http://127.0.0.1:8000/", { waitUntil: "networkidle" });
await p.waitForTimeout(3000);
await p.setInputFiles('input[type=file]', "/tmp/annual_report.txt");
await p.waitForTimeout(6000);
const ta = p.locator("#question");
await ta.fill("What was the gross margin?");
await p.getByRole("button", { name: /launch geodesics/i }).click();
await p.waitForTimeout(6000);
const u = await p.evaluate(() => window.__u);
console.log("LIVE SHADER UNIFORMS:");
for (const k of Object.keys(u).sort()) {
  const v = u[k];
  console.log("  " + k.padEnd(18), Array.isArray(v) ? JSON.stringify(v.map(x=>typeof x==="number"?+x.toFixed(3):x)) : (typeof v==="number"? +v.toFixed(4):v));
}
await b.close();
