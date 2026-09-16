import { chromium } from "playwright";
const FILE = process.argv[2] || "/tmp/ops.txt";
const TAG  = process.argv[3] || "ui";
const b = await chromium.launch({
  executablePath: "/opt/pw-browsers/chromium",
  args: ["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader","--ignore-gpu-blocklist"],
});
const p = await b.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
const errs = [];
p.on("console", m => { if (m.type() === "error") errs.push(m.text()); });
p.on("pageerror", e => errs.push("PAGEERROR: " + e.message));

await p.goto("http://127.0.0.1:8000/", { waitUntil: "networkidle" });
await p.waitForTimeout(4000);
await p.screenshot({ path: `/tmp/${TAG}-01-empty.png` });

await p.setInputFiles('input[type=file]', FILE);
await p.waitForTimeout(6000);
await p.screenshot({ path: `/tmp/${TAG}-02-loaded.png` });

const ta = p.locator("#question");
if (await ta.count()) {
  await ta.fill("What was the gross margin?");
  await p.getByRole("button", { name: /launch geodesics/i }).click();
  await p.waitForTimeout(6000);
  await p.screenshot({ path: `/tmp/${TAG}-03-answer.png` });
}
await p.keyboard.press("p");
await p.waitForTimeout(800);
await p.screenshot({ path: `/tmp/${TAG}-04-params.png` });

const t = await p.evaluate(() =>
  [...document.querySelectorAll("span.u-label")].map(e => {
    const v = e.nextElementSibling;
    return v ? `${e.textContent} = ${v.textContent}` : null;
  }).filter(Boolean));
console.log("TELEMETRY:"); t.forEach(r => console.log("  " + r));
console.log("\nCONSOLE ERRORS:", errs.length ? errs.join("\n  ") : "none");
await b.close();
