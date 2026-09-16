// ============================================================================
// GARGANTUA — Schwarzschild null-geodesic raytracer
// ----------------------------------------------------------------------------
// A full-screen fragment shader. For every pixel we launch one photon
// backwards from the observer's eye and integrate its path through the
// curved spacetime around a non-rotating (Schwarzschild) black hole until
// it either falls through the horizon, strikes the accretion disk, or
// escapes to infinity.
//
// The document is the mass. Everything the shader draws is driven by real
// RAG state through the uniforms below — see src/sim/mapping.ts for the
// table that produces them. Nothing here animates on a timer that isn't
// either wall-clock camera motion or a value the backend reported.
//
// PHYSICS NOTE ON THE INTEGRATION METHOD
// --------------------------------------
// The exact null geodesic in Schwarzschild geometry, in the orbital plane
// of the photon, obeys the Binet-form orbit equation
//
//     d²u/dφ² + u = 3 M u²                                            (1)
//
// where u = 1/r and M is the geometrised mass (r_s = 2M). The right-hand
// term is the entire general-relativistic correction: drop it and you get
// d²u/dφ² + u = 0, whose solution is a straight line. Keeping it is what
// bends the light.
//
// Integrating (1) directly in (u, φ) is both cheaper and better-behaved
// than marching a 3D position with a small step, because the photon's
// motion is planar: the plane is fixed by the initial position and
// direction, so we solve a 2D ODE and lift the result back into 3D. This
// avoids the classic artefact where a naively-marched ray tunnels through
// the photon sphere near r = 1.5 r_s and produces a hard black disc with
// no ring.
//
// We use a leapfrog/velocity-Verlet step in φ, which is symplectic and so
// conserves the orbit's shape over long integrations far better than
// forward Euler at the same step count. That matters: at grazing impact
// parameters a photon can wind several times around the hole, and Euler
// visibly spirals it inward.
// ============================================================================

precision highp float;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

uniform vec2  uResolution;      // backing-store size in device pixels
uniform float uTime;            // seconds since scene mount (camera motion only)

// --- Camera (measured, and reported verbatim in the HUD) -------------------
uniform vec3  uCamPos;          // observer position, units of r_s
uniform mat3  uCamBasis;        // right / up / forward
uniform float uFov;             // vertical field of view, radians

// --- Quality (from AdaptiveQuality; GEODESIC STEPS is exactly uSteps) ------
uniform int   uSteps;           // integration steps per geodesic
uniform int   uDiskSamples;     // lensing passes across the disk

// --- The mass: the document ------------------------------------------------
uniform float uRs;              // Schwarzschild radius, scene units.
                                // From document size. MassState.schwarzschildRadius.

// --- The disk: the vector index -------------------------------------------
uniform float uDiskInner;       // ISCO, 3 r_s for Schwarzschild
uniform float uDiskOuter;       // outer edge; grows with chunk count
uniform float uDiskLuminosity;  // 0..1 from index density. 0 => dark disk.
uniform float uDiskIntegrity;   // 0..1 from page extraction quality. Low
                                // integrity mottles the disk with gaps —
                                // an honest visual for a document we could
                                // only partially read.
uniform float uParticleSeed;    // stable per-document seed (hashed doc id)

// --- Collapse: the ingestion cinematic ------------------------------------
uniform float uCollapse;        // 0..1, from the REAL job progress. At 0 the
                                // disk has not ignited; at 1 the index is
                                // crystallised. Never advanced by a timer.

// --- Signal: the current answer -------------------------------------------
uniform float uGrounding;       // 0 / .33 / .66 / 1 — the grounding ramp.
                                // This is the ONLY uniform that may tint
                                // the disk's core colour.
uniform float uAbstained;       // 1 => SIGNAL LOST. Disk goes dark.
uniform float uTraceProgress;   // 0..1 geodesic-trace animation for a query
uniform int   uHotspotCount;    // === citations.length (max 8)
uniform vec3  uHotspots[8];     // (theta 0..1, intensity 0..1, unused)

// --- Frequency shift: the translate control -------------------------------
uniform float uRedshift;        // -1..1. 0 = source language. Negative
                                // blueshifts, positive redshifts. Applied
                                // to the disk's emitted spectrum only.

varying vec2 vUv;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const float PI       = 3.14159265359;
// Escape threshold. Deliberately not smaller: below about 1e-3 the
// integration variables have lost enough precision that the asymptotic
// direction becomes noise. r = 1000 r_s is asymptotic for our purposes.
const float ESCAPE_U = 1e-3;
const int   MAX_STEPS = 512;   // hard loop bound; WebGL1 needs a constant

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

float hash21(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// Value noise, used for the disk's turbulent banding.
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 w = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 4; i++) {
    v += a * noise(p);
    p *= 2.03;
    a *= 0.5;
  }
  return v;
}

// ---------------------------------------------------------------------------
// Blackbody-ish emission ramp
// ---------------------------------------------------------------------------
// Not a physically calibrated Planck curve — a hand-tuned ramp that lands
// on the reference's measured palette: deep amber -> orange -> white-hot.
// `t` is normalised temperature 0..1.
vec3 emissionRamp(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 cold = vec3(0.141, 0.082, 0.020);  // #241505  grounding-none
  vec3 dim  = vec3(0.541, 0.361, 0.122);  // #8a5c1f  grounding-weak
  vec3 mid  = vec3(1.000, 0.549, 0.165);  // #ff8c2a  grounding-moderate
  vec3 hot  = vec3(1.000, 0.953, 0.839);  // #fff3d6  grounding-strong
  // Blend points tuned against the reference: the disk should read amber
  // through its outer half, orange through most of the inner half, and
  // white-hot only in a thin band at the ISCO. Earlier blend points put
  // the whole disk in the gold between orange and white, which is the
  // single most common way this render goes wrong.
  vec3 c = mix(cold, dim, smoothstep(0.0, 0.22, t));
  c = mix(c, mid, smoothstep(0.30, 0.70, t));
  c = mix(c, hot, smoothstep(0.84, 1.0, t));
  return c;
}

// Relativistic frequency shift applied to an emitted colour. Positive
// `z` reddens (recession / climbing out of the well), negative blueshifts.
// Drives the Translate panel's "frequency shift" control.
vec3 applyShift(vec3 c, float z) {
  float w = clamp(z, -1.0, 1.0);
  // Blueshift lifts the blue channel and suppresses red; redshift the
  // reverse. Luminance is roughly preserved so the control reads as a
  // colour change, not a brightness change.
  vec3 shifted = vec3(
    c.r * (1.0 + 0.45 * w),
    c.g * (1.0 - 0.10 * abs(w)),
    c.b * (1.0 - 0.55 * w)
  );
  float lum0 = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float lum1 = max(dot(shifted, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
  return shifted * (lum0 / lum1);
}

// ---------------------------------------------------------------------------
// Starfield — the background the lensing distorts
// ---------------------------------------------------------------------------
vec3 starfield(vec3 dir) {
  // Project the escaped ray direction onto a lat/long grid and scatter
  // point sources. Sampling in direction space (not screen space) is what
  // makes the Einstein ring work: strongly-bent rays sample the sky from
  // behind the hole, so stars appear duplicated around the shadow.
  vec2 uv = vec2(atan(dir.z, dir.x) / (2.0 * PI) + 0.5, acos(clamp(dir.y, -1.0, 1.0)) / PI);
  vec3 col = vec3(0.0);
  for (int layer = 0; layer < 3; layer++) {
    float scale = 180.0 * pow(2.0, float(layer));
    // uv.x spans 2pi and uv.y spans pi, so the grid must be
    // stretched 2:1 or every star renders as a horizontal dash.
    vec2 g = vec2(uv.x * 2.0, uv.y) * scale;
    vec2 cell = floor(g);
    float h = hash21(cell + float(layer) * 71.3);
    if (h > 0.9955) {
      vec2 pos = fract(g) - 0.5;
      float d = length(pos);
      float bright = (h - 0.9955) / 0.0045;
      float s = smoothstep(0.28, 0.0, d) * bright;
      // Faint colour variation so the field isn't a grid of identical dots
      vec3 tint = mix(vec3(0.72, 0.80, 1.0), vec3(1.0, 0.92, 0.78), hash11(h * 41.0));
      col += tint * s * 0.55;
    }
  }
  return col;
}

// ---------------------------------------------------------------------------
// Accretion disk sample
// ---------------------------------------------------------------------------
// Called when a geodesic crosses the equatorial plane between the inner
// and outer radius. `r` is in units of r_s, `phi` the azimuth.
vec3 sampleDisk(float r, float phi) {
  // --- Radial temperature profile ---------------------------------------
  // Shakura-Sunyaev thin disk: T ∝ r^(-3/4), with the inner-edge cutoff
  // factor (1 - sqrt(r_in/r))^(1/4) that takes emission to zero at the
  // ISCO instead of diverging.
  float x = r / uDiskInner;
  float cutoff = pow(max(1.0 - sqrt(1.0 / max(x, 1.0001)), 0.0), 0.25);
  float temp = pow(max(x, 1.0), -0.75) * cutoff;
  temp = clamp(temp * 1.05, 0.0, 1.0);

  // --- The disk is the index --------------------------------------------
  // Luminosity comes from index density. A sparse index makes a dim disk.
  // With uDiskLuminosity == 0 (nothing indexed) the disk does not glow at
  // all — which is the honest reading of "there is nothing to retrieve
  // from", not a decorative default.
  temp *= mix(0.15, 1.0, uDiskLuminosity);

  // --- Grounding drives colour, exclusively ------------------------------
  // uGrounding is the only term allowed to move the disk up the ramp.
  temp = mix(temp * 0.55, temp, 0.35 + 0.65 * uGrounding);

  // --- Turbulent banding: individual chunks in orbit ---------------------
  // The band frequency scales with the seed so two different documents
  // have visibly different disk structure, and it is stable across frames
  // for the same document.
  float orbitalPhase = phi + uTime * 0.06 / pow(max(r, 0.5), 1.5);
  // Higher radial frequency gives the fine concentric striation the
  // reference shows, rather than a smooth wash.
  float bands = fbm(vec2(orbitalPhase * 5.0, r * 7.5 + uParticleSeed));
  temp *= 0.70 + 0.55 * bands;

  // --- Data integrity ----------------------------------------------------
  // Pages we could only read badly leave gaps in the disk. This is not
  // decoration: uDiskIntegrity comes from PageInfo.is_low_quality, and a
  // document with a third of its pages flagged has a visibly holed disk.
  if (uDiskIntegrity < 0.999) {
    float gap = step(uDiskIntegrity, hash11(floor(orbitalPhase * 9.0) + uParticleSeed));
    temp *= 1.0 - 0.75 * gap;
  }

  // --- The collapse sequence --------------------------------------------
  // During ingestion the disk ignites from the inside out, tracking real
  // job progress. uCollapse is never advanced by a timer.
  // The ignition front sweeps outward as the index builds. It must
  // OVERSHOOT the outer edge at uCollapse == 1, otherwise a fully indexed
  // document still renders with its outer disk dark — the front stops
  // exactly at the rim and the smoothstep never resolves to 1 there. The
  // 1.35 factor is what makes "collapse complete" mean "disk fully lit".
  float radialFrac = (r - uDiskInner) / max(uDiskOuter - uDiskInner, 0.001);
  float ignited = smoothstep(0.0, 1.0, (uCollapse * 1.35 - radialFrac) * 4.0);
  temp *= ignited;

  vec3 col = emissionRamp(temp) * temp * 1.15;

  // --- Citation hot spots ------------------------------------------------
  // The surviving rays that struck the disk. Each is positioned by its
  // citation's page number and heated by its real relevance_score.
  for (int i = 0; i < 8; i++) {
    if (i >= uHotspotCount) break;
    float theta = uHotspots[i].x * 2.0 * PI;
    float intensity = uHotspots[i].y;
    float angular = abs(mod(phi - theta + PI, 2.0 * PI) - PI);
    // Hot spots sit mid-disk, radially, so they read as points on the ring
    float radial = abs(r - mix(uDiskInner * 1.4, uDiskOuter * 0.75, hash11(float(i) + uParticleSeed)));
    float spot = exp(-angular * angular * 90.0) * exp(-radial * radial * 9.0);
    // The trace animation reveals spots as the rays arrive.
    float arrived = smoothstep(float(i) / 8.0, float(i) / 8.0 + 0.25, uTraceProgress);
    col += emissionRamp(0.88 + 0.12 * intensity) * spot * intensity * 2.4 * arrived;
  }

  // --- Abstention --------------------------------------------------------
  // Not an error state. The disk going dark IS the system reporting,
  // correctly, that the document does not support the query. It is
  // rendered as a deliberate, composed state rather than a fault.
  col *= mix(1.0, 0.06, uAbstained);

  // --- Frequency shift (Translate) ---------------------------------------
  col = applyShift(col, uRedshift);

  return col;
}

// ---------------------------------------------------------------------------
// The geodesic integrator
// ---------------------------------------------------------------------------
// Integrates equation (1) for one photon and returns the accumulated
// colour. `ro` is the observer position and `rd` the initial direction,
// both in units where the horizon sits at r = uRs.
vec3 traceGeodesic(vec3 ro, vec3 rd) {
  vec3 accum = vec3(0.0);

  // --- Set up the orbital plane ------------------------------------------
  // The photon's motion is confined to the plane spanned by its position
  // and direction vectors. We build an orthonormal basis (e1, e2) in that
  // plane and integrate a 2D problem.
  float r0 = length(ro);
  vec3 e1 = ro / r0;
  vec3 normal = cross(e1, rd);
  float nlen = length(normal);
  if (nlen < 1e-6) {
    // Radial ray: no angular momentum, falls straight in. No lensing to do.
    return vec3(0.0);
  }
  normal /= nlen;
  vec3 e2 = normalize(cross(normal, e1));

  // Initial conditions in (u, φ) with u = 1/r.
  float u = 1.0 / r0;
  // du/dφ from the initial direction: the radial component of rd relative
  // to the tangential component sets the initial slope of the orbit.
  float rdotDir = dot(rd, e1);
  float tanDir  = dot(rd, e2);
  float du = -u * rdotDir / max(tanDir, 1e-6);

  float phi = 0.0;
  // Step size in φ. Total sweep is bounded at ~4π (two full windings)
  // which is generous for anything not asymptotically on the photon
  // sphere, and keeps the loop bound honest.
  float dphi = (4.0 * PI) / float(uSteps);

  float M = uRs * 0.5;   // geometrised mass; r_s = 2M

  float prevY = dot(ro, e1) * 0.0 + (1.0 / u) * 0.0;  // previous plane-y
  vec3  prevPos = ro;

  for (int i = 0; i < MAX_STEPS; i++) {
    if (i >= uSteps) break;

    // --- Velocity-Verlet step on d²u/dφ² = 3 M u² - u -------------------
    float acc = 3.0 * M * u * u - u;
    float uNext = u + du * dphi + 0.5 * acc * dphi * dphi;
    float accNext = 3.0 * M * uNext * uNext - uNext;
    du = du + 0.5 * (acc + accNext) * dphi;
    u = uNext;
    phi += dphi;

    // --- Horizon ---------------------------------------------------------
    // u > 1/r_s means the photon is inside the Schwarzschild radius. It is
    // captured; nothing it carries reaches the observer. This is the black
    // of the shadow, and it is genuinely black — not a dark grey disc.
    if (u > 1.0 / uRs) {
      return accum;
    }

    // --- Escape ----------------------------------------------------------
    if (u < ESCAPE_U) {
      // Sample the sky along the ray's ASYMPTOTIC direction — the tangent
      // to the orbit, not the position angle.
      //
      // This distinction is not pedantry. The position angle φ only takes
      // values at multiples of dphi, so using normalize(P(φ)) quantises the
      // sampled sky direction to the integration step: adjacent pixels snap
      // to different steps and every star smears into a horizontal dash.
      // The tangent is continuous in φ and has no such artefact.
      //
      //   P(φ)     = (1/u)·(e1 cosφ + e2 sinφ)
      //   dP/dφ    = (dr/dφ)·(e1 cosφ + e2 sinφ) + (1/u)·(-e1 sinφ + e2 cosφ)
      //   dr/dφ    = -(du/dφ)/u²
      // Written in the scaled form u²·(dP/dφ) rather than dP/dφ itself.
      // The unscaled expression contains -du/u² and 1/u, both of which
      // explode as u -> 0 and lose all their significant bits to float
      // cancellation just as the ray escapes — which is precisely when we
      // need the direction. Multiplying through by u² leaves two small,
      // comparable terms and the artefact disappears.
      vec3 radial    = e1 * cos(phi) + e2 * sin(phi);
      vec3 tangentia = -e1 * sin(phi) + e2 * cos(phi);
      vec3 dir = normalize(-radial * du + tangentia * u);
      accum += starfield(dir);
      return accum;
    }

    float r = 1.0 / u;
    vec3 pos = (e1 * cos(phi) + e2 * sin(phi)) * r;

    // --- Equatorial-plane crossing => disk hit ---------------------------
    // The disk lies in y = 0. We detect a sign change in the plane
    // coordinate between steps and interpolate the crossing point, rather
    // than testing |y| < epsilon — the latter misses crossings entirely at
    // low step counts and is why cheap raytracers show a broken disk when
    // quality drops.
    if (prevPos.y * pos.y < 0.0) {
      float t = prevPos.y / (prevPos.y - pos.y);
      vec3 hit = mix(prevPos, pos, t);
      float hr = length(hit.xz);
      if (hr > uDiskInner && hr < uDiskOuter) {
        float hphi = atan(hit.z, hit.x);
        vec3 c = sampleDisk(hr, hphi);

        // Gravitational + Doppler beaming. The approaching side of the
        // disk is brightened and the receding side dimmed — the asymmetry
        // that makes the reference's left side white-hot.
        vec3 orbitDir = normalize(vec3(-hit.z, 0.0, hit.x));
        vec3 toObs = normalize(ro - hit);
        // v/c = sqrt(M/r); with M = r_s/2 and hr in units of r_s this is
        // sqrt(0.5/hr). Using the physical value rather than a tamed one:
        // the approaching/receding asymmetry is the single most
        // recognisable feature of the reference image.
        float beta = 0.707 / sqrt(max(hr, 1.2));
        float cosA = dot(orbitDir, toObs);
        float doppler = 1.0 / max(1.0 - beta * cosA, 0.15);
        // Beaming has to move the emission up the COLOUR ramp, not just
        // scale its luminance. Scaling alone is squashed flat by the
        // filmic tonemap — the approaching and receding limbs end up the
        // same gold, which is exactly the failure visible in the first
        // pass of this shader. Physically this is right too: the observed
        // temperature is Doppler-shifted, so the approaching side is not
        // merely brighter, it is bluer/whiter.
        float boost = clamp((doppler - 1.0) * 1.5, -1.0, 1.0);
        float lum = length(c);
        c = mix(c, emissionRamp(1.0) * lum * 1.15, max(boost, 0.0));
        c = mix(c, emissionRamp(0.30) * lum * 0.85, max(-boost, 0.0));
        // Residual luminance term, at a reduced exponent since the ramp
        // shift above now carries most of the asymmetry.
        c *= pow(doppler, 1.7);

        // Gravitational redshift climbing out of the well.
        float gshift = sqrt(max(1.0 - uRs / max(hr, uRs * 1.001), 0.02));
        c *= gshift;

        // Inner-rim blueshift fringe. Emission from just outside the ISCO
        // on the approaching limb reaches the observer strongly blueshifted,
        // which reads as the violet edge along the shadow in the reference.
        // Weighted by both proximity to the inner edge and approach angle so
        // it appears exactly where the physics puts it.
        float rim = smoothstep(uDiskInner * 1.9, uDiskInner, hr);
        float approach = max(cosA, 0.0);
        c += vec3(0.22, 0.10, 0.62) * rim * approach * length(c) * 0.55;

        accum += c;

        // Optically thick: stop at the first crossing unless the quality
        // budget allows tracing the secondary image (the light that passes
        // over the top of the hole and shows the disk's far side).
        if (uDiskSamples <= 1) return accum;
      }
    }

    prevPos = pos;
    prevY = pos.y;
  }

  return accum;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
void main() {
  vec2 frag = vUv * uResolution;
  vec2 p = (2.0 * frag - uResolution) / uResolution.y;

  // Per-pixel jitter breaks up the banding that a fixed step size
  // otherwise produces along the photon sphere. Cheaper and better-looking
  // than raising uSteps.
  float jitter = hash21(frag) - 0.5;
  p += jitter / uResolution.y;

  float tanHalf = tan(uFov * 0.5);
  vec3 rd = normalize(uCamBasis * vec3(p.x * tanHalf, p.y * tanHalf, -1.0));

  vec3 col = traceGeodesic(uCamPos, rd);

  // --- Bloom ------------------------------------------------------------
  // Cheap energy-preserving highlight bloom. The reference's disk blooms
  // heavily; without this the inner ring clips to flat white.
  vec3 over = max(col - 1.0, 0.0);
  col += over * 0.28;

  // --- Tonemap ----------------------------------------------------------
  // ACES-ish filmic curve. Reinhard washes the white-hot inner ring out to
  // grey, which kills the one place in the composition that is meant to be
  // pure light.
  col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
  col = clamp(col, 0.0, 1.0);

  // Slight vignette, matching the reference's falloff into the corners.
  float vig = 1.0 - 0.28 * dot(p * 0.5, p * 0.5);
  col *= clamp(vig, 0.0, 1.0);

  // Dithering in the final 8-bit step. Without it the near-black gradient
  // around the shadow shows visible banding on OLED displays.
  col += (hash21(frag + uTime) - 0.5) / 255.0;

  gl_FragColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
