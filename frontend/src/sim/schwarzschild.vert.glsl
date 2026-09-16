// Full-screen triangle. Not a quad: a single oversized triangle covers the
// viewport with one primitive, avoids the diagonal seam where two triangles
// meet, and skips the redundant vertex shading along that edge.
attribute vec2 aPosition;
varying vec2 vUv;

void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
