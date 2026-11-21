#version 300 es
precision highp float;

// Quad corner in [-1, 1]^2
layout(location = 0) in vec2 aQuad;
// Per-instance attributes
layout(location = 1) in vec4 aPosVel;     // (x_gl, y_gl, vx_gl, vy_gl) in plate units
layout(location = 2) in vec4 aShape;      // (r_pulse, E, pressure, 0)
layout(location = 3) in vec3 aAbsorbance; // (-ln(max(c, 0.04)))

uniform vec2 uResolution;
uniform vec4 uPlateRect; // (ox_px, oy_px, size_px, size_px) in GL framebuffer pixels (bottom-left origin)
uniform float uStoreScale;

out vec2 vLocal;
out vec3 vAbs;
out float vAmount;

void main() {
  vec2 pos = aPosVel.xy;
  vec2 vel = aPosVel.zw;
  float r = aShape.x;
  float E = aShape.y;
  float pressure = aShape.z;

  float speed = length(vel);
  // Default long axis vertically away from DJ at bottom (upward TLC elution streak).
  vec2 dir = (speed > 0.001) ? (vel / speed) : vec2(0.0, 1.0);
  vec2 perp = vec2(-dir.y, dir.x);

  float longAxis = r * min(1.0 + 6.0 * speed, 2.6);
  float shortAxis = r / (1.0 + 2.0 * pressure);

  // Quad spans +-2.3 sigma-radii so Gaussian smoothly hits 0 before quad border.
  float span = 2.3;
  vec2 offsetPlate = dir * (aQuad.x * longAxis * span) + perp * (aQuad.y * shortAxis * span);
  vec2 worldPlate = pos + offsetPlate;

  vec2 px = uPlateRect.xy + worldPlate * uPlateRect.zw;
  vec2 ndc = (px / uResolution) * 2.0 - 1.0;
  gl_Position = vec4(ndc, 0.0, 1.0);

  vLocal = aQuad * span;
  vAbs = aAbsorbance;
  vAmount = 0.06 * (0.3 + 0.7 * E) * uStoreScale;
}
