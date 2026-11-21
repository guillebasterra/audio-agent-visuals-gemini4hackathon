#version 300 es
precision highp float;

in vec2 vLocal;
in vec3 vAbs;
in float vAmount;

out vec4 fragColor;

void main() {
  float r2 = dot(vLocal, vLocal);
  if (r2 > 5.29) discard;
  // Soft anisotropic Gaussian splat (sigma = 0.58 in normalized axis units)
  float g = exp(-r2 / (2.0 * 0.58 * 0.58));
  fragColor = vec4(vAbs * (vAmount * g), 1.0);
}
