#version 300 es
precision highp float;

uniform sampler2D uPrevTrail;
uniform vec2 uResolution;
uniform float uDecay;
uniform float uAdvectPx; // pixels per frame away from DJ (upward, toward +y in GL)

in vec2 vUv;
out vec4 fragColor;

void main() {
  // DJ is at the bottom (y=0 in GL): sample slightly below (-y) so trails stream upward away from DJ.
  vec2 srcUv = vUv - vec2(0.0, uAdvectPx / uResolution.y);
  if (srcUv.y < 0.0) {
    fragColor = vec4(0.0);
    return;
  }
  vec3 prev = texture(uPrevTrail, srcUv).rgb;
  fragColor = vec4(prev * uDecay, 1.0);
}
