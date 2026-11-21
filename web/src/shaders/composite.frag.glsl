#version 300 es
precision highp float;

uniform sampler2D uTrailTex;
uniform sampler2D uAgentTex; // 128 x 2 RGBA32F
uniform int uAgentCount;
uniform vec2 uResolution;
uniform vec4 uPlateRect;     // (ox_px, oy_px, size_px, size_px) in GL framebuffer pixels
uniform float uDpr;
uniform float uTime;
uniform float uTrailDecode;  // 1.0 for float FBO, 4.0 for RGBA8 fallback

in vec2 vUv;
out vec4 fragColor;

// Polynomial smooth minimum for rounded polygon corners (Inigo Quilez)
float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-5), 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

// Lightweight hash for animated subtle paper grain
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec2 fragPx = gl_FragCoord.xy;
  vec2 p = (fragPx - uPlateRect.xy) / uPlateRect.zw;

  // 1. Trail absorbance (soft translucent TLC elution streaks)
  vec3 rawTrail = texture(uTrailTex, vUv).rgb * uTrailDecode;
  vec3 trailAbs = 0.24 * 1.35 * tanh(rawTrail / 1.35);

  // 2. Body pass: soft power-diagram cells that squash into rounded polygons & NEVER merge
  vec3 bodyAbs = vec3(0.0);
  float seamMask = 0.0;

  if (uAgentCount > 0 && p.x > -0.06 && p.x < 1.06 && p.y > -0.06 && p.y < 1.06) {
    int bestIdx = -1;
    float bestPow = 1e9;
    vec2 bestPos = vec2(0.0);
    float bestR = 0.028;
    float bestE = 0.8;
    float bestD2 = 1e9;

    for (int i = 0; i < 128; i++) {
      if (i >= uAgentCount) break;
      vec4 row0 = texelFetch(uAgentTex, ivec2(i, 0), 0);
      vec2 dp = p - row0.xy;
      float d2 = dot(dp, dp);
      float r = row0.z;
      float pw = d2 - r * r;
      if (pw < bestPow) {
        bestPow = pw;
        bestIdx = i;
        bestPos = row0.xy;
        bestR = r;
        bestE = row0.w;
        bestD2 = d2;
      }
    }

    float outerReach = bestR * 1.25;
    if (bestIdx >= 0 && bestD2 < outerReach * outerReach) {
      float distCenter = sqrt(bestD2);
      float dCircle = bestR - distCenter;
      float dPlanes = 1e9;
      float minRawPlane = 1e9;

      // ~1.5 CSS px paper-colored gap between touching bodies (half on each side of bisector)
      float pxInPlate = uDpr / uPlateRect.z;
      float halfGap = 0.75 * pxInPlate;
      float kRound = bestR * 0.28;

      for (int j = 0; j < 128; j++) {
        if (j >= uAgentCount) break;
        if (j == bestIdx) continue;
        vec4 rowJ = texelFetch(uAgentTex, ivec2(j, 0), 0);
        vec2 diff = rowJ.xy - bestPos;
        float sep2 = dot(diff, diff);
        float maxRange = bestR + rowJ.z + 0.05;
        if (sep2 > maxRange * maxRange || sep2 < 1e-8) continue;

        vec2 dpJ = p - rowJ.xy;
        float powJ = dot(dpJ, dpJ) - rowJ.z * rowJ.z;
        float dPlane = (powJ - bestPow) / (2.0 * sqrt(sep2));
        minRawPlane = min(minRawPlane, dPlane);
        dPlanes = (dPlanes > 1e8) ? (dPlane - halfGap) : smin(dPlanes, dPlane - halfGap, kRound);
      }

      // Squash softly against the DJ booth rail (y_gl = 0.97 corresponds to physics y = 0.03)
      float dDjWall = (0.97 - p.y) - halfGap;
      dPlanes = (dPlanes > 1e8) ? dDjWall : smin(dPlanes, dDjWall, kRound);

      // Combine free circular boundary with neighbor/wall contact planes via smooth-min
      float dCell = smin(dCircle, dPlanes, kRound);

      // Free outer rims get lush TLC diffusion; contact facets stay distinct & soft-edged
      float isContactFacet = smoothstep(bestR * 0.32, -bestR * 0.05, dPlanes - dCircle);
      float outerSoft = max(bestR * 0.44, 4.5 * pxInPlate);
      float contactSoft = max(bestR * 0.19, 2.5 * pxInPlate);
      float edgeSoft = mix(outerSoft, contactSoft, isContactFacet);

      float mask = smoothstep(0.0, edgeSoft, dCell);

      // Richer chromatographic dome core; energy E controls pigment density (core saturation)
      float dome = 0.56 + 0.44 * smoothstep(0.0, bestR * 0.72, dCell);
      float energyDensity = 0.24 + 1.24 * bestE;

      vec3 agentAbs = texelFetch(uAgentTex, ivec2(bestIdx, 1), 0).rgb;
      bodyAbs = agentAbs * (mask * dome * energyDensity);

      // Gently reduce trail stain inside active contact gaps without creating a bright halo
      if (dCircle > -0.002 && minRawPlane < halfGap * 2.0) {
        seamMask = (1.0 - smoothstep(0.0, halfGap * 2.0, minRawPlane)) * smoothstep(-0.002, 0.004, dCircle);
      }
    }
  }

  trailAbs *= (1.0 - 0.22 * seamMask);

  // 3. Warm pinkish TLC silica paper (#f6ebe6) + animated grain (+-3.5%) + slight vignette
  vec3 paper = vec3(0.9647, 0.9216, 0.9020);
  vec2 centered = (vUv - 0.5) * 2.0;
  float vig = 1.0 - 0.075 * dot(centered, centered);
  paper *= mix(vec3(0.965, 0.935, 0.905), vec3(1.0), clamp(vig, 0.0, 1.0));

  float grainFrame = floor(uTime * 16.0);
  vec2 grainCoord = floor(fragPx / max(uDpr, 1.0));
  float g = hash12(grainCoord + vec2(grainFrame * 17.31, grainFrame * 43.17)) - 0.5;
  paper *= (1.0 + g * 0.07);

  vec3 color = paper * exp(-(trailAbs + bodyAbs));
  fragColor = vec4(color, 1.0);
}
