#version 300 es
precision highp float;

uniform sampler2D uTrailTex;
uniform sampler2D uAgentTex; // 128 x 3 RGBA32F:
                             // row 0: (x, y, rPulse, density)
                             // row 1: (absR, absG, absB, djSquish)
                             // row 2: (vx, vy, wobble2, wobble3)
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

  // 1. Trail absorbance (soft translucent TLC elution streaks drifting upward away from DJ)
  vec3 rawTrail = texture(uTrailTex, vUv).rgb * uTrailDecode;
  vec3 trailAbs = 0.26 * 1.45 * tanh(rawTrail / 1.45);

  // 2. Body pass: soft power-diagram blobby cells with wobble, velocity stretch, DJ squish & NO merge
  vec3 bodyAbs = vec3(0.0);
  float seamMask = 0.0;

  if (uAgentCount > 0 && p.x > -0.09 && p.x < 1.09 && p.y > -0.09 && p.y < 1.09) {
    int bestIdx = -1;
    float bestPow = 1e9;
    vec2 bestPos = vec2(0.0);
    float bestR = 0.036;
    float bestDensity = 1.0;
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
        bestDensity = row0.w;
        bestD2 = d2;
      }
    }

    float outerReach = bestR * 1.42;
    if (bestIdx >= 0 && bestD2 < outerReach * outerReach) {
      vec4 row1 = texelFetch(uAgentTex, ivec2(bestIdx, 1), 0);
      vec4 row2 = texelFetch(uAgentTex, ivec2(bestIdx, 2), 0);
      vec3 agentAbs = row1.rgb;
      float djSquish = row1.a;
      vec2 vel = row2.xy;
      float wobble2 = row2.z;
      float wobble3 = row2.w;

      // Transform local offset for organic velocity stretch + DJ-line horizontal bulge
      vec2 dpShape = p - bestPos;

      // Squish against DJ line: widen horizontally as body presses flat against y = 0.03
      if (djSquish > 0.001) {
        dpShape.x /= (1.0 + 0.26 * djSquish);
        dpShape.y *= (1.0 + 0.14 * djSquish);
      }

      // Stretch along velocity vector (volume-preserving mild elongation)
      float spd = length(vel);
      if (spd > 0.002) {
        vec2 vDir = vel / spd;
        vec2 vPerp = vec2(-vDir.y, vDir.x);
        float stretch = 1.0 + min(spd * 2.4, 0.28);
        float along = dot(dpShape, vDir) / stretch;
        float across = dot(dpShape, vPerp) * sqrt(stretch);
        dpShape = vDir * along + vPerp * across;
      }

      // 3 low-order wobble modes (quadrupole jiggle + trefoil organic + slow liquid breathing)
      float ang = atan(dpShape.y, dpShape.x);
      float idf = float(bestIdx);
      float ph2 = idf * 1.91 + uTime * 2.1;
      float ph3 = idf * 2.73 - uTime * 2.8;
      float phB = idf * 1.37 + uTime * 1.65;
      float rMod =
        1.0 +
        wobble2 * cos(2.0 * ang + ph2) +
        wobble3 * cos(3.0 * ang + ph3) +
        0.028 * sin(2.0 * ang - phB) +
        0.018 * cos(3.0 * ang + phB * 0.7);
      float rWobble = bestR * clamp(rMod, 0.78, 1.26);

      float distShape = length(dpShape);
      float dCircle = rWobble - distShape;
      float dPlanes = 1e9;
      float minRawPlane = 1e9;

      // ~1.6 CSS px paper-colored gap between touching bodies (half on each side of bisector)
      float pxInPlate = uDpr / uPlateRect.z;
      float halfGap = 0.80 * pxInPlate;
      float kRound = bestR * 0.26;

      for (int j = 0; j < 128; j++) {
        if (j >= uAgentCount) break;
        if (j == bestIdx) continue;
        vec4 rowJ = texelFetch(uAgentTex, ivec2(j, 0), 0);
        vec2 diff = rowJ.xy - bestPos;
        float sep2 = dot(diff, diff);
        float maxRange = bestR + rowJ.z + 0.08;
        if (sep2 > maxRange * maxRange || sep2 < 1e-8) continue;

        vec2 dpJ = p - rowJ.xy;
        float powJ = dot(dpJ, dpJ) - rowJ.z * rowJ.z;
        float dPlane = (powJ - bestPow) / (2.0 * sqrt(sep2));
        minRawPlane = min(minRawPlane, dPlane);
        dPlanes = (dPlanes > 1e8) ? (dPlane - halfGap) : smin(dPlanes, dPlane - halfGap, kRound);
      }

      // Flatten crisply against the DJ booth rail at the BOTTOM (y_gl = 0.03)
      float dDjWall = (p.y - 0.03) - halfGap;
      float kDj = mix(kRound, bestR * 0.14, djSquish);
      dPlanes = (dPlanes > 1e8) ? dDjWall : smin(dPlanes, dDjWall, kDj);

      // Combine blobby outer contour with neighbor/DJ contact planes via smooth-min (squash-without-merge)
      float dCell = smin(dCircle, dPlanes, kRound);

      // Free outer rims get soft TLC diffusion; packed facets flatten cleanly into rounded polygons
      float isContactFacet = smoothstep(bestR * 0.34, -bestR * 0.04, dPlanes - dCircle);
      float outerSoft = max(bestR * 0.38, 3.8 * pxInPlate);
      float contactSoft = max(bestR * 0.13, 1.9 * pxInPlate);
      float edgeSoft = mix(outerSoft, contactSoft, isContactFacet);

      float mask = smoothstep(0.0, edgeSoft, dCell);

      // Rich TLC core; bestDensity encodes E * proximity vibrancy (vivid front, delicate back)
      float dome = 0.56 + 0.44 * smoothstep(0.0, bestR * 0.72, dCell);
      bodyAbs = agentAbs * (mask * dome * bestDensity);

      // Gently lighten trail inside active contact gaps so squashed cells stay distinct
      if (dCircle > -0.002 && minRawPlane < halfGap * 2.0) {
        seamMask = (1.0 - smoothstep(0.0, halfGap * 2.0, minRawPlane)) * smoothstep(-0.002, 0.004, dCircle);
      }
    }
  }

  trailAbs *= (1.0 - 0.24 * seamMask);

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
