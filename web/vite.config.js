import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";

const tracksDir = path.resolve(__dirname, "../tracks");

function localTracksPlugin() {
  return {
    name: "local-tracks-middleware",
    configureServer(server) {
      server.middlewares.use("/api/tracks", (_req, res) => {
        let files = [];
        try {
          files = fs
            .readdirSync(tracksDir)
            .filter((f) => /\.(mp3|wav|m4a|ogg)$/i.test(f))
            .sort();
        } catch {
          files = [];
        }
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ tracks: files }));
      });

      server.middlewares.use("/tracks", (req, res, next) => {
        const clean = decodeURIComponent((req.url || "").split("?")[0]).replace(/^\/+/, "");
        if (!clean || clean.includes("..")) return next();
        const filePath = path.join(tracksDir, clean);
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return next();

        const stat = fs.statSync(filePath);
        const total = stat.size;
        const ext = path.extname(filePath).toLowerCase();
        const mime =
          ext === ".mp3"
            ? "audio/mpeg"
            : ext === ".wav"
              ? "audio/wav"
              : ext === ".ogg"
                ? "audio/ogg"
                : "application/octet-stream";

        const range = req.headers.range;
        if (range) {
          const [startStr, endStr] = range.replace(/bytes=/, "").split("-");
          const start = parseInt(startStr, 10);
          const end = endStr ? parseInt(endStr, 10) : total - 1;
          res.writeHead(206, {
            "Content-Range": `bytes ${start}-${end}/${total}`,
            "Accept-Ranges": "bytes",
            "Content-Length": end - start + 1,
            "Content-Type": mime,
          });
          fs.createReadStream(filePath, { start, end }).pipe(res);
        } else {
          res.writeHead(200, {
            "Content-Length": total,
            "Accept-Ranges": "bytes",
            "Content-Type": mime,
          });
          fs.createReadStream(filePath).pipe(res);
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [localTracksPlugin()],
  server: {
    port: 5173,
    proxy: {
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
      },
    },
  },
});
