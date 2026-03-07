const esbuild = require("esbuild");

esbuild.build({
  entryPoints: ["./src/js/codemirror/cm6-puzzlescript.js"],
  bundle: true,
  format: "iife",
  platform: "browser",
  target: ["es2019"],
  sourcemap: true,
  outfile: "./src/js/codemirror/cm6.bundle.js"
}).catch(function(error) {
  console.error(error);
  process.exit(1);
});
