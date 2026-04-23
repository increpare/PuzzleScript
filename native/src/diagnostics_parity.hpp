#pragma once

#include <filesystem>

// Read NDJSON from build_parser_corpus_bundle.js; for each line run ps_frontend_parse and compare
// canonicalized diagnostics to the embedded "reference" array. Returns 0 if all match.
int diagnosticsParityMain(const std::filesystem::path& bundleNdjsonPath);
