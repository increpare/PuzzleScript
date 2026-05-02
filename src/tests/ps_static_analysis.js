#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { compileSemanticSource } = require('../canonicalize');

const SCHEMA = 'ps-static-analysis-v1';

function emptyFacts() {
    return {
        mergeability: [],
        movement_action: [],
        count_layer_invariants: [],
        transient_boundary: [],
    };
}

function analyzeSource(source, options = {}) {
    const sourcePath = options.sourcePath || '<memory>';
    const compiled = compileSemanticSource(source, {
        includeWinConditions: true,
        throwOnError: false,
    });

    if (compiled.errorCount > 0 || compiled.state === null || compiled.state.invalid) {
        return {
            schema: SCHEMA,
            source: { path: sourcePath },
            status: 'compile_error',
            errors: compiled.errorStrings.slice(),
            ps_tagged: null,
            facts: emptyFacts(),
            summary: { proved: 0, candidate: 0, rejected: 0 },
        };
    }

    const psTagged = { game: { tags: {} } };
    const report = {
        schema: SCHEMA,
        source: { path: sourcePath },
        status: 'ok',
        ps_tagged: psTagged,
        facts: emptyFacts(),
        summary: { proved: 0, candidate: 0, rejected: 0 },
    };

    if (options.includePsTagged === false) {
        delete report.ps_tagged;
    }
    return report;
}

function analyzeFile(filePath, options = {}) {
    const resolved = path.resolve(filePath);
    const source = fs.readFileSync(resolved, 'utf8');
    return analyzeSource(source, Object.assign({}, options, { sourcePath: filePath }));
}

module.exports = {
    SCHEMA,
    analyzeFile,
    analyzeSource,
};
