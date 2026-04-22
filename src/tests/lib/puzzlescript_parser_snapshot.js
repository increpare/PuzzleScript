'use strict';

function parseOnly(source) {
    resetParserErrorState();
    const processor = new codeMirrorFn();
    const parserState = processor.startState();
    const lines = source.split('\n');
    for (const line of lines) {
        const stream = new CodeMirror.StringStream(line, 4);
        do {
            processor.token(stream, parserState);
        } while (stream.eol() === false);
    }
    return parserState;
}

/**
 * Tokenize-only pass with `compiling` enabled so parser `logError` / `logWarning`
 * calls populate `errorStrings` (matches in-editor behavior during compile's loadFile).
 */
function collectParserPhaseDiagnostics(source) {
    resetParserErrorState();
    compiling = true;
    try {
        const processor = new codeMirrorFn();
        const parserState = processor.startState();
        const lines = source.split('\n');
        for (const line of lines) {
            const stream = new CodeMirror.StringStream(line, 4);
            do {
                processor.token(stream, parserState);
            } while (stream.eol() === false);
        }
        return errorStrings.slice();
    } finally {
        compiling = false;
    }
}

function serializeObjects(objects) {
    return Object.keys(objects || {}).sort().map(name => ({
        name,
        line_number: objects[name].lineNumber,
        colors: Array.isArray(objects[name].colors) ? objects[name].colors.slice() : [],
        spritematrix: Array.isArray(objects[name].spritematrix) ? objects[name].spritematrix.slice() : [],
    }));
}

function serializeLegendEntries(entries) {
    return (entries || []).map(entry => ({
        name: entry[0],
        items: entry.slice(1),
        line_number: entry.lineNumber || 0,
    }));
}

function serializeSounds(entries) {
    return (entries || []).map(entry => ({
        tokens: entry.slice(0, -1).map(token => ({
            text: Array.isArray(token) ? token[0] : String(token),
            kind: Array.isArray(token) ? token[1] : '',
        })),
        line_number: entry[entry.length - 1],
    }));
}

function serializeRules(entries) {
    return (entries || []).map(entry => ({
        rule: entry[0],
        line_number: entry[1],
        mixed_case: entry[2],
    }));
}

function serializeWinconditions(entries) {
    return (entries || []).map(entry => ({
        tokens: entry.slice(0, -1),
        line_number: entry[entry.length - 1],
    }));
}

function serializeLevels(levels) {
    return (levels || []).map(level => {
        if (level[0] === '\n') {
            return {
                kind: 'message',
                line_number: level[2] ?? null,
                message: level[1] || '',
            };
        }
        return {
            kind: 'level',
            line_number: typeof level[0] === 'number' ? level[0] : null,
            rows: level.length > 0 ? level.slice(1) : [],
        };
    });
}

function toStringMap(map) {
    const result = {};
    for (const key of Object.keys(map || {}).sort()) {
        result[key] = map[key];
    }
    return result;
}

function buildParserStateSnapshot(source) {
    const state = parseOnly(source);
    return {
        schema_version: 1,
        parser_state: {
            line_number: state.lineNumber,
            comment_level: state.commentLevel,
            section: state.section,
            visited_sections: Array.isArray(state.visitedSections) ? state.visitedSections.slice() : [],
            line_should_end: Boolean(state.line_should_end),
            line_should_end_because: state.line_should_end_because || '',
            sol_after_comment: Boolean(state.sol_after_comment),
            inside_cell: Boolean(state.inside_cell),
            bracket_balance: state.bracket_balance || 0,
            arrow_passed: Boolean(state.arrow_passed),
            rule_prelude: Boolean(state.rule_prelude),
            objects_candname: state.objects_candname || '',
            objects_section: state.objects_section || 0,
            objects_spritematrix: Array.isArray(state.objects_spritematrix) ? state.objects_spritematrix.slice() : [],
            collision_layers: Array.isArray(state.collisionLayers) ? state.collisionLayers.map(layer => layer.slice()) : [],
            token_index: state.tokenIndex || 0,
            current_line_wip_array: Array.isArray(state.current_line_wip_array) ? state.current_line_wip_array.map(String) : [],
            metadata_pairs: Array.isArray(state.metadata) ? state.metadata.slice() : [],
            metadata_lines: toStringMap(state.metadata_lines),
            original_case_names: toStringMap(state.original_case_names),
            original_line_numbers: toStringMap(state.original_line_numbers),
            names: Array.isArray(state.names) ? state.names.slice() : [],
            abbrev_names: Array.isArray(state.abbrevNames) ? state.abbrevNames.slice() : [],
            objects: serializeObjects(state.objects),
            legend_synonyms: serializeLegendEntries(state.legend_synonyms),
            legend_aggregates: serializeLegendEntries(state.legend_aggregates),
            legend_properties: serializeLegendEntries(state.legend_properties),
            sounds: serializeSounds(state.sounds),
            rules: serializeRules(state.rules),
            winconditions: serializeWinconditions(state.winconditions),
            levels: serializeLevels(state.levels),
            subsection: state.subsection || '',
        },
    };
}

module.exports = {
    buildParserStateSnapshot,
    collectParserPhaseDiagnostics,
};
