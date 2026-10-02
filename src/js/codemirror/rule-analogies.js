// Repeat a previous rule using a consistent substitution between declared names.
// There is no vocabulary of colours/materials/states: the names supply the evidence.
class RuleAnalogies {
    static DIRECTIONAL_COMPONENTS = [
        ['up', 'down', 'left', 'right'], ['u', 'd', 'l', 'r'],
        ['north', 'south', 'west', 'east'], ['n', 's', 'w', 'e'],
        ['horizontal', 'vertical'], ['h', 'v']
    ];

    static isDirectionalSubstitution(substitution) {
        const to = substitution.to.toLowerCase();
        return RuleAnalogies.DIRECTIONAL_COMPONENTS.some(family =>
            family.includes(substitution.from) && family.includes(to));
    }

    static parts(name) {
        const split = name.replace(/(\p{Ll})(\p{Lu})/gu, '$1\0$2')
            .replace(/(\p{Lu})(\p{Lu}\p{Ll})/gu, '$1\0$2');
        const parts = split.match(/\p{L}+|\p{N}+|_+/gu) || [];
        return parts.join('') === name ? parts : [];
    }

    static substitution(source, target) {
        if (source.length < 2 || source.length !== target.length) return null;
        const changed = source.map((part, i) => part.toLowerCase() === target[i].toLowerCase() ? -1 : i).filter(i => i >= 0);
        if (changed.length !== 1) return null;
        const i = changed[0], from = source[i], to = target[i];
        if (from[0] === '_' || to[0] === '_') return null;
        const numeric = /^\d+$/.test(from) && /^\d+$/.test(to);
        if (/^\d+$/.test(from) !== /^\d+$/.test(to)) return null;
        return { from: from.toLowerCase(), to, index: i, offset: numeric ? Number(to) - Number(from) : null };
    }

    static replaceParts(parts, substitution) {
        return parts.map((part, index) => {
            if (substitution.offset !== null) {
                if (index !== substitution.index || !/^\d+$/.test(part)) return part;
                const value = Number(part) + substitution.offset;
                if (!Number.isSafeInteger(value) || value < 0) return '?';
                return part[0] === '0' ? String(value).padStart(part.length, '0') : String(value);
            }
            return part.toLowerCase() === substitution.from ? substitution.to : part;
        }).join('');
    }

    static signature(name, index) {
        return JSON.stringify([name.symbol.kind, index,
            name.parts.map((part, i) => i === index ? null : part.toLowerCase())]);
    }

    static ruleKey(line) {
        // Ignore formatting and comments, but keep message text case-sensitive.
        const tokens = RuleCompletion.tokens(line);
        const key = [];
        for (const token of tokens) {
            if (token.word === '(') break;
            key.push(token.word);
            if (token.word === 'message') {
                key.push(line.slice(token.end).trim());
                break;
            }
        }
        return JSON.stringify(key);
    }

    static suggestions(semantics, prefix, range = 500, limit = 12) {
        const state = semantics.state;
        const names = new Map();
        for (const name of [...Object.keys(state.objects), ...semantics.definitions.keys()]) {
            const symbol = semantics.resolve(name);
            const display = state.original_case_names[name] || name;
            if (symbol) names.set(name, { display, symbol, parts: RuleAnalogies.parts(display) });
        }
        // Index only components matching what is being typed. Unmatched prefixes
        // should be cheap even in a game containing hundreds of rules and names.
        const targets = new Map();
        for (const target of names.values()) {
            if (target.parts.length < 2) continue;
            target.parts.forEach((part, index) => {
                if (part[0] === '_' || !part.toLowerCase().startsWith(prefix)) return;
                const signature = RuleAnalogies.signature(target, index);
                if (!targets.has(signature)) targets.set(signature, []);
                targets.get(signature).push(target);
            });
        }
        if (!targets.size) return [];
        const substitutions = new Map();
        const variants = name => {
            if (!substitutions.has(name)) {
                const source = names.get(name);
                const matches = source.parts.flatMap((part, index) =>
                    (targets.get(RuleAnalogies.signature(source, index)) || [])
                        .map(target => RuleAnalogies.substitution(source.parts, target.parts)).filter(Boolean));
                substitutions.set(name, matches);
            }
            return substitutions.get(name);
        };
        const rules = [];
        const existing = new Set();
        // The live state includes the current, unfinished line. Only use earlier rules.
        for (const entry of state.rules) {
            if (entry[1] >= state.lineNumber) continue;
            const line = (entry[2] || entry[0]).trim();
            const key = RuleAnalogies.ruleKey(line);
            existing.add(key);
            if (entry[1] >= state.lineNumber - range) rules.push({ line, key });
        }
        const results = [];
        const processed = new Set();
        for (let i = rules.length - 1; i >= 0 && results.length < limit; i--) {
            const { line, key } = rules[i];
            if (processed.has(key)) continue;
            processed.add(key);
            const parsed = semantics.parse(line);
            if (!semantics.validRule(parsed)) continue;
            const hasRuleDirection = Boolean(RuleTransform.getFirstRuleDirection(line));
            const used = [...new Set(parsed.names.map(item => item.name))];
            const tried = new Set();
            for (const name of used) {
                for (const substitution of variants(name)) {
                    // Directional rule completion rotates/mirrors the whole rule.
                    // A suffix-only repeat would leave its rule direction unchanged.
                    if (hasRuleDirection && RuleAnalogies.isDirectionalSubstitution(substitution)) continue;
                    // A numbered repeat is labelled by its first stage, not by a later RHS stage.
                    if (substitution.offset !== null) {
                        const firstNumber = used.map(name => names.get(name).parts[substitution.index]).find(part => /^\d+$/.test(part));
                        if (substitution.from !== firstNumber) continue;
                    }
                    const key = substitution.offset === null ? substitution.from + ':' + substitution.to.toLowerCase() : 'offset:' + substitution.index + ':' + substitution.offset;
                    if (tried.has(key)) continue;
                    tried.add(key);
                    const replacements = new Map();
                    let valid = true;
                    for (const usedName of used) {
                        const original = names.get(usedName);
                        if (!original.parts.length) continue;
                        const replacementName = RuleAnalogies.replaceParts(original.parts, substitution).toLowerCase();
                        if (replacementName === usedName) continue;
                        const replacement = names.get(replacementName);
                        if (!replacement || replacement.symbol.kind !== original.symbol.kind) { valid = false; break; }
                        replacements.set(usedName, replacement.display);
                    }
                    if (!valid || !replacements.size) continue;
                    let text = line;
                    for (let n = parsed.names.length - 1; n >= 0; n--) {
                        const item = parsed.names[n];
                        if (replacements.has(item.name)) {
                            text = text.slice(0, item.token.start) + replacements.get(item.name) + text.slice(item.token.end);
                        }
                    }
                    const resultKey = RuleAnalogies.ruleKey(text);
                    if (existing.has(resultKey)) continue;
                    if (!semantics.validRule(semantics.parse(text))) continue;
                    existing.add(resultKey);
                    results.push({ text, displayText: 'Repeat for ' + substitution.to,
                        extra: [...replacements].map(([from, to]) => names.get(from).display + ' → ' + to).join('; ') });
                    if (results.length >= limit) return results;
                }
            }
        }
        return results;
    }
}
