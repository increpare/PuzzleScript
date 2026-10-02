// Semantic information for completion, built from the editor's live parser state.
// Unknown or unfinished syntax is deliberately not treated as a compiler error.
class RuleCompletion {
    constructor(state) {
        this.state = state;
        this.definitions = new Map();
        this.resolved = new Map();
        this.layers = new Map();
        for (const [kind, entries] of [
            ['synonym', state.legend_synonyms], ['property', state.legend_properties],
            ['aggregate', state.legend_aggregates]
        ]) {
            for (const entry of entries) this.definitions.set(entry[0], { kind, members: entry.slice(1) });
        }
        state.collisionLayers.forEach((layer, index) => {
            for (const name of layer) {
                // A multiply assigned object is not reliable evidence of a conflict.
                this.layers.set(name, this.layers.has(name) ? null : index);
            }
        });
    }

    resolve(name, visiting = new Set()) {
        name = name.toLowerCase();
        if (this.resolved.has(name)) return this.resolved.get(name);
        if (visiting.has(name)) return null;
        visiting.add(name);
        let result = null;
        if (Object.prototype.hasOwnProperty.call(this.state.objects, name)) {
            result = { kind: 'object', key: name, objects: [name] };
        } else if (this.definitions.has(name)) {
            const definition = this.definitions.get(name);
            const members = definition.members.map(member => this.resolve(member, visiting));
            if (members.length && members.every(Boolean)) {
                if (definition.kind === 'synonym') {
                    // The compiler promotes aliases of properties/aggregates to
                    // their own definitions; only concrete object aliases collapse.
                    result = members[0].kind === 'object' ? members[0] :
                        { kind: members[0].kind, key: name, objects: members[0].objects };
                } else if (members.every(member => member.kind === 'object' || member.kind === definition.kind)) {
                    result = { kind: definition.kind, key: name,
                        objects: [...new Set(members.flatMap(member => member.objects))] };
                }
            }
        }
        visiting.delete(name);
        this.resolved.set(name, result);
        return result;
    }

    // Positions let analogy completion replace names without touching prose or spacing.
    static tokens(line) {
        return [...line.matchAll(/->|[\[\]|+()]|[^\s\[\]|()+]+/gu)]
            .map(match => ({ text: match[0], word: match[0].toLowerCase(), start: match.index, end: match.index + match[0].length }));
    }

    parse(line) {
        const result = { lhs: [], rhs: [], side: 'lhs', cell: null, row: -1, column: -1,
            modifier: '', late: false, rigid: false, valid: true, lhsComplete: false,
            names: [], commands: [], arrow: false, comment: false };
        let inPrelude = true;
        for (const token of RuleCompletion.tokens(line)) {
            const word = token.word;
            if (word === '(') { result.comment = true; break; }
            if (word === '[') {
                if (result.cell || result.commands.length) result.valid = false;
                inPrelude = false;
                result.row = result[result.side].length;
                result.column = 0;
                result.cell = [];
                result[result.side].push([result.cell]);
            } else if (word === '|' || word === ']') {
                if (!result.cell || result.modifier) result.valid = false;
                if (word === '|' && result.cell) {
                    result.cell = [];
                    result[result.side][result.row].push(result.cell);
                    result.column++;
                } else {
                    result.cell = null;
                }
                result.modifier = '';
            } else if (word === '->') {
                if (result.arrow || result.cell || !result.lhs.length) result.valid = false;
                result.lhsComplete = result.valid;
                result.arrow = true;
                result.side = 'rhs';
                result.cell = null;
                result.modifier = '';
                result.row = -1;
            } else if (inPrelude) {
                if (!['+', 'late', 'rigid', 'random', 'up', 'down', 'left', 'right', 'horizontal', 'vertical', 'orthogonal'].includes(word)) result.valid = false;
                if (word === 'late') result.late = true;
                if (word === 'rigid') result.rigid = true;
            } else if (result.cell) {
                if (reg_directions_only.test(word)) {
                    if (result.modifier) result.valid = false;
                    result.modifier = word;
                } else if (word === '...') {
                    if (result.modifier || result.cell.length) result.valid = false;
                    result.cell.push({ name: word, modifier: '', symbol: null });
                } else {
                    const symbol = this.resolve(word);
                    if (!symbol) result.valid = false;
                    const item = { name: word, modifier: result.modifier, symbol, token };
                    result.cell.push(item);
                    result.names.push(item);
                    result.modifier = '';
                }
            } else if (result.arrow && commandwords.includes(word)) {
                result.commands.push(word);
                if (word === 'message') break;
            } else {
                result.valid = false;
            }
        }
        return result;
    }

    patternWords(context, words) {
        return words.filter((word, index) => index === 0 || (
            !context.modifier && (word !== 'random' || context.side === 'rhs') &&
            (!context.late || ['no', 'random', 'randomdir'].includes(word))
        ));
    }

    binding(context, symbol) {
        if (!context.lhsComplete || !context.valid) return 'unknown';
        const matches = cell => cell && cell.some(item => item.symbol &&
            item.symbol.key === symbol.key && item.modifier !== 'no' && item.modifier !== 'random');
        if (matches((context.lhs[context.row] || [])[context.column])) return 'cell';
        const count = context.lhs.flat().filter(matches).length;
        return count === 1 ? 'unique' : 'unbound';
    }

    // Each group requires one of its objects. Aggregates require every member.
    groups(symbol) {
        return symbol.kind === 'aggregate' ? symbol.objects.map(name => [name]) : [symbol.objects];
    }

    conflicts(a, b) {
        return this.groups(a).some(groupA => this.groups(b).some(groupB =>
            groupA.every(nameA => groupB.every(nameB =>
                this.layers.get(nameA) != null && this.layers.get(nameA) === this.layers.get(nameB)))
        ));
    }

    nameHint(context, name) {
        const symbol = this.resolve(name);
        if (!symbol || !context.cell) return { allowed: true, extra: '' };
        const modifier = context.modifier;
        if (modifier === 'no' && symbol.kind === 'aggregate') return { allowed: false };
        // Multiple RHS random terms contribute to a single choice pool. A random
        // spawn can also replace an occupant, so it is not a required co-occupant.
        if (context.valid && modifier !== 'no' && modifier !== 'random') {
            for (const item of context.cell) {
                if (item.symbol && item.modifier !== 'no' && this.conflicts(symbol, item.symbol)) return { allowed: false };
            }
        }
        let extra = '';
        if (symbol.kind === 'property') {
            if (modifier === 'no') extra = 'none of the objects in this property';
            else if (modifier === 'random') extra = 'choose a random object from this property';
            else if (context.side === 'rhs') {
                const binding = this.binding(context, symbol);
                if (binding === 'unbound') return { allowed: false };
                extra = binding === 'cell' ? 'the same object matched on the left in this cell' :
                    binding === 'unique' ? 'the same object matched on the left (one match in the rule)' :
                    'needs a matching property on the left';
            } else extra = 'matches one of: ' + symbol.objects.map(name => this.state.original_case_names[name] || name).join(', ');
        }
        return { allowed: true, extra };
    }

    completeRule(context) {
        return context.valid && context.lhsComplete && !context.cell && !context.modifier &&
            (context.rhs.length ? context.rhs.length === context.lhs.length &&
                context.rhs.every((row, i) => row.length === context.lhs[i].length) : context.commands.length > 0);
    }

    validRule(context) {
        if (!this.completeRule(context) || (context.late && context.rigid)) return false;
        for (const side of ['lhs', 'rhs']) {
            for (let row = 0; row < context[side].length; row++) {
                for (let column = 0; column < context[side][row].length; column++) {
                    const cell = context[side][row][column];
                    for (let i = 0; i < cell.length; i++) {
                        const item = cell[i];
                        if (!item.symbol) continue;
                        if (side === 'lhs' && item.modifier === 'random') return false;
                        if (context.late && item.modifier && !['no', 'random', 'randomdir'].includes(item.modifier)) return false;
                        const position = Object.assign({}, context, { side, row, column,
                            modifier: item.modifier, cell: cell.slice(0, i) });
                        if (!this.nameHint(position, item.name).allowed) return false;
                    }
                }
            }
        }
        return true;
    }
}
