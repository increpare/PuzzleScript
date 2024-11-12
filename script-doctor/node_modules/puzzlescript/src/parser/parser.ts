import * as nearley from 'nearley'
import { RULE_DIRECTION_WITH_RELATIVE } from '../util'
import { AstBuilder } from './astBuilder'
import * as ast from './astTypes'
import compiledGrammar from './grammar'

export enum ValidationLevel {
    ERROR,
    WARNING,
    INFO
}

class Parser {
    private grammar: nearley.Grammar
    constructor() {
        this.grammar = nearley.Grammar.fromCompiled(compiledGrammar)
    }
    public parseToAST(code: string) {
        const parser = new nearley.Parser(this.grammar)
        parser.feed(code)
        parser.feed('\n')
        parser.finish()
        const results = parser.results as Array<ast.IASTGame<RULE_DIRECTION_WITH_RELATIVE, string, string, number | '.'>>
        if (results.length === 1) {
            return results[0]
        } else if (results.length === 0) {
            throw new Error(`ERROR: Could not parse`)
        } else {
            throw new Error(`AMBIGUOUS: has ${results.length} results`)
        }

    }
    public parse(code: string) {
        const node = this.parseToAST(code)

        const builder = new AstBuilder(code)
        const gameData = builder.build(node)

        return { data: gameData }
    }
}

export default new Parser()
