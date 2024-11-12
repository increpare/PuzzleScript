interface NearleyToken {
    value: any;
    [key: string]: any;
}
interface NearleyLexer {
    reset: (chunk: string, info: any) => void;
    next: () => NearleyToken | undefined;
    save: () => any;
    formatError: (token: never) => string;
    has: (tokenType: string) => boolean;
}
interface NearleyRule {
    name: string;
    symbols: NearleySymbol[];
    postprocess?: (d: any[], loc?: number, reject?: {}) => any;
}
declare type NearleySymbol = string | {
    literal: any;
} | {
    test: (token: any) => boolean;
};
interface Grammar {
    Lexer: NearleyLexer | undefined;
    ParserRules: NearleyRule[];
    ParserStart: string;
}
declare const grammar: Grammar;
export default grammar;
