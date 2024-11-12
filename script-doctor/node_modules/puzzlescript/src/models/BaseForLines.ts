import { Optional } from '../util'

export interface IGameCode {
    code: string
    sourceOffset: number
}

export interface IGameCodeWithSource {
    sourceString: string
}
// Return an object with the line and column information for the given
// offset in `str`.
// From https://github.com/harc/ohm/blob/b88336faf69e7bd89e309931b60445c3dfd495ab/src/util.js#L56
export function getLineAndColumn(str: string, offset: number) {
    let lineNum = 1
    let colNum = 1

    let currOffset = 0
    let lineStartOffset = 0

    let nextLine = null
    let prevLine = null
    let prevLineStartOffset = -1

    while (currOffset < offset) {
        const c = str.charAt(currOffset++)
        if (c === '\n') {
            lineNum++
            colNum = 1
            prevLineStartOffset = lineStartOffset
            lineStartOffset = currOffset
        } else if (c !== '\r') {
            colNum++
        }
    }
    // Find the end of the target line.
    let lineEndOffset = str.indexOf('\n', lineStartOffset)
    if (lineEndOffset === -1) {
        lineEndOffset = str.length
    } else {
        // Get the next line.
        const nextLineEndOffset = str.indexOf('\n', lineEndOffset + 1)
        nextLine = nextLineEndOffset === -1 ? str.slice(lineEndOffset)
            : str.slice(lineEndOffset, nextLineEndOffset)
        // Strip leading and trailing EOL char(s).
        nextLine = nextLine.replace(/^\r?\n/, '').replace(/\r$/, '')
    }

    // Get the previous line.
    if (prevLineStartOffset >= 0) {
        prevLine = str.slice(prevLineStartOffset, lineStartOffset)
            .replace(/\r?\n$/, '')  // Strip trailing EOL char(s).
    }

    // Get the target line, stripping a trailing carriage return if necessary.
    const line = str.slice(lineStartOffset, lineEndOffset).replace(/\r$/, '')

    return {
        lineNum,
        colNum,
        line,
        prevLine,
        nextLine
    }
}

export class BaseForLines {
    public readonly __source: IGameCode // tslint:disable-line:variable-name
    public __coverageCount: Optional<number> // tslint:disable-line:variable-name
    constructor(source: IGameCode) {
        if (!source) {
            throw new Error(`BUG: failed to provide the source when constructing this object`)
        }
        this.__source = source
        this.__coverageCount = null
        // This is only used for code coverage
        if (process.env.NODE_ENV === 'development') {
            this.__coverageCount = 0
        }
    }
    public __getSourceLineAndColumn() {
        const s = this.__source
        return getLineAndColumn(s.code, s.sourceOffset)
    }
    public toString() {
        const s = this.__source
        const { lineNum, colNum, line } = getLineAndColumn(s.code, s.sourceOffset)
        return `${line}\nSource: ${lineNum}:${colNum}`
    }
    public toSourceString() {
        const s = this.__source
        const { lineNum } = getLineAndColumn(s.code, s.sourceOffset)
        return s.code.split('\n')[lineNum - 1]
    }
    // This is mostly used for creating code coverage for the games. So we know which Rules (or objects) are not being matched
    public __getLineAndColumnRange() {
        const s = this.__source
        const { lineNum, colNum } = getLineAndColumn(s.code, s.sourceOffset)
        return {
            start: { line: lineNum, col: colNum },
            end: { line: lineNum, col: colNum + 1 }
        }
    }
    public __incrementCoverage() {
        if (process.env.NODE_ENV === 'development') {
            if (!this.__coverageCount) {
                this.__coverageCount = 0
            }
            this.__coverageCount++
        }
    }
}
