import { GameData, IGameNode } from './models/game'
import { IRule } from './models/rule'

// These types are just so that the Code Coverage JSON objects are strongly-typed
interface ICoverageLocation {
    line: number
    col: number
}
interface ICoverageLocationRange {
    start: ICoverageLocation
    end: ICoverageLocation
}
interface ICoverageFunction {
    name: string,
    decl: ICoverageLocationRange,
    loc: ICoverageLocationRange,
    line: number
}
interface ICoverageCount { [id: string]: number }
interface ICoverageStatements { [id: string]: ICoverageLocationRange }
interface ICoverageFunctions { [id: string]: ICoverageFunction }
interface ICoverageEntry {
    path: string
    s: ICoverageCount
    f: ICoverageCount
    statementMap: ICoverageStatements
    fnMap: ICoverageFunctions
    branchMap: object
    b: object
}

export function saveCoverageFile(data: GameData, absPath: string, pathRelative: (p: string) => string) {
    // record the appliedRules in a coverage.json file
    // key = Line number, value = count of times the rule executed
    const codeCoverageTemp = new Map<string, {count: number, node: IGameNode}>()

    // First add all the Tiles, Legend Items, collisionLayers, Rules, and Levels.
    // Then, after running, add all the matched rules.
    function coverageKey(node: IGameNode) {
        // the HTML reporter does not like multiline fields.
        // Rather than report multiple times, we just report the 1st line
        // This is a problem with `startloop`
        const { start, end } = node.__getLineAndColumnRange()
        if (start.line !== end.line) {
            return JSON.stringify({
                end: {
                    col: start.col + 3,
                    line: start.line
                },
                start: {
                    col: start.col - 1,
                    line: start.line
                }
            })
        } else {
            return JSON.stringify({
                end: {
                    col: end.col - 1,
                    line: end.line
                },
                start: {
                    col: start.col - 1,
                    line: start.line
                }
            })
        }
    }
    function addNodeToCoverage(node: IGameNode) {
        codeCoverageTemp.set(coverageKey(node), { count: 0, node })
    }
    // data.objects.forEach(addNodeToCoverage)
    // data.legends.forEach(addNodeToCoverage)
    // data.sounds.forEach(addNodeToCoverage)
    // data.collisionLayers.forEach(addNodeToCoverage) // these entries are sometimes (always?) null
    data.rules.forEach(addNodeToCoverage)
    data.winConditions.forEach(addNodeToCoverage)
    // data.levels.forEach(addNodeToCoverage)

    function recursivelyGetRules(rules: IRule[]) {
        let ret: IRule[] = []
        for (const rule of rules) {
            ret.push(rule)
            ret = ret.concat(recursivelyGetRules(rule.getChildRules()))
        }
        return ret
    }

    // record the tick coverage
    const ary = new Array<IGameNode>()
    const nodesToCover = ary.concat(recursivelyGetRules(data.rules))
        /*.concat(data.objects).concat(data.legends)*/
        .concat(data.winConditions)
        /*.concat(data.levels)*/

    for (const node of nodesToCover) {
        const line = coverageKey(node)
        const nodeCount = node.__coverageCount || 0
        const existingEntry = codeCoverageTemp.get(line)
        if (existingEntry) {
            codeCoverageTemp.set(line, { count: existingEntry.count + nodeCount, node })
        } else {
            codeCoverageTemp.set(line, { count: nodeCount, node })
        }
    }

    const codeCoverage2 = [...codeCoverageTemp.entries()].map(([key, { count, node }]) => {
        const loc = JSON.parse(key) as ICoverageLocationRange
        return { loc, count, node }
    })
    // Generate the coverage.json file from which Rules were applied
    const statementMap: ICoverageStatements = {}
    const fnMap: ICoverageFunctions = {}
    const f: ICoverageCount = {}
    const s: ICoverageCount = {}

    // Add all the matched rules
    codeCoverage2.forEach((entry, index) => {
        const { loc, node } = entry
        let { count } = entry

        // sometimes count can be null
        if (!(count >= 0)) {
            count = 0
        }
        s[index] = count
        statementMap[index] = loc
        f[index] = count
        fnMap[index] = {
            decl: loc,
            line: loc.start.line,
            loc,
            name: node.toSourceString()
        }
    })

    const relPath = pathRelative(absPath)

    const codeCoverageEntry: ICoverageEntry = {
        b: {},
        branchMap: {},
        f,
        fnMap,
        path: relPath,
        s,
        statementMap
    }
    const codeCoverageObj: { [path: string]: ICoverageEntry } = {}
    codeCoverageObj[relPath] = codeCoverageEntry
    return codeCoverageObj
}
