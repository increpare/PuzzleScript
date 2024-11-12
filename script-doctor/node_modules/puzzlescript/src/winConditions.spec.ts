/* eslint-env jasmine */
import { LevelEngine } from './engine'
import Parser from './parser/parser'

function parseEngine(code: string) {
    const { data } = Parser.parse(code)

    const engine = new LevelEngine(data)
    engine.setLevel(0)
    return { engine, data }
}

function buildGame(winConditions: string[]) {
    return `title foo

    ========
    OBJECTS
    ========

    Background .
    gray

    Player
    transparent

    Hat H
    transparent

    Glasses
    transparent

    Dog D
    transparent

    Cat
    transparent

    =======
    LEGEND
    =======

    P = Player AND Glasses AND Hat
    L = Player

    =======
    SOUNDS
    =======

    ================
    COLLISIONLAYERS
    ================
    Background
    Player
    Hat
    Glasses
    Dog
    Cat

    ======
    RULES
    ======

    ==============
    WINCONDITIONS
    ==============

    ${winConditions.join('\n')}

    =======
    LEVELS
    =======

    PDHL

`
}

describe('Win Conditions', () => {

    it('detects conditions for simple checks', () => {
        function simple(conditions: string[], expected: boolean) {
            const { engine } = parseEngine(buildGame(conditions))
            const { isWinning } = engine.tick()
            expect(isWinning).toBe(expected)
        }
        simple(['NO Player'], false)
        simple(['NO Cat'], true)
        simple(['SOME Cat'], false)
        simple(['SOME Player'], true)
        simple(['ALL Glasses ON Player'], true)
        simple(['SOME Glasses ON Player'], true)
        simple(['NO Glasses ON Player'], false)
        // All Target on CleanDishes
        simple(['ALL Player ON Hat'], false)
        simple(['NO Dog ON Player'], true)
        simple(['ALL Dog ON Player'], false)
        simple(['SOME Dog ON Player'], false)

    })

})
