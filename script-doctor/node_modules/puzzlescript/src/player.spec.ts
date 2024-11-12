/* eslint-env jasmine */
import fs from 'fs'
import path from 'path'
import { LevelEngine } from './engine'
import Parser from './parser/parser'
import { INPUT_BUTTON } from './util'

function parseEngine(code: string, levelNum = 0) {
    const { data } = Parser.parse(code)

    const engine = new LevelEngine(data)
    engine.setLevel(levelNum)
    return { engine, data }
}

describe('player movement', () => {
    it('evaluates a simple game', () => {
        const { engine, data } = parseEngine(`title foo

        (verbose_logging)
        (debug)

        (run_rules_on_level_start)

        realtime_interval 0.1


        ===
        OBJECTS
        ===

        background
        green

        player
        Yellow

        ===
        LEGEND
        ===

        . = background
        P = Player

        ====
        SOUNDS
        ====

        ====
        COLLISIONLAYERS
        ====

        background
        player

        ====
        RULES
        ====

        ===
        WINCONDITIONS
        ===

        ===
        LEVELS
        ===

        P.

        `) // end game

        const playerSprite = data.getSpriteByName('player')
        engine.press(INPUT_BUTTON.RIGHT)
        engine.tick()

        expect(engine.getCurrentLevel().getCells()[0][1].getSpritesAsSet().has(playerSprite)).toBe(true)
        expect(engine.getCurrentLevel().getCells()[0][0].getSpritesAsSet().has(playerSprite)).toBe(false)
    })

    it('players next to each other should move in unison', () => {
        const { engine, data } = parseEngine(`title foo

        (verbose_logging)
        (debug)

        (run_rules_on_level_start)

        realtime_interval 0.1

        ===
        OBJECTS
        ===

        background
        green

        player
        Yellow

        ===
        LEGEND
        ===

        . = background
        P = Player

        ====
        SOUNDS
        ====

        ====
        COLLISIONLAYERS
        ====

        background
        player

        ====
        RULES
        ====

        ===
        WINCONDITIONS
        ===

        ===
        LEVELS
        ===

        PP..

        `) // end game

        const playerSprite = data.getSpriteByName('player')
        engine.press(INPUT_BUTTON.RIGHT)
        engine.tick()

        expect(engine.getCurrentLevel().getCells()[0][0].getSpritesAsSet().has(playerSprite)).toBe(false)
        expect(engine.getCurrentLevel().getCells()[0][1].getSpritesAsSet().has(playerSprite)).toBe(true)
        expect(engine.getCurrentLevel().getCells()[0][2].getSpritesAsSet().has(playerSprite)).toBe(true)
        expect(engine.getCurrentLevel().getCells()[0][3].getSpritesAsSet().has(playerSprite)).toBe(false)

        engine.press(INPUT_BUTTON.RIGHT)
        engine.tick()

        expect(engine.getCurrentLevel().getCells()[0][0].getSpritesAsSet().has(playerSprite)).toBe(false)
        expect(engine.getCurrentLevel().getCells()[0][1].getSpritesAsSet().has(playerSprite)).toBe(false)
        expect(engine.getCurrentLevel().getCells()[0][2].getSpritesAsSet().has(playerSprite)).toBe(true)
        expect(engine.getCurrentLevel().getCells()[0][3].getSpritesAsSet().has(playerSprite)).toBe(true)
    })

    it('wantsToMove should become applied to sprites in another bracket', () => {
        const { engine, data } = parseEngine(`title foo

        (verbose_logging)
        (debug)

        (run_rules_on_level_start)

        realtime_interval 0.1

        ===
        OBJECTS
        ===

        background
        green

        player
        Yellow

        shadow
        black

        ===
        LEGEND
        ===

        . = background
        P = Player
        S = shadow

        ====
        SOUNDS
        ====

        ====
        COLLISIONLAYERS
        ====

        background
        player
        shadow

        ====
        RULES
        ====

        [ > player ] [ shadow ] -> [ > player ] [ > shadow ]

        ===
        WINCONDITIONS
        ===

        ===
        LEVELS
        ===

        P.
        S.

        `) // end game

        const playerSprite = data.getSpriteByName('player')
        const shadowSprite = data.getSpriteByName('shadow')
        engine.press(INPUT_BUTTON.RIGHT)
        engine.tick()

        expect(engine.getCurrentLevel().getCells()[0][0].getSpritesAsSet().has(playerSprite)).toBe(false)
        expect(engine.getCurrentLevel().getCells()[0][1].getSpritesAsSet().has(playerSprite)).toBe(true)

        expect(engine.getCurrentLevel().getCells()[1][0].getSpritesAsSet().has(shadowSprite)).toBe(false)
        expect(engine.getCurrentLevel().getCells()[1][1].getSpritesAsSet().has(shadowSprite)).toBe(true)
    })

    it('wantsToMove should remain when updating sprites', () => {
        const { engine, data } = parseEngine(`title foo

        (verbose_logging)
        (debug)

        (run_rules_on_level_start)

        realtime_interval 0.1

        ===
        OBJECTS
        ===

        background
        green

        player
        Yellow

        ===
        LEGEND
        ===

        . = background
        P = Player

        ====
        SOUNDS
        ====

        ====
        COLLISIONLAYERS
        ====

        background
        player

        ====
        RULES
        ====

        [ player ] -> [ player ]

        ===
        WINCONDITIONS
        ===

        ===
        LEVELS
        ===

        P.

        `) // end game

        const playerSprite = data.getSpriteByName('player')
        engine.press(INPUT_BUTTON.RIGHT)
        engine.tick()

        expect(engine.getCurrentLevel().getCells()[0][0].getSpritesAsSet().has(playerSprite)).toBe(false)
        expect(engine.getCurrentLevel().getCells()[0][1].getSpritesAsSet().has(playerSprite)).toBe(true)
    })

    it('wantsToMove should be removed when the condition has a direction but the right does not', () => {
        const { engine, data } = parseEngine(`title foo

        (verbose_logging)
        (debug)

        (run_rules_on_level_start)

        realtime_interval 0.1

        ===
        OBJECTS
        ===

        background
        green

        player
        Yellow

        ===
        LEGEND
        ===

        . = background
        P = Player

        ====
        SOUNDS
        ====

        ====
        COLLISIONLAYERS
        ====

        background
        player

        ====
        RULES
        ====

        [ > Player ] -> [ Player ]

        ===
        WINCONDITIONS
        ===

        ===
        LEVELS
        ===

        P.

        `) // end game

        const playerSprite = data.getSpriteByName('player')
        engine.press(INPUT_BUTTON.RIGHT)
        engine.tick()

        expect(engine.getCurrentLevel().getCells()[0][0].getSpritesAsSet().has(playerSprite)).toBe(true)
        expect(engine.getCurrentLevel().getCells()[0][1].getSpritesAsSet().has(playerSprite)).toBe(false)
    })

    it('only creates one Player when Player is an OR tile', () => {
        const { engine, data } = parseEngine(`title foo

        ===
        OBJECTS
        ===

        background
        green

        player1
        Yellow

        player2
        blue

        ===
        LEGEND
        ===

        . = background
        P = Player1
        Player = player1 OR player2

        ====
        SOUNDS
        ====

        ====
        COLLISIONLAYERS
        ====

        background
        player

        ====
        RULES
        ====

        ===
        WINCONDITIONS
        ===

        ===
        LEVELS
        ===

        P..

        `) // end game

        const player1 = data.getSpriteByName('player1')
        const player2 = data.getSpriteByName('player2')
        engine.press(INPUT_BUTTON.RIGHT)
        engine.tick()

        expect(player1.getCellsThatMatch().size).toBe(1)
        expect(player2.getCellsThatMatch().size).toBe(0)
        expect(engine.getCurrentLevel().getCells()[0][1].getSpritesAsSet().has(player1)).toBe(true)
    })

    it('preserves wantsToMove when sprite changes', () => {
        const { engine, data } = parseEngine(`title foo

        ===
        OBJECTS
        ===

        background
        green

        player1
        Yellow

        player2
        blue

        ===
        LEGEND
        ===

        . = background
        P = Player1 AND Background
        Player = player1 OR player2

        ====
        SOUNDS
        ====

        ====
        COLLISIONLAYERS
        ====

        background
        player

        ====
        RULES
        ====

        [ Player ] -> [ Player1 ]

        ===
        WINCONDITIONS
        ===

        ===
        LEVELS
        ===

        P.

        `) // end game

        const player1 = data.getSpriteByName('player1')
        engine.press(INPUT_BUTTON.RIGHT)
        engine.tick()

        expect(player1.getCellsThatMatch().size).toBe(1)
        expect(engine.getCurrentLevel().getCells()[0][1].getSpritesAsSet().has(player1)).toBe(true)
    })

    it('plays a level of Beam Islands', () => {
        const LEVEL_NUM = 3
        const LEVEL_SOLUTION = 'lluuuxlduruuxddddd'
        const { engine } = parseEngine(fs.readFileSync(path.join(__dirname, '../games/beam-islands/script.txt'), 'utf-8'), LEVEL_NUM) // end game
        let didWin = false

        const keypresses = LEVEL_SOLUTION.split('')
        for (const key of keypresses) {
            switch (key) {
                case 'u': engine.press(INPUT_BUTTON.UP); break
                case 'd': engine.press(INPUT_BUTTON.DOWN); break
                case 'l': engine.press(INPUT_BUTTON.LEFT); break
                case 'r': engine.press(INPUT_BUTTON.RIGHT); break
                case 'x': engine.press(INPUT_BUTTON.ACTION); break
            }
            const { isWinning } = engine.tick()
            if (isWinning) {
                didWin = true
            }
        }
        expect(didWin).toBe(true)
    })

})
