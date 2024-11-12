/* eslint-env jasmine */
import Parser from './parser'
import Serializer from './serializer'

function checkGrammar(code: string) {
    // check that it does not throw an Error
    const { data } = Parser.parse(code)
    const json = new Serializer(data).toJson()
    expect(json).toMatchSnapshot()
    const game2 = Serializer.fromJson(json, code)

    // verify the toKey representation of all the rules is the same as before
    if (data.rules.length !== game2.rules.length) {
        throw new Error(`BUG: rule lengths do not match`)
    }
    data.rules.forEach((rule, index) => {
        const rule2 = game2.rules[index]
        // if (rule.toKey() !== rule2.toKey()) {
        //     debugger
        //     throw new Error(`BUG: rule.toKey mismatch.\norig=${rule.toKey()}\nnew =${rule2.toKey()}`)
        // }
        expect(rule.toKey()).toEqual(rule2.toKey())
    })

    // const json2 = new Serializer(game2).toJson()
    // expect(json2).toEqual(json)
}

describe('serializer', () => {
    it('parses an empty game', () => {
        checkGrammar(`title Test Game`)
    })

    it('parses a simple game', () => {
        checkGrammar(`title Test Game
===
OBJECTS
===

background .
black

player P
yellow

===
COLLISIONLAYERS
===

background
player

===
RULES
===

[ NO player | background ] -> [ RANDOM player | > background ]

===
LEVELS
===

MESSAGE hello

..P..

`)
    })

    it('parses a game with RANDOM, loops, and debug statements', () => {
        checkGrammar(`title Test Game
===
OBJECTS
===

background .
black

player
yellow

===
LEGEND
===

P = player AND background

===
COLLISIONLAYERS
===

background
player

===
RULES
===

STARTLOOP

RANDOM RIGID LATE UP [ > player ] -> [ RANDOM player ] DEBUGGER

ENDLOOP

===
LEVELS
===

MESSAGE hello

P.

`)
    })
})
