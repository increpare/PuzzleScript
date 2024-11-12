import { GameData } from '.'
import { Cell, CellSaveState } from './engine'
import { GameMetadata } from './models/metadata'
import { A11Y_MESSAGE } from './models/rule'
import { GameSprite } from './models/tile'
import { Soundish } from './parser/astTypes'

export type Optional<T> = T | null

export enum RULE_DIRECTION {
    UP = 'UP',
    DOWN = 'DOWN',
    LEFT = 'LEFT',
    RIGHT = 'RIGHT',
    ACTION = 'ACTION',
    STATIONARY = 'STATIONARY',
    RANDOMDIR = 'RANDOMDIR'
}

export enum INPUT_BUTTON {
    UP = 'UP',
    DOWN = 'DOWN',
    LEFT = 'LEFT',
    RIGHT = 'RIGHT',
    ACTION = 'ACTION',
    UNDO = 'UNDO',
    RESTART = 'RESTART'
}

export enum RULE_DIRECTION_RELATIVE {
    RELATIVE_LEFT = '<',
    RELATIVE_RIGHT = '>',
    RELATIVE_UP = '^',
    RELATIVE_DOWN = 'V'
}

export type RULE_DIRECTION_WITH_RELATIVE = RULE_DIRECTION | RULE_DIRECTION_RELATIVE

// From https://stackoverflow.com/questions/10865025/merge-flatten-an-array-of-arrays-in-javascript/39000004#39000004
export function _flatten<T>(arrays: T[][]) {
    // return [].concat.apply([], arrays) as T[]
    const ret: T[] = []
    arrays.forEach((ary) => {
        ary.forEach((item) => {
            ret.push(item)
        })
    })
    return ret
}

// export function filterNulls<T>(items: Array<Optional<T>>) {
//     const ret: T[] = []
//     items.forEach((x) => {
//         if (x) { ret.push(x) }
//     })
//     return ret
// }

// export function _zip<T1, T2>(array1: T1[], array2: T2[]) {
//     if (array1.length < array2.length) {
//         throw new Error(`BUG: Zip array length mismatch ${array1.length} != ${array2.length}`)
//     }
//     return array1.map((v1, index) => {
//         return [v1, array2[index]]
//     })
// }

// export function _extend(dest: any, ...rest: any[]) {
//     for (const obj of rest) {
//         for (const key of Object.keys(obj)) {
//             dest[key] = obj[key]
//         }
//     }
//     return dest
// }

export function _debounce(callback: () => any) {
    let timeout: any// NodeJS.Timer
    return () => {
        if (timeout) {
            clearTimeout(timeout)
        }
        timeout = setTimeout(() => {
            callback()
        }, 10)
    }
}

export function opposite(dir: RULE_DIRECTION) {
    switch (dir) {
        case RULE_DIRECTION.UP:
            return RULE_DIRECTION.DOWN
        case RULE_DIRECTION.DOWN:
            return RULE_DIRECTION.UP
        case RULE_DIRECTION.LEFT:
            return RULE_DIRECTION.RIGHT
        case RULE_DIRECTION.RIGHT:
            return RULE_DIRECTION.LEFT
        default:
            throw new Error(`BUG: Invalid direction: "${dir}"`)
    }
}

export function setEquals<T>(set1: Set<T>, set2: Set<T>) {
    if (set1.size !== set2.size) return false
    for (const elem of set2) {
        if (!set1.has(elem)) return false
    }
    return true
}

export function setAddAll<T>(setA: Set<T>, iterable: Iterable<T>) {
    const newSet = new Set(setA)
    for (const elem of iterable) {
        newSet.add(elem)
    }
    return newSet
}

export function setIntersection<T>(setA: Set<T>, setB: Iterable<T>) {
    const intersection = new Set<T>()
    for (const elem of setB) {
        if (setA.has(elem)) {
            intersection.add(elem)
        }
    }
    return intersection
}

export function setDifference<T>(setA: Set<T>, setB: Iterable<T>) {
    const difference = new Set(setA)
    for (const elem of setB) {
        difference.delete(elem)
    }
    return difference
}

// From https://stackoverflow.com/a/19303725
let seed = 1
let randomValuesForTesting: Optional<number[]> = null
export function nextRandom(maxNonInclusive: number) {
    if (randomValuesForTesting) {
        if (randomValuesForTesting.length <= seed - 1) {
            throw new Error(`BUG: the list of random values for testing was too short.
            See calls to setRandomValuesForTesting([...]).
            The list was [${randomValuesForTesting}]. Index being requested is ${seed - 1}`)
        }
        const ret = randomValuesForTesting[seed - 1]
        seed++
        // console.log(`Sending "random" value of "${ret}"`);

        return ret
    }
    const x = Math.sin(seed++) * 10000
    return Math.round((x - Math.floor(x)) * (maxNonInclusive - 1))
    // return Math.round(Math.random() * (maxNonInclusive - 1))
}
export function resetRandomSeed() {
    seed = 1
}
export function setRandomValuesForTesting(values: number[]) {
    randomValuesForTesting = values
    resetRandomSeed()
}
export function clearRandomValuesForTesting() {
    randomValuesForTesting = null
    resetRandomSeed()
}
export function getRandomSeed() {
    return seed
}

/**
 * A `DEBUGGER` flag in the game source that causes the evaluation to pause.
 * It works like the
 * [debugger](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/debugger)
 * keyword in JavaScript.
 *
 * **Note:** the game needs to run in debug mode (`node --inspect-brk path/to/puzzlescript.js` or `npm run play-debug`)
 * for this flag to have any effect.
 *
 * This string can be added to:
 *
 * - A Rule. Example: `DEBUGGER [ > player | cat ] -> [ > player | > cat ]`
 * - A bracket when the condition is updated: `[ > player | cat ] DEBUGGER -> [ > player | > cat ]`
 * - A bracket when it is evaluated: `[ > player | cat ] -> [ > player | > cat ] DEBUGGER`
 * - A neighbor when the condition is updated: `[ > player DEBUGGER | cat ] -> [ > player | > cat ]`
 * - A neighbor when it is evaluated: `[ > player | cat ] -> [ > player | > cat DEBUGGER ]`
 * - A tile when the condition is updated: `[ > player | DEBUGGER cat ] -> [ > player | > cat ]`
 * - A tile when it is matched: `[ > player | cat ] -> [ > player | DEBUGGER > cat ]`
 */
export enum DEBUG_FLAG {
    BREAKPOINT = 'DEBUGGER', // only when the rule matches elements
    /**
     * Pause when a Cell causes an entry to be removed from the set of matches for this rule/bracket/neighbor/tile
     */
    BREAKPOINT_REMOVE = 'DEBUGGER_REMOVE'
}

export interface ICacheable {
    toKey: () => string
}

export function spritesThatInteractWithPlayer(game: GameData) {
    const playerSprites = game.getPlayer().getSprites()
    const interactsWithPlayer = new Set<GameSprite>(playerSprites)

    // Add all the sprites in the same collision layer as the Player
    for (const playerSprite of interactsWithPlayer) {
        const collisionLayer = playerSprite.getCollisionLayer()
        for (const sprite of game.objects) {
            if (sprite.getCollisionLayer() === collisionLayer) {
                interactsWithPlayer.add(sprite)
            }
        }
    }

    // Add all the winCondition sprites
    for (const win of game.winConditions) {
        for (const tile of win.a11yGetTiles()) {
            for (const sprite of tile.getSprites()) {
                interactsWithPlayer.add(sprite)
            }
        }
    }

    // Add all the other sprites that interact with the player
    for (const rule of game.rules) {
        for (const sprites of rule.a11yGetConditionSprites()) {
            if (setIntersection(sprites, interactsWithPlayer).size > 0) {
                for (const sprite of sprites) {
                    interactsWithPlayer.add(sprite)
                }
            }
        }
    }

    // remove the background sprite (even though it transitively interacts)
    const background = game.getMagicBackgroundSprite()
    if (background) {
        interactsWithPlayer.delete(background)
    }

    // remove transparent sprites once the dependecies are found
    return new Set([...interactsWithPlayer].filter((s) => !s.isTransparent()))
}

// Webworker message interfaces

// Polls until a condition is true
export function pollingPromise<T>(ms: number, fn: () => T) {
    return new Promise<T>((resolve) => {
        const timer = setInterval(() => {
            const value = fn()
            if (value) {
                clearInterval(timer)
                resolve(value)
            }
        }, ms)
    })
}

export interface TypedMessageEvent<T> extends MessageEvent {
    data: T
}

export enum MESSAGE_TYPE {
    PAUSE = 'PAUSE',
    RESUME = 'RESUME',
    TICK = 'TICK',
    PRESS = 'PRESS',
    CLOSE = 'CLOSE',
    // Event handler events
    ON_GAME_CHANGE = 'ON_GAME_CHANGE',
    ON_PRESS = 'ON_PRESS',
    ON_MESSAGE = 'ON_MESSAGE',
    ON_MESSAGE_DONE = 'ON_MESSAGE_DONE',
    ON_LEVEL_LOAD = 'ON_LEVEL_LOAD',
    ON_LEVEL_CHANGE = 'ON_LEVEL_CHANGE',
    ON_WIN = 'ON_WIN',
    ON_SOUND = 'ON_SOUND',
    ON_TICK = 'ON_TICK',
    ON_PAUSE = 'ON_PAUSE',
    ON_RESUME = 'ON_RESUME'
}

export interface CellishJson {
    colIndex: number,
    rowIndex: number,
    spriteNames: string[]
}

export interface SerializedTickResult {
    changedCells: CellishJson[]
    soundToPlay: Optional<number>
    messageToShow: Optional<string>
    didWinGame: boolean
    didLevelChange: boolean
    wasAgainTick: boolean,
    a11yMessages: A11Y_MESSAGE<CellishJson, string>
}

export type WorkerMessage = {
    type: MESSAGE_TYPE.ON_GAME_CHANGE
    code: ArrayBuffer
    level: number
    checkpoint: Optional<CellSaveState>
} | {
    type: MESSAGE_TYPE.PRESS
    button: INPUT_BUTTON
} | {
    type: MESSAGE_TYPE.CLOSE
} | {
    type: MESSAGE_TYPE.PAUSE
} | {
    type: MESSAGE_TYPE.RESUME
} | {
    type: MESSAGE_TYPE.ON_MESSAGE_DONE
}

export type WorkerResponse = {
    type: MESSAGE_TYPE.ON_GAME_CHANGE
    payload: ArrayBuffer // IGraphJson
} | {
    type: MESSAGE_TYPE.TICK
    payload: SerializedTickResult
} | {
    type: MESSAGE_TYPE.PRESS
    payload: void
} | {
    type: MESSAGE_TYPE.CLOSE
    payload: void
} | {
    type: MESSAGE_TYPE.PAUSE
    payload: void
} | {
    type: MESSAGE_TYPE.RESUME
    payload: void
} | {
    type: MESSAGE_TYPE.ON_PRESS
    direction: INPUT_BUTTON
} | {
    type: MESSAGE_TYPE.ON_MESSAGE
    message: string
} | {
    type: MESSAGE_TYPE.ON_LEVEL_LOAD
    level: number
    levelSize: Optional<{rows: number, cols: number}>
} | {
    type: MESSAGE_TYPE.ON_LEVEL_CHANGE
    level: number
    cells: Optional<CellishJson[][]>
    message: Optional<string>
} | {
    type: MESSAGE_TYPE.ON_WIN
} | {
    type: MESSAGE_TYPE.ON_PAUSE
} | {
    type: MESSAGE_TYPE.ON_RESUME
} | {
    type: MESSAGE_TYPE.ON_SOUND
    soundCode: number
} | {
    type: MESSAGE_TYPE.ON_TICK
    changedCells: CellishJson[]
    checkpoint: Optional<CellSaveState>
    hasAgain: boolean
    a11yMessages: Array<A11Y_MESSAGE<CellishJson, string>>
}

export interface PuzzlescriptWorker {
    postMessage(msg: WorkerMessage, transferrables?: Transferable[]): void
    addEventListener(type: 'message', handler: (msg: {data: WorkerResponse}) => void): void
}

export const shouldTick = (metadata: GameMetadata, lastTick: number) => {
    const now = Date.now()
    let minTime = Math.min(metadata.realtimeInterval || 1000, metadata.keyRepeatInterval || 1000, metadata.againInterval || 1000)
    if (minTime > 100 || Number.isNaN(minTime)) {
        minTime = .01
    }
    return (now - lastTick) >= (minTime * 1000)
}

// This interface is so the WebWorker does not have to instantiate Cells just to render to the screen
export interface Cellish {
    colIndex: number
    rowIndex: number
    getSprites(): GameSprite[]
    getSpritesAsSet(): Set<GameSprite>
    getWantsToMove(sprite: GameSprite): Optional<RULE_DIRECTION>
}

export interface GameEngineHandler {
    onGameChange(gameData: GameData): void
    onPress(dir: INPUT_BUTTON): void
    onMessage(msg: string): Promise<void>
    onLevelLoad(level: number, newLevelSize: Optional<{rows: number, cols: number}>): void
    onLevelChange(level: number, cells: Optional<Cellish[][]>, message: Optional<string>): void
    onWin(): void
    onSound(sound: Soundish): Promise<void>
    onTick(changedCells: Set<Cellish>, checkpoint: Optional<CellSaveState>, hasAgain: boolean, a11yMessages: Array<A11Y_MESSAGE<Cell, GameSprite>>): void
    onPause(): void
    onResume(): void
    // onGameChange(data: GameData): void
}

export interface GameEngineHandlerOptional {
    onGameChange?(gameData: GameData): void
    onPress?(dir: INPUT_BUTTON): void
    onMessage?(msg: string): Promise<void>
    onLevelLoad?(level: number, newLevelSize: Optional<{rows: number, cols: number}>): void
    onLevelChange?(level: number, cells: Optional<Cellish[][]>, message: Optional<string>): void
    onWin?(): void
    onSound?(sound: Soundish): Promise<void>
    onTick?(changedCells: Set<Cellish>, checkpoint: Optional<CellSaveState>, hasAgain: boolean, a11yMessages: Array<A11Y_MESSAGE<Cellish, GameSprite>>): void
    onPause?(): void
    onResume?(): void
    // onGameChange?(data: GameData): void
}

export class EmptyGameEngineHandler implements GameEngineHandler {
    private subHandlers: GameEngineHandlerOptional[]
    constructor(subHandlers?: GameEngineHandlerOptional[]) {
        this.subHandlers = subHandlers || []
    }
    public onGameChange(gameData: GameData) { for (const h of this.subHandlers) { h.onGameChange && h.onGameChange(gameData) } }
    public onPress(dir: INPUT_BUTTON) { for (const h of this.subHandlers) { h.onPress && h.onPress(dir) } }
    public async onMessage(msg: string) { for (const h of this.subHandlers) { h.onMessage && await h.onMessage(msg) } }
    public onLevelLoad(level: number, newLevelSize: Optional<{rows: number, cols: number}>) { for (const h of this.subHandlers) { h.onLevelLoad && h.onLevelLoad(level, newLevelSize) } }
    public onLevelChange(level: number, cells: Optional<Cellish[][]>, message: Optional<string>) { for (const h of this.subHandlers) { h.onLevelChange && h.onLevelChange(level, cells, message) } }
    public onWin() { for (const h of this.subHandlers) { h.onWin && h.onWin() } }
    public async onSound(sound: Soundish) { for (const h of this.subHandlers) { h.onSound && h.onSound(sound) } }
    public onTick(changedCells: Set<Cellish>, checkpoint: Optional<CellSaveState>, hasAgain: boolean, a11yMessages: Array<A11Y_MESSAGE<Cellish, GameSprite>>) {
        for (const h of this.subHandlers) { h.onTick && h.onTick(changedCells, checkpoint, hasAgain, a11yMessages) }
    }
    public onPause() { for (const h of this.subHandlers) { h.onPause && h.onPause() } }
    public onResume() { for (const h of this.subHandlers) { h.onResume && h.onResume() } }
    // public onGameChange(data: GameData) { this.subHandlers.forEach(h => h.onGameChange && h.onGameChange(data)) }
}

export interface Engineish {
    setGame(code: string, level: number, checkpoint: Optional<CellSaveState>): void
    dispose(): void
    pause?(): void
    resume?(): void
    press?(dir: INPUT_BUTTON): void
    tick?(): void
}
