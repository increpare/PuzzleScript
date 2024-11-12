import { Optional, RULE_DIRECTION } from '..'
import { CollisionLayer } from '../models/collisionLayer'
import { HexColor, IColor, TransparentColor } from '../models/colors'
import { GameData } from '../models/game'
import { Dimension, GameMetadata } from '../models/metadata'
import { IRule, ISimpleBracket, SimpleBracket, SimpleEllipsisBracket, SimpleNeighbor, SimpleRule, SimpleRuleGroup, SimpleRuleLoop, SimpleTileWithModifier } from '../models/rule'
import { GameLegendTileAnd, GameLegendTileOr, GameLegendTileSimple, GameSprite, GameSpritePixels, IGameTile } from '../models/tile'
import { WinConditionOn, WinConditionSimple } from '../models/winCondition'
import * as ast from './astTypes'

// const EXAMPLE = {
//     metadata: { author: 'Phil' },
//     colors: {
//         123: '#cc99cc'
//     },
//     sounds: {
//         12: { /* ... */ }
//     },
//     collisionLayers: [
//         0, 1, 2
//     ],
//     commands: {
//         901: { type: 'MESSAGE', message: 'hello world'}
//     },

//     sprites: {
//         234: { name, pixels: [[123, 123]], collisionLayer: 0, sounds: {} }
//     },
//     tiles: {
//         345: { type: 'OR', sprites: [234, 234, 234], collisionLayer: 0}
//     },
//     tilesWithModifiers: {
//         456: { direction: 'UP', negation: false, tile: 345}
//     },
//     neighbors: {
//         567: { tilesWithModifiers: [456, 456]}
//     },
//     conditions: {
//         678: { type: 'NORMAL', direction: 'UP', neighbors: [567]}
// //        6789: { type: 'ELLIPSIS', direction: 'UP', beforeNeighbors: [567], afterNeighbors: [567]}
//     },
//     actionMutations: {
//         abc: {
//             bcd: {type: 'SET', wantsToMove: 'UP', tileOrSprite: 234 },
//             cde: {type: 'REMOVE', tileOrSprite: 345}
//         }
//     },
//     ellipsisBrackets: {
//         6789: { type: 'ELLIPSIS', direction: 'UP', beforeCondition: 678, afterCondition: 678}
//     },
//     bracketPairs: {
//         def: { condition: 678, actionMutations: ['abc']}
//     },
//     simpleRules: {
//         789: { type: 'SIMPLE', bracketPairs: [
//             {conditionBracket: 678, actionMutations: ['bcd']},
//             {conditionBracket: 6789, actionMutations: ['bcd', 'cde'], commands: [890]}
//         ]}
//     },
//     clusteredRules: { // maybe we should make groups and loops separate ()
//         890: { type: 'GROUP', rules: [789, 789, 789, 789]},
//         8901: { type: 'LOOP', rules: [7890, 7890, 7890]}
//     },
//     orderedRules: [
//         890, 8901
//     ],
//     levels: [
//         {type: 'MESSAGE', message: 'hello world'},
//         {type: 'MAP', mapTilesOrSprites: [[345, 234], [345, 345]]}
//     ]
// }

type ColorId = string
type SpriteId = string
type TileId = string
type CollisionId = string
type BracketId = string
type NeighborId = string
type TileWithModifierId = string
type CommandId = string
type RuleId = string
type SoundId = string
// type ActionMutationsId = string
// type BracketPairId = string

interface ISourceNode {
    _sourceOffset: number
}

interface IGraphSprite extends ISourceNode {
    name: string,
    pixels: Array<Array<Optional<ColorId>>>,
    collisionLayer: CollisionId,
    // sounds: {}
}

enum TILE_TYPE {
    OR = 'OR',
    AND = 'AND',
    SIMPLE = 'SIMPLE',
    SPRITE = 'SPRITE'
}
type GraphTile = ISourceNode & ({
    type: TILE_TYPE.OR
    name: string
    sprites: SpriteId[]
    collisionLayers: CollisionId[]
} | {
    type: TILE_TYPE.AND
    name: string
    sprites: SpriteId[]
    collisionLayers: CollisionId[]
} | {
    type: TILE_TYPE.SIMPLE
    name: string
    sprite: SpriteId
    collisionLayers: CollisionId[]
} | {
    type: TILE_TYPE.SPRITE
    name: string
    sprite: SpriteId
    collisionLayer: CollisionId
})

// type BracketPair = {
//     conditionBracket: BracketId,
//     actionMutations: BracketId, // ActionMutation
//     commands: CommandId[]
// }

interface IGraphGameMetadata {
    author: Optional<string>
    homepage: Optional<string>
    youtube: Optional<string>
    zoomscreen: Optional<Dimension>
    flickscreen: Optional<Dimension>
    colorPalette: Optional<string>
    backgroundColor: Optional<ColorId>
    textColor: Optional<ColorId>
    realtimeInterval: Optional<number>
    keyRepeatInterval: Optional<number>
    againInterval: Optional<number>
    noAction: boolean
    noUndo: boolean
    runRulesOnLevelStart: Optional<string>
    noRepeatAction: boolean
    throttleMovement: boolean
    noRestart: boolean
    requirePlayerMovement: boolean
    verboseLogging: boolean
}

function toRULE_DIRECTION(dir: RULE_DIRECTION): RULE_DIRECTION {
    return dir
}

class MapWithId<T, TJson> {
    private counter: number
    private prefix: string
    private idMap: Map<T, string>
    private jsonMap: Map<T, TJson>
    constructor(prefix: string) {
        this.counter = 0
        this.prefix = prefix
        this.idMap = new Map()
        this.jsonMap = new Map()
    }

    public set(key: T, value: TJson, id?: string) {
        if (!this.idMap.has(key)) {
            this.idMap.set(key, id || this.freshId())
        }
        this.jsonMap.set(key, value)
        return this.getId(key)
    }

    public get(key: T) {
        const value = this.jsonMap.get(key)
        if (!value) {
            debugger; throw new Error(`BUG: Element has not been added to the set`) // tslint:disable-line:no-debugger
        }
        return value
    }

    public getId(key: T) {
        const value = this.idMap.get(key)
        if (!value) {
            debugger; throw new Error(`BUG: Element has not been added to the set`) // tslint:disable-line:no-debugger
        }
        return value
    }

    public toJson() {
        const ret: {[key: string]: TJson} = {}
        for (const [obj, id] of this.idMap) {
            const json = this.jsonMap.get(obj)
            if (!json) {
                debugger; throw new Error(`BUG: Could not find matching json representation for "${id}"`) // tslint:disable-line:no-debugger
            }
            ret[id] = json
        }
        return ret
    }

    private freshId() {
        return `${this.prefix}-${this.counter++}`
    }
}

class DefiniteMap<K, V> {
    private map: Map<K, V>
    constructor() {
        this.map = new Map()
    }
    public get(key: K) {
        const v = this.map.get(key)
        if (!v) {
            throw new Error(`ERROR: JSON is missing key "${key}". Should have already been added`)
        }
        return v
    }
    public set(k: K, v: V) {
        this.map.set(k, v)
    }
    public values() {
        return this.map.values()
    }
}

export default class Serializer {

    public static fromJson(source: IGraphJson, code: string): GameData {
        // First, build up all of the lookup maps
        const colorMap: DefiniteMap<string, IColor> = new DefiniteMap()
        const spritesMap: DefiniteMap<string, GameSprite> = new DefiniteMap()
        const soundMap: DefiniteMap<string, ast.SoundItem<IGameTile>> = new DefiniteMap()
        const collisionLayerMap: DefiniteMap<string, CollisionLayer> = new DefiniteMap()
        const bracketMap: DefiniteMap<string, ISimpleBracket> = new DefiniteMap()
        const neighborsMap: DefiniteMap<string, SimpleNeighbor> = new DefiniteMap()
        const tileWithModifierMap: DefiniteMap<string, SimpleTileWithModifier> = new DefiniteMap()
        const tileMap: DefiniteMap<string, IGameTile> = new DefiniteMap()
        const ruleMap: DefiniteMap<string, IRule> = new DefiniteMap()
        const commandMap: DefiniteMap<string, ast.Command<ast.SoundItem<IGameTile>>> = new DefiniteMap()

        for (const [key, val] of Object.entries(source.colors)) {
            colorMap.set(key, new HexColor({ code, sourceOffset: 0 }, val))
        }
        const layers: DefiniteMap<string, IGameTile[]> = new DefiniteMap()
        for (const [key] of Object.entries(source.collisionLayers)) {
            layers.set(key, [])
        }

        const transparent = new TransparentColor({ code, sourceOffset: 0 })
        let spriteIndex = 0
        for (const [key, val] of Object.entries(source.sprites)) {
            const { _sourceOffset: sourceOffset, name, pixels } = val
            const sprite = new GameSpritePixels(
                { code, sourceOffset },
                name,
                null,
                pixels.map((row) => row.map((color) => {
                    if (color) {
                        return colorMap.get(color) || transparent
                    } else {
                        return transparent
                    }
                }))
            )
            // assign an index to each GameSprite
            sprite.allSpritesBitSetIndex = spriteIndex

            spriteIndex++
            spritesMap.set(key, sprite)
            layers.get(val.collisionLayer).push(sprite)
        }

        for (const [key, val] of Object.entries(source.tiles)) {
            const { _sourceOffset: sourceOffset } = val
            let tile
            switch (val.type) {
                case 'OR':
                    tile = new GameLegendTileOr({ code, sourceOffset }, val.name, val.sprites.map((item) => spritesMap.get(item)))
                    break
                case 'AND':
                    tile = new GameLegendTileAnd({ code, sourceOffset }, val.name, val.sprites.map((item) => spritesMap.get(item)))
                    break
                case 'SIMPLE':
                    tile = new GameLegendTileSimple({ code, sourceOffset }, val.name, spritesMap.get(val.sprite))
                    break
                case 'SPRITE':
                    tile = new GameLegendTileSimple({ code, sourceOffset }, val.name, spritesMap.get(val.sprite))
                    break
                default:
                    throw new Error(`ERROR: Unsupported tile type`)
            }
            tileMap.set(key, tile)
            // layers.get(val.collisionLayer).push(tile)
        }

        for (const [key, val] of Object.entries(source.sounds)) {
            switch (val.type) {
                case ast.SOUND_TYPE.SFX:
                case ast.SOUND_TYPE.WHEN:
                    soundMap.set(key, { ...val })
                    break
                case ast.SOUND_TYPE.SPRITE_DIRECTION:
                case ast.SOUND_TYPE.SPRITE_EVENT:
                case ast.SOUND_TYPE.SPRITE_MOVE:
                    soundMap.set(key, { ...val, sprite: tileMap.get(val.sprite) })
                    break
            }
        }
        for (const [key, val] of Object.entries(source.commands)) {
            switch (val.type) {
                case ast.COMMAND_TYPE.MESSAGE:
                    commandMap.set(key, val)
                    break
                case ast.COMMAND_TYPE.SFX:
                    commandMap.set(key, { ...val, sound: soundMap.get(val.sound) })
                    break
                case ast.COMMAND_TYPE.RESTART:
                case ast.COMMAND_TYPE.AGAIN:
                case ast.COMMAND_TYPE.CANCEL:
                case ast.COMMAND_TYPE.CHECKPOINT:
                case ast.COMMAND_TYPE.WIN:
                    commandMap.set(key, val)
                    break
                default:
                    throw new Error(`ERROR: Unsupported command type`)
            }
        }

        for (const [key, val] of Object.entries(source.collisionLayers)) {
            const { _sourceOffset: sourceOffset } = val
            collisionLayerMap.set(key, new CollisionLayer({ code, sourceOffset }, layers.get(key)))
        }

        for (const [key, val] of Object.entries(source.tilesWithModifiers)) {
            const { _sourceOffset: sourceOffset } = val
            tileWithModifierMap.set(key, new SimpleTileWithModifier({ code, sourceOffset }, val.isNegated, val.isRandom, val.direction, tileMap.get(val.tile), val.debugFlag))
        }

        for (const [key, val] of Object.entries(source.neighbors)) {
            const { _sourceOffset: sourceOffset } = val
            neighborsMap.set(key, new SimpleNeighbor({ code, sourceOffset }, new Set(val.tileWithModifiers.map((item) => tileWithModifierMap.get(item))), val.debugFlag))
        }

        for (const [key, val] of Object.entries(source.brackets)) {
            const { _sourceOffset: sourceOffset } = val
            switch (val.type) {
                case ast.BRACKET_TYPE.SIMPLE:
                    bracketMap.set(key, new SimpleBracket({ code, sourceOffset }, val.direction, val.neighbors.map((item) => neighborsMap.get(item)), val.debugFlag))
                    break
                case ast.BRACKET_TYPE.ELLIPSIS:
                    bracketMap.set(key, new SimpleEllipsisBracket(
                        { code, sourceOffset },
                        val.direction,
                        val.beforeNeighbors.map((item) => neighborsMap.get(item)),
                        val.afterNeighbors.map((item) => neighborsMap.get(item)),
                        val.debugFlag))
                    break
                default:
                    throw new Error(`ERROR: Unsupported bracket type`)
            }
        }

        for (const [key, val] of Object.entries(source.ruleDefinitions)) {
            const { _sourceOffset: sourceOffset } = val
            switch (val.type) {
                case ast.RULE_TYPE.SIMPLE:
                    ruleMap.set(key, new SimpleRule(
                        { code, sourceOffset },
                        val.conditions.map((item) => bracketMap.get(item)),
                        val.actions.map((item) => bracketMap.get(item)),
                        val.commands.map((item) => commandMap.get(item)),
                        val.isLate,
                        val.isRigid,
                        val.debugFlag))
                    break
                case ast.RULE_TYPE.GROUP:
                    ruleMap.set(key, new SimpleRuleGroup({ code, sourceOffset }, val.isRandom, val.rules.map((item) => ruleMap.get(item))))
                    break
                case ast.RULE_TYPE.LOOP:
                    ruleMap.set(key, new SimpleRuleLoop({ code, sourceOffset }, false/*TODO: Figure out if loops need isRandom*/, val.rules.map((item) => ruleMap.get(item))))
                    break
                default:
                    throw new Error(`ERROR: Unsupported rule type`)
            }
        }

        const levels = source.levels.map((item) => {
            switch (item.type) {
                case ast.LEVEL_TYPE.MAP:
                    return { ...item, cells: item.cells.map((row) => row.map((tile) => tileMap.get(tile))) }
                case ast.LEVEL_TYPE.MESSAGE:
                    return item
                default:
                    throw new Error(`ERROR: Invalid level type`)
            }
        })

        const winConditions = source.winConditions.map((item) => {
            const { _sourceOffset: sourceOffset } = item
            switch (item.type) {
                case ast.WIN_CONDITION_TYPE.SIMPLE:
                    return new WinConditionSimple({ code, sourceOffset }, item.qualifier, tileMap.get(item.tile))
                case ast.WIN_CONDITION_TYPE.ON:
                    return new WinConditionOn({ code, sourceOffset }, item.qualifier, tileMap.get(item.tile), tileMap.get(item.onTile))
                default:
                    throw new Error(`ERROR: Unsupported Win Condition type`)
            }
        })

        const metadata = new GameMetadata()
        for (const [key, val] of Object.entries(source.metadata)) {
            if (val) {
                switch (key) {
                    case 'backgroundColor':
                    case 'textColor':
                        metadata._setValue(key, colorMap.get(val as string))
                        break
                    case 'zoomScreen':
                    case 'flickScreen':
                        const { width, height } = val
                        metadata._setValue(key, new Dimension(width, height))
                        break
                    default:
                        metadata._setValue(key, val)
                }
            }
        }

        return new GameData(
            { code, sourceOffset: 0 },
            source.title, metadata,
            [...spritesMap.values()],
            [...tileMap.values()],
            [...soundMap.values()],
            [...collisionLayerMap.values()],
            source.rules.map((item) => ruleMap.get(item)) as SimpleRuleGroup[],
            winConditions, levels)
    }

    private readonly game: GameData
    private readonly colorsMap: Map<string, ColorId>
    private readonly spritesMap: MapWithId<GameSprite, IGraphSprite>
    private readonly soundMap: MapWithId<ast.SoundItem<IGameTile>, ast.SoundItem<string>>
    private readonly collisionLayerMap: MapWithId<CollisionLayer, ISourceNode>
    private readonly conditionsMap: MapWithId<ISimpleBracket, ast.Bracket<NeighborId>>
    private readonly neighborsMap: MapWithId<SimpleNeighbor, ast.Neighbor<TileWithModifierId>>
    private readonly tileWithModifierMap: MapWithId<SimpleTileWithModifier, ast.TileWithModifier<RULE_DIRECTION, TileId>>
    private readonly tileMap: MapWithId<IGameTile, GraphTile>
    private readonly ruleMap: MapWithId<IRule, ast.Rule<RuleId, RuleId, BracketId, CommandId>>
    private readonly commandMap: MapWithId<ast.Command<ast.SoundItem<IGameTile>>, ast.Command<SoundId>>
    private readonly winConditions: Array<ast.WinCondition<TileId>>
    private orderedRules: RuleId[]
    private levels: Array<ast.Level<TileId>>

    constructor(game: GameData) {
        this.game = game
        this.colorsMap = new Map()
        this.spritesMap = new MapWithId('sprite')
        this.soundMap = new MapWithId('sound')
        this.collisionLayerMap = new MapWithId('collision')
        this.conditionsMap = new MapWithId('bracket')
        this.neighborsMap = new MapWithId('neighbor')
        this.tileWithModifierMap = new MapWithId('twm')
        this.tileMap = new MapWithId('tile')
        this.ruleMap = new MapWithId('rule')
        this.commandMap = new MapWithId('command')

        if (this.game.metadata.backgroundColor) {
            const hex = this.game.metadata.backgroundColor.toHex()
            this.colorsMap.set(hex, hex)
        }
        if (this.game.metadata.textColor) {
            const hex = this.game.metadata.textColor.toHex()
            this.colorsMap.set(hex, hex)
        }
        // Load up the colors and sprites
        this.game.collisionLayers.forEach((item) => this.buildCollisionLayer(item))
        this.game.sounds.forEach((item) => {
            this.buildSound(item)
        })
        this.game.objects.forEach((sprite) => {
            this.buildSprite(sprite)
        })
        this.orderedRules = this.game.rules.map((item) => this.recBuildRule(item))

        this.game.legends.forEach((tile) => {
            this.buildTile(tile)
        })

        this.levels = this.game.levels.map((item) => this.buildLevel(item))

        this.winConditions = this.game.winConditions.map((item) => {
            if (item instanceof WinConditionOn) {
                const ret: ast.WinCondition<TileId> = {
                    _sourceOffset: item.__source.sourceOffset,
                    type: ast.WIN_CONDITION_TYPE.ON,
                    qualifier: item.qualifier,
                    tile: this.buildTile(item.tile),
                    onTile: this.buildTile(item.onTile)
                }
                return ret
            } else {
                const ret: ast.WinCondition<TileId> = {
                    _sourceOffset: item.__source.sourceOffset,
                    type: ast.WIN_CONDITION_TYPE.SIMPLE,
                    qualifier: item.qualifier,
                    tile: this.buildTile(item.tile)
                }
                return ret
            }
        })
    }
    public buildCollisionLayer(item: CollisionLayer) {
        return this.collisionLayerMap.set(item, { _sourceOffset: item.__source.sourceOffset })
    }
    public metadataToJson(): IGraphGameMetadata {
        return {
            author: this.game.metadata.author,
            homepage: this.game.metadata.homepage,
            youtube: this.game.metadata.youtube,
            zoomscreen: this.game.metadata.zoomscreen,
            flickscreen: this.game.metadata.flickscreen,
            colorPalette: this.game.metadata.colorPalette,
            backgroundColor: this.game.metadata.backgroundColor ? this.buildColor(this.game.metadata.backgroundColor) : null,
            textColor: this.game.metadata.textColor ? this.buildColor(this.game.metadata.textColor) : null,
            realtimeInterval: this.game.metadata.realtimeInterval,
            keyRepeatInterval: this.game.metadata.keyRepeatInterval,
            againInterval: this.game.metadata.againInterval,
            noAction: this.game.metadata.noAction,
            noUndo: this.game.metadata.noUndo,
            runRulesOnLevelStart: this.game.metadata.runRulesOnLevelStart,
            noRepeatAction: this.game.metadata.noRepeatAction,
            throttleMovement: this.game.metadata.throttleMovement,
            noRestart: this.game.metadata.noRestart,
            requirePlayerMovement: this.game.metadata.requirePlayerMovement,
            verboseLogging: this.game.metadata.verboseLogging
        }
    }
    public toJson(): IGraphJson {
        const colors: {[key: string]: string} = {}
        for (const [key, value] of this.colorsMap) {
            colors[key] = value
        }
        return {
            version: 1,
            title: this.game.title,
            metadata: this.metadataToJson(),
            colors,
            sounds: this.soundMap.toJson(),
            collisionLayers: this.collisionLayerMap.toJson(),
            commands: this.commandMap.toJson(),
            sprites: this.spritesMap.toJson(),
            tiles: this.tileMap.toJson(),
            tilesWithModifiers: this.tileWithModifierMap.toJson(),
            neighbors: this.neighborsMap.toJson(),
            brackets: this.conditionsMap.toJson(),
            ruleDefinitions: this.ruleMap.toJson(),
            winConditions: this.winConditions,
            rules: this.orderedRules,
            levels: this.levels
        }
    }
    private buildLevel(level: ast.Level<IGameTile>): ast.Level<TileId> {
        switch (level.type) {
            case ast.LEVEL_TYPE.MAP:
                return { ...level, cells: level.cells.map((row) => row.map((cell) => this.buildTile(cell))) }
            case ast.LEVEL_TYPE.MESSAGE:
                return level
            default:
                debugger; throw new Error(`BUG: Unsupported level subtype`) // tslint:disable-line:no-debugger
        }
    }
    private recBuildRule(rule: IRule): string {
        if (rule instanceof SimpleRule) {
            return this.ruleMap.set(rule, {
                type: ast.RULE_TYPE.SIMPLE,
                directions: [], // Simplified rules do not have directions
                conditions: rule.conditionBrackets.map((item) => this.buildConditionBracket(item)),
                actions: rule.actionBrackets.map((item) => this.buildConditionBracket(item)),
                commands: rule.commands.map((item) => this.buildCommand(item)),
                isRandom: null,
                isLate: rule.isLate(),
                isRigid: rule.hasRigid(),
                _sourceOffset: rule.__source.sourceOffset,
                debugFlag: rule.debugFlag
            })
        } else if (rule instanceof SimpleRuleGroup) {
            return this.ruleMap.set(rule, {
                type: ast.RULE_TYPE.GROUP,
                isRandom: rule.isRandom,
                rules: rule.getChildRules().map((item) => this.recBuildRule(item)),
                _sourceOffset: rule.__source.sourceOffset,
                debugFlag: null // TODO: Unhardcode me
            })
        } else if (rule instanceof SimpleRuleLoop) {
            const x: ast.RuleLoop<string> = {
                type: ast.RULE_TYPE.LOOP,
                // isRandom: rule.isRandom,
                rules: rule.getChildRules().map((item) => this.recBuildRule(item)),
                _sourceOffset: rule.__source.sourceOffset,
                debugFlag: null // TODO: unhardcode me
            }
            return this.ruleMap.set(rule, x)
        } else {
            debugger; throw new Error(`BUG: Unsupported rule type`) // tslint:disable-line:no-debugger
        }

    }
    private buildCommand(command: ast.Command<ast.SoundItem<IGameTile>>) {
        switch (command.type) {
            case ast.COMMAND_TYPE.SFX:
                return this.commandMap.set(command, { ...command, sound: this.soundMap.getId(command.sound) })
            case ast.COMMAND_TYPE.AGAIN:
            case ast.COMMAND_TYPE.CANCEL:
            case ast.COMMAND_TYPE.CHECKPOINT:
            case ast.COMMAND_TYPE.MESSAGE:
            case ast.COMMAND_TYPE.RESTART:
            case ast.COMMAND_TYPE.WIN:
                return this.commandMap.set(command, command)
            default:
                debugger; throw new Error(`BUG: Unsupoprted command type`) // tslint:disable-line:no-debugger
        }
    }
    private buildConditionBracket(bracket: ISimpleBracket): BracketId {
        if (bracket instanceof SimpleEllipsisBracket) {
            const b = bracket
            const before = b.beforeEllipsisBracket.getNeighbors().map((item) => this.buildNeighbor(item)) // this.buildConditionBracket(b.beforeEllipsisBracket)
            const after = b.afterEllipsisBracket.getNeighbors().map((item) => this.buildNeighbor(item)) // this.buildConditionBracket(b.afterEllipsisBracket)
            return this.conditionsMap.set(bracket, {
                type: ast.BRACKET_TYPE.ELLIPSIS,
                direction: toRULE_DIRECTION(b.direction),
                beforeNeighbors: before,
                afterNeighbors: after,
                _sourceOffset: bracket.__source.sourceOffset,
                debugFlag: b.debugFlag
            })
        } else if (bracket instanceof SimpleBracket) {
            return this.conditionsMap.set(bracket, {
                type: ast.BRACKET_TYPE.SIMPLE,
                direction: toRULE_DIRECTION(bracket.direction),
                neighbors: bracket.getNeighbors().map((item) => this.buildNeighbor(item)),
                _sourceOffset: bracket.__source.sourceOffset,
                debugFlag: bracket.debugFlag
            })
        } else {
            debugger; throw new Error(`BUG: Unsupported bracket type`) // tslint:disable-line:no-debugger
        }
    }
    private buildNeighbor(neighbor: SimpleNeighbor): NeighborId {
        return this.neighborsMap.set(neighbor, {
            tileWithModifiers: [...neighbor._tilesWithModifier].map((item) => this.buildTileWithModifier(item)),
            _sourceOffset: neighbor.__source.sourceOffset,
            debugFlag: null // TODO: Pull it out of the neighbor
        })
    }
    private buildTileWithModifier(t: SimpleTileWithModifier): TileWithModifierId {
        return this.tileWithModifierMap.set(t, {
            direction: t._direction ? toRULE_DIRECTION(t._direction) : null,
            isNegated: t._isNegated,
            isRandom: t._isRandom,
            tile: this.buildTile(t._tile),
            _sourceOffset: t.__source.sourceOffset,
            debugFlag: t._debugFlag
        })
    }
    private buildTile(tile: IGameTile): TileId {
        if (tile instanceof GameLegendTileOr) {
            return this.tileMap.set(tile, {
                type: TILE_TYPE.OR,
                name: tile.getName(),
                sprites: tile.getSprites().map((item) => this.buildSprite(item)),
                collisionLayers: tile.getCollisionLayers().map((item) => this.buildCollisionLayer(item)),
                _sourceOffset: tile.__source.sourceOffset
            })
        } else if (tile instanceof GameLegendTileAnd) {
            return this.tileMap.set(tile, {
                type: TILE_TYPE.AND,
                name: tile.getName(),
                sprites: tile.getSprites().map((item) => this.buildSprite(item)),
                collisionLayers: tile.getCollisionLayers().map((item) => this.buildCollisionLayer(item)),
                _sourceOffset: tile.__source.sourceOffset
            })
        } else if (tile instanceof GameLegendTileSimple) {
            return this.tileMap.set(tile, {
                type: TILE_TYPE.SIMPLE,
                name: tile.getName(),
                sprite: this.buildSprite(tile.getSprites()[0]),
                collisionLayers: tile.getCollisionLayers().map((item) => this.buildCollisionLayer(item)),
                _sourceOffset: tile.__source.sourceOffset
            })
        } else if (tile instanceof GameSprite) {
            return this.tileMap.set(tile, {
                type: TILE_TYPE.SPRITE,
                name: tile.getName(),
                sprite: this.buildSprite(tile),
                collisionLayer: this.buildCollisionLayer(tile.getCollisionLayer()),
                _sourceOffset: tile.__source.sourceOffset
            })
        } else {
            debugger; throw new Error(`BUG: Invalid tile type`) // tslint:disable-line:no-debugger
        }
    }
    private buildSprite(sprite: GameSprite): SpriteId {
        const { spriteHeight, spriteWidth } = this.game.getSpriteSize()
        return this.spritesMap.set(sprite, {
            name: sprite.getName(),
            collisionLayer: this.collisionLayerMap.getId(sprite.getCollisionLayer()),
            pixels: sprite.getPixels(spriteHeight, spriteWidth).map((row) => row.map((pixel) => {
                if (pixel.isTransparent()) {
                    return null
                } else {
                    return this.buildColor(pixel)
                }
            })),
            _sourceOffset: sprite.__source.sourceOffset
        })
    }
    private buildColor(color: IColor) {
        const hex = color.toHex()
        this.colorsMap.set(hex, hex)
        return hex
    }
    private buildSound(sound: ast.SoundItem<IGameTile>) {
        switch (sound.type) {
            case ast.SOUND_TYPE.SFX:
            case ast.SOUND_TYPE.WHEN:
                return this.soundMap.set(sound, { ...sound })
            case ast.SOUND_TYPE.SPRITE_DIRECTION:
            case ast.SOUND_TYPE.SPRITE_EVENT:
            case ast.SOUND_TYPE.SPRITE_MOVE:
                return this.soundMap.set(sound, { ...sound, sprite: this.buildTile(sound.sprite) })
        }
    }
}

export interface IGraphJson {
    version: number,
    title: string,
    metadata: IGraphGameMetadata,
    colors: {[key: string]: string},
    sounds: {[key: string]: ast.SoundItem<string>},
    collisionLayers: {[key: string]: ISourceNode},
    commands: {[key: string]: ast.Command<SoundId>},
    sprites: {[key: string]: IGraphSprite},
    tiles: {[key: string]: GraphTile},
    winConditions: Array<ast.WinCondition<TileId>>,
    tilesWithModifiers: {[key: string]: ast.TileWithModifier<RULE_DIRECTION, TileId>},
    neighbors: {[key: string]: ast.Neighbor<TileWithModifierId>},
    brackets: {[key: string]: ast.Bracket<NeighborId>},
    ruleDefinitions: {[key: string]: ast.Rule<RuleId, RuleId, BracketId, CommandId>},
    rules: RuleId[],
    levels: Array<ast.Level<TileId>>
}
