import { lookupColorPalette } from '../colors'
import { CollisionLayer } from '../models/collisionLayer'
import { HexColor, TransparentColor } from '../models/colors'
import { GameData } from '../models/game'
import { Dimension, GameMetadata } from '../models/metadata'
import { ISimpleBracket, SimpleBracket, SimpleEllipsisBracket, SimpleNeighbor, SimpleRule, SimpleRuleGroup, SimpleRuleLoop, SimpleTileWithModifier } from '../models/rule'
import { GameLegendTileAnd, GameLegendTileOr, GameLegendTileSimple, GameSprite, GameSpritePixels, GameSpriteSingleColor, IGameTile } from '../models/tile'
import { WinConditionOn, WinConditionSimple } from '../models/winCondition'
import { ICacheable, Optional, RULE_DIRECTION, RULE_DIRECTION_RELATIVE, RULE_DIRECTION_WITH_RELATIVE } from '../util'
import * as ast from './astTypes'

// Shorthand because the type signatures are long
type AST_Tile = ast.TileWithModifier<RULE_DIRECTION_WITH_RELATIVE, string>
type AST_Neighbor = ast.Neighbor<AST_Tile>
type AST_BasicRule = ast.SimpleRule<ast.Bracket<AST_Neighbor>, ast.Command<string>>
type AST_Rule = ast.Rule<
    ast.RuleGroup<
    ast.SimpleRule<
    ast.Bracket<AST_Neighbor>,
    ast.Command<string>
    >
    >,
    ast.SimpleRule<
    ast.Bracket<AST_Neighbor>,
    ast.Command<string>>,
    ast.Bracket<AST_Neighbor>, ast.Command<string>>

const RULE_DIRECTION_LIST = [
    RULE_DIRECTION.UP,
    RULE_DIRECTION.DOWN,
    RULE_DIRECTION.LEFT,
    RULE_DIRECTION.RIGHT
]

const RULE_DIRECTION_SET: Set<string> = new Set(RULE_DIRECTION_LIST)

function removeNulls<T>(ary: Array<Optional<T>>) {
    // return ary.filter(a => !!a)

    // TypeScript-friendly version
    const ret: T[] = []
    for (const item of ary) {
        if (item) {
            ret.push(item)
        }
    }
    return ret
}

function cacheSetAndGet<A extends ICacheable>(cache: Map<string, A>, obj: A) {
    const key = obj.toKey()
    if (!cache.has(key)) {
        cache.set(key, obj)
    }
    return cache.get(key) as A
}

function relativeDirectionToAbsolute(currentDirection: RULE_DIRECTION, relativeModifier: RULE_DIRECTION_RELATIVE) {
    let currentDir
    switch (currentDirection) {
        case RULE_DIRECTION.RIGHT:
            currentDir = 0
            break
        case RULE_DIRECTION.UP:
            currentDir = 1
            break
        case RULE_DIRECTION.LEFT:
            currentDir = 2
            break
        case RULE_DIRECTION.DOWN:
            currentDir = 3
            break
        default:
            throw new Error(`BUG! Invalid rule direction "${currentDirection}`)
    }

    switch (relativeModifier) {
        case RULE_DIRECTION_RELATIVE.RELATIVE_RIGHT:
            currentDir += 0
            break
        case RULE_DIRECTION_RELATIVE.RELATIVE_UP:
            currentDir += 1
            break
        case RULE_DIRECTION_RELATIVE.RELATIVE_LEFT:
            currentDir += 2
            break
        case RULE_DIRECTION_RELATIVE.RELATIVE_DOWN:
            currentDir += 3
            break
        default:
            throw new Error(`BUG! invalid relative direction "${relativeModifier}"`)
    }
    switch (currentDir % 4) {
        case 0:
            return RULE_DIRECTION.RIGHT
        case 1:
            return RULE_DIRECTION.UP
        case 2:
            return RULE_DIRECTION.LEFT
        case 3:
            return RULE_DIRECTION.DOWN
        default:
            throw new Error(`BUG! Incorrectly computed rule direction "${currentDirection}" "${relativeModifier}"`)
    }
}

export class AstBuilder {
    private readonly code: string
    private readonly tileCache: Map<string, IGameTile>
    private readonly soundCache: Map<string, ast.SoundItem<IGameTile>>
    constructor(code: string) {
        this.code = code
        this.tileCache = new Map()
        this.soundCache = new Map()
    }
    public build(root: ast.IASTGame<RULE_DIRECTION_WITH_RELATIVE, string, string, number | '.'>) {
        const source = this.toSource({ _sourceOffset: 0 })

        const metadata = new GameMetadata()
        root.metadata.forEach((pair) => {
            let value
            if (typeof pair.value === 'object' && pair.value.type) {
                switch (pair.value.type) {
                    case ast.COLOR_TYPE.HEX8:
                    case ast.COLOR_TYPE.HEX6:
                    case ast.COLOR_TYPE.NAME:
                        {
                            const v = pair.value
                            value = this.buildColor(v, metadata.colorPalette)
                        }
                        break
                    case 'WIDTH_AND_HEIGHT':
                        {
                            const v = pair.value
                            const v2 = v
                            value = new Dimension(v2.width, v2.height)
                        }
                        break
                    default:
                        throw new Error(`BUG: Invalid type at this point in time: ${pair.value}`)
                }
            } else {
                value = pair.value
            }
            metadata._setValue(pair.type, value)
        })

        const sprites = root.sprites.map((n) => this.buildSprite(n, metadata.colorPalette))
        // assign an index to each GameSprite
        sprites.forEach((sprite, index) => {
            sprite.allSpritesBitSetIndex = index
        })

        // Load the legend items up (used in Rules and Levels later on)
        const legendItems = root.legendItems.map((n) => this.buildLegendItem(n))
        const sounds = root.sounds.map((n) => this.buildSound(n))

        const collisionLayers = root.collisionLayers.map((n) => this.buildCollisionLayer(n))
        const simpleRules = this.buildSimpleRules(root.rules)

        const winConditions = root.winConditions.map((n) => this.buildWinConditon(n))
        const levels = root.levels.map((n) => this.buildLevel(n))

        return new GameData(source, root.title, metadata, sprites, legendItems, sounds, collisionLayers, simpleRules, winConditions, levels)
    }

    private buildSprite(node: ast.Sprite<number | '.'>, colorPalette: Optional<string>) {
        let ret: GameSprite
        if (node.type === ast.SPRITE_TYPE.WITH_PIXELS) {
            const source = this.toSource(node)
            const colors = node.colors.map((n) => this.buildColor(n, colorPalette))
            const pixels = node.pixels.map((row) => {
                return row.map((col) => {
                    if (col === '.') {
                        return new TransparentColor(source)
                    } else {
                        return colors[col] || new TransparentColor(source)
                    }
                })
            }) // Pixel colors are 0-indexed.

            ret = new GameSpritePixels(source, node.name, node.mapChar, pixels)
        } else {
            ret = new GameSpriteSingleColor(this.toSource(node), node.name, node.mapChar, node.colors.map((n) => this.buildColor(n, colorPalette)))
        }
        this.cacheAdd(node.name, ret)
        if (node.mapChar) {
            this.cacheAdd(node.mapChar, ret)
        }
        return ret
    }

    private buildColor(node: ast.IColor, colorPalette: Optional<string>) {
        const source = this.toSource(node)
        const currentColorPalette = colorPalette || 'arnecolors'
        switch (node.type) {
            case ast.COLOR_TYPE.HEX8:
            case ast.COLOR_TYPE.HEX6:
                return new HexColor(source, node.value)
            case ast.COLOR_TYPE.NAME:
                if (node.value.toUpperCase() === 'TRANSPARENT') {
                    return new TransparentColor(source)
                } else {
                    // Look up the color
                    const hex = lookupColorPalette(currentColorPalette, node.value)
                    if (hex) {
                        return new HexColor(source, hex)
                    } else {
                        return new TransparentColor(source)
                    }
                }
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private buildLegendItem(node: ast.LegendItem<string>) {
        const source = this.toSource(node)
        switch (node.type) {
            case ast.TILE_TYPE.SIMPLE:
                if (!node.tile) { throw new Error(`BUG!!!!!!`) }
                {
                    const ret = new GameLegendTileSimple(source, node.name, this.cacheGet(node.tile) as GameSprite)
                    this.cacheAdd(node.name, ret)
                    return ret
                }
            case ast.TILE_TYPE.AND:
                if (!node.tiles) { throw new Error(`BUG!!!!!!`) }
                {
                    const ret = new GameLegendTileAnd(source, node.name, node.tiles.map((n) => this.cacheGet(n)))
                    this.cacheAdd(node.name, ret)
                    return ret
                }
            case ast.TILE_TYPE.OR:
                if (!node.tiles) { throw new Error(`BUG!!!!!!`) }
                {
                    const ret = new GameLegendTileOr(source, node.name, node.tiles.map((n) => this.cacheGet(n)))
                    this.cacheAdd(node.name, ret)
                    return ret
                }
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private buildCollisionLayer(node: ast.CollisionLayer<string>) {
        const source = this.toSource(node)
        return new CollisionLayer(source, node.tiles.map((n) => this.cacheGet(n)))
    }

    private buildSound(node: ast.SoundItem<string>): ast.SoundItem<IGameTile> {
        switch (node.type) {
            case 'SOUND_SFX':
                this.soundCacheAdd(node.soundEffect, node)
                return node
            case 'SOUND_WHEN':
                return node
            case 'SOUND_SPRITE_MOVE':
            case 'SOUND_SPRITE_DIRECTION':
            case 'SOUND_SPRITE_EVENT':
                return { ...node, sprite: this.cacheGet(node.sprite) }
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private buildCommand(node: ast.Command<string>): Optional<ast.Command<ast.SoundItem<IGameTile>>> {
        switch (node.type) {
            case ast.COMMAND_TYPE.SFX:
                if (!this.soundCacheHas(node.sound)) {
                    return null
                }
                const sound = this.soundCacheGet(node.sound)
                return { ...node, sound }
            case ast.COMMAND_TYPE.AGAIN:
            case ast.COMMAND_TYPE.CANCEL:
            case ast.COMMAND_TYPE.MESSAGE:
            case ast.COMMAND_TYPE.CHECKPOINT:
            case ast.COMMAND_TYPE.RESTART:
            case ast.COMMAND_TYPE.WIN:
                return node
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private buildWinConditon(node: ast.WinCondition<string>) {
        const source = this.toSource(node)
        switch (node.type) {
            case ast.WIN_CONDITION_TYPE.ON:
                return new WinConditionOn(source, node.qualifier, this.cacheGet(node.tile), this.cacheGet(node.onTile))
            case ast.WIN_CONDITION_TYPE.SIMPLE:
                return new WinConditionSimple(source, node.qualifier, this.cacheGet(node.tile))
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private buildLevel(node: ast.Level<string>) {
        switch (node.type) {
            case ast.LEVEL_TYPE.MESSAGE:
                return node
            case ast.LEVEL_TYPE.MAP:
                return { ...node, cells: node.cells.map((row) => row.map((cell) => this.cacheGet(cell))) }
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private buildSimpleRules(rules: AST_Rule[]) {

        // Simplify the rules by de-duplicating them
        const ruleCache = new Map()
        const bracketCache = new Map()
        const neighborCache = new Map()
        const tileCache = new Map()

        const simpleRules = rules.map((n) => this.simplifyRule(n, ruleCache, bracketCache, neighborCache, tileCache))
        return simpleRules
    }

    private simplifyRule(node: AST_Rule,
                         ruleCache: Map<string, SimpleRule>,
                         bracketCache: Map<string, ISimpleBracket>,
                         neighborCache: Map<string, SimpleNeighbor>,
                         tileCache: Map<string, SimpleTileWithModifier>): SimpleRuleGroup {

        const source = this.toSource(node)
        switch (node.type) {
            case ast.RULE_TYPE.LOOP: {
                    const subRules = node.rules.map((n) => this.simplifyRule(n, ruleCache, bracketCache, neighborCache, tileCache))
                    return new SimpleRuleLoop(source, false/*isRandom*/, subRules)
                }
            case ast.RULE_TYPE.GROUP:
                // Extra checks to make TypeScript happy
                if (node.rules[0]) {
                    const firstRule = node.rules[0]
                    const isRandom = !!firstRule.isRandom
                    const subRules = node.rules.map((n) => this.simplifyRule(n, ruleCache, bracketCache, neighborCache, tileCache))

                    // if (rules.length === 1) {
                    //     return rules[0]
                    // }
                    return new SimpleRuleGroup(source, isRandom, subRules)
                }
                throw new Error(`BUG!!!!!!`)
            case ast.RULE_TYPE.SIMPLE: {
                    /**
                     * Expands this Rule into multiple `SimpleRule` objects.
                     *
                     * For example, `HORIZONTAL [ > player ] -> [ < crate ]` gets expanded to the following `SimpleRule`s:
                     *
                     * -  `LEFT  [ LEFT  player ] -> [ RIGHT crate ]`
                     * -  `RIGHT [ RIGHT player ] -> [ LEFT  crate ]`
                     *
                     * The `SimpleRule`s only have absolute directions
                     * rather than the relative ones specified in the original game code.
                     */
                    const isRandom = !!node.isRandom
                    const conditions = node.conditions
                    const actions = node.actions

                    // Check if valid
                    if (conditions.length !== actions.length && actions.length !== 0) {
                        throw new Error(`Left side has "${conditions.length}" conditions and right side has "${actions.length}" actions!`)
                    }

                    if (conditions.length === actions.length) {
                        // do nothing
                    } else if (actions.length !== 0) {
                        throw new Error(`Invalid Rule. The number of brackets on the right must match the structure of the left hand side or be 0`)
                    }

                    const simpleRules = this.rule_convertToMultiple(node).map((r) => this.rule_toSimple(r, ruleCache, bracketCache, neighborCache, tileCache))
                    // If the brackets are all the same object then that means we can just output 1 rule
                    // (the brackets don't have any directions. Otherwise they would not have been
                    // deduplicated as part of the .toKey() and cacheGetAndSet)
                    const isDuplicate = simpleRules.length === 1 || (!node.isRandom && simpleRules[1] && simpleRules[0].canCollapseBecauseBracketsMatch(simpleRules[1]))
                    if (isDuplicate) {
                        simpleRules[0].subscribeToCellChanges()
                        // we still need it to be in a RuleGroup
                        // so the Rule can be evaluated multiple times (not just once)
                        return new SimpleRuleGroup(source, isRandom, [simpleRules[0]])
                    } else {
                        for (const rule of simpleRules) {
                            rule.subscribeToCellChanges()
                        }
                        return new SimpleRuleGroup(source, isRandom, simpleRules)
                    }
                }
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private rule_toSimple(node: AST_BasicRule,
                          ruleCache: Map<string, SimpleRule>,
                          bracketCache: Map<string, ISimpleBracket>,
                          neighborCache: Map<string, SimpleNeighbor>,
                          tileCache: Map<string, SimpleTileWithModifier>) {

        const source = this.toSource(node)
        const directions = this.rule_getDirectionModifiers(node)
        const commands = removeNulls(node.commands.map((n) => this.buildCommand(n)))

        if (directions.length !== 1) {
            throw new Error(`BUG: should have exactly 1 direction by now but found the following: "${directions}"`)
        }

        // Check if the condition matches the action. If so, we can simplify evaluation.
        const conditionBrackets = node.conditions.map((x) => this.bracket_toSimple(x, directions[0], ruleCache, bracketCache, neighborCache, tileCache))
        const actionBrackets = node.actions.map((x) => this.bracket_toSimple(x, directions[0], ruleCache, bracketCache, neighborCache, tileCache))

        for (let index = 0; index < conditionBrackets.length; index++) {
            const action = actionBrackets[index]
            // Skip rules with no action bracket `[ > Player ] -> CHECKPOINT`
            if (!action) {
                continue
            }
        }
        return cacheSetAndGet(ruleCache, new SimpleRule(source, conditionBrackets, actionBrackets, commands, node.isLate, node.isRigid, node.debugFlag))
    }

    private rule_convertToMultiple(node: AST_BasicRule) {
        let rulesToConvert = []
        let convertedRules: AST_BasicRule[] = []

        for (const direction of this.rule_getDirectionModifiers(node)) {
            const expandedDirection = this.rule_clone(node, direction, null, null)
            rulesToConvert.push(expandedDirection)
        }

        const expandModifiers = new Map()
        expandModifiers.set(ast.RULE_MODIFIER.HORIZONTAL, [RULE_DIRECTION.LEFT, RULE_DIRECTION.RIGHT])
        expandModifiers.set(ast.RULE_MODIFIER.VERTICAL, [RULE_DIRECTION.UP, RULE_DIRECTION.DOWN])
        expandModifiers.set(ast.RULE_MODIFIER.MOVING, [RULE_DIRECTION.UP, RULE_DIRECTION.DOWN, RULE_DIRECTION.LEFT, RULE_DIRECTION.RIGHT, RULE_DIRECTION.ACTION])

        let didExpandRulesToConvert
        do {
            didExpandRulesToConvert = false
            for (const rule of rulesToConvert) {
                let didExpand = false
                const direction = this.rule_getDirectionModifiers(rule)[0]
                if (this.rule_getDirectionModifiers(rule).length !== 1) {
                    throw new Error(`BUG: should have already expanded the rule to only contian one direction`)
                }
                for (const [nameToExpand, variations] of expandModifiers) {
                    if (this.rule_hasModifier(rule, nameToExpand)) {
                        for (const variation of variations) {
                            convertedRules.push(this.rule_clone(rule, direction, nameToExpand, variation))
                            didExpand = true
                            didExpandRulesToConvert = true
                        }
                    }
                }
                if (!didExpand) {
                    // Try expanding PARALLEL and ORTHOGONAL (since they depend on the rule direction)
                    let perpendiculars
                    let parallels
                    switch (direction) {
                        case RULE_DIRECTION.UP:
                        case RULE_DIRECTION.DOWN:
                            perpendiculars = [RULE_DIRECTION.LEFT, RULE_DIRECTION.RIGHT]
                            parallels = [RULE_DIRECTION.UP, RULE_DIRECTION.DOWN]
                            break
                        case RULE_DIRECTION.LEFT:
                        case RULE_DIRECTION.RIGHT:
                            perpendiculars = [RULE_DIRECTION.UP, RULE_DIRECTION.DOWN]
                            parallels = [RULE_DIRECTION.LEFT, RULE_DIRECTION.RIGHT]
                            break
                        default:
                            throw new Error(`BUG: There must be some direction`)
                    }
                    if (perpendiculars && parallels) {
                        const orthoParallels = [
                            { nameToExpand: ast.RULE_MODIFIER.ORTHOGONAL, variations: perpendiculars },
                            { nameToExpand: ast.RULE_MODIFIER.PERPENDICULAR, variations: perpendiculars },
                            { nameToExpand: ast.RULE_MODIFIER.PARALLEL, variations: parallels }
                        ]
                        for (const { nameToExpand, variations } of orthoParallels) {

                            if (this.rule_hasModifier(rule, nameToExpand)) {
                                for (const variation of variations) {
                                    convertedRules.push(this.rule_clone(rule, direction, nameToExpand, variation))
                                    didExpand = true
                                    didExpandRulesToConvert = true
                                }
                            }
                        }
                    }

                }
                // If nothing was expanded and this is the current rule
                // then just keep it
                if (!didExpand) {
                    convertedRules.push(rule)
                }
            }
            rulesToConvert = convertedRules
            convertedRules = []
        } while (didExpandRulesToConvert)

        return rulesToConvert
    }

    private rule_clone(node: AST_BasicRule, direction: RULE_DIRECTION, nameToExpand: Optional<ast.RULE_MODIFIER>, newName: Optional<RULE_DIRECTION>) {
        const conditions = node.conditions.map((bracket) => this.bracket_clone(bracket, direction, nameToExpand, newName))
        const actions = node.actions.map((bracket) => this.bracket_clone(bracket, direction, nameToExpand, newName))
        // retain LATE and RIGID but discard the rest of the modifiers
        let directionModifier
        switch (direction) {
            case RULE_DIRECTION.UP:
                directionModifier = ast.RULE_MODIFIER.UP
                break
            case RULE_DIRECTION.DOWN:
                directionModifier = ast.RULE_MODIFIER.DOWN
                break
            case RULE_DIRECTION.LEFT:
                directionModifier = ast.RULE_MODIFIER.LEFT
                break
            case RULE_DIRECTION.RIGHT:
                directionModifier = ast.RULE_MODIFIER.RIGHT
                break
            default:
                throw new Error(`BUG: Invalid direction "${direction}"`)
        }
        return { ...node, directions: [directionModifier], conditions, actions }
    }

    private rule_getDirectionModifiers(node: AST_BasicRule) {
        // Convert HORIZONTAL and VERTICAL to 2:
        if (node.directions.indexOf(ast.RULE_MODIFIER.HORIZONTAL) >= 0) {
            return [RULE_DIRECTION.LEFT, RULE_DIRECTION.RIGHT]
        }
        if (node.directions.indexOf(ast.RULE_MODIFIER.VERTICAL) >= 0) {
            return [RULE_DIRECTION.UP, RULE_DIRECTION.DOWN]
        }
        const directions = node.directions.filter((m) => RULE_DIRECTION_SET.has(m)).map((d) => {
            switch (d) {
                case ast.RULE_MODIFIER.UP:
                    return RULE_DIRECTION.UP
                case ast.RULE_MODIFIER.DOWN:
                    return RULE_DIRECTION.DOWN
                case ast.RULE_MODIFIER.LEFT:
                    return RULE_DIRECTION.LEFT
                case ast.RULE_MODIFIER.RIGHT:
                    return RULE_DIRECTION.RIGHT
                default:
                    throw new Error(`BUG: Invalid rule direction "${d}"`)
            }
        })
        if (directions.length === 0) {
            return RULE_DIRECTION_LIST
        } else {
            return directions
        }
    }

    private rule_hasModifier(node: AST_BasicRule, modifier: ast.RULE_MODIFIER) {
        for (const bracket of node.conditions) {
            for (const neighbor of this.rule_getAllBracketNeighbors(bracket)) {
                for (const t of neighbor.tileWithModifiers) {
                    if (t.direction as string === modifier) { // HACK: cast to string
                        return true
                    }
                }
            }
        }
        return false
    }

    private rule_getAllBracketNeighbors(node: ast.Bracket<AST_Neighbor>) {
        switch (node.type) {
            case ast.BRACKET_TYPE.SIMPLE:
                return node.neighbors
            case ast.BRACKET_TYPE.ELLIPSIS:
                return [...node.beforeNeighbors, ...node.afterNeighbors]
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private bracket_clone(node: ast.Bracket<AST_Neighbor>, direction: RULE_DIRECTION, nameToExpand: Optional<ast.RULE_MODIFIER>, newName: Optional<RULE_DIRECTION>): ast.Bracket<AST_Neighbor> {
        switch (node.type) {
            case ast.BRACKET_TYPE.SIMPLE:
                const neighbors = node.neighbors.map((n) => this.neighbor_clone(n, direction, nameToExpand, newName))
                return { ...node, neighbors }
            case ast.BRACKET_TYPE.ELLIPSIS:
                const beforeNeighbors = node.beforeNeighbors.map((n) => this.neighbor_clone(n, direction, nameToExpand, newName))
                const afterNeighbors = node.afterNeighbors.map((n) => this.neighbor_clone(n, direction, nameToExpand, newName))
                return { ...node, beforeNeighbors, afterNeighbors }
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private neighbor_clone(node: AST_Neighbor, direction: RULE_DIRECTION, nameToExpand: Optional<ast.RULE_MODIFIER>, newName: Optional<RULE_DIRECTION>): AST_Neighbor {
        return { ...node, tileWithModifiers: node.tileWithModifiers.map((t) => this.tile_clone(t, direction, nameToExpand, newName)) }
    }

    private tile_clone(node: AST_Tile, direction: RULE_DIRECTION, nameToExpand: Optional<ast.RULE_MODIFIER>, newName: Optional<RULE_DIRECTION>): AST_Tile {
        switch (node.direction) {
            case RULE_DIRECTION_RELATIVE.RELATIVE_UP:
            case RULE_DIRECTION_RELATIVE.RELATIVE_DOWN:
            case RULE_DIRECTION_RELATIVE.RELATIVE_LEFT:
            case RULE_DIRECTION_RELATIVE.RELATIVE_RIGHT:
                const modifier = relativeDirectionToAbsolute(direction, node.direction)
                return { ...node, direction: modifier }
            case nameToExpand:
                return { ...node, direction: newName }
            case RULE_DIRECTION.UP:
            case RULE_DIRECTION.DOWN:
            case RULE_DIRECTION.LEFT:
            case RULE_DIRECTION.RIGHT:
            case RULE_DIRECTION.ACTION:
            case RULE_DIRECTION.STATIONARY:
            case RULE_DIRECTION.RANDOMDIR:
            case null:
            case undefined:
                return node
            default:
                return node // throw new Error(`BUG: Unsupported tile direction ${JSON.stringify(node)}`)
        }
    }

    private bracket_toSimple(node: ast.Bracket<AST_Neighbor>, direction: RULE_DIRECTION, ruleCache: Map<string, SimpleRule>,
                             bracketCache: Map<string, ISimpleBracket>, neighborCache: Map<string, SimpleNeighbor>,
                             tileCache: Map<string, SimpleTileWithModifier>) {

        const source = this.toSource(node)
        switch (node.type) {
            case ast.BRACKET_TYPE.SIMPLE:
                const neighbors = node.neighbors.map((x) => this.neighbor_toSimple(x, neighborCache, tileCache))
                return cacheSetAndGet(bracketCache, new SimpleBracket(source, direction, neighbors, node.debugFlag))
            case ast.BRACKET_TYPE.ELLIPSIS:
                const beforeEllipsis = node.beforeNeighbors.map((x) => this.neighbor_toSimple(x, neighborCache, tileCache))
                const afterEllipsis = node.afterNeighbors.map((x) => this.neighbor_toSimple(x, neighborCache, tileCache))
                return cacheSetAndGet(bracketCache, new SimpleEllipsisBracket(source, direction, beforeEllipsis, afterEllipsis, node.debugFlag))
            default:
                throw new Error(`Unsupported type ${node}`)
        }
    }

    private neighbor_toSimple(node: AST_Neighbor, neighborCache: Map<string, SimpleNeighbor>, tileCache: Map<string, SimpleTileWithModifier>) {
        const source = this.toSource(node)

        // Collapse duplicate tiles into one.
        // e.g. Cyber-Lasso has the following rule:
        // ... -> [ ElectricFloor Powered no ElectricFloor Claimed ]
        //
        // ElectricFloor occurs twice (one is negated)
        // We keep the first and remove the rest
        const tilesMap = new Map()
        for (const t of node.tileWithModifiers) {
            if (!tilesMap.has(t.tile)) {
                tilesMap.set(t.tile, t)
            }
        }
        const tileWithModifiers = [...tilesMap.values()]

        const simpleTilesWithModifier = new Set(removeNulls(tileWithModifiers.map((x) => this.tile_toSimple(x, tileCache))))
        return cacheSetAndGet(neighborCache, new SimpleNeighbor(source, simpleTilesWithModifier, node.debugFlag))
    }

    private tile_toSimple(node: AST_Tile, tileCache: Map<string, SimpleTileWithModifier>) {
        const source = this.toSource(node)
        // Some games mistakenly use SFX# in a bracket when the SFX should be in the commands list after the brackets
        if (!this.cacheHas(node.tile)) {
            return null
        }
        const tile = this.cacheGet(node.tile)

        let direction
        switch (node.direction) {
            case RULE_DIRECTION_RELATIVE.RELATIVE_UP:
            case RULE_DIRECTION_RELATIVE.RELATIVE_DOWN:
            case RULE_DIRECTION_RELATIVE.RELATIVE_LEFT:
            case RULE_DIRECTION_RELATIVE.RELATIVE_RIGHT:
                throw new Error(`BUG: Relative directions should have been resolved by now`)
            default:
                direction = node.direction || null // could be undefined (causes problems when evaluating)
        }
        return cacheSetAndGet(tileCache, new SimpleTileWithModifier(source, node.isNegated, node.isRandom, direction, tile, node.debugFlag))
    }

    private toSource(node: ast.IASTNode) {
        return {
            code: this.code,
            sourceOffset: node._sourceOffset
        }
    }

    private cacheHas(name: string) {
        return this.tileCache.has(name.toLowerCase())
    }

    private cacheAdd(name: string, value: IGameTile) {
        if (this.tileCache.has(name.toLowerCase())) {
            throw new Error(`BUG??? duplicate definition of ${name}`)
        }
        this.tileCache.set(name.toLowerCase(), value)
    }

    private cacheGet(name: string) {
        const value = this.tileCache.get(name.toLowerCase())
        if (value) {
            return value
        } else {
            throw new Error(`BUG: Could not find tile named ${name}`)
        }
    }

    private soundCacheAdd(name: string, value: ast.SoundItem<IGameTile>) {
        if (this.soundCache.has(name.toLowerCase())) {
            throw new Error(`BUG??? duplicate definition of ${name}`)
        }
        this.soundCache.set(name.toLowerCase(), value)
    }

    private soundCacheGet(name: string) {
        const value = this.soundCache.get(name.toLowerCase())
        if (value) {
            return value
        } else {
            throw new Error(`BUG: Could not find sound named ${name}`)
        }
    }

    private soundCacheHas(name: string) {
        return this.soundCache.has(name.toLowerCase())
    }
}
