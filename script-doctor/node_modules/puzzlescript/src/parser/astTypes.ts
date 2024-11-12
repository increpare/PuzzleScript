import { Optional } from '..'
import { WIN_QUALIFIER } from '../models/winCondition'
import { DEBUG_FLAG, RULE_DIRECTION } from '../util'

export enum RULE_MODIFIER {
    RANDOM = 'RANDOM',
    UP = 'UP',
    DOWN = 'DOWN',
    LEFT = 'LEFT',
    RIGHT = 'RIGHT',
    VERTICAL = 'VERTICAL',
    HORIZONTAL = 'HORIZONTAL',
    ORTHOGONAL = 'ORTHOGONAL',
    PERPENDICULAR = 'PERPENDICULAR',
    PARALLEL = 'PARALLEL',
    MOVING = 'MOVING',
    LATE = 'LATE',
    RIGID = 'RIGID'
}

export enum TILE_MODIFIER {
    NO = 'NO',
    LEFT = 'LEFT',
    RIGHT = 'RIGHT',
    UP = 'UP',
    DOWN = 'DOWN',
    RANDOMDIR = 'RANDOMDIR',
    RANDOM = 'RANDOM',
    STATIONARY = 'STATIONARY',
    MOVING = 'MOVING',
    ACTION = 'ACTION',
    VERTICAL = 'VERTICAL',
    HORIZONTAL = 'HORIZONTAL',
    PERPENDICULAR = 'PERPENDICULAR',
    PARALLEL = 'PARALLEL',
    ORTHOGONAL = 'ORTHOGONAL',
    ARROW_ANY = 'ARROW_ANY'
}

export enum SOUND_WHEN {
    RESTART = 'RESTART',
    UNDO = 'UNDO',
    TITLESCREEN = 'TITLESCREEN',
    STARTGAME = 'STARTGAME',
    STARTLEVEL = 'STARTLEVEL',
    ENDLEVEL = 'ENDLEVEL',
    ENDGAME = 'ENDGAME',
    SHOWMESSAGE = 'SHOWMESSAGE',
    CLOSEMESSAGE = 'CLOSEMESSAGE'
}

export enum SOUND_SPRITE_DIRECTION {
    UP = 'UP',
    DOWN = 'DOWN',
    LEFT = 'LEFT',
    RIGHT = 'RIGHT',
    HORIZONTAL = 'HORIZONTAL',
    VERTICAL = 'VERTICAL'
}

export enum SOUND_SPRITE_EVENT {
    CREATE = 'CREATE',
    DESTROY = 'DESTROY',
    CANTMOVE = 'CANTMOVE'
}

export interface IASTNode {
    _sourceOffset: number // | undefined
}

export enum COLOR_TYPE {
    HEX8 = 'HEX8',
    HEX6 = 'HEX6',
    NAME = 'NAME'
}

export type IColor = IASTNode & ({
    type: COLOR_TYPE.HEX8
    value: string
} | {
    type: COLOR_TYPE.HEX6
    value: string
} | {
    type: COLOR_TYPE.NAME
    value: string
})

export enum SPRITE_TYPE {
    NO_PIXELS = 'NO_PIXELS',
    WITH_PIXELS = 'WITH_PIXELS'
}

export type Sprite<Pixel> = IASTNode & {
    name: string
    mapChar: Optional<string>
    colors: IColor[]
} & ({
    type: SPRITE_TYPE.NO_PIXELS
} | {
    type: SPRITE_TYPE.WITH_PIXELS
    pixels: Pixel[][]
})

export enum TILE_TYPE {
    SIMPLE = 'LEGEND_ITEM_SIMPLE',
    OR = 'LEGEND_ITEM_OR',
    AND = 'LEGEND_ITEM_AND'
}
export type LegendItem<TileRef> = IASTNode & { name: string } & ({
    type: TILE_TYPE.SIMPLE
    tile: TileRef
} | {
    type: TILE_TYPE.OR
    tiles: TileRef[]
} | {
    type: TILE_TYPE.AND
    tiles: TileRef[]
})

export enum SOUND_TYPE {
    WHEN = 'SOUND_WHEN',
    SFX = 'SOUND_SFX',
    SPRITE_DIRECTION = 'SOUND_SPRITE_DIRECTION',
    SPRITE_MOVE = 'SOUND_SPRITE_MOVE',
    SPRITE_EVENT = 'SOUND_SPRITE_EVENT'
}

export interface SfxSoundItem<TileRef> {
    type: SOUND_TYPE.SFX
    soundEffect: string
}

export interface Soundish {soundCode: number}
export type SoundItem<TileRef> = IASTNode & Soundish & ({
    type: SOUND_TYPE.WHEN
    when: SOUND_WHEN
} | {
    type: SOUND_TYPE.SPRITE_DIRECTION
    sprite: TileRef
    spriteDirection: SOUND_SPRITE_DIRECTION
} | {
    type: SOUND_TYPE.SPRITE_MOVE
    sprite: TileRef
} | {
    type: SOUND_TYPE.SPRITE_EVENT
    sprite: TileRef
    spriteEvent: SOUND_SPRITE_EVENT
} | SfxSoundItem<TileRef>)

export type CollisionLayer<TileRef> = IASTNode & {
    type: 'COLLISION_LAYER'
    tiles: TileRef[]
}

// Rules have an optional debugFlag
export type Debuggable = IASTNode & {
    debugFlag: Optional<DEBUG_FLAG>
}

export enum RULE_TYPE {
    GROUP = 'RULE_GROUP',
    LOOP = 'RULE_LOOP',
    SIMPLE = 'RULE_SIMPLE'
}

export type Rule<RuleGroupRef, SimpleRuleRef, BracketRef, CommandRef> = RuleGroup<SimpleRuleRef> | RuleLoop<RuleGroupRef> | SimpleRule<BracketRef, CommandRef>

export type RuleGroup<SimpleRuleRef> = Debuggable & {
    type: RULE_TYPE.GROUP
    rules: SimpleRuleRef[]
    isRandom: boolean
}

export type RuleLoop<RuleGroupRef> = Debuggable & {
    type: RULE_TYPE.LOOP
    rules: RuleGroupRef[]
}

export type SimpleRule<BracketRef, CommandRef> = Debuggable & {
    type: RULE_TYPE.SIMPLE
    conditions: BracketRef[]
    actions: BracketRef[]
    commands: CommandRef[]
    directions: RULE_MODIFIER[]
    isRandom: Optional<boolean>
    isLate: boolean
    isRigid: boolean
}

export enum BRACKET_TYPE {
    SIMPLE = 'BRACKET_SIMPLE',
    ELLIPSIS = 'BRACKET_ELLIPSIS'
}
export type Bracket<NeighborRef> = Debuggable & ({
    type: BRACKET_TYPE.SIMPLE
    direction: RULE_DIRECTION
    neighbors: NeighborRef[]
} | {
    type: BRACKET_TYPE.ELLIPSIS
    direction: RULE_DIRECTION
    beforeNeighbors: NeighborRef[]
    afterNeighbors: NeighborRef[]
})

export type Neighbor<TileWithModifierRef> = Debuggable & {
    tileWithModifiers: TileWithModifierRef[]
}

export type TileWithModifier<TileDirections, TileRef> = Debuggable & {
    direction: Optional<TileDirections>
    isNegated: boolean
    isRandom: boolean
    tile: TileRef
}

export enum COMMAND_TYPE {
    MESSAGE = 'COMMAND_MESSAGE',
    AGAIN = 'COMMAND_AGAIN',
    CANCEL = 'COMMAND_CANCEL',
    CHECKPOINT = 'COMMAND_CHECKPOINT',
    RESTART = 'COMMAND_RESTART',
    WIN = 'COMMAND_WIN',
    SFX = 'COMMAND_SFX'
}
export type Command<SoundRef> = MessageCommand | AgainCommand | CancelCommand | CheckpointCommand | RestartCommand | WinCommand | SFXCommand<SoundRef>

export type MessageCommand = IASTNode & {
    type: COMMAND_TYPE.MESSAGE
    message: string
}

export type AgainCommand = IASTNode & {
    type: COMMAND_TYPE.AGAIN
}

export type CancelCommand = IASTNode & {
    type: COMMAND_TYPE.CANCEL
}

export type CheckpointCommand = IASTNode & {
    type: COMMAND_TYPE.CHECKPOINT
}

export type RestartCommand = IASTNode & {
    type: COMMAND_TYPE.RESTART
}

export type WinCommand = IASTNode & {
    type: COMMAND_TYPE.WIN
}

export type SFXCommand<SoundRef> = IASTNode & {
    type: COMMAND_TYPE.SFX
    sound: SoundRef
}

export enum WIN_CONDITION_TYPE {
    SIMPLE = 'WINCONDITION_SIMPLE',
    ON = 'WINCONDITION_ON'
}

export type WinCondition<TileRef> = IASTNode & {
    qualifier: WIN_QUALIFIER
    tile: TileRef
} & ({
    type: WIN_CONDITION_TYPE.SIMPLE
} | {
    type: WIN_CONDITION_TYPE.ON
    onTile: TileRef
})

export enum LEVEL_TYPE {
    MESSAGE = 'LEVEL_MESSAGE',
    MAP = 'LEVEL_MAP'
}

export type Level<TileRef> = IASTNode & ({
    type: LEVEL_TYPE.MESSAGE
    message: string
} | {
    type: LEVEL_TYPE.MAP
    cells: TileRef[][]
})

export interface IDimension {
    type: 'WIDTH_AND_HEIGHT'
    width: number
    height: number
}

type B1<TileDirections, TileRef> = Bracket<Neighbor<TileWithModifier<TileDirections, TileRef>>>

export interface IASTGame<TileDirections, TileRef, SoundRef, PixelRef> {
    title: string
    metadata: Array<{type: string, value: string | boolean | IDimension | IColor}>
    sprites: Array<Sprite<PixelRef>>
    legendItems: Array<LegendItem<TileRef>>
    collisionLayers: Array<CollisionLayer<TileRef>>
    sounds: Array<SoundItem<TileRef>>
    rules: Array<
        Rule<
            RuleGroup<SimpleRule<B1<TileDirections, TileRef>, Command<SoundRef>>>,
            SimpleRule<B1<TileDirections, TileRef>, Command<SoundRef>>,
            B1<TileDirections, TileRef>,
            Command<SoundRef>
        >
    >
    winConditions: Array<WinCondition<TileRef>>
    levels: Array<Level<TileRef>>
}
