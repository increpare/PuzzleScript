import { getLetterSprites } from '../letters'
import { Level, SoundItem } from '../parser/astTypes'
import { Optional } from '../util'
import { IGameCode } from './BaseForLines'
import { CollisionLayer } from './collisionLayer'
import { GameMetadata } from './metadata'
import { SimpleRuleGroup } from './rule'
import { GameSprite, IGameTile } from './tile'
import { WinConditionSimple } from './winCondition'

export interface IGameNode {
    __source: {code: string, sourceOffset: number}
    __coverageCount: Optional<number>
    __getSourceLineAndColumn(): { lineNum: number, colNum: number }
    __getLineAndColumnRange(): { start: { line: number, col: number }, end: { line: number, col: number } }
    toString(): string
    toSourceString(): string
}

export class GameData {
    public readonly title: string
    public readonly metadata: GameMetadata
    public readonly objects: GameSprite[]
    public readonly legends: IGameTile[]
    public readonly sounds: Array<SoundItem<IGameTile>>
    public readonly collisionLayers: CollisionLayer[]
    public readonly rules: SimpleRuleGroup[]
    public readonly winConditions: WinConditionSimple[]
    public readonly levels: Array<Level<IGameTile>>
    private readonly cacheSpriteSize: {spriteHeight: number, spriteWidth: number}
    private cachedBackgroundSprite: Optional<GameSprite>
    private readonly letterSprites: Map<string, GameSprite>

    constructor(
        source: IGameCode,
        title: string,
        metadata: GameMetadata,
        objects: GameSprite[],
        legends: IGameTile[],
        sounds: Array<SoundItem<IGameTile>>,
        collisionLayers: CollisionLayer[],
        rules: SimpleRuleGroup[],
        winConditions: WinConditionSimple[],
        levels: Array<Level<IGameTile>>
    ) {
        this.title = title
        this.metadata = metadata
        this.objects = objects
        this.legends = legends
        this.sounds = sounds
        this.collisionLayers = collisionLayers
        this.winConditions = winConditions
        this.levels = levels
        this.rules = rules
        this.cachedBackgroundSprite = null

        const firstSpriteWithPixels = this.objects.filter((sprite) => sprite.hasPixels())[0]
        if (firstSpriteWithPixels) {
            const firstSpritePixels = firstSpriteWithPixels.getPixels(1, 1) // We don't care about these args
            this.cacheSpriteSize = {
                spriteHeight: firstSpritePixels.length,
                spriteWidth: firstSpritePixels[0].length
            }
        } else {
            // All the sprites are just a single color, so set the size to be 5x5
            this.cacheSpriteSize = {
                spriteHeight: 1,
                spriteWidth: 1
            }
        }

        // Create a collisionlayer for the letter sprites
        let spriteIndexCounter = this.objects.length // 1 more than all the game sprites

        this.letterSprites = getLetterSprites(source)
        for (const letterSprite of this.letterSprites.values()) {
            letterSprite.allSpritesBitSetIndex = spriteIndexCounter++
        }
        const letterCollisionLayer = new CollisionLayer(source, [...this.letterSprites.values()])
        this.collisionLayers.push(letterCollisionLayer)
    }

    public _getSpriteByName(name: string) {
        return this.objects.find((sprite) => sprite.getName().toLowerCase() === name.toLowerCase()) || null
    }
    public _getTileByName(name: string) {
        return this.legends.find((tile) => tile.getName().toLowerCase() === name.toLowerCase())
    }
    public getSpriteByName(name: string) {
        const sprite = this._getSpriteByName(name)
        if (!sprite) {
            throw new Error(`BUG: Could not find sprite "${name}" but expected one to exist.`)
        }
        return sprite
    }
    public getTileByName(name: string) {
        const tile = this._getTileByName(name)
        if (!tile) {
            throw new Error(`BUG: Could not find tile "${name}" but expected one to exist.`)
        }
        return tile
    }

    public getMagicBackgroundSprite() {
        if (this.cachedBackgroundSprite) {
            return this.cachedBackgroundSprite
        } else {
            const background: Optional<GameSprite> = this._getSpriteByName('background')
            if (!background) {
                const legendBackground = this.legends.find((tile) => tile.getName().toLowerCase() === 'background')
                if (legendBackground) {
                    if (legendBackground.isOr()) {
                        return null
                    } else {
                        return legendBackground.getSprites()[0]
                    }
                }
            }
            if (!background) {
                throw new Error(`ERROR: Game does not have a Background Sprite or Tile`)
            }
            this.cachedBackgroundSprite = background
            return background
        }
    }
    public getPlayer(): IGameTile {
        const player = this._getSpriteByName('player') || this.legends.find((tile) => tile.getName().toLowerCase() === 'player')
        if (!player) {
            throw new Error(`BUG: Could not find the Player sprite or tile in the game`)
        }
        return player
    }

    public clearCaches() {
        for (const rule of this.rules) {
            rule.clearCaches()
        }
        for (const sprite of this.objects) {
            sprite.clearCaches()
        }
    }

    public getSpriteSize() {
        return this.cacheSpriteSize
    }

    public getLetterSprite(char: string) {
        const sprite = this.letterSprites.get(char)
        if (!sprite) {
            throw new Error(`BUG: Cannot find sprite for letter "${char}"`)
        }
        return sprite
    }
}
