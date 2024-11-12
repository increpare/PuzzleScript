import { _flatten } from '../util'
import { BaseForLines, IGameCode } from './BaseForLines'
import { GameSprite, IGameTile } from './tile'

let collisionIdCounter = 0
export class CollisionLayer extends BaseForLines {
    public readonly id: number // Used for sorting collision layers for rendering
    private sprites: GameSprite[]

    constructor(source: IGameCode, tiles: IGameTile[]) {
        super(source)
        this.id = collisionIdCounter++

        // Map all the Objects to the layer
        tiles.forEach((tile: IGameTile) => {
            tile.setCollisionLayer(this)
            tile._getDescendantTiles().forEach((subTile) => {
                subTile.setCollisionLayer(this)
            })
        })

        // build an array of Sprites so we can index to them in a BitSet
        this.sprites = [...new Set(_flatten(tiles.map((t) => t.getSprites())))]

        this.sprites.forEach((sprite, index) => sprite.setCollisionLayerAndIndex(this, index))
    }

    // isInvalid(): Optional<string> {
    //     return null
    // }

    public getBitSetIndexOf(sprite: GameSprite) {
        const index = this.sprites.indexOf(sprite)
        if (index < 0) {
            throw new Error(`BUG: Sprite is not in this CollisionLayer`)
        }
        return index
    }

    // bitSetToSprites(bitSet: BitSet) {
    //     const ret = []
    //     let index = 0
    //     for (const sprite of this.sprites) {
    //         if (bitSet.get(index)) {
    //             ret.push(sprite)
    //         }
    //         index++
    //     }
    //     return ret
    // }
}
