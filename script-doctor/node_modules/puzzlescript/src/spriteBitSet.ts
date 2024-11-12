import { BitSet } from 'bitset'
import { GameData } from './models/game'
import { GameSprite } from './models/tile'
// BitSet does not export a default so import does not work in webpack-built file
const BitSet2 = require('bitset') // tslint:disable-line:no-var-requires

abstract class CustomBitSet<T> {
    protected readonly bitSet: BitSet
    constructor(items?: Iterable<T>, bitSet?: BitSet) {
        if (bitSet) {
            this.bitSet = bitSet
        } else {
            this.bitSet = new BitSet2()
        }

        if (items) {
            this.addAll(items)
        }
    }

    // Unused
    // public clear() {
    //     this.bitSet.clear()
    // }

    public isEmpty() {
        return this.bitSet.isEmpty()
    }

    public addAll(items: Iterable<T>) {
        for (const sprite of items) {
            this.add(sprite)
        }
    }

    // Unused
    // public removeAll(items: Iterable<T>) {
    //     for (const sprite of items) {
    //         this.remove(sprite)
    //     }
    // }

    public add(item: T) {
        this.bitSet.set(this._indexOf(item))
    }

    public remove(item: T) {
        this.bitSet.clear(this._indexOf(item))
    }

    public has(item: T) {
        return !!this.bitSet.get(this._indexOf(item))
    }

    public containsAll(other: CustomBitSet<T>) {
        return other.bitSet.cardinality() === this.bitSet.and(other.bitSet).cardinality()
    }

    public containsAny(other: CustomBitSet<T>) {
        return !this.bitSet.and(other.bitSet).isEmpty()
    }

    public containsNone(other: CustomBitSet<T>) {
        return other.bitSet.and(this.bitSet).isEmpty()
    }

    protected abstract indexOf(item: T): number

    private _indexOf(item: T) {
        const index = this.indexOf(item)
        if (index < 0) {
            throw new Error(`BUG: Expected the item index to be >= 0 but it was ${index}`)
        }
        return index
    }
}

export class SpriteBitSet extends CustomBitSet<GameSprite> {

    public indexOf(item: GameSprite) {
        return item.allSpritesBitSetIndex
    }

    public union(bitSets: Iterable<SpriteBitSet>) {
        let ret: SpriteBitSet = this // tslint:disable-line:no-this-assignment
        for (const bitSet of bitSets) {
            ret = ret.or(bitSet)
        }
        return ret
    }

    protected toString(gameData: GameData) {
        const str = []
        for (const sprite of this.getSprites(gameData)) {
            str.push(sprite.getName())
        }
        return str.join(' ')
    }

    private getSprites(gameData: GameData) {
        const sprites = new Set<GameSprite>()
        for (const sprite of gameData.objects) {
            if (this.has(sprite)) {
                sprites.add(sprite)
            }
        }
        return sprites
    }

    private or(bitSet: SpriteBitSet) {
        return new SpriteBitSet(undefined, this.bitSet.or(bitSet.bitSet))
    }

}
