import { IGameCode } from '../models/BaseForLines'
import { GameLegendTileSimple, GameSprite, IGameTile } from '../models/tile'
import { SfxSoundItem } from './astTypes'

export class LookupHelper {
    public _allSoundEffects: Map<string, SfxSoundItem<IGameTile>>
    public _allObjects: Map<string, GameSprite>
    public _allLegendTiles: Map<string, IGameTile>
    public _allLevelChars: Map<string, IGameTile>

    constructor() {
        this._allSoundEffects = new Map()
        this._allObjects = new Map()
        this._allLegendTiles = new Map()
        this._allLevelChars = new Map()
    }

    public _addToHelper<A>(map: Map<string, A>, key: string, value: A) {
        if (map.has(key)) {
            throw new Error(`ERROR: Duplicate object is defined named "${key}". They are case-sensitive!`)
        }
        map.set(key, value)
    }
    public addSoundEffect(key: string, soundEffect: SfxSoundItem<IGameTile>) {
        this._addToHelper(this._allSoundEffects, key.toLowerCase(), soundEffect)
    }
    public addToAllObjects(gameObject: GameSprite) {
        this._addToHelper(this._allObjects, gameObject.getName().toLowerCase(), gameObject)
    }
    public addToAllLegendTiles(legendTile: GameLegendTileSimple) {
        this._addToHelper(this._allLegendTiles, legendTile.spriteNameOrLevelChar.toLowerCase(), legendTile)
    }
    public addObjectToAllLevelChars(levelChar: string, gameObject: GameSprite) {
        this._addToHelper(this._allLegendTiles, levelChar.toLowerCase(), gameObject)
        this._addToHelper(this._allLevelChars, levelChar.toLowerCase(), gameObject)
    }
    public addLegendToAllLevelChars(legendTile: GameLegendTileSimple) {
        this._addToHelper(this._allLevelChars, legendTile.spriteNameOrLevelChar.toLowerCase(), legendTile)
    }
    public lookupObjectOrLegendTile(source: IGameCode, key: string) {
        key = key.toLowerCase()
        const value = this._allObjects.get(key) || this._allLegendTiles.get(key)
        if (!value) {
            throw new Error(`ERROR: Could not look up "${key}". Has it been defined in the Objects section or the Legend section?`)
        }
        return value
    }
    public lookupByLevelChar(key: string) {
        const value = this._allLevelChars.get(key.toLowerCase())
        if (!value) {
            throw new Error(`ERROR: Could not look up "${key}" in the levelChars map. Has it been defined in the Objects section or the Legend section?`)
        }
        return value
    }
    public lookupSoundEffect(key: string) {
        return this._allSoundEffects.get(key.toLowerCase())
    }
}
