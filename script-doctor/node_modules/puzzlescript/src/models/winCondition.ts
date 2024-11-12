import { Cell } from '../engine'
import { BaseForLines, IGameCode } from './BaseForLines'
import { IGameTile } from './tile'

export enum WIN_QUALIFIER {
    NO = 'NO',
    ALL = 'ALL',
    ANY = 'ANY',
    SOME = 'SOME'
}

export class WinConditionSimple extends BaseForLines {
    public readonly qualifier: WIN_QUALIFIER
    public readonly tile: IGameTile

    constructor(source: IGameCode, qualifierEnum: WIN_QUALIFIER, tile: IGameTile) {
        super(source)
        this.qualifier = qualifierEnum
        this.tile = tile
        if (!tile) {
            throw new Error('BUG: Could not find win condition tile')
        }
    }

    public cellsThatMatchTile(cells: Iterable<Cell>, tile: IGameTile) {
        return [...cells].filter((cell) => tile.matchesCell(cell))
    }

    public isSatisfied(cells: Iterable<Cell>) {
        const ret = this._isSatisfied(cells)
        if (ret) {
            if (process.env.NODE_ENV === 'development') {
                this.__incrementCoverage()
            }
        }
        return ret
    }

    public a11yGetTiles() {
        return [this.tile]
    }

    protected _isSatisfied(cells: Iterable<Cell>) {
        const tileCells = this.cellsThatMatchTile(cells, this.tile)
        switch (this.qualifier) {
            case WIN_QUALIFIER.NO:
                return tileCells.length === 0
            case WIN_QUALIFIER.ANY:
            case WIN_QUALIFIER.SOME:
                return tileCells.length > 0
            // case WIN_QUALIFIER.ALL:
            //     return ![...cells].filter(cell => !this.matchesTile(cell, this._tile))[0]
            default:
                throw new Error(`BUG: Invalid qualifier: "${this.qualifier}"`)
        }
    }
}

export class WinConditionOn extends WinConditionSimple {
    public readonly onTile: IGameTile

    constructor(source: IGameCode, qualifierEnum: WIN_QUALIFIER, tile: IGameTile, onTile: IGameTile) {
        super(source, qualifierEnum, tile)
        this.onTile = onTile
    }

    public a11yGetTiles() {
        return [this.tile, this.onTile]
    }

    protected _isSatisfied(cells: Iterable<Cell>) {
        // ALL Target ON CleanDishes
        const tileCells = this.cellsThatMatchTile(cells, this.tile)
        const onTileCells = this.cellsThatMatchTile(tileCells, this.onTile)

        switch (this.qualifier) {
            case WIN_QUALIFIER.NO:
                return onTileCells.length === 0
            case WIN_QUALIFIER.ANY:
            case WIN_QUALIFIER.SOME:
                return onTileCells.length > 0
            case WIN_QUALIFIER.ALL:
                return onTileCells.length === tileCells.length
            default:
                throw new Error(`BUG: Invalid qualifier: "${this.qualifier}"`)
        }
    }
}
