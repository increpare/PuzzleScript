import { Optional } from '../util'
import { BaseForLines, IGameCode } from './BaseForLines'
import { IGameNode } from './game'

export class RGB {
    public readonly r: number
    public readonly g: number
    public readonly b: number
    public readonly a: Optional<number>

    constructor(r: number, g: number, b: number, a: Optional<number>) {
        this.r = r
        this.g = g
        this.b = b
        this.a = a
    }

    public isDark() {
        return this.r + this.g + this.b < 128 * 3
    }
}

export interface IColor extends IGameNode {
    isTransparent: () => boolean
    hasAlpha: () => boolean
    toRgb: () => RGB
    toHex: () => string
}

export class HexColor extends BaseForLines implements IColor {
    private hex: string

    constructor(source: IGameCode, hex: string) {
        super(source)
        this.hex = hex
    }

    public isTransparent() { return false }
    public hasAlpha() { return this.hex.length > 7 /*#123456aa*/ }
    public toRgb() {
        return hexToRgb(this.hex)
    }
    public toHex() {
        return this.hex
    }
}

export class TransparentColor extends BaseForLines implements IColor {
    constructor(source: IGameCode) {
        super(source)
    }

    public isTransparent() { return true }
    public hasAlpha() { return false }
    public toRgb(): RGB {
        throw new Error('BUG: Transparent colors do not have RGB data')
    }
    public toHex(): string {
        throw new Error('BUG: Transparent colors do not have a hex color value')
    }
}

export function hexToRgb(hex: string) {
    // https://stackoverflow.com/a/5624139
    // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
    const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i
    hex = hex.replace(shorthandRegex, (m, r, g, b) => {
        return r + r + g + g + b + b
    })

    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})?$/i.exec(hex)
    if (result) {
        return new RGB(
            parseInt(result[1], 16),
            parseInt(result[2], 16),
            parseInt(result[3], 16),
            result[4] ? parseInt(result[4], 16) / 255 : null
        )
    } else {
        throw new Error(`BUG: Invalid hex color: '${hex}'`)
    }
}
