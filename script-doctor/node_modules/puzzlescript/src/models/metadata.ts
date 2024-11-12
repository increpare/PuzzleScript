import { Optional } from '../util'
import { IColor } from './colors'

export class Dimension {
    public readonly width: number
    public readonly height: number

    constructor(width: number, height: number) {
        this.width = width
        this.height = height
    }
}

export class GameMetadata {
    public author: Optional<string>
    public homepage: Optional<string>
    public youtube: Optional<string>
    public zoomscreen: Optional<Dimension>
    public flickscreen: Optional<Dimension>
    public colorPalette: Optional<string>
    public backgroundColor: Optional<IColor>
    public textColor: Optional<IColor>
    public realtimeInterval: Optional<number>
    public keyRepeatInterval: Optional<number>
    public againInterval: Optional<number>
    public noAction: boolean
    public noUndo: boolean
    public runRulesOnLevelStart: Optional<string>
    public noRepeatAction: boolean
    public scanline: boolean
    public throttleMovement: boolean
    public noRestart: boolean
    public requirePlayerMovement: boolean
    public verboseLogging: boolean

    constructor() {
        this.noAction = false
        this.noUndo = false
        this.noRepeatAction = false
        this.scanline = false
        this.throttleMovement = false
        this.noRestart = false
        this.requirePlayerMovement = false
        this.verboseLogging = false

        this.author = null
        this.homepage = null
        this.youtube = null
        this.zoomscreen = null
        this.flickscreen = null
        this.colorPalette = null
        this.backgroundColor = null
        this.textColor = null
        this.realtimeInterval = null
        this.keyRepeatInterval = null
        this.againInterval = null
        this.runRulesOnLevelStart = null
    }

    public _setValue(key: string, value: boolean | number | string | Dimension | IColor) {
        switch (key.toLowerCase()) {
            case 'author': this.author = value as string; break
            case 'homepage': this.homepage = value as string; break
            case 'youtube': this.youtube = value as string; break
            case 'zoomscreen': this.zoomscreen = value as Dimension; break
            case 'flickscreen': this.flickscreen = value as Dimension; break
            case 'colorpalette':
            case 'color_palette': this.colorPalette = value as string; break
            case 'backgroundcolor':
            case 'background_color': this.backgroundColor = value as IColor; break
            case 'textcolor':
            case 'text_color': this.textColor = value as IColor; break
            case 'realtimeinterval':
            case 'realtime_interval': this.realtimeInterval = value as number; break
            case 'keyrepeatinterval':
            case 'key_repeat_interval': this.keyRepeatInterval = value as number; break
            case 'againinterval':
            case 'again_interval': this.againInterval = value as number; break
            case 'noaction': this.noAction = value as boolean; break
            case 'noundo': this.noUndo = value as boolean; break
            case 'runrulesonlevelstart':
            case 'run_rules_on_level_start': this.runRulesOnLevelStart = value as string; break
            case 'norepeataction':
            case 'norepeat_action': this.noRepeatAction = value as boolean; break
            case 'scanline': this.scanline = value as boolean; break
            case 'throttlemovement':
            case 'throttle_movement': this.throttleMovement = value as boolean; break
            case 'norestart': this.noRestart = value as boolean; break
            case 'requireplayermovement':
            case 'require_player_movement': this.requirePlayerMovement = value as boolean; break
            case 'verboselogging':
            case 'verbose_logging': this.verboseLogging = value as boolean; break
            default:
                throw new Error(`BUG: Unsupported config field "${key}" with value "${value}"`)
        }
    }
}
