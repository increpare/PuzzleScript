export enum LOG_LEVEL {
    SEVERE = 'SEVERE',
    WARN = 'WARN',
    INFO = 'INFO',
    DEBUG = 'DEBUG',
    TRACE = 'TRACE'
}

const LEVELS = [
    LOG_LEVEL.SEVERE,
    LOG_LEVEL.WARN,
    LOG_LEVEL.INFO,
    LOG_LEVEL.DEBUG,
    LOG_LEVEL.TRACE
]

function toNum(level: LOG_LEVEL) {
    return LEVELS.indexOf(level)
}

function toLevel(level: string) {
    switch (level.toUpperCase()) {
        case LOG_LEVEL.SEVERE: return LOG_LEVEL.SEVERE
        case LOG_LEVEL.WARN: return LOG_LEVEL.WARN
        case LOG_LEVEL.INFO: return LOG_LEVEL.INFO
        case LOG_LEVEL.DEBUG: return LOG_LEVEL.DEBUG
        case LOG_LEVEL.TRACE: return LOG_LEVEL.TRACE
        default:
            throw new Error(`ERROR: Invalid log level. valid levels are ${JSON.stringify(LEVELS)} but was given '${level}'`)
    }
}

type LogMessage = (() => any) | any

class Logger {
    private readonly currentLevelNum: number
    constructor() {
        this.currentLevelNum = toNum(toLevel(process.env.LOG_LEVEL ?? LOG_LEVEL.SEVERE))
    }
    public isLevel(level: LOG_LEVEL) {
        return toNum(level) <= this.currentLevelNum
    }

    public warn(message: LogMessage) {
        this.log(LOG_LEVEL.WARN, message)
    }
    public info(message: LogMessage) {
        this.log(LOG_LEVEL.INFO, message)
    }
    public debug(message: LogMessage) {
        this.log(LOG_LEVEL.DEBUG, message)
    }
    public trace(message: LogMessage) {
        this.log(LOG_LEVEL.TRACE, message)
    }
    private logFn(level: LOG_LEVEL, fn: () => string) {
        if (this.isLevel(level)) {
            console.warn(fn()) // tslint:disable-line:no-console
        }
    }
    private log(level: LOG_LEVEL, message: LogMessage) {
        if (typeof message === 'string') {
            this.logFn(level, () => message)
        } else {
            this.logFn(level, message)
        }
    }
}

export const logger = new Logger()
