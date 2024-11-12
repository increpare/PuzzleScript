class QuickLru<V> {
    private size: number
    private cache: { [key: string]: V}
    private oldCache: { [key: string]: V}
    private readonly maxSize: number
    constructor(maxSize: number) {
        this.maxSize = maxSize
        this.size = 0
        this.cache = {}
        this.oldCache = {}
    }

    public set(key: string, value: V) {
        const has = typeof this.cache[key] !== 'undefined' || typeof this.oldCache[key] !== 'undefined'
        if (this.size > this.maxSize) {
            this.oldCache = this.cache
            this.size = 0
        }
        if (!has) {
            this.cache[key] = value
            this.size++
        }
    }
    public get(key: string): V | undefined {
        const value = this.cache[key]
        if (typeof value !== 'undefined') {
            return value
        }
        return this.oldCache[key]
    }
}

export default class LruCache<Value> {
    private lru: QuickLru<Value>
    constructor(maxSize: number) {
        this.lru = new QuickLru(maxSize)
    }

    public get(key: string, valueFn: () => Value) {
        const val = this.lru.get(key)
        // speed up by combining .has(key) and .get(key)
        if (val !== undefined) {
            return val
        }
        const value = valueFn()
        this.lru.set(key, value)
        return value
    }

    // has(key: Key) {
    //     return key
    // }
}
