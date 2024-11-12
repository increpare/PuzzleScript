import { Optional } from './util'

export type Comparator<T> = (a: T, b: T) => number

export class SortedArray<T> implements Iterable<T> {
    private readonly comparator: Comparator<T>
    private ary: T[]
    constructor(comparator: Comparator<T>) {
        this.comparator = comparator
        this.ary = []
    }
    public [Symbol.iterator]() {
        // We only need to sort when we are iterating
        this.ary.sort(this.comparator)
        return this.ary[Symbol.iterator]()
    }

    public add(item: T) {
        const index = this.indexOf(item)
        if (index < 0) {
            this.ary.push(item)
        }
    }

    public delete(theItem: T) {
        const index = this.indexOf(theItem)
        if (index >= 0) {
            this.ary.splice(index, 1)
        }
    }

    public size() {
        return this.ary.length
    }
    // Unused
    // public has(item: T) {
    //     return this.indexOf(item) >= 0
    // }
    // public isEmpty() {
    //     return this.ary.length === 0
    // }
    // public clear() {
    //     this.ary = []
    // }

    private indexOf(theItem: T) {
        return this.ary.indexOf(theItem)
    }
}

class ListItem<T> {
    public item: T
    public previous: Optional<ListItem<T>>
    public next: Optional<ListItem<T>>
    constructor(item: T, previous: Optional<ListItem<T>>, next: Optional<ListItem<T>>) {
        this.item = item
        this.previous = previous
        this.next = next
    }
}

class IteratorResultDone<T> implements IteratorReturnResult<T> {
    public done: true
    public value: T
    constructor() {
        this.done = true
        this.value = {} as T // tslint:disable-line:no-object-literal-type-assertion
    }
}
class ListIteratorResult<T> implements IteratorYieldResult<T> {
    public value: T
    public done: false
    constructor(value: T) {
        this.done = false
        this.value = value
    }
}

class ListIterator<T> implements Iterator<T> {
    private listHead: Optional<ListItem<T>>
    private current: Optional<ListItem<T>>
    constructor(listHead: Optional<ListItem<T>>) {
        this.listHead = listHead
        this.current = null
    }
    public next(value?: any) {
        if (this.listHead) {
            this.current = this.listHead
            this.listHead = null
        } else if (this.current) {
            // increment right before we return the item, not earlier because folks could have add items in
            this.current = this.current.next
        }
        if (this.current) {
            return new ListIteratorResult<T>(this.current.item)
        } else {
            return new IteratorResultDone<T>()
        }
    }
}

export class SortedList<T> implements Iterable<T> {
    private readonly comparator: Comparator<T>
    private head: Optional<ListItem<T>>
    constructor(comparator: Comparator<T>) {
        this.comparator = comparator
        this.head = null
    }
    public [Symbol.iterator]() {
        return new ListIterator<T>(this.head)
    }
    public add(newItem: T) {
        let prev: Optional<ListItem<T>> = null
        let current = this.head

        while (current) {
            // check if the current node is less than the new node
            const cmp = current.item === newItem ? 0 : this.comparator(current.item, newItem)
            if (cmp === 0) {
                return false // item already exists in our list
            } else if (cmp > 0) {
                break // add here
            }
            prev = current
            current = current.next
        }

        // Cases:
        // - add to middle of list (current and prev) add just before current
        // - add to end of list (!current and prev)
        // - add to beginning of list (current and !prev)
        // - empty list (!current and !prev)
        if (prev && current) {
            // insert before the prev
            const node = new ListItem<T>(newItem, prev, current)
            prev.next = node
            current.previous = node
            // this.head = node
        } else if (prev) {
            // same as previous case except we don't repoint current.previous
            const node = new ListItem<T>(newItem, prev, current)
            prev.next = node
            // current.previous = node
            // this.head = node
        } else if (current) {
            // insert before the prev
            const node = new ListItem<T>(newItem, prev, current)
            // prev.next = node
            current.previous = node
            this.head = node
        } else {
            const node = new ListItem<T>(newItem, prev, current)
            // prev.next = node
            // current.previous = node
            this.head = node
        }
        return true // added
    }

    public delete(item: T) {
        const node = this.findNode(item)
        if (node) {

            // detach
            if (node.previous) {
                node.previous.next = node.next
            } else if (this.head === node) {
                this.head = node.next
            } else {
                throw new Error(`BUG: Invariant violation`)
            }
            if (node.next) {
                node.next.previous = node.previous
            }
            return true
        } else {
            throw new Error(`BUG: Item was not in the list`)
            // return false
        }
    }

    // Unused
    // public has(item: T) {
    //     return !!this.findNode(item)
    // }

    public isEmpty() {
        return !this.head
    }
    public size() {
        let size = 0
        for (const _item of this) {
            size++
        }
        return size
    }

    // Unused
    // public clear() {
    //     this.head = null
    // }

    // Unused
    // public first() {
    //     if (this.head) {
    //         return this.head.item
    //     } else {
    //         throw new Error(`BUG: List was empty so cannot get first cell`)
    //     }
    // }

    private findNode(item: T) {
        let current = this.head
        while (current) {
            if (current.item === item || this.comparator(current.item, item) === 0) {
                return current
            }
            current = current.next
        }
        return null
    }
}
