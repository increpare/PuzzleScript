import { SortedArray, SortedList } from './sortedList'

function numberComparator(a: number, b: number) {
    return a - b
}

function objComparator(a: {value: number}, b: {value: number}) {
    return a.value - b.value
}

describe('SortedList', () => {
    it('does not add duplicates', () => {
        const list = new SortedList(numberComparator)
        expect(list.size()).toBe(0)
        list.add(1)
        expect(list.size()).toBe(1)
        list.add(1)
        expect(list.size()).toBe(1)
    })

    it('validates simple', () => {
        const list = new SortedList(objComparator)
        expect(list.size()).toBe(0)
        expect(list.isEmpty()).toBe(true)
        list.add({ value: 1 })
        expect(list.size()).toBe(1)
        expect(list.isEmpty()).toBe(false)
        list.add({ value: 1 })
        expect(list.size()).toBe(1)

        expect([{ value: 1 }]).toEqual([...list])

        list.delete({ value: 1 })
        expect(list.size()).toBe(0)
    })

    it('validates intermediate', () => {
        const list = new SortedList(numberComparator)

        list.add(6)
        list.add(5)
        list.add(4)
        list.add(3)
        list.add(2)
        list.add(1)

        list.delete(6)
        expect(list.size()).toBe(5)

        list.delete(1)
        expect(list.size()).toBe(4)

        list.delete(3)
        expect(list.size()).toBe(3)
    })

    it('iterates properly when the list is modified during iteration', () => {
        const list = new SortedList(numberComparator)

        list.add(6)
        list.add(5)
        list.add(4)
        list.add(3)
        list.add(2)
        list.add(1)

        const ret = []
        for (const x of list) {
            ret.push(x)
            if (x === 1) {
                list.add(1.5) // add after the current index (should show up)
            }
            if (x === 2) {
                list.add(1.9) // add in the current position (should not show up)
            }
            if (x === 3) {
                list.add(0) // add earlier (should not show up)
            }
            if (x === 4) {
                list.delete(4) // delete the current item (should still iterate over 5)
            }
            if (x === 5) {
                list.delete(1) // delete an already processed item (should still iterate over 5)
            }
        }

        expect(ret).toEqual([ 1, 1.5, 2, 3, 4, 5, 6 ])

        // ensure 1 & 4 were removed, and 0 & 1.5 were added
        expect([...list]).toEqual([ 0, 1.5, 1.9, 2, 3, 5, 6 ])
    })
})

describe('SortedArray', () => {
    it('does not add duplicates', () => {
        const list = new SortedArray(numberComparator)
        expect(list.size()).toBe(0)
        list.add(1)
        expect(list.size()).toBe(1)
        list.add(1)
        expect(list.size()).toBe(1)
    })

    it('validates intermediate', () => {
        const list = new SortedArray(numberComparator)

        list.add(6)
        list.add(5)
        list.add(4)
        list.add(3)
        list.add(2)
        list.add(1)

        list.delete(6)
        expect(list.size()).toBe(5)

        list.delete(1)
        expect(list.size()).toBe(4)

        list.delete(3)
        expect(list.size()).toBe(3)
    })

    it('iterates in sorted order', () => {
        const list = new SortedArray(numberComparator)

        list.add(6)
        list.add(1)
        list.add(5)
        list.add(4)
        list.add(2)
        list.add(3)

        expect([...list]).toEqual([1, 2, 3, 4, 5, 6])
    })
})
