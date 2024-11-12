# BitSet.js

[![NPM Package](https://img.shields.io/npm/v/bitset.svg?style=flat)](https://npmjs.org/package/bitset "View this project on npm")
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)


BitSet.js is an infinite [Bit-Array](http://en.wikipedia.org/wiki/Bit_array) (aka bit vector, bit string, bit set) implementation in JavaScript. Infinite means that if you invert a bit vector, the leading ones get remembered. As far as I can tell, BitSet.js is the only library which has this feature. It is also heavily benchmarked against other implementations and is the most performant implementation to date.

## Examples

### Basic usage

```javascript
let bs = new BitSet;
bs.set(128, 1); // Set bit at position 128
console.log(bs.toString(16)); // Print out a hex dump with one bit set
```

### Flipping bits

```javascript
let bs = new BitSet;
bs
  .flip(0, 62)
  .flip(29, 35);

let str = bs.toString();

if (str === "111111111111111111111111111000000011111111111111111111111111111") {
   console.log("YES!");
}
```

### Range Set

```javascript
let bs = new BitSet;
bs.setRange(10, 18, 1); // Set a 1 between 10 and 18, inclusive
```

### User permissions

If you want to store user permissions in your database and use BitSet for the bit twiddling, you can start with the following Linux-style snippet:
```javascript
let P_READ  = 2; // Bit pos
let P_WRITE = 1;
let P_EXEC  = 0;

let user = new BitSet;
user.set(P_READ); // Give read perms
user.set(P_WRITE); // Give write perms

let group = new BitSet(P_READ);
let world = new BitSet(P_EXEC);

console.log("0" + user.toString(8) + group.toString(8) + world.toString(8));
```

## Installation


```
npm install bitset
```

## Using BitSet.js with the browser

```html
<script src="bitset.js"></script>
<script>
    console.log(BitSet("111"));
</script>
```

## Using BitSet.js with require.js

```html
<script src="require.js"></script>
<script>
requirejs(['bitset.js'],
function(BitSet) {
    console.log(BitSet("1111"));
});
</script>
```

## Constructor

The default `BitSet` constructor accepts a single value of one the following types :

- String
  - Binary strings : `new BitSet("010101")`
  - Binary strings with prefix : `new BitSet("0b010101")`
  - Hexadecimal strings with prefix `new BitSet("0xaffe")`
- Array
  - The values of the array are the indices to be set to 1 : `new BitSet([1,12,9])`
- [Uint8Array](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Uint8Array)
  - A binary representation in 8 bit form
- Number
  - A binary value
- BitSet
  - A BitSet object, which get copied over


## Functions


The data type Mixed can be either a BitSet object, a String or an integer representing a native bitset with 31 bits.


### BitSet set(ndx[, value=1])

Mutable; Sets value 0 or 1 to index `ndx` of the bitset

int get(ndx)
---
Gets the value at index ndx

### BitSet setRange(from, to[, value=1])

Mutable; Helper function for set, to set an entire range to a given value

### BitSet clear([from[, to]])

Mutable; Sets a portion of a given bitset to zero

- If no param is given, the whole bitset gets cleared
- If one param is given, the bit at this index gets cleared
- If two params are given, the range is cleared

### BitSet slice([from[, to]])

Immutable; Extracts a portion of a given bitset as a new bitset

- If no param is given, the bitset is getting cloned
- If one param is given, the index is used as offset
- If two params are given, the range is returned as new BitSet

### BitSet flip([from[, to]])

Mutable; Toggles a portion of a given bitset

- If no param is given, the bitset is inverted
- If one param is given, the bit at the index is toggled
- If two params are given, the bits in the given range are toggled

### BitSet not()

Immutable; Calculates the bitwise complement

### BitSet and(Mixed x)

Immutable; Calculates the bitwise intersection of two bitsets

### BitSet or(Mixed x)

Immutable; Calculates the bitwise union of two bitsets

### BitSet xor(Mixed x)

Immutable; Calculates the bitwise xor between two bitsets

### BitSet andNot(Mixed x)

Immutable; Calculates the bitwise difference of two bitsets (this is not the nand operation!)

### BitSet clone()

Immutable; Clones the actual object

### Array toArray()

Returns an array with all indexes set in the bitset

### String toString([base=2])

Returns a string representation with respect to the base

### int cardinality()

Calculates the number of bits set

### int msb()

Calculates the most significant bit (the left most)

### int ntz()

Calculates the number of trailing zeros (zeros on the right). If all digits are zero, `Infinity` is returned, since BitSet.js is an arbitrary large bit vector implementation.

### int lsb()

Calculates the least significant bit (the right most)

### bool isEmpty()

Checks if the bitset has all bits set to zero

### bool equals()

Checks if two bitsets are the same

### BitSet.fromBinaryString(str)

Alternative constructor to pass with a binary string

### BitSet.fromHexString(str)

Alternative constructor to pass a hex string

### BitSet.Random([n=32])

Create a random BitSet with a maximum length of n bits

## Iterator Interface

A `BitSet` object is iterable. The iterator gets all bits up to the most significant bit. If no bits are set, the iteration stops immediately.

```js
let bs = BitSet.Random(55);
for (let b of bs) {
  console.log(b);
} 
```

Note: If the bitset is inverted so that all leading bits are 1, the iterator must be stopped by the user!


## Coding Style

As every library I publish, BitSet.js is also built to be as small as possible after compressing it with Google Closure Compiler in advanced mode. Thus the coding style orientates a little on maxing-out the compression rate. Please make sure you keep this style if you plan to extend the library.

## Building the library

After cloning the Git repository run:

```
npm install
npm run build
```

## Run a test

Testing the source against the shipped test suite is as easy as

```
npm run test
```

## Copyright and Licensing

Copyright (c) 2024, [Robert Eisele](https://raw.org/)
Licensed under the MIT license.
