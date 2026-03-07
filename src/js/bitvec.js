'use strict';

//GENERAL PRINCIPLE
// Split bit index "shift" into: position within the word vs which word.
// inner_idx: internal index - which bit within that int (0..0b11111)
// outer_idx: external index - which int our index is in ( not a "layer" index)

// for movements: 5 bits per layer: layer 6, 12, … start at bit 5*6 = 30 and straddle into next word

function BitVec(init) {
	this.data = new Int32Array(init);
}

BitVec.prototype.cloneInto = function (target) {
	for (let i = 0; i < this.data.length; ++i) {
		target.data[i] = this.data[i];
	}
	return target;
}
BitVec.prototype.clone = function () {
	return new BitVec(this.data);
}

BitVec.prototype.iand = function (other) {
	for (let i = 0; i < this.data.length; ++i) {
		this.data[i] &= other.data[i];
	}
}


BitVec.prototype.inot = function () {
	for (let i = 0; i < this.data.length; ++i) {
		this.data[i] = ~this.data[i];
	}
}

BitVec.prototype.ior = function (other) {
	for (let i = 0; i < this.data.length; ++i) {
		this.data[i] |= other.data[i];
	}
}

BitVec.prototype.iclear = function (other) {
	for (let i = 0; i < this.data.length; ++i) {
		this.data[i] &= ~other.data[i];
	}
}


BitVec.prototype.ibitset = function (ind) {
	const outer_idx = ind >> 5;
	const inner_idx = ind & 0b11111;
	this.data[outer_idx] |= 1 << inner_idx;
}



function IBITSET(tok, index) {
	return `${tok}.data[${index}>>5] |= 1 << (${index} & 0b11111);`;
}


BitVec.prototype.ibitclear = function (ind) {
	const outer_idx = ind >> 5;
	const inner_idx = ind & 0b11111;
	// clears that one bit, leaves others unchanged
	this.data[outer_idx] &= ~(1 << inner_idx);
}

BitVec.prototype.get = function (ind) {
	const outer_idx = ind >> 5;
	const inner_idx = ind & 0b11111;
	return (this.data[outer_idx] & 1 << inner_idx) !== 0;
}

function GET(tok, index) {
	const outer_idx = index >> 5;
	const inner_idx = index & 0b11111;
	return `((${tok}.data[${outer_idx}] & 1 << ${inner_idx}) !== 0)`;
}


BitVec.prototype.getshiftor = function (mask, shift) {
	const inner_idx = shift & 0b11111;
	const outer_idx = shift >> 5;
	let ret = this.data[outer_idx] >>> inner_idx;
	if (inner_idx > 27) {//32 - inner_idx > 5
		ret |= this.data[outer_idx + 1] << (32 - inner_idx);
	}
	return ret & mask;
}

function GETSHIFTOR(tok, mask, shift) {
	const inner_idx = shift & 0b11111;
	const outer_idx = shift >> 5;
	if (inner_idx > 27) {//32 - inner_idx > 5
		return `${mask}&((${tok}.data[${outer_idx}] >>> ${inner_idx}) | (${tok}.data[${outer_idx}+1] << (32-${inner_idx})))`;
	} else {
		return `${mask}&(${tok}.data[${outer_idx}] >>> ${inner_idx})`;
	}
}

BitVec.prototype.ishiftor = function (mask, shift) {
	const inner_idx = shift & 0b11111;
	const outer_idx = shift >> 5;
	// low: mask shifted left so it lands at bit position (inner_idx) in word outer_idx
	const low = mask << inner_idx;
	this.data[outer_idx] |= low;
	// if we have overflow into the next word
	if (inner_idx > 27) {//32 - inner_idx > 5
		// high: part of mask that overflows into next word (mask >>> (32-inner_idx))
		const high = mask >> (32 - inner_idx);
		this.data[outer_idx + 1] |= high;
	}
}

function ISHIFTOR(tok, mask, shift) {
	return `{
		const inner_idx = ${shift}&0b11111;
		const outer_idx = ${shift}>>5;
		const low = ${mask} << inner_idx;
		${tok}.data[outer_idx] |= low;
		if (inner_idx > 27) {//32 - inner_idx > 5
			const high = ${mask} >> (32 - inner_idx);
			${tok}.data[outer_idx+1] |= high;
		}
	}`;
}


BitVec.prototype.ishiftclear = function (mask, shift) {
	const inner_idx = shift & 0b11111;
	const outer_idx = shift >> 5;
	const low = mask << inner_idx;
	this.data[outer_idx] &= ~low;
	if (inner_idx > 27) {//32 - inner_idx > 5
		const high = mask >> (32 - inner_idx);
		this.data[outer_idx + 1] &= ~high;
	}
}

function WEIRDNESS_FOUND(msg) {
	throw new Error(`found ${msg}`);
}


function ISHIFTCLEAR(tok, mask, shift) {
	const inner_idx = shift & 0b11111;
	const outer_idx = shift >> 5;
	const low = mask + "<<" + inner_idx;
	let result = `${tok}.data[${outer_idx}] &= ~(${low});\n`
	if (inner_idx > 27) {//32 - inner_idx > 5
		const high = mask + ">>>" + (32 - inner_idx);
		const idx = outer_idx + 1;
		result += `${tok}.data[${idx}] &= ~(${high});\n`;
	}
	return result;
}

BitVec.prototype.equals = function (other) {
	if (this.data.length !== other.data.length)
		return false;
	for (let i = 0; i < this.data.length; ++i) {
		if (this.data[i] !== other.data[i])
			return false;
	}
	return true;
}


function EQUALS(tok, other, array_size) {
	let result = "(true";
	for (let i = 0; i < array_size; i++) {
		result += `&&(${tok}.data[${i}] === ${other}.data[${i}])`;
	}
	return result + ")";
}

function EQUALS_TOK_REAL(tok, other) {
	let result = "(true";
	for (let i = 0; i < other.data.length; i++) {
		result += `&&(${tok}.data[${i}] === ${other.data[i]})`;
	}
	return result + ")";
}

function NOT_EQUALS(tok, other, array_size) {
	let result = "(false";
	for (let i = 0; i < array_size; i++) {
		result += `||(${tok}.data[${i}] !== ${other}.data[${i}])`;
	}
	return result + ")";
}


BitVec.prototype.setZero = function () {
	this.data.fill(0);
}

function ARRAY_SET_ZERO(tok) {
	return tok + ".fill(0);\n";
}

function SET_ZERO(tok) {
	return tok + ".data.fill(0);\n";
}

BitVec.prototype.iszero = function () {
	for (let i = 0; i < this.data.length; ++i) {
		if (this.data[i] !== 0)
			return false;
	}
	return true;
}

function IS_ZERO(tok, array_size) {
	let result = "(true";
	for (let i = 0; i < array_size; i++) {
		result += `&&(${tok}.data[${i}]===0)`;
	}
	return result + ")";
}

function IS_NONZERO(tok, array_size) {
	let result = "(false";
	for (let i = 0; i < array_size; i++) {
		result += `||(${tok}.data[${i}]!==0)`;
	}
	return result + ")";
}

BitVec.prototype.bitsSetInArray = function (arr) {
	for (let i = 0; i < this.data.length; ++i) {
		if ((this.data[i] & arr[i]) !== this.data[i]) {
			return false;
		}
	}
	return true;
}

function BITS_SET_IN_ARRAY(tok, arr, array_size) {

	let result = "(true";
	for (let i = 0; i < array_size; i++) {
		result += `&&((${tok}.data[${i}] & ${arr}[${i}]) === ${tok}.data[${i}])`;
	}
	return result + ")";
}

function NOT_BITS_SET_IN_ARRAY(tok, arr, array_size) {
	let result = "(false";
	for (let i = 0; i < array_size; i++) {
		result += `||((${tok}.data[${i}] & ${arr}[${i}]) !== ${tok}.data[${i}])`;
	}
	return result + ")";
}

BitVec.prototype.bitsClearInArray = function (arr) {
	for (let i = 0; i < this.data.length; ++i) {
		if (this.data[i] & arr[i]) {
			return false;
		}
	}
	return true;
}

function BITS_CLEAR_IN_ARRAY(tok, arr, array_size) {
	if (array_size === 0)
		return "true";
	let result = "(true";
	for (let i = 0; i < array_size; i++) {
		result += `&&((${tok}.data[${i}] & ${arr}.data[${i}]) === 0)`;
	}
	return result + ")";
}

BitVec.prototype.anyBitsInCommon = function (other) {
	for (let i = 0; i < this.data.length; ++i) {
		if (this.data[i] & other.data[i]) {
			return true;
		}
	}
	return false;
}

BitVec.prototype.prettyPrint = function () {
	var result = "";
	//print string as bit array, grouped into fives
	for (let i = 0; i < this.data.length; i++) {
		for (let j = 0; j < 32; j++) {
			// 1<<j = bit mask for bit j in the word
			result += (this.data[i] & (1 << j)) ? "1" : "0";
		}
		result += " ";
	}
	return result;
}

function ANY_BITS_IN_COMMON(tok, arr, array_size) {
	if (array_size === 0) {
		return "false";
	}
	let result = "(false";
	for (let i = 0; i < array_size; i++) {
		result += `||((${tok}.data[${i}] & ${arr}.data[${i}]) !== 0)`;
	}
	return result + ")";
}

function ANY_BITS_IN_COMMON_TOK_REAL(tok, arr) {
	if (arr.length === 0) {
		return "false";
	}
	let result = "(false";
	for (let i = 0; i < arr.length; i++) {
		result += `||((${tok}.data[${i}] & ${arr[i]}) !== 0)`;
	}
	return result + ")";
}

// this function is used to unroll loops in parallel from bitvec - it returns a string
// representation of the javascript unrolled code
function UNROLL(command, array_size) {
	const toks = command.split(" ");
	let result = "";
	for (let i = 0; i < array_size; i++) {
		result += `${toks[0]}.data[${i}] ${toks[1]} ${toks[2]}.data[${i}];\n`;
	}
	return result;
}

function UNROLL_TOK_REAL(tok, op, val, array_size) {
	let result = "";
	for (let i = 0; i < array_size; i++) {
		result += tok + ".data[" + i + "]" + op + val.data[i] + ";\n";
	}
	return result;
}

function LEVEL_GET_CELL_INTO(level, index, targetarray, OBJECT_SIZE) {
	let result = "";
	for (let i = 0; i < OBJECT_SIZE; i++) {
		result += targetarray + `.data[${i}]=level.objects[${index}*${OBJECT_SIZE}+${i}];\n`;
	}
	return result;
}

function LEVEL_GET_MOVEMENTS_INTO(index, targetarray, MOVEMENT_SIZE) {
	let result = "";
	for (let i = 0; i < MOVEMENT_SIZE; i++) {
		result += targetarray + `.data[${i}]=level.movements[${index}*${MOVEMENT_SIZE}+${i}];\n`;
	}
	return result;
}


function LEVEL_SET_CELL(level, index, vec, array_size) {
	let result = "";
	for (let i = 0; i < array_size; i++) {
		result += `\t${level}.objects[${index}*${array_size}+${i}]=${vec}.data[${i}];\n`;
	}
	return result;
}


function IMPORT_COMPILE_TIME_ARRAY(runtime, compiletime, array_size) {
	let result = "";
	for (let i = 0; i < array_size; i++) {
		result += `${runtime}.data[${i}]=${compiletime.data[i]};\n`;
	}
	return result;
}

