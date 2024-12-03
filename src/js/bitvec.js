
function BitVec(init) {
	this.data = new Int32Array(init);
}

BitVec.prototype.cloneInto = function(target) {
	for (var i=0;i<this.data.length;++i) {
		target.data[i]=this.data[i];
	}
	return target;
}
BitVec.prototype.clone = function() {
	return new BitVec(this.data);
}

BitVec.prototype.iand = function(other) {
	for (var i = 0; i < this.data.length; ++i) {
		this.data[i] &= other.data[i];
	}
}


BitVec.prototype.inot = function() {
	for (var i = 0; i < this.data.length; ++i) {
		this.data[i] = ~this.data[i];
	}
}

BitVec.prototype.ior = function(other) {
	for (var i = 0; i < this.data.length; ++i) {
		this.data[i] |= other.data[i];
	}
}

BitVec.prototype.iclear = function(other) {
	for (var i = 0; i < this.data.length; ++i) {
		this.data[i] &= ~other.data[i];
	}
}

BitVec.prototype.ibitset = function(ind) {
	this.data[ind>>5] |= 1 << (ind & 31);
}

BitVec.prototype.ibitclear = function(ind) {
	this.data[ind>>5] &= ~(1 << (ind & 31));
}

BitVec.prototype.get = function(ind) {
	return (this.data[ind>>5] & 1 << (ind & 31)) !== 0;
}

BitVec.prototype.getshiftor = function(mask, shift) {
	var toshift = shift & 31;
	var ret = this.data[shift>>5] >>> (toshift);
	if (toshift) {
		ret |= this.data[(shift>>5)+1] << (32 - toshift);
	}
	return ret & mask;
}

BitVec.prototype.ishiftor = function(mask, shift) {
	var toshift = shift&31;
	var low = mask << toshift;
	this.data[shift>>5] |= low;
	if (toshift) {
		var high = mask >> (32 - toshift);
		this.data[(shift>>5)+1] |= high;
	}
}

BitVec.prototype.ishiftclear = function(mask, shift) {
	var toshift = shift & 31;
	var low = mask << toshift;
	this.data[shift>>5] &= ~low;
	if (toshift){
		var high = mask >> (32 - (shift & 31));
		this.data[(shift>>5)+1] &= ~high;
	}
}

BitVec.prototype.equals = function(other) {
	if (this.data.length !== other.data.length)
		return false;
	for (var i = 0; i < this.data.length; ++i) {
		if (this.data[i] !== other.data[i])
			return false;
	}
	return true;
}

BitVec.prototype.setZero = function() {
	for (var i = 0; i < this.data.length; ++i) {
		this.data[i]=0;
	}
}

BitVec.prototype.iszero = function() {
	for (var i = 0; i < this.data.length; ++i) {
		if (this.data[i])
			return false;
	}
	return true;
}

BitVec.prototype.bitsSetInArray = function(arr) {
	for (var i = 0; i < this.data.length; ++i) {
		if ((this.data[i] & arr[i]) !== this.data[i]) {
			return false;
		}
	}
	return true;
}

BitVec.prototype.bitsClearInArray = function(arr) {
	for (var i = 0; i < this.data.length; ++i) {
		if (this.data[i] & arr[i]) {
			return false;
		}
	}
	return true;
}

BitVec.prototype.anyBitsInCommon = function(other) {
	return !this.bitsClearInArray(other.data);
}