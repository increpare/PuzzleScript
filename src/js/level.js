'use strict';

function Level(lineNumber, width, height, layerCount, objects) {
	this.lineNumber = lineNumber;
	this.width = width;
	this.height = height;
	this.n_tiles = width * height;
	this.objects = objects;
	this.layerCount = layerCount;
	this.commandQueue = [];
	this.commandQueueSourceRules = [];
}

Level.prototype.delta_index = function (direction) {
	const [dx, dy] = dirMasksDelta[direction]
	return dx * this.height + dy
}

Level.prototype.clone = function () {
	let clone = new Level(this.lineNumber, this.width, this.height, this.layerCount, null);
	clone.objects = new Int32Array(this.objects);
	return clone;
}

Level.prototype.getCell = function (index) {
	return new BitVec(this.objects.subarray(index * STRIDE_OBJ, index * STRIDE_OBJ + STRIDE_OBJ));
}

Level.prototype.getCellInto = function (index, targetarray) {
	for (let i = 0; i < STRIDE_OBJ; i++) {
		targetarray.data[i] = this.objects[index * STRIDE_OBJ + i];
	}
	return targetarray;
}

Level.prototype.setCell = function (index, vec) {
	for (let i = 0; i < vec.data.length; ++i) {
		this.objects[index * STRIDE_OBJ + i] = vec.data[i];
	}
}

let _movementVecs;
let _movementVecIndex = 0;
Level.prototype.getMovements = function (index) {
	let _movementsVec = _movementVecs[_movementVecIndex];
	_movementVecIndex = (_movementVecIndex + 1) % _movementVecs.length;

	for (let i = 0; i < STRIDE_MOV; i++) {
		_movementsVec.data[i] = this.movements[index * STRIDE_MOV + i];
	}
	return _movementsVec;
}

Level.prototype.getRigids = function (index) {
	return this.rigidMovementAppliedMask[index].clone();
}

Level.prototype.getMovementsInto = function (index, targetarray) {
	let _movementsVec = targetarray;

	for (let i = 0; i < STRIDE_MOV; i++) {
		_movementsVec.data[i] = this.movements[index * STRIDE_MOV + i];
	}
	return _movementsVec;
}

Level.prototype.setMovements = function (index, vec) {
	for (let i = 0; i < vec.data.length; ++i) {
		this.movements[index * STRIDE_MOV + i] = vec.data[i];
	}

	//corresponding object stuff in repositionEntitiesOnLayer
	let colIndex = (index / this.height) | 0;
	let rowIndex = (index % this.height);
	level.colCellContents_Movements[colIndex].ior(vec);
	level.rowCellContents_Movements[rowIndex].ior(vec);
	level.mapCellContents_Movements.ior(vec);
}


function LEVEL_SET_MOVEMENTS(index, vec, array_size) {
	var result = "{";
	for (let i = 0; i < array_size; i++) {
		result += `\tlevel.movements[${index}*${array_size}+${i}]=${vec}.data[${i}];\n`;
	}
	result += `
	const colIndex=(${index}/level.height)|0;
	const rowIndex=(${index}%level.height);

	${UNROLL(`level.colCellContents_Movements[colIndex] |= ${vec}`, array_size)}
	${UNROLL(`level.rowCellContents_Movements[rowIndex] |= ${vec}`, array_size)}
	${UNROLL(`level.mapCellContents_Movements |= ${vec}`, array_size)}
}`

	return result;
}

Level.prototype.calcBackgroundMask = function (state) {
	if (state.backgroundlayer === undefined) {
		logError("you have to have a background layer");
	}

	let backgroundMask = state.layerMasks[state.backgroundlayer];
	for (let i = 0; i < this.n_tiles; i++) {
		let cell = this.getCell(i);
		cell.iand(backgroundMask);
		if (!cell.iszero()) {
			return cell;
		}
	}
	const cell = new BitVec(STRIDE_OBJ);
	cell.ibitset(state.backgroundid);
	return cell;
}