/*

The connected textures work by using a bitmask of the neighbouring cells. Depending if a
neighbour is present, the bit is set:

     North: 1
      East: 2
     South: 4
      West: 8
North-West: 16
North-East: 32
South-East: 64
South-West: 128

The algorithm accepts 3 textures:

1. The 'main' texture, which is just the normal wall, without any borders
2. The 'borders' texture, which contains the borders of a wall
3. the 'corners' textures, which is only used for the (round) corners.

Based on those textures, the algorithm calculates all possible 47 connected textures which
are then used accordingly when drawing the cell.

*/


/**
 * Adjust the colors in `sprite` to the one in the given `colors` palette
 * @param colors {Object} Color palette
 * @param sprite {Object} the sprite
 */
function adjustColors(colors, sprite) {
  var mapping = {};
  var dat = sprite.spritematrix;
  sprite.colors.forEach(function(col, idx) {
    var colNr = colors.indexOf(col);
    if (colNr < 0) {
      colNr = colors.length;
      colors.push(col);
    }
    mapping[idx] = colNr;
  });
  for (var i=0; i<dat.length; i++) {
    for (var j=0; j<dat[0].length; j++) {
      dat[i][j] = mapping[dat[i][j]];
    }
  }
}

/**
 * Prepare the connected textures for the given object.
 * @param main {Object} the main sprite
 * @param borders {Object} the 'borders' sprite
 * @param corners {Object} the 'corners' sprite
 */
function generateConnectedTextures(main, borders, corners) {
  // first, create a common color palette.
  var colors = main.colors;
  adjustColors(colors, borders);
  adjustColors(colors, corners);

  var connected = main.connected = {
    textures: [],
    lookup: [],
  };

  var templates = [
    main.spritematrix,
    borders.spritematrix,
    corners.spritematrix
  ];

  // create all 47 different variations and the lookup table
  // for now, hardcode the sprite sizes...
  var CW = 5;
  var CH = 5;

  var mapping = {};
  for (var p = 0; p < 256; p++) {
    // calculate the texture key. the key takes into account the 16 basic connections and the
    // additional various corner permutations.
    var code = p;
    if ((p&1) === 0) {
      // if north border is set, then also NW and NE corner is set
      code |= 16 + 32;
    }
    if ((p&2) === 0) {
      // if east border is set, then also NE and SE corner is set
      code |= 32 + 64;
    }
    if ((p&4) === 0) {
      // if south border is set, then also SE and SW corner is set
      code |= 64 + 128;
    }
    if ((p&8) === 0) {
      // if west border is set, then also SW and NW corner is set
      code |= 128 + 16;
    }
    if (code in mapping) {
      connected.lookup[p] = mapping[code];
      continue;
    } else {
      mapping[code] = connected.textures.length;
      connected.lookup[p] = connected.textures.length;
    }
    // since code does not exist yet, create a new texture
    var text = [];
    connected.textures.push(text);
    for (var j=0; j<CH; j++) {
      var col = [];
      text.push(col);
      for (var i=0; i<CW; i++) {
        var txt = 0;
        if ((p&1)===0 && j === 0) {
          txt = 1;
        }
        if ((p&2)===0 && i === CW-1) {
          txt = 1;
        }
        if ((p&4)===0 && j === CH-1) {
          txt = 1;
        }
        if ((p&8)===0 && i === 0) {
          txt = 1;
        }
        // border corners
        if (i===0 && j===0 && (p & (8+16+1)) === (8+1)) {
          txt = 1;
        }
        if (i===CW-1 && j===0 && (p & (1+32+2)) === (1+2)) {
          txt = 1;
        }
        if (i===CW-1 && j===CH-1 && (p & (2+64+4)) === (2+4)) {
          txt = 1;
        }
        if (i===0 && j===CH-1 && (p & (4+128+8)) === (4+8)) {
          txt = 1;
        }
        // round corner
        if (i===0 && j===0 && (p&(1+8)) === 0) {
          txt = 2;
        }
        if (i===CW-1 && j===0 && (p&(1+2)) === 0) {
          txt = 2;
        }
        if (i===CW-1 && j===CH-1 && (p&(2+4)) === 0) {
          txt = 2;
        }
        if (i===0 && j===CH-1 && (p&(4+8)) === 0) {
          txt = 2;
        }
        col.push(templates[txt][j][i]);
      }
    }
  }
}
