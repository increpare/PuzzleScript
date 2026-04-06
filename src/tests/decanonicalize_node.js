#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');

const { canonicalizeSource } = require('../canonicalize');
const { decanonicalizeSemantic } = require('../decanonicalize');

const source = fs.readFileSync('src/demo/notsnake.txt', 'utf8');
const canonical = canonicalizeSource(source, 'semantic');
const rehydrated = decanonicalizeSemantic(canonical);
const roundTripped = canonicalizeSource(rehydrated, 'semantic');

assert.deepStrictEqual(roundTripped, canonical, 'decanonicalized source should preserve semantic canonical form');

console.log('decanonicalize_node: ok');
