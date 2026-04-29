#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const vm = require('vm');

function usage() {
  return [
    'Usage: node scripts/list_compact_codegen_frontier.js [testdata.js] [--limit N] [--after N]',
    '',
    'Ranks simulation-corpus cases by rough compile bring-up simplicity.',
  ].join('\n');
}

function parseArgs(argv) {
  const out = {
    testdataPath: path.resolve('src/tests/resources/testdata.js'),
    limit: 40,
    after: 0,
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (arg === '--limit' && i + 1 < argv.length) {
      out.limit = Number(argv[++i]);
    } else if (arg === '--after' && i + 1 < argv.length) {
      out.after = Number(argv[++i]);
    } else if (!arg.startsWith('--')) {
      out.testdataPath = path.resolve(arg);
    } else {
      throw new Error(`Unknown argument: ${arg}\n${usage()}`);
    }
  }
  if (!Number.isInteger(out.limit) || out.limit <= 0) {
    throw new Error('--limit must be a positive integer');
  }
  if (!Number.isInteger(out.after) || out.after < 0) {
    throw new Error('--after must be a non-negative integer');
  }
  return out;
}

function loadTestdata(filePath) {
  const code = fs.readFileSync(filePath, 'utf8');
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox, { filename: filePath });
  if (!Array.isArray(sandbox.testdata)) {
    throw new Error(`Could not load testdata array from ${filePath}`);
  }
  return sandbox.testdata;
}

function countSectionLines(source, sectionName) {
  const lines = source.split(/\r?\n/);
  let inSection = false;
  let count = 0;
  for (const line of lines) {
    const trimmed = line.trim();
    if (/^=+$/.test(trimmed)) {
      continue;
    }
    if (/^[A-Z][A-Z ]+$/.test(trimmed)) {
      inSection = trimmed === sectionName;
      continue;
    }
    if (inSection && trimmed && !trimmed.startsWith('(')) {
      count++;
    }
  }
  return count;
}

function countLevels(source) {
  const marker = source.match(/=+\s*LEVELS\s*=+/i);
  if (!marker) return 0;
  const levelsText = source.slice(marker.index + marker[0].length);
  return levelsText
    .split(/\n\s*\n\s*\n+/)
    .map(part => part.trim())
    .filter(part => part && !part.toLowerCase().startsWith('message '))
    .length;
}

function summarizeCase(entry, index) {
  const name = entry[0];
  const payload = entry[1];
  const source = Array.isArray(payload) ? payload[0] : entry[1];
  const inputs = Array.isArray(payload) ? payload[1] : entry[2];
  if (typeof source !== 'string') {
    throw new Error(`Unexpected testdata source shape at case ${index + 1} (${name})`);
  }
  const sourceLines = source.split(/\r?\n/).length;
  const sourceChars = source.length;
  const inputCount = Array.isArray(inputs) ? inputs.length : 0;
  const ruleLines = countSectionLines(source, 'RULES');
  const objectLines = countSectionLines(source, 'OBJECTS');
  const levelCount = countLevels(source);
  return {
    index: index + 1,
    name,
    sourceLines,
    sourceChars,
    inputCount,
    ruleLines,
    objectLines,
    levelCount,
    score: sourceChars + inputCount * 20 + ruleLines * 80 + levelCount * 120,
  };
}

function main() {
  const options = parseArgs(process.argv);
  const cases = loadTestdata(options.testdataPath)
    .map(summarizeCase)
    .filter(item => item.index > options.after)
    .sort((a, b) => (
      a.score - b.score
      || a.sourceChars - b.sourceChars
      || a.inputCount - b.inputCount
      || a.index - b.index
    ))
    .slice(0, options.limit);

  console.log('index\tscore\tlines\tchars\tinputs\trules\tlevels\tname');
  for (const item of cases) {
    console.log([
      item.index,
      item.score,
      item.sourceLines,
      item.sourceChars,
      item.inputCount,
      item.ruleLines,
      item.levelCount,
      item.name,
    ].join('\t'));
  }
}

main();
