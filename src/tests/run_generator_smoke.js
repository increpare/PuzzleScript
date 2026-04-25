const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

if (process.argv.length < 4) {
  console.error('Usage: node src/tests/run_generator_smoke.js <puzzlescript_generator> <game.txt>');
  process.exit(2);
}

const generatorPath = path.resolve(process.argv[2]);
const gamePath = path.resolve(process.argv[3]);
const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'psgen-smoke-'));

function run(args, label, expectedStatus = 0) {
  const result = spawnSync(generatorPath, args, { encoding: 'utf8' });
  if (result.status !== expectedStatus) {
    throw new Error(`${label} exited ${result.status}, expected ${expectedStatus}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
  }
  return result;
}

function writeSpec(name, lines) {
  const specPath = path.join(tempDir, name);
  fs.writeFileSync(specPath, lines.join('\n'));
  return specPath;
}

const tests = [];

function test(name, body) {
  tests.push({ name, body });
}

const initLevel = [
  '(INIT LEVEL)',
  '######',
  '#....#',
  '#....#',
  '#....#',
  '#....#',
  '######',
  '',
];

test('fixed sample output is deterministic across job counts', () => {
  const smokeSpecPath = writeSpec('smoke.gen', [
    ...initLevel,
    '(GENERATION RULES)',
    'choose 1 [ no wall ] -> [ player ]',
    'choose 2 [ no wall no player no crate ] [ no wall no player no target ] -> [ crate ] [ target ]',
    '',
  ]);

  const out1 = path.join(tempDir, 'jobs1.json');
  const out2 = path.join(tempDir, 'jobs2.json');
  const common = [
    gamePath,
    smokeSpecPath,
    '--samples', '20',
    '--seed', '7',
    '--solver-timeout-ms', '50',
    '--top-k', '5',
    '--quiet',
  ];

  run([...common, '--jobs', '1', '--json-out', out1], 'jobs=1');
  run([...common, '--jobs', '2', '--json-out', out2], 'jobs=2');

  const json1 = fs.readFileSync(out1, 'utf8');
  const json2 = fs.readFileSync(out2, 'utf8');
  assert.strictEqual(json1, json2, 'fixed-sample output should be deterministic across job counts');

  const parsed = JSON.parse(json1);
  assert.strictEqual(parsed.totals.samples_attempted, 20);
  assert.strictEqual(parsed.totals.valid_generated, 20);
  assert.ok(parsed.totals.solved > 0, 'smoke fixture should produce at least one solved level');
  assert.ok(parsed.top.length > 0, 'top results should be retained');
  assert.strictEqual(parsed.totals.invalid_generation, parsed.totals.rejected);
  assert.strictEqual(parsed.totals.duplicate_levels, parsed.totals.deduped);
  assert.strictEqual(parsed.totals.unsolved, parsed.totals.exhausted);
  assert.strictEqual(parsed.totals.solver_timeouts, parsed.totals.timeouts);
});

function runAcceptedSpec(name, ruleLines) {
  const specPath = writeSpec(`${name}.gen`, [
    ...initLevel,
    '(GENERATION RULES)',
    ...ruleLines,
    '',
  ]);
  const outPath = path.join(tempDir, `${name}.json`);
  run([
    gamePath,
    specPath,
    '--samples', '4',
    '--seed', '11',
    '--solver-timeout-ms', '50',
    '--top-k', '2',
    '--quiet',
    '--json-out', outPath,
  ], name);

  const json = JSON.parse(fs.readFileSync(outPath, 'utf8'));
  assert.strictEqual(json.totals.samples_attempted, 4, `${name} should run the requested samples`);
  assert.ok(json.totals.valid_generated > 0, `${name} should generate at least one valid level`);
}

test('option p is accepted in generation rules', () => runAcceptedSpec('option-probability', [
  'choose 1 option 1 [ no wall ] -> [ player ]',
  'choose 1 option 1 [ no wall no player no crate ] [ no wall no player no target ] -> [ crate ] [ target ]',
]));

test('grouped or alternatives are accepted', () => runAcceptedSpec('grouped-or', [
  'choose 1 [ no wall ] -> [ player ] or [ no wall ] -> [ player ]',
  'choose 1 [ no wall no player no crate ] [ no wall no player no target ] -> [ crate ] [ target ]',
]));

test('directional row patterns with pipe separators are accepted', () => runAcceptedSpec('directional-row-pipe', [
  'choose 1 right [ no wall | no wall ] -> [ player | crate ]',
  'choose 1 [ no wall no player no crate ] -> [ target ]',
]));

test('generator preset fixtures run', () => {
  const presetDir = path.join(__dirname, 'generator_presets');
  if (fs.existsSync(presetDir)) {
    for (const file of fs.readdirSync(presetDir).filter((name) => name.endsWith('.gen')).sort()) {
      const result = run([
        gamePath,
        path.join(presetDir, file),
        '--samples', '8',
        '--seed', '11',
        '--solver-timeout-ms', '20',
        '--top-k', '3',
        '--jobs', '1',
        '--quiet',
      ], `preset ${file}`);
      const presetJson = JSON.parse(result.stdout);
      assert.strictEqual(presetJson.totals.samples_attempted, 8, `${file} should run the requested samples`);
      assert.strictEqual(presetJson.totals.invalid_generation, presetJson.totals.rejected);
    }
  }
});

test('timed preset run does not crash during repeated solver searches', () => {
  const result = run([
    gamePath,
    path.join(__dirname, 'generator_presets', 'sokoban_room_scatter.gen'),
    '--time-ms', '250',
    '--jobs', '1',
    '--quiet',
  ], 'timed preset regression');
  const timedJson = JSON.parse(result.stdout);
  assert.ok(timedJson.totals.samples_attempted > 0, 'timed run should attempt samples');
});

test('square bracket sections are rejected', () => {
  const badSpecPath = writeSpec('bad-sections.gen', [
    '[ INIT LEVEL ]',
    '###',
    '#.#',
    '###',
    '',
    '[ GENERATION RULES ]',
    'choose 1 [ no wall ] -> [ player ]',
    '',
  ]);

  const bad = run([gamePath, badSpecPath, '--samples', '1', '--quiet'], 'bad sections', 1);
  assert.match(bad.stderr, /must use \(INIT LEVEL\) and \(GENERATION RULES\)/);
});

let failures = 0;
for (const { name, body } of tests) {
  try {
    body();
    process.stdout.write(`ok - ${name}\n`);
  } catch (error) {
    failures += 1;
    process.stderr.write(`not ok - ${name}\n${error.stack || error}\n`);
  }
}

if (failures > 0) {
  process.exit(1);
}

process.stdout.write('generator_smoke passed\n');
