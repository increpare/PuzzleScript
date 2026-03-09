/**
 * Lightweight PuzzleScript syntax highlighter for documentation.
 * Tokenizer rule: keyword (fixed list) -> cm-* token; everything else -> cm-NAME.
 * No CodeMirror dependency. Token classes (cm-*) are styled in doc-code.css.
 */
(function () {
  'use strict';

  // Keyword -> token type (class names used by doc-code.css)
  var SECTION_NAMES = ['objects', 'legend', 'sounds', 'collisionlayers', 'rules', 'winconditions', 'levels'];
  var METADATA_KEYS = ['title', 'author', 'homepage', 'require_player_movement', 'key_repeat_interval'];
  var KEYWORD_TO_TOKEN = Object.create(null);

  function addKeywords(words, token) {
    for (var i = 0; i < words.length; i++) {
      KEYWORD_TO_TOKEN[words[i].toLowerCase()] = token;
    }
  }

  addKeywords(SECTION_NAMES, 'HEADER');
  addKeywords(METADATA_KEYS, 'METADATA');
  addKeywords([
    'late', 'rigid', 'startloop', 'endloop', 'horizontal', 'vertical', 'orthogonal', 'noaction',
    'move', 'action', 'create', 'destroy', 'cantmove', 'cancel', 'restart', 'win', 'again',
    'undo', 'titlescreen', 'startgame', 'endgame', 'startlevel', 'endlevel',  'closemessage',
    'checkpoint', 'no', 'random', 'randomdir', 'any', 'all', 'some','on',
    'sfx0', 'sfx1', 'sfx2', 'sfx3', 'sfx4', 'sfx5', 'sfx6', 'sfx7', 'sfx8', 'sfx9', 'sfx10'], 'COMMAND');
  addKeywords(['up', 'down', 'left', 'right', 'moving', 'stationary', 'parallel', 'perpendicular'], 'DIRECTION');
  addKeywords(['message', 'showmessage'], 'MESSAGE_VERB');

  // Logic words and quantifiers
  KEYWORD_TO_TOKEN['or'] = 'LOGICWORD';
  KEYWORD_TO_TOKEN['and'] = 'LOGICWORD';
  addKeywords(['any', 'all', 'some'], 'LOGICWORD');

  // Color names for parsing sprite palette lines (lowercase for lookup)
  var COLOR_NAMES = ['black', 'white', 'gray', 'grey', 'darkgray', 'darkgrey', 'lightgray', 'lightgrey',
    'red', 'darkred', 'lightred', 'brown', 'darkbrown', 'lightbrown', 'orange', 'yellow', 'green',
    'darkgreen', 'lightgreen', 'blue', 'lightblue', 'darkblue', 'purple', 'pink', 'transparent'];
  var REGEX_HEX = /^#[0-9a-f]{6}$/i;

  function parsePaletteLine(line) {
    var palette = [];
    var tokens = line.trim().split(/\s+/);
    for (var i = 0; i < tokens.length; i++) {
      var raw = tokens[i];
      var t = raw.toLowerCase();
      if (COLOR_NAMES.indexOf(t) !== -1 && t !== 'transparent') {
        palette.push('COLOR-' + raw.toUpperCase().replace(/GREY/g, 'GRAY'));
      } else if (REGEX_HEX.test(raw)) {
        palette.push({ type: 'hex', value: raw });
      }
    }
    return palette;
  }

  function isSpriteMatrixLine(line) {
    var s = line.trim();
    return s.length === 5 && /^[.0-9]+$/.test(s);
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function span(cls, text) {
    return '<span class="cm-' + cls + '">' + escapeHtml(text) + '</span>';
  }

  // Mirror CodeMirror behaviour: avoid full-black on dark bg by lightening until contrast >= 2.361
  var DOC_BG_HEX = '#0F192A';
  var hexStyleCache = {};
  function hexColorStyle(hexCode) {
    if (hexStyleCache[hexCode]) return hexStyleCache[hexCode];
    function parseColor(input) {
      if (input.length < 7) return null;
      return [
        parseInt(input.slice(1, 3), 16),
        parseInt(input.slice(3, 5), 16),
        parseInt(input.slice(5, 7), 16)
      ];
    }
    function luminance(rgb) {
      var a = [rgb[0], rgb[1], rgb[2]].map(function (v) {
        v /= 255;
        return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
      });
      return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
    }
    var bgLum = luminance(parseColor(DOC_BG_HEX));
    function contrast(rgb) { return (luminance(rgb) + 0.05) / (bgLum + 0.05); }
    function rgbToHsl(rgb) {
      var r = rgb[0] / 255, g = rgb[1] / 255, b = rgb[2] / 255;
      var max = Math.max(r, g, b), min = Math.min(r, g, b);
      var h, s, l = (max + min) / 2;
      if (max === min) { h = s = 0; } else {
        var d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
        else if (max === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h /= 6;
      }
      return [h, s, l];
    }
    function hslToRgb(hsl) {
      var h = hsl[0], s = hsl[1], l = hsl[2];
      var r, g, b;
      if (s === 0) { r = g = b = l; } else {
        function hue2rgb(p, q, t) {
          if (t < 0) t += 1; if (t > 1) t -= 1;
          if (t < 1/6) return p + (q - p) * 6 * t;
          if (t < 1/2) return q;
          if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
          return p;
        }
        var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        var p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
      }
      return [r * 255, g * 255, b * 255];
    }
    var col = parseColor(hexCode);
    if (!col) { hexStyleCache[hexCode] = 'color:' + hexCode; return hexStyleCache[hexCode]; }
    var style;
    var r = contrast(col);
    if (r < 2.361) {
      var hsl = rgbToHsl(col);
      do { hsl[2] += 0.01; r = contrast(hslToRgb(hsl)); } while (r < 2.361 && hsl[2] <= 1);
      style = 'color:hsl(' + Math.round(hsl[0] * 360) + ',' + Math.round(hsl[1] * 100) + '%,' + Math.round(hsl[2] * 100) + '%)';
    } else {
      style = 'color:' + hexCode;
    }
    hexStyleCache[hexCode] = style;
    return style;
  }

  /**
   * Tokenize one line; returns array of { type, value }.
   */
  function tokenizeLine(line) {
    var tokens = [];
    var i = 0;
    var len = line.length;
    var sol = true;

    while (i < len) {
      // Skip whitespace
      if (/\s/.test(line[i])) {
        var wsStart = i;
        while (i < len && /\s/.test(line[i])) i++;
        tokens.push({ type: null, value: line.slice(wsStart, i) });
        sol = false;
        continue;
      }

      // Comment: ( ... )
      if (line[i] === '(') {
        var depth = 1;
        var start = i;
        i++;
        while (i < len && depth > 0) {
          if (line[i] === '(') depth++;
          else if (line[i] === ')') depth--;
          i++;
        }
        tokens.push({ type: 'comment', value: line.slice(start, i) });
        sol = false;
        continue;
      }

      if (line[i] === ')') {
        tokens.push({ type: 'comment', value: line[i] });
        i++;
        sol = false;
        continue;
      }

      // Equals: run at start of line -> EQUALSBIT; single = elsewhere -> ASSIGNMENT
      if (line[i] === '=') {
        if (sol) {
          var eqStart = i;
          while (i < len && line[i] === '=') i++;
          tokens.push({ type: 'EQUALSBIT', value: line.slice(eqStart, i) });
        } else {
          tokens.push({ type: 'ASSIGNMENT', value: '=' });
          i++;
        }
        sol = false;
        continue;
      }

      // Section name at start of line (match word only; next iteration emits trailing space)
      if (sol) {
        var sectionMatch = line.slice(i).match(/^([a-z]+)/i);
        if (sectionMatch) {
          var word = sectionMatch[1].toLowerCase();
          if (SECTION_NAMES.indexOf(word) !== -1) {
            tokens.push({ type: 'HEADER', value: sectionMatch[1] });
            i += sectionMatch[1].length;
            sol = false;
            continue;
          }
        }
      }

      // Metadata keyword at start of line (match word only; next iteration emits trailing space)
      if (sol) {
        var metaMatch = line.slice(i).match(/^([a-z]+)/i);
        if (metaMatch) {
          var m = metaMatch[1].toLowerCase();
          if (METADATA_KEYS.indexOf(m) !== -1) {
            tokens.push({ type: 'METADATA', value: metaMatch[1] });
            i += metaMatch[1].length;
            sol = false;
            continue;
          }
        }
      }

      // ->
      if (line[i] === '-' && line[i + 1] === '>') {
        tokens.push({ type: 'ARROW', value: '->' });
        i += 2;
        sol = false;
        continue;
      }

      // [ ] |
      if (line[i] === '[' || line[i] === ']' || line[i] === '|') {
        tokens.push({ type: 'BRACKET', value: line[i] });
        i++;
        sol = false;
        continue;
      }

      // ...
      if (line[i] === '.' && line[i + 1] === '.' && line[i + 2] === '.') {
        tokens.push({ type: 'COMMAND', value: '...' });
        i += 3;
        sol = false;
        continue;
      }

      // Direction symbols ^ v < >
      var sym = line[i];
      if (sym === '^' || sym === 'v' || sym === '<' || sym === '>') {
        tokens.push({ type: 'DIRECTION', value: sym });
        i++;
        sol = false;
        continue;
      }

      // Multi-digit numbers (e.g. rule numbers, sfx indices) -> SFX styling
      var numMatch = line.slice(i).match(/^\d{2,}/);
      if (numMatch) {
        tokens.push({ type: 'SOUND', value: numMatch[0] });
        i += numMatch[0].length;
        sol = false;
        continue;
      }

      // # followed by 6 hex digits -> colour by that hex
      if (line[i] === '#' && line.slice(i + 1).match(/^[0-9a-fA-F]{6}(?![0-9a-fA-F])/)) {
        var hex = line.slice(i, i + 7);
        tokens.push({ type: 'HEXCOLOR', value: hex });
        i += 7;
        sol = false;
        continue;
      }

      // Word: letters, numbers, underscore
      var wordMatch = line.slice(i).match(/^[\p{L}\p{N}_]+/u);
      if (wordMatch) {
        var w = wordMatch[0];
        var wLower = w.toLowerCase();
        var tok = KEYWORD_TO_TOKEN[wLower];
        if (!tok && COLOR_NAMES.indexOf(wLower) !== -1) {
          tok = (wLower === 'transparent') ? 'FADECOLOR' : 'COLOR-' + w.toUpperCase().replace(/GREY/g, 'GRAY');
        }
        tokens.push({ type: tok || 'NAME', value: w });
        i += w.length;
        sol = false;
        continue;
      }

      // Single character fallback
      tokens.push({ type: null, value: line[i] });
      i++;
      sol = false;
    }

    // High priority: anything after a METADATA keyword on this line -> METADATATEXT
    var afterMetadata = false;
    for (var j = 0; j < tokens.length; j++) {
      if (tokens[j].type === 'METADATA') afterMetadata = true;
      else if (afterMetadata) tokens[j].type = 'METADATATEXT';
    }

    // High priority: anything after a METADATA or MESSAGE_VERB keyword on this line -> METADATATEXT
    var afterMetadata = false;
    for (var j = 0; j < tokens.length; j++) {
      if (tokens[j].type === 'METADATA' || tokens[j].type === 'MESSAGE_VERB') afterMetadata = true;
      else if (afterMetadata) tokens[j].type = 'METADATATEXT';
    }
    //everything to the left of an cm-ASSIGNMENT is cm-NAME
    var beforeAssignment = false
    for (var j=tokens.length-1; j>=0; j--) {
      if (tokens[j].type === 'ASSIGNMENT') beforeAssignment = true;
      else if (beforeAssignment) tokens[j].type = 'NAME';

    }

    return tokens;
  }

  /**
   * Highlight full source: split into lines, tokenize each line, return HTML (spans only).
   * Sprite matrix lines (5 chars of . and 0-9) are colored using the previous line as palette.
   * Lines starting with # (level data) are rendered entirely as LEVEL (high priority).
   */
  function highlightPuzzleScript(code) {
    var lines = code.split(/\n/);
    var out = [];
    for (var L = 0; L < lines.length; L++) {
      var line = lines[L];
      //if line begins with # and has no = sign in it
      if (/^\s*#/.test(line) && !line.includes('=')) {
        out.push(span('LEVEL', line));
      } else if (isSpriteMatrixLine(line)) {
        var prevLine = L > 0 ? lines[L - 1] : '';
        var palette = parsePaletteLine(prevLine);
        var s = line.trim();
        for (var i = 0; i < s.length; i++) {
          var ch = s[i];
          if (ch === '.') {
            out.push(span('FADECOLOR', ch));
          } else {
            var n = parseInt(ch, 10);
            var entry = palette[n];
            if (entry === undefined) {
              entry = (n === 0) ? 'COLOR-PINK' : (n === 1) ? 'COLOR-YELLOW' : (n === 2) ? 'COLOR-BLACK' : 'MATRIX';
            }
            if (typeof entry === 'object' && entry.type === 'hex') {
              out.push('<span style="', hexColorStyle(entry.value), '">', escapeHtml(ch), '</span>');
            } else {
              out.push(span(entry, ch));
            }
          }
        }
      } else {
        var tokens = tokenizeLine(line);
        for (var t = 0; t < tokens.length; t++) {
          var tok = tokens[t];
          if (tok.type === 'HEXCOLOR') {
            out.push('<span style="', hexColorStyle(tok.value), '">', escapeHtml(tok.value), '</span>');
          } else if (tok.type) {
            out.push(span(tok.type, tok.value));
          } else {
            out.push('<span class="cm-ws">', escapeHtml(tok.value), '</span>');
          }
        }
      }
      if (L < lines.length - 1) out.push('\n');
    }
    return out.join('');
  }

  /**
   * Run highlighter on all pre/code blocks and preserve Edit links (order preserved).
   * Adds .doc-code to the <pre> so doc-code.css applies.
   */
  function init() {
    var blocks = document.querySelectorAll('pre code');
    for (var b = 0; b < blocks.length; b++) {
      var codeEl = blocks[b];
      var result = '';
      for (var c = 0; c < codeEl.childNodes.length; c++) {
        var node = codeEl.childNodes[c];
        if (node.nodeType === 1) {
          result += node.outerHTML;
        } else if (node.nodeType === 3 && node.textContent) {
          result += highlightPuzzleScript(node.textContent);
        }
      }
      if (result.length === 0) continue;
      codeEl.innerHTML = result;
      var preParent = codeEl.parentElement;
      if (preParent && preParent.tagName === 'PRE') {
        preParent.classList.add('doc-code');
      }
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.highlightPuzzleScript = highlightPuzzleScript;
})();
