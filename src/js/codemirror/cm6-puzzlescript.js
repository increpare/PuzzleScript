import { EditorSelection, EditorState, RangeSetBuilder } from "@codemirror/state";
import { moveLineDown, moveLineUp, defaultKeymap, history, historyKeymap, indentWithTab, toggleComment } from "@codemirror/commands";
import { autocompletion, completionKeymap, startCompletion } from "@codemirror/autocomplete";
import { closeSearchPanel, highlightSelectionMatches, search, searchKeymap } from "@codemirror/search";
import {
  Decoration,
  Direction,
  EditorView,
  ViewPlugin,
  drawSelection,
  highlightActiveLineGutter,
  keymap,
  layer,
  lineNumbers,
  RectangleMarker
} from "@codemirror/view";

function ensureEditorCodeMirrorCompat() {
  if (!window.CodeMirror) {
    window.CodeMirror = function CodeMirror() {};
  }

  const cm = window.CodeMirror;
  if (!cm.modes) {
    cm.modes = {};
  }
  if (!cm.hint) {
    cm.hint = {};
  }
  if (!cm.commands) {
    cm.commands = {};
  }
  if (!cm.defaults) {
    cm.defaults = {};
  }
  if (cm.defaults.mode == null) {
    cm.defaults.mode = null;
  }
  if (!cm.Pos) {
    cm.Pos = function(line, ch) {
      return { line: line, ch: ch };
    };
  }
  if (!cm.defineMode) {
    cm.defineMode = function(name, modeFactory) {
      cm.modes[name] = modeFactory;
      if (cm.defaults.mode == null) {
        cm.defaults.mode = name;
      }
    };
  }
  if (!cm.registerHelper) {
    cm.registerHelper = function(type, name, value) {
      if (!cm[type]) {
        cm[type] = {};
      }
      cm[type][name] = value;
    };
  }

  return cm;
}

ensureEditorCodeMirrorCompat();

function countColumn(string, end, tabSize, startIndex, startValue) {
  if (end == null) {
    end = string.search(/[^\s\u00a0]/);
    if (end === -1) {
      end = string.length;
    }
  }
  let n = startValue || 0;
  for (let i = startIndex || 0; i < end; i++) {
    if (string.charCodeAt(i) === 9) {
      n += tabSize - (n % tabSize);
    } else {
      n++;
    }
  }
  return n;
}

if (!window.CodeMirror.StringStream) {
  window.CodeMirror.StringStream = function(string, tabSize) {
    this.pos = this.start = 0;
    this.string = string;
    this.tabSize = tabSize || 8;
    this.lastColumnPos = this.lastColumnValue = 0;
    this.lineStart = 0;
  };

  window.CodeMirror.StringStream.prototype = {
    eol: function() { return this.pos >= this.string.length; },
    sol: function() { return this.pos === this.lineStart; },
    peek: function() { return this.string.charAt(this.pos) || undefined; },
    next: function() {
      if (this.pos < this.string.length) {
        return this.string.charAt(this.pos++);
      }
    },
    eat: function(match) {
      const ch = this.string.charAt(this.pos);
      let ok;
      if (typeof match === "string") {
        ok = ch === match;
      } else {
        ok = ch && (match.test ? match.test(ch) : match(ch));
      }
      if (ok) {
        ++this.pos;
        return ch;
      }
    },
    eatWhile: function(match) {
      const start = this.pos;
      while (this.eat(match)) {}
      return this.pos > start;
    },
    eatSpace: function() {
      const start = this.pos;
      while (/[\s\u00a0]/.test(this.string.charAt(this.pos))) {
        ++this.pos;
      }
      return this.pos > start;
    },
    skipToEnd: function() { this.pos = this.string.length; },
    skipTo: function(ch) {
      const found = this.string.indexOf(ch, this.pos);
      if (found > -1) {
        this.pos = found;
        return true;
      }
    },
    backUp: function(n) { this.pos -= n; },
    column: function() {
      if (this.lastColumnPos < this.start) {
        this.lastColumnValue = countColumn(this.string, this.start, this.tabSize, this.lastColumnPos, this.lastColumnValue);
        this.lastColumnPos = this.start;
      }
      return this.lastColumnValue - (this.lineStart ? countColumn(this.string, this.lineStart, this.tabSize) : 0);
    },
    indentation: function() {
      return countColumn(this.string, null, this.tabSize) -
        (this.lineStart ? countColumn(this.string, this.lineStart, this.tabSize) : 0);
    },
    match: function(pattern, consume, caseInsensitive) {
      if (typeof pattern === "string") {
        const cased = function(str) { return caseInsensitive ? str.toLowerCase() : str; };
        const substr = this.string.substr(this.pos, pattern.length);
        if (cased(substr) === cased(pattern)) {
          if (consume !== false) {
            this.pos += pattern.length;
          }
          return true;
        }
      } else {
        const match = this.string.slice(this.pos).match(pattern);
        if (match && match.index > 0) {
          return null;
        }
        if (match && consume !== false) {
          this.pos += match[0].length;
        }
        return match;
      }
    },
    current: function() { return this.string.slice(this.start, this.pos); },
    hideFirstChars: function(n, inner) {
      this.lineStart += n;
      try {
        return inner();
      } finally {
        this.lineStart -= n;
      }
    }
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHTML(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

const colorCache = new Map();

function styleFromHexCode(hexCode) {
  function parseColor(input) {
    input = input.trim();
    if (input.length < 4) {
      return null;
    } else if (input.length < 7) {
      return [
        parseInt(input.charAt(1), 16) * 0x11,
        parseInt(input.charAt(2), 16) * 0x11,
        parseInt(input.charAt(3), 16) * 0x11
      ];
    }
    return [
      parseInt(input.substr(1, 2), 16),
      parseInt(input.substr(3, 2), 16),
      parseInt(input.substr(5, 2), 16)
    ];
  }

  function luminance(rgb) {
    const a = rgb.map(function(v) {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
  }

  function contrast(rgb) {
    return (luminance(rgb) + 0.05) / (bgLuminance + 0.05);
  }

  function rgbToHsl(rgb) {
    let [r, g, b] = rgb;
    r /= 255;
    g /= 255;
    b /= 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h;
    let s;
    let l = (max + min) / 2;

    if (max === min) {
      h = 0;
      s = 0;
    } else {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case r:
          h = (g - b) / d + (g < b ? 6 : 0);
          break;
        case g:
          h = (b - r) / d + 2;
          break;
        default:
          h = (r - g) / d + 4;
          break;
      }
      h /= 6;
    }

    return [h, s, l];
  }

  function hslToRgb(hsl) {
    const [h, s, l] = hsl;
    let r;
    let g;
    let b;

    if (s === 0) {
      r = l;
      g = l;
      b = l;
    } else {
      function hue2rgb(p, q, t) {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      }

      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1 / 3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1 / 3);
    }

    return [r * 255, g * 255, b * 255];
  }

  const bgLuminance = luminance(parseColor("#0F192A"));
  if (colorCache.has(hexCode)) {
    return colorCache.get(hexCode);
  }

  let style = "";
  const color = parseColor(hexCode);
  if (color) {
    let ratio = contrast(color);
    if (ratio < 2.361) {
      const hsl = rgbToHsl(color);
      while (ratio < 2.361 && hsl[2] < 1) {
        hsl[2] += 0.01;
        ratio = contrast(hslToRgb(hsl));
      }
      style = "color: hsl(" + Math.trunc(hsl[0] * 360) + "," + Math.trunc(hsl[1] * 100) + "%," + Math.trunc(hsl[2] * 100) + "%)";
    } else {
      style = "color:" + hexCode;
    }
  }

  colorCache.set(hexCode, style);
  return style;
}

function cloneModeState(mode, state) {
  return mode.copyState ? mode.copyState(state) : state;
}

function createMode() {
  const modeFactory = window.CodeMirror && window.CodeMirror.modes && window.CodeMirror.modes.puzzle;
  if (!modeFactory) {
    throw new Error("PuzzleScript mode is not registered on CodeMirror.modes.puzzle.");
  }
  return modeFactory();
}

class PuzzleTokenizer {
  constructor(doc) {
    this.mode = createMode();
    this.reset(doc);
  }

  reset(doc) {
    this.doc = doc;
    this.states = [cloneModeState(this.mode, this.mode.startState())];
    this.builtThrough = 0;
  }

  ensureStateBeforeLine(lineNumber) {
    while (this.builtThrough < lineNumber) {
      const previousState = cloneModeState(this.mode, this.states[this.builtThrough]);
      this.tokenizeText(this.doc.line(this.builtThrough + 1).text, previousState);
      this.states[this.builtThrough + 1] = cloneModeState(this.mode, previousState);
      this.builtThrough++;
    }
    return cloneModeState(this.mode, this.states[lineNumber]);
  }

  tokenizeText(text, state) {
    const tokens = [];
    if (text.length === 0) {
      if (this.mode.blankLine) {
        this.mode.blankLine(state);
      }
      return tokens;
    }

    const stream = new window.CodeMirror.StringStream(text, 4);
    while (!stream.eol()) {
      stream.start = stream.pos;
      const style = this.mode.token(stream, state) || "";
      if (stream.pos <= stream.start) {
        stream.pos = stream.start + 1;
      }
      tokens.push({
        from: stream.start,
        to: stream.pos,
        style: style
      });
    }
    return tokens;
  }

  tokenizeLine(lineNumber) {
    const state = this.ensureStateBeforeLine(lineNumber);
    const line = this.doc.line(lineNumber + 1);
    return this.tokenizeText(line.text, state);
  }

  getTokenAt(lineNumber, ch) {
    const state = this.ensureStateBeforeLine(lineNumber);
    const line = this.doc.line(lineNumber + 1);
    const stream = new window.CodeMirror.StringStream(line.text, 4);
    let lastToken = {
      start: 0,
      end: 0,
      string: "",
      type: "",
      state: cloneModeState(this.mode, state)
    };

    if (line.text.length === 0) {
      if (this.mode.blankLine) {
        this.mode.blankLine(state);
      }
      lastToken.state = cloneModeState(this.mode, state);
      return lastToken;
    }

    while (!stream.eol()) {
      stream.start = stream.pos;
      const style = this.mode.token(stream, state) || "";
      if (stream.pos <= stream.start) {
        stream.pos = stream.start + 1;
      }
      lastToken = {
        start: stream.start,
        end: stream.pos,
        string: line.text.slice(stream.start, stream.pos),
        type: style,
        state: cloneModeState(this.mode, state)
      };
      if (ch <= stream.pos) {
        return lastToken;
      }
    }

    return lastToken;
  }
}

function tokenStyleSpec(style) {
  if (!style) {
    return null;
  }

  const parts = style.split(/\s+/).filter(Boolean);
  if (parts.length === 0) {
    return null;
  }

  const classes = [];
  let inlineStyle = "";

  for (const part of parts) {
    if (part.indexOf("MULTICOLOR") === 0) {
      const color = part.slice("MULTICOLOR".length);
      classes.push("cm-COLOR");
      inlineStyle = styleFromHexCode(color);
      continue;
    }

    classes.push("cm-" + part);
    if (part.indexOf("COLOR-#") === 0) {
      inlineStyle = styleFromHexCode(part.slice("COLOR-".length));
      if (classes.indexOf("cm-COLOR") === -1) {
        classes.push("cm-COLOR");
      }
    }
  }

  return {
    className: classes.join(" "),
    inlineStyle: inlineStyle
  };
}

const decorationCache = new Map();

function tokenDecoration(style) {
  const spec = tokenStyleSpec(style);
  if (!spec) {
    return null;
  }

  const cacheKey = spec.className + "|" + spec.inlineStyle;
  if (!decorationCache.has(cacheKey)) {
    const config = {};
    if (spec.className) {
      config.class = spec.className;
    }
    if (spec.inlineStyle) {
      config.attributes = { style: spec.inlineStyle };
    }
    decorationCache.set(cacheKey, Decoration.mark(config));
  }
  return decorationCache.get(cacheKey);
}

function completionKind(tag) {
  switch ((tag || "").toUpperCase()) {
    case "COLOR":
      return "constant";
    case "DIRECTION":
      return "keyword";
    case "COMMAND":
    case "MESSAGE_VERB":
    case "SOUNDEVENT":
    case "SOUNDVERB":
      return "keyword";
    case "NAME":
    case "IDENTIFIER":
      return "variable";
    default:
      return "text";
  }
}

function completionInfo(option) {
  if (!option.extra && !option.tag) {
    return null;
  }
  const bits = [];
  if (option.tag) {
    bits.push("<strong>" + escapeHTML(option.tag) + "</strong>");
  }
  if (option.extra) {
    bits.push("<div>" + escapeHTML(option.extra) + "</div>");
  }
  return function() {
    const wrap = document.createElement("div");
    wrap.className = "puzzlescript-completion-info";
    wrap.innerHTML = bits.join("");
    return wrap;
  };
}

function makePuzzleCompletionSource(getEditorFacade) {
  return function(context) {
    if (!window.CodeMirror || !window.CodeMirror.hint || !window.CodeMirror.hint.anyword) {
      return null;
    }

    const editorFacade = getEditorFacade();
    if (!editorFacade) {
      return null;
    }

    const line = context.state.doc.lineAt(context.pos);
    const result = window.CodeMirror.hint.anyword(editorFacade, {});
    if (!result || !result.list || result.list.length === 0) {
      return null;
    }

    return {
      from: line.from + result.from.ch,
      to: line.from + result.to.ch,
      filter: false,
      options: result.list.map(function(option) {
        return {
          label: option.text,
          apply: option.text,
          detail: option.extra || "",
          type: completionKind(option.tag),
          info: completionInfo(option)
        };
      })
    };
  };
}

function toOffset(doc, pos) {
  if (pos == null) {
    return doc.length;
  }
  const lineNumber = clamp(pos.line, 0, doc.lines - 1) + 1;
  const line = doc.line(lineNumber);
  const ch = clamp(pos.ch || 0, 0, line.length);
  return line.from + ch;
}

function fromOffset(doc, offset) {
  const line = doc.lineAt(offset);
  return {
    line: line.number - 1,
    ch: offset - line.from
  };
}

function createEditor(textarea) {
  const listeners = {
    change: [],
    keyup: [],
    mousedown: [],
    drop: [],
    beforeChange: []
  };
  let editorFacade = null;
  let tokenizer = null;

  const mount = document.createElement("div");
  mount.className = "puzzlescript-cm6-host";
  mount.style.height = "100%";
  mount.style.width = "100%";

  const form = textarea.parentNode;
  if (form && form.style) {
    form.style.height = "100%";
  }
  textarea.style.display = "none";
  textarea.insertAdjacentElement("afterend", mount);

  function emit(name, payload) {
    for (const callback of listeners[name]) {
      callback(editorFacade, payload);
    }
  }

  function buildDecorations(view) {
    const builder = new RangeSetBuilder();
    let lastLine = -1;

    for (const range of view.visibleRanges) {
      let startLine = view.state.doc.lineAt(range.from).number - 1;
      const endLine = view.state.doc.lineAt(range.to).number - 1;
      if (startLine <= lastLine) {
        startLine = lastLine + 1;
      }
      for (let lineNumber = startLine; lineNumber <= endLine; lineNumber++) {
        const line = view.state.doc.line(lineNumber + 1);
        const tokens = tokenizer.tokenizeLine(lineNumber);
        for (const token of tokens) {
          const decoration = tokenDecoration(token.style);
          if (!decoration || token.from === token.to) {
            continue;
          }
          builder.add(line.from + token.from, line.from + token.to, decoration);
        }
      }
      lastLine = endLine;
    }

    return builder.finish();
  }

  const syntaxPlugin = ViewPlugin.fromClass(class {
    constructor(view) {
      this.decorations = buildDecorations(view);
    }

    update(update) {
      if (update.docChanged) {
        tokenizer.reset(update.state.doc);
      }
      if (update.docChanged || update.viewportChanged) {
        this.decorations = buildDecorations(update.view);
      }
    }
  }, {
    decorations: function(value) {
      return value.decorations;
    }
  });

  function isDarkTheme() {
    return document.body.classList.contains("dark-theme") ||
      (!document.body.classList.contains("light-theme") && document.body.style.colorScheme !== "light");
  }

  // Coordinate base for layer markers (matches @codemirror/view internal getBase).
  function getLayerBase(view) {
    let rect = view.scrollDOM.getBoundingClientRect();
    let left = view.textDirection === Direction.LTR ? rect.left : rect.right - view.scrollDOM.clientWidth * view.scaleX;
    return {
      left: left - view.scrollDOM.scrollLeft * view.scaleX,
      top: rect.top - view.scrollDOM.scrollTop * view.scaleY
    };
  }

  // Active line layer drawn *behind* the selection (CM5 behaviour). Must come after drawSelection()
  // in extensions so it gets a lower z-index. Spans full editor width (content area).
  const activeLineBehindLayer = layer({
    above: false,
    class: "cm-activeLineLayer",
    markers(view) {
      let lastLineStart = -1;
      let markers = [];
      let content = view.contentDOM;
      let contentRect = content.getBoundingClientRect();
      let lineElt = content.querySelector(".cm-line");
      let lineStyle = lineElt ? window.getComputedStyle(lineElt) : null;
      let paddingLeft = (lineStyle ? parseInt(lineStyle.paddingLeft, 10) : 0) || 0;
      let paddingRight = (lineStyle ? parseInt(lineStyle.paddingRight, 10) : 0) || 0;
      let leftSide = contentRect.left + paddingLeft;
      let rightSide = contentRect.right - paddingRight;
      let base = getLayerBase(view);
      for (let r of view.state.selection.ranges) {
        let line = view.lineBlockAt(r.head);
        if (line.from > lastLineStart) {
          let coords = view.coordsAtPos(line.from);
          if (coords) {
            let left = leftSide - base.left;
            let top = coords.top - base.top;
            let width = Math.max(0, rightSide - leftSide);
            let height = coords.bottom - coords.top;
            markers.push(new RectangleMarker("cm-activeLine", left, top, width, height));
          }
          lastLineStart = line.from;
        }
      }
      return markers;
    },
    update(update, dom) {
      return update.docChanged || update.selectionSet || update.viewportChanged;
    }
  });

  const darkTheme = EditorView.theme({
    "&": {
      height: "100%",
      backgroundColor: "#0F192A",
      color: "#D1EDFF",
      fontFamily: "'Consolas', 'Lucida Console', monospace"
    },
    ".cm-scroller": {
      overflow: "auto",
      fontFamily: "inherit",
      lineHeight: "1"
    },
    ".cm-content": {
      minHeight: "100%",
      caretColor: "#F8F8F0",
      lineHeight: "1"
    },
    ".cm-line": {
      lineHeight: "1",
      paddingLeft: "4px"
    },
    ".cm-cursor, .cm-dropCursor": {
      borderLeftColor: "#F8F8F0"
    },
    "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection": {
      backgroundColor: "#314D67"
    },
    ".cm-activeLine": {
      backgroundColor: "#203040"
    },
    ".cm-gutters": {
      backgroundColor: "#0F192A",
      color: "#999",
      borderRight: "1px solid #d1edff"
    },
    ".cm-gutterElement": {
      color: "#999"
    },
    ".cm-activeLineGutter": {
      backgroundColor: "#19315b"
    },
    ".cm-panels": {
      backgroundColor: "#203040",
      color: "yellow"
    },
    ".cm-panels.cm-panels-top": {
      borderBottom: "2px solid black"
    },
    ".cm-panels.cm-panels-bottom": {
      borderTop: "2px solid black"
    },
    ".cm-searchMatch": {
      backgroundColor: "#494949",
      outline: "1px solid #b0b0b0"
    },
    ".cm-searchMatch.cm-searchMatch-selected": {
      backgroundColor: "#314D67"
    },
    ".cm-selectionMatch": {
      backgroundColor: "#49494930"
    },
    ".cm-tooltip": {
      border: "1px solid silver",
      backgroundColor: "#0F192A",
      color: "#FFc540"
    },
    ".cm-tooltip-autocomplete": {
      "& > ul > li[aria-selected]": {
        backgroundColor: "#3f5883",
        color: "orange"
      }
    },
    ".cm-foldPlaceholder": {
      backgroundColor: "transparent",
      border: "none",
      color: "#D1EDFF"
    }
  }, { dark: true });

  const lightTheme = EditorView.theme({
    "&": {
      height: "100%",
      backgroundColor: "#ffffff",
      color: "#848484",
      fontFamily: "'Consolas', 'Lucida Console', monospace"
    },
    ".cm-scroller": {
      overflow: "auto",
      fontFamily: "inherit",
      lineHeight: "1"
    },
    ".cm-content": {
      minHeight: "100%",
      caretColor: "#164",
      lineHeight: "1"
    },
    ".cm-line": {
      lineHeight: "1",
      paddingLeft: "4px"
    },
    ".cm-cursor, .cm-dropCursor": {
      borderLeftColor: "#164"
    },
    "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection": {
      backgroundColor: "#eeffee"
    },
    ".cm-activeLine": {
      backgroundColor: "#e8f2ff"
    },
    ".cm-gutters": {
      backgroundColor: "#f5f5f5",
      color: "#8b8b8b",
      borderRight: "1px solid #ddd"
    },
    ".cm-activeLineGutter": {
      backgroundColor: "#c1c3d1"
    },
    ".cm-panels": {
      backgroundColor: "#f5f5f5",
      color: "#333"
    },
    ".cm-searchMatch": {
      backgroundColor: "#ffff0054",
      outline: "1px solid #999"
    },
    ".cm-searchMatch.cm-searchMatch-selected": {
      backgroundColor: "#eeffee"
    },
    ".cm-selectionMatch": {
      backgroundColor: "#e8f2ff"
    },
    ".cm-tooltip": {
      border: "1px solid black",
      backgroundColor: "#ffffff",
      color: "#c38a06"
    },
    ".cm-tooltip-autocomplete": {
      "& > ul > li[aria-selected]": {
        backgroundColor: "#e8f2ff",
        color: "rgb(233, 105, 0)"
      }
    },
    ".cm-foldPlaceholder": {
      backgroundColor: "transparent",
      border: "none",
      color: "#848484"
    }
  }, { dark: false });

  const puzzleTheme = isDarkTheme() ? darkTheme : lightTheme;

  function clickToken(event, view) {
    const pos = view.posAtCoords({ x: event.clientX, y: event.clientY });
    if (pos == null) {
      return false;
    }

    const line = view.state.doc.lineAt(pos);
    const token = tokenizer.getTokenAt(line.number - 1, pos - line.from);
    const tokenClasses = token.type.split(/\s+/).filter(Boolean);
    const tokenText = line.text.slice(token.start, token.end);

    if (tokenClasses.indexOf("SOUND") >= 0) {
      const seed = parseInt(tokenText, 10);
      if (!Number.isNaN(seed)) {
        playSound(seed, true);
        event.preventDefault();
        return true;
      }
    }

    if (tokenClasses.indexOf("LEVEL") >= 0 && (event.ctrlKey || event.metaKey)) {
      document.activeElement.blur();
      view.contentDOM.blur();
      event.preventDefault();
      if (typeof prevent === "function") {
        prevent(event);
      }
      compile(["levelline", line.number - 1]);
      return true;
    }

    return false;
  }

  function createState(docText) {
    return EditorState.create({
      doc: docText,
      extensions: [
        EditorState.allowMultipleSelections.of(false),
        EditorState.languageData.of(function() {
          return [{
            commentTokens: {
              block: {
                open: "(",
                close: ")"
              }
            }
          }];
        }),
        EditorView.editorAttributes.of({
          class: "cm-s-midnight"
        }),
        EditorView.lineWrapping,
        lineNumbers(),
        drawSelection(),
        activeLineBehindLayer,
        highlightActiveLineGutter(),
        history(),
        search(),
        highlightSelectionMatches(),
        syntaxPlugin,
        puzzleTheme,
        EditorView.updateListener.of(function(update) {
          if (update.docChanged) {
            textarea.value = update.state.doc.toString();
            emit("change", update);
          }
        }),
        EditorView.domEventHandlers({
          keyup: function(event, view) {
            emit("keyup", event);
            return false;
          },
          drop: function(event, view) {
            emit("drop", event);
            return false;
          },
          mousedown: function(event, view) {
            emit("mousedown", event);
            return clickToken(event, view);
          }
        }),
        autocompletion({
          override: [makePuzzleCompletionSource(function() {
            return editorFacade;
          })],
          activateOnTyping: true
        }),
        keymap.of([
          {
            key: "Ctrl-/",
            run: toggleComment
          },
          {
            key: "Cmd-/",
            run: toggleComment
          },
          {
            key: "Esc",
            run: closeSearchPanel
          },
          {
            key: "Shift-Ctrl-ArrowUp",
            run: moveLineUp
          },
          {
            key: "Shift-Ctrl-ArrowDown",
            run: moveLineDown
          },
          {
            key: "Shift-Cmd-ArrowUp",
            run: moveLineUp
          },
          {
            key: "Shift-Cmd-ArrowDown",
            run: moveLineDown
          },
          indentWithTab,
          ...completionKeymap,
          ...searchKeymap,
          ...historyKeymap,
          ...defaultKeymap
        ])
      ]
    });
  }

  tokenizer = new PuzzleTokenizer(EditorState.create({ doc: textarea.value }).doc);
  let view = new EditorView({
    state: createState(textarea.value),
    parent: mount
  });

  function ensureView() {
    return view;
  }

  editorFacade = {
    view: view,
    display: {
      input: {
        blur: function() {
          ensureView().contentDOM.blur();
        },
        focus: function() {
          ensureView().focus();
        }
      }
    },
    doc: {
      lastLine: function() {
        return ensureView().state.doc.lines - 1;
      }
    },
    on: function(name, callback) {
      if (!listeners[name]) {
        listeners[name] = [];
      }
      listeners[name].push(callback);
    },
    getValue: function() {
      return ensureView().state.doc.toString();
    },
    setValue: function(value) {
      textarea.value = value;
      tokenizer.reset(EditorState.create({ doc: value }).doc);
      view.setState(createState(value));
      editorFacade.view = view;
    },
    clearHistory: function() {
      const currentText = ensureView().state.doc.toString();
      const currentSelection = ensureView().state.selection.main;
      tokenizer.reset(EditorState.create({ doc: currentText }).doc);
      view.setState(createState(currentText));
      view.dispatch({
        selection: {
          anchor: currentSelection.anchor,
          head: currentSelection.head
        }
      });
      editorFacade.view = view;
    },
    setOption: function(name, value) {
      if (name === "theme" && value !== "midnight") {
        console.warn("Only the midnight theme is currently supported by the CM6 bridge.");
      }
    },
    getLine: function(lineNumber) {
      return ensureView().state.doc.line(lineNumber + 1).text;
    },
    firstLine: function() {
      return 0;
    },
    lastLine: function() {
      return ensureView().state.doc.lines - 1;
    },
    getCursor: function() {
      return fromOffset(ensureView().state.doc, ensureView().state.selection.main.head);
    },
    getTokenAt: function(pos) {
      if (tokenizer.doc !== ensureView().state.doc) {
        tokenizer.reset(ensureView().state.doc);
      }
      return tokenizer.getTokenAt(pos.line, pos.ch);
    },
    listSelections: function() {
      return ensureView().state.selection.ranges.map(function(range) {
        return {
          anchor: fromOffset(ensureView().state.doc, range.anchor),
          head: fromOffset(ensureView().state.doc, range.head),
          from: function() {
            return fromOffset(ensureView().state.doc, range.from);
          },
          to: function() {
            return fromOffset(ensureView().state.doc, range.to);
          },
          empty: function() {
            return range.empty;
          }
        };
      });
    },
    operation: function(callback) {
      callback();
    },
    replaceRange: function(text, from, to) {
      const doc = ensureView().state.doc;
      const fromOffsetValue = toOffset(doc, from);
      const toOffsetValue = to == null ? fromOffsetValue : toOffset(doc, to);
      ensureView().dispatch({
        changes: {
          from: fromOffsetValue,
          to: toOffsetValue,
          insert: text
        }
      });
    },
    setSelections: function(selections) {
      const ranges = selections.map(function(selection) {
        return EditorSelection.range(
          toOffset(ensureView().state.doc, selection.anchor),
          toOffset(ensureView().state.doc, selection.head)
        );
      });
      ensureView().dispatch({
        selection: EditorSelection.create(ranges)
      });
    },
    scrollIntoView: function(target) {
      let pos = 0;
      if (typeof target === "number") {
        const line = ensureView().state.doc.line(clamp(target, 0, ensureView().state.doc.lines - 1) + 1);
        pos = line.from;
      } else if (target && typeof target.line === "number") {
        pos = toOffset(ensureView().state.doc, target);
      }
      ensureView().dispatch({
        effects: EditorView.scrollIntoView(pos, { y: "center" })
      });
    },
    setCursor: function(line, ch) {
      const offset = toOffset(ensureView().state.doc, { line: line, ch: ch });
      ensureView().dispatch({
        selection: { anchor: offset },
        effects: EditorView.scrollIntoView(offset, { y: "center" })
      });
    },
    posFromMouse: function(event) {
      const pos = ensureView().posAtCoords({ x: event.clientX, y: event.clientY });
      if (pos == null) {
        return {
          line: 0,
          ch: 0
        };
      }
      return fromOffset(ensureView().state.doc, pos);
    },
    refresh: function() {},
    focus: function() {
      ensureView().focus();
    },
    startAutocomplete: function() {
      startCompletion(ensureView());
    },
    closeSearch: function() {
      closeSearchPanel(ensureView());
    }
  };

  window.CodeMirror.commands.autocomplete = function(editor) {
    editor.startAutocomplete();
  };
  window.CodeMirror.commands.clearSearch = function(editor) {
    editor.closeSearch();
  };
  window.CodeMirror.commands.toggleComment = function(editor) {
    toggleComment(editor.view);
  };
  window.CodeMirror.commands.swapLineUp = function(editor) {
    moveLineUp(editor.view);
  };
  window.CodeMirror.commands.swapLineDown = function(editor) {
    moveLineDown(editor.view);
  };
  textarea.value = view.state.doc.toString();
  return editorFacade;
}

window.PuzzleScriptCM6 = {
  createEditor: createEditor
};
