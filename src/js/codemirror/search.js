// CodeMirror, copyright (c) by Marijn Haverbeke and others
// Distributed under an MIT license: https://codemirror.net/5/LICENSE

// PuzzleScript-specific backport of the CM6 search panel to CM5.

// Define search commands. Depends on searchcursor.js and panel.js.

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"), require("./searchcursor"), require("../display/panel"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror", "./searchcursor", "../display/panel"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // default search panel location
  CodeMirror.defineOption("search", {bottom: false});

  function isWordChar(ch) {
    return !!ch && CodeMirror.isWordChar(ch);
  }

  function isWholeWordMatch(beforeStart, first, last, afterEnd) {
    return (!isWordChar(beforeStart) || !isWordChar(first)) &&
           (!isWordChar(last) || !isWordChar(afterEnd));
  }

  function charBefore(cm, pos) {
    if (pos.ch) return cm.getLine(pos.line).charAt(pos.ch - 1);
    return pos.line > cm.firstLine() ? "\n" : "";
  }

  function charAfter(cm, pos) {
    var line = cm.getLine(pos.line);
    if (pos.ch < line.length) return line.charAt(pos.ch);
    return pos.line < cm.lastLine() ? "\n" : "";
  }

  function isWholeWordRange(cm, from, to) {
    return isWholeWordMatch(charBefore(cm, from), charAfter(cm, from),
                             charBefore(cm, to), charAfter(cm, to));
  }

  function searchOverlay(query, caseInsensitive, wholeWord) {
    if (typeof query == "string")
      query = new RegExp(query.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&"), caseInsensitive ? "gi" : "g");
    else if (!query.global)
      query = new RegExp(query.source, query.ignoreCase ? "gi" : "g");

    return {token: function(stream) {
      for (;;) {
        query.lastIndex = stream.pos;
        var match = query.exec(stream.string);
        if (!match) {
          stream.skipToEnd();
          return;
        }
        var end = match.index + match[0].length;
        if (wholeWord && !isWholeWordMatch(stream.string.charAt(match.index - 1),
                                            stream.string.charAt(match.index),
                                            stream.string.charAt(end - 1),
                                            stream.string.charAt(end))) {
          stream.pos = match.index + (match[0].length || 1);
          continue;
        }
        if (match.index == stream.pos) {
          stream.pos += match[0].length || 1;
          return "searching";
        }
        stream.pos = match.index;
        return;
      }
    }};
  }

  function SearchState() {
    this.posFrom = this.posTo = null;
    this.query = null;
    this.queryText = "";
    this.replaceText = "";
    this.regexp = false;
    this.wholeWord = false;
    this.zeroWidthMatch = false;
    this.overlay = null;
    this.annotate = null;
    this.panel = null;
    this.closePanel = null;
    this.panelHandle = null;
    this.panelObserver = null;
    this.searchField = null;
    this.replaceField = null;
    this.regexpField = null;
    this.wholeWordField = null;
    this.loggedInvalidQueries = null;
  }

  function getSearchState(cm) {
    return cm.state.search || (cm.state.search = new SearchState());
  }

  function getSearchCursor(cm, query, pos, wholeWord) {
    var cursor = cm.getSearchCursor(query, pos, {
      caseFold: true,
      multiline: true
    });
    if (!wholeWord) return cursor;

    return {
      find: function(reverse) {
        var found;
        while (found = cursor.find(reverse)) {
          if (isWholeWordRange(cm, cursor.from(), cursor.to())) return found;
        }
        return false;
      },
      findNext: function() { return this.find(false); },
      findPrevious: function() { return this.find(true); },
      from: function() { return cursor.from(); },
      to: function() { return cursor.to(); },
      replace: function(text, origin) { cursor.replace(text, origin); }
    };
  }

  function parseString(string) {
    return string.replace(/\\([nrt\\])/g, function(match, ch) {
      if (ch == "n") return "\n"
      if (ch == "r") return "\r"
      if (ch == "t") return "\t"
      if (ch == "\\") return "\\"
      return match
    })
  }

  function compileQuery(text, regexp) {
    if (!text) return {query: null, error: null};
    if (!regexp) return {query: text, error: null};
    try {
      return {query: new RegExp(text, "i"), error: null};
    } catch (error) {
      return {query: null, error: error};
    }
  }

  function clearOverlay(cm, state) {
    if (state.overlay) {
      cm.removeOverlay(state.overlay);
      state.overlay = null;
    }
    if (state.annotate) {
      state.annotate.clear();
      state.annotate = null;
    }
  }

  function reportInvalidQuery(state, error) {
    var key = state.queryText;
    if (Object.prototype.hasOwnProperty.call(state.loggedInvalidQueries, key)) return;
    state.loggedInvalidQueries[key] = true;
    if (typeof console != "undefined" && console.error) {
      console.error("Invalid search regular expression \"" + key + "\": " + error.message);
    }
  }

  function startSearch(cm, state) {
    var compiled = compileQuery(state.queryText, state.regexp);
    clearOverlay(cm, state);
    state.zeroWidthMatch = false;
    if (compiled.error) {
      reportInvalidQuery(state, compiled.error);
      state.query = null;
      return compiled;
    }
    state.query = compiled.query;
    if (!compiled.query) return compiled;

    state.overlay = searchOverlay(state.query, true, state.wholeWord);
    cm.addOverlay(state.overlay);
    if (cm.showMatchesOnScrollbar) {
      state.annotate = cm.showMatchesOnScrollbar(state.query, true);
    }
    return compiled;
  }

  function samePosition(a, b) {
    return a.line == b.line && a.ch == b.ch;
  }

  function findCursor(cm, state, reverse, start) {
    var cursor = getSearchCursor(cm, state.query, start, state.wholeWord);
    var match = cursor.find(reverse);
    if (match) {
      cursor.match = match;
      return cursor;
    }
    var line = reverse ? cm.lastLine() : cm.firstLine();
    var wrap = reverse ? CodeMirror.Pos(line, cm.getLine(line).length) : CodeMirror.Pos(line, 0);
    cursor = getSearchCursor(cm, state.query, wrap, state.wholeWord);
    match = cursor.find(reverse);
    if (!match) return null;
    cursor.match = match;
    return cursor;
  }

  function replacementText(state, match) {
    var text = parseString(state.replaceText);
    if (typeof state.query == "string") return text;
    return text.replace(/\$(\d)/g, function(_, index) {
      return match[index] == null ? "" : match[index];
    });
  }

  function selectCursor(cm, cursor) {
    cm.setSelection(cursor.from(), cursor.to());
    cm.scrollIntoView({from: cursor.from(), to: cursor.to()}, 20);
  }

  function findNext(cm, rev, callback) {cm.operation(function() {
    var state = getSearchState(cm);
    if (!state.query) return;
    var pos = rev ? state.posFrom : state.posTo;
    var cursor = findCursor(cm, state, rev, pos);
    if (cursor && state.zeroWidthMatch && samePosition(cursor.from(), pos) && samePosition(cursor.to(), pos)) {
      if (!cursor.find(rev)) {
        var line = rev ? cm.lastLine() : cm.firstLine();
        cursor = findCursor(cm, state, rev,
                            rev ? CodeMirror.Pos(line, cm.getLine(line).length) : CodeMirror.Pos(line, 0));
      }
    }
    if (!cursor) return;
    selectCursor(cm, cursor);
    state.posFrom = cursor.from(); state.posTo = cursor.to();
    state.zeroWidthMatch = CodeMirror.cmpPos(state.posFrom, state.posTo) == 0;
    if (callback) callback(cursor.from(), cursor.to())
  });}

  function el(tag, attrs) {
    var element = tag ? document.createElement(tag) : document.createDocumentFragment();
    for (var key in attrs) {
      element[key] = attrs[key];
    }
    for (var i = 2; i < arguments.length; i++) {
      var child = arguments[i]
      element.appendChild(typeof child == "string" ? document.createTextNode(child) : child);
    }
    return element;
  }

  function setAriaLabel(element, text) {
    element.setAttribute("aria-label", text);
    return element;
  }

  function textField(name, placeholder, value) {
    var field = el("input", {
      type: "text",
      name: name,
      value: value,
      className: "CodeMirror-search-field"
    });
    field.setAttribute("form", "");
    field.setAttribute("placeholder", placeholder);
    return setAriaLabel(field, placeholder);
  }

  function searchButton(name, text, action, title) {
    var button = el("button", {
      type: "button",
      name: name,
      className: "CodeMirror-search-button",
      title: title
    }, text);
    CodeMirror.on(button, "click", function(event) {
      CodeMirror.e_preventDefault(event);
      action();
    });
    return button;
  }

  function platformShortcut(pc, mac) {
    return CodeMirror.keyMap["default"] == CodeMirror.keyMap.macDefault ? mac : pc;
  }

  function checkbox(name, text, checked) {
    var input = el("input", {
      type: "checkbox",
      name: name,
      checked: checked
    });
    input.setAttribute("form", "");
    return {input: input, label: el("label", null, input, text)};
  }

  function updateSearchFromPanel(cm) {
    var state = getSearchState(cm);
    state.queryText = state.searchField.value;
    state.replaceText = state.replaceField.value;
    state.regexp = state.regexpField.checked;
    state.wholeWord = state.wholeWordField.checked;
    state.posFrom = state.posTo = cm.getCursor();
    startSearch(cm, state);
  }

  function listenForPanelChanges(cm, state) {
    CodeMirror.on(state.searchField, "input", function() { updateSearchFromPanel(cm); });
    CodeMirror.on(state.replaceField, "input", function() {
      state.replaceText = state.replaceField.value;
    });
    CodeMirror.on(state.regexpField, "change", function() { updateSearchFromPanel(cm); });
    CodeMirror.on(state.wholeWordField, "change", function() { updateSearchFromPanel(cm); });
  }

  function focusAndSelect(field) {
    field.focus();
    field.select();
  }

  function ctrlShortcut(event, shift) {
    return event.ctrlKey && !event.metaKey && !event.altKey && !!event.shiftKey == shift;
  }

  function cmdShortcut(event, shift) {
    return event.metaKey && !event.ctrlKey && !event.altKey && !!event.shiftKey == shift;
  }

  function cmdAltShortcut(event, shift) {
    return event.metaKey && !event.ctrlKey && event.altKey && !!event.shiftKey == shift;
  }

  function panelKeyDown(cm, state, event) {
    var handled = true;
    if (event.keyCode == 27) {
      closeSearchPanel(cm);
    } else if (event.keyCode == 13 && !event.ctrlKey && !event.metaKey && !event.altKey &&
               event.target == state.searchField) {
      findNext(cm, event.shiftKey);
    } else if (event.keyCode == 13 && !event.ctrlKey && !event.metaKey && !event.altKey &&
               !event.shiftKey && event.target == state.replaceField) {
      replaceNext(cm);
    } else if (event.keyCode == 70 && (ctrlShortcut(event, false) || cmdShortcut(event, false))) {
      focusAndSelect(state.searchField);
    } else if ((event.keyCode == 82 && ctrlShortcut(event, true)) ||
               (event.keyCode == 70 && cmdAltShortcut(event, true))) {
      replaceAllMatches(cm);
    } else if (event.keyCode == 71 && (ctrlShortcut(event, !!event.shiftKey) ||
                                      cmdShortcut(event, !!event.shiftKey))) {
      findNext(cm, event.shiftKey);
    } else {
      handled = false;
    }
    if (handled) CodeMirror.e_stop(event);
  }

  function makeSearchPanel(cm, state) {
    var regexp = checkbox("regexp", cm.phrase("regex"), state.regexp);
    var wholeWord = checkbox("wholeWord", cm.phrase("whole word"), state.wholeWord);
    var options = el("span", {className: "CodeMirror-search-options"},
                     regexp.label, wholeWord.label);

    state.searchField = textField("search", cm.phrase("Find"), state.queryText);
    state.replaceField = textField("replace", cm.phrase("Replace"), state.replaceText);
    state.regexpField = regexp.input;
    state.wholeWordField = wholeWord.input;

    state.panel = el("div", {className: "CodeMirror-search-panel"},
      state.searchField,
      searchButton("next", cm.phrase("next"), function() { findNext(cm, false); },
                   platformShortcut("Ctrl+G", "\u2318+G")),
      searchButton("prev", cm.phrase("previous"), function() { findNext(cm, true); },
                   platformShortcut("Ctrl+Shift+G", "\u2318+Shift+G")),
      options,
      el("br", null),
      state.replaceField,
      searchButton("replace", cm.phrase("replace"), function() { replaceNext(cm); }, "Enter"),
      searchButton("replaceAll", cm.phrase("replace all"), function() { replaceAllMatches(cm); },
                   platformShortcut("Ctrl+Shift+R", "\u2318+Alt+Shift+F")),
      searchButton("close", "\u00d7", function() { closeSearchPanel(cm); }, "Esc")
    );
    state.panel.querySelector("[name=close]").setAttribute("aria-label", cm.phrase("close"));
    CodeMirror.on(state.panel, "keydown", function(event) {
      panelKeyDown(cm, state, event);
    });
    listenForPanelChanges(cm, state);
    return state.panel;
  }

  function mountSearchPanel(cm, state, panel) {
    if (typeof cm.addPanel != "function")
      throw new Error("CodeMirror search requires addon/display/panel.js");

    var bottom = cm.options.search.bottom;
    var dialog = el("div", {
      className: "CodeMirror-dialog CodeMirror-dialog-" +
                 (bottom ? "bottom" : "top") + " CodeMirror-search-dialog"
    }, panel);
    var originalHeight = cm.getWrapperElement().style.height;
    var handle = cm.addPanel(dialog, {position: bottom ? "bottom" : "top"});
    var closed = false;
    state.panelHandle = handle;
    cm.setSize(null, "100%");
    if (typeof window != "undefined" && typeof window.ResizeObserver == "function") {
      state.panelObserver = new window.ResizeObserver(function() {
        if (state.panelHandle == handle && !handle.cleared) handle.changed();
      });
      state.panelObserver.observe(dialog);
    }

    return function() {
      if (closed) return;
      closed = true;
      if (state.panelObserver) {
        state.panelObserver.disconnect();
        state.panelObserver = null;
      }
      if (state.panelHandle == handle) state.panelHandle = null;
      handle.clear();
      cm.setSize(null, originalHeight);
      panelClosed(cm);
      cm.focus();
    };
  }

  function panelClosed(cm) {
    var state = getSearchState(cm);
    state.panel = state.closePanel = state.panelHandle = state.panelObserver = null;
    state.searchField = state.replaceField = null;
    state.regexpField = state.wholeWordField = null;
    state.loggedInvalidQueries = null;
    clearActiveSearch(cm, state);
  }

  function closeSearchPanel(cm) {
    var state = getSearchState(cm);
    if (state.closePanel) state.closePanel();
    else clearActiveSearch(cm, state);
  }

  function queryFromSelection(selection) {
    if (!/[\r\n]/.test(selection)) return selection;

    var lines = selection.split(/\r\n?|\n/);
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (line) return line;
    }
    return "";
  }

  function openSearchPanel(cm, focusReplace) {
    var state = getSearchState(cm);
    if (state.panel && state.panel.parentNode) {
      var existing = focusReplace ? state.replaceField : state.searchField;
      focusAndSelect(existing);
      return;
    }

    var selection = cm.getSelection();
    if (selection) state.queryText = queryFromSelection(selection);
    var panel = makeSearchPanel(cm, state);
    state.loggedInvalidQueries = Object.create(null);
    state.closePanel = mountSearchPanel(cm, state, panel);
    state.posFrom = state.posTo = cm.getCursor();
    startSearch(cm, state);
    focusAndSelect(focusReplace ? state.replaceField : state.searchField);
  }

  function clearActiveSearch(cm, state) {
    cm.operation(function() {
      state.query = null;
      clearOverlay(cm, state);
    });
  }

  function clearSearch(cm) {
    closeSearchPanel(cm);
  }

  function replaceNext(cm) {
    var state = getSearchState(cm);
    if (cm.getOption("readOnly") || !state.query) return;

    cm.operation(function() {
      var from = cm.getCursor("from");
      var to = cm.getCursor("to");
      var cursor = findCursor(cm, state, false, from);
      if (!cursor) return;

      if (samePosition(cursor.from(), from) && samePosition(cursor.to(), to)) {
        cursor.replace(replacementText(state, cursor.match), "replace");
        if (!cursor.findNext())
          cursor = findCursor(cm, state, false, CodeMirror.Pos(cm.firstLine(), 0));
      }
      if (cursor) {
        selectCursor(cm, cursor);
        state.posFrom = cursor.from(); state.posTo = cursor.to();
        state.zeroWidthMatch = CodeMirror.cmpPos(state.posFrom, state.posTo) == 0;
      }
    });
  }

  function replaceAllMatches(cm) {
    var state = getSearchState(cm);
    if (cm.getOption("readOnly") || !state.query) return;

    cm.operation(function() {
      var cursor = getSearchCursor(cm, state.query, CodeMirror.Pos(cm.firstLine(), 0), state.wholeWord);
      var found = [], match;
      while (match = cursor.findNext())
        found.push({from: cursor.from(), to: cursor.to(), match: match});
      for (var i = found.length - 1; i >= 0; i--) {
        match = found[i];
        cm.replaceRange(replacementText(state, match.match), match.from, match.to, "replace");
      }
    });
  }

  CodeMirror.commands.find = function(cm) { openSearchPanel(cm, false); };
  CodeMirror.commands.findPersistent = CodeMirror.commands.find;
  CodeMirror.commands.findNext = function(cm) { findNext(cm, false); };
  CodeMirror.commands.findPrev = function(cm) { findNext(cm, true); };
  CodeMirror.commands.findPersistentNext = CodeMirror.commands.findNext;
  CodeMirror.commands.findPersistentPrev = CodeMirror.commands.findPrev;
  CodeMirror.commands.clearSearch = clearSearch;
  CodeMirror.commands.replace = function(cm) {
    openSearchPanel(cm, true);
  };
  CodeMirror.commands.replaceAll = function(cm) {
    var state = getSearchState(cm);
    if (!state.panel) openSearchPanel(cm, true);
    else replaceAllMatches(cm);
  };
});
