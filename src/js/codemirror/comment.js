// CodeMirror, copyright (c) by Marijn Haverbeke and others
// Distributed under an MIT license: https://codemirror.net/LICENSE

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  var noOptions = {
    lineComment:"//",
  };
  var nonWS = /[^\s\u00a0]/;
  var Pos = CodeMirror.Pos, cmp = CodeMirror.cmpPos;

  function firstNonWS(str) {
    var found = str.search(nonWS);
    return found == -1 ? 0 : found;
  }

  CodeMirror.commands.toggleComment = function(cm) {
    cm.toggleComment();
  };

  CodeMirror.defineExtension("toggleComment", function(options) {
    if (!options) options = noOptions;
    var cm = this;
    var minLine = Infinity, ranges = this.listSelections(), mode = null;
    for (var i = ranges.length - 1; i >= 0; i--) {
      var from = ranges[i].from(), to = ranges[i].to();
      if (from.line >= minLine) continue;
      if (to.line >= minLine) to = Pos(minLine, 0);
      minLine = from.line;
      if (mode == null) {
        if (cm.uncomment(from, to, options)) mode = "un";
        else { cm.lineComment(from, to, options); mode = "line"; }
      } else if (mode == "un") {
        cm.uncomment(from, to, options);
      } else {
        cm.lineComment(from, to, options);
      }
    }
  });

  // Rough heuristic to try and detect lines that are part of multi-line string
  function probablyInsideString(cm, pos, line) {
    return /\bstring\b/.test(cm.getTokenTypeAt(Pos(pos.line, 0))) && !/^[\'\"\`]/.test(line)
  }

  function getMode(cm, pos) {
    var mode = cm.getMode()
    return mode.useInnerComments === false || !mode.innerMode ? mode : cm.getModeAt(pos)
  }

  CodeMirror.defineExtension("lineComment", function(from, to, options) {
    if (!options) options = noOptions;
    var self = this, mode = getMode(self, from);
    var firstLine = self.getLine(from.line);
    if (firstLine == null || probablyInsideString(self, from, firstLine)) return;

    var commentStart = "(";
    var commentEnd = ")";

    var end = Math.min(to.ch != 0 || to.line == from.line ? to.line + 1 : to.line, self.lastLine() + 1);
    var pad = options.padding == null ? " " : options.padding;
    var blankLines = options.commentBlankLines || from.line == to.line;

    self.operation(function() {
      
        for (var i = from.line; i < end; ++i) {
          var l = self.getLine(i);
          if (blankLines || nonWS.test(l)){
            self.replaceRange(commentStart + pad, Pos(i, 0));
            l = self.getLine(i);

            self.replaceRange(pad+commentEnd, Pos(i, l.length));
          }
      
      }
    });
  });

  CodeMirror.defineExtension("uncomment", function(from, to, options) {
    if (!options) options = noOptions;
    var self = this, mode = getMode(self, from);
    var end = Math.min(to.ch != 0 || to.line == from.line ? to.line : to.line - 1, self.lastLine()), start = Math.min(from.line, end);

    // Try finding line comments
    var commentStart = "(";
    var commentEnd = ")";
    var lines = [];
    var pad = options.padding == null ? " " : options.padding, didSomething;
    lineComment: {
      for (var i = start; i <= end; ++i) {
        var line = self.getLine(i);
        
        var found_s = line.indexOf(commentStart);
        var found_e = line.lastIndexOf(commentEnd);

        if (found_s > -1 && !/comment/.test(self.getTokenTypeAt(Pos(i, found_s + 1)))) found_s = -1;
        if (found_s == -1 && nonWS.test(line)) break lineComment;
        if (found_s > -1 && nonWS.test(line.slice(0, found_s))) break lineComment;
        
        
        if (found_e > -1 && !/comment/.test(self.getTokenTypeAt(Pos(i, found_e )))) found_e = -1;
        if (found_e == -1 && nonWS.test(line)) break lineComment;
        if (found_e > -1 && nonWS.test(line.slice(found_e+1))) break lineComment;

        lines.push(line);
      }
      self.operation(function() {
        for (var i = start; i <= end; ++i) {
          var line = lines[i - start];
          var pos_s = line.indexOf(commentStart), endPos_s = pos_s + commentStart.length;
          var pos_e = line.indexOf(commentEnd), endPos_e = pos_e + commentEnd.length;

          if (pos_s < 0 || pos_e < 0) continue;


          if (line.slice(endPos_e-1-pad.length, endPos_e-1) == pad) pos_e -= pad.length;
          if (line.slice(endPos_s, endPos_s + pad.length) == pad) endPos_s += pad.length;
          didSomething = true;
          self.replaceRange("", Pos(i, pos_e), Pos(i, endPos_e));
          self.replaceRange("", Pos(i, pos_s), Pos(i, endPos_s));
        }
      });
      if (didSomething) return true;
    }
    
    return false;
  });
});
