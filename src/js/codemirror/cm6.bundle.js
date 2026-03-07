(() => {
  // node_modules/@marijn/find-cluster-break/src/index.js
  var rangeFrom = [];
  var rangeTo = [];
  (() => {
    let numbers = "lc,34,7n,7,7b,19,,,,2,,2,,,20,b,1c,l,g,,2t,7,2,6,2,2,,4,z,,u,r,2j,b,1m,9,9,,o,4,,9,,3,,5,17,3,3b,f,,w,1j,,,,4,8,4,,3,7,a,2,t,,1m,,,,2,4,8,,9,,a,2,q,,2,2,1l,,4,2,4,2,2,3,3,,u,2,3,,b,2,1l,,4,5,,2,4,,k,2,m,6,,,1m,,,2,,4,8,,7,3,a,2,u,,1n,,,,c,,9,,14,,3,,1l,3,5,3,,4,7,2,b,2,t,,1m,,2,,2,,3,,5,2,7,2,b,2,s,2,1l,2,,,2,4,8,,9,,a,2,t,,20,,4,,2,3,,,8,,29,,2,7,c,8,2q,,2,9,b,6,22,2,r,,,,,,1j,e,,5,,2,5,b,,10,9,,2u,4,,6,,2,2,2,p,2,4,3,g,4,d,,2,2,6,,f,,jj,3,qa,3,t,3,t,2,u,2,1s,2,,7,8,,2,b,9,,19,3,3b,2,y,,3a,3,4,2,9,,6,3,63,2,2,,1m,,,7,,,,,2,8,6,a,2,,1c,h,1r,4,1c,7,,,5,,14,9,c,2,w,4,2,2,,3,1k,,,2,3,,,3,1m,8,2,2,48,3,,d,,7,4,,6,,3,2,5i,1m,,5,ek,,5f,x,2da,3,3x,,2o,w,fe,6,2x,2,n9w,4,,a,w,2,28,2,7k,,3,,4,,p,2,5,,47,2,q,i,d,,12,8,p,b,1a,3,1c,,2,4,2,2,13,,1v,6,2,2,2,2,c,,8,,1b,,1f,,,3,2,2,5,2,,,16,2,8,,6m,,2,,4,,fn4,,kh,g,g,g,a6,2,gt,,6a,,45,5,1ae,3,,2,5,4,14,3,4,,4l,2,fx,4,ar,2,49,b,4w,,1i,f,1k,3,1d,4,2,2,1x,3,10,5,,8,1q,,c,2,1g,9,a,4,2,,2n,3,2,,,2,6,,4g,,3,8,l,2,1l,2,,,,,m,,e,7,3,5,5f,8,2,3,,,n,,29,,2,6,,,2,,,2,,2,6j,,2,4,6,2,,2,r,2,2d,8,2,,,2,2y,,,,2,6,,,2t,3,2,4,,5,77,9,,2,6t,,a,2,,,4,,40,4,2,2,4,,w,a,14,6,2,4,8,,9,6,2,3,1a,d,,2,ba,7,,6,,,2a,m,2,7,,2,,2,3e,6,3,,,2,,7,,,20,2,3,,,,9n,2,f0b,5,1n,7,t4,,1r,4,29,,f5k,2,43q,,,3,4,5,8,8,2,7,u,4,44,3,1iz,1j,4,1e,8,,e,,m,5,,f,11s,7,,h,2,7,,2,,5,79,7,c5,4,15s,7,31,7,240,5,gx7k,2o,3k,6o".split(",").map((s) => s ? parseInt(s, 36) : 1);
    for (let i = 0, n = 0; i < numbers.length; i++)
      (i % 2 ? rangeTo : rangeFrom).push(n = n + numbers[i]);
  })();
  function isExtendingChar(code) {
    if (code < 768) return false;
    for (let from = 0, to = rangeFrom.length; ; ) {
      let mid = from + to >> 1;
      if (code < rangeFrom[mid]) to = mid;
      else if (code >= rangeTo[mid]) from = mid + 1;
      else return true;
      if (from == to) return false;
    }
  }
  function isRegionalIndicator(code) {
    return code >= 127462 && code <= 127487;
  }
  var ZWJ = 8205;
  function findClusterBreak(str, pos, forward = true, includeExtending = true) {
    return (forward ? nextClusterBreak : prevClusterBreak)(str, pos, includeExtending);
  }
  function nextClusterBreak(str, pos, includeExtending) {
    if (pos == str.length) return pos;
    if (pos && surrogateLow(str.charCodeAt(pos)) && surrogateHigh(str.charCodeAt(pos - 1))) pos--;
    let prev = codePointAt(str, pos);
    pos += codePointSize(prev);
    while (pos < str.length) {
      let next = codePointAt(str, pos);
      if (prev == ZWJ || next == ZWJ || includeExtending && isExtendingChar(next)) {
        pos += codePointSize(next);
        prev = next;
      } else if (isRegionalIndicator(next)) {
        let countBefore = 0, i = pos - 2;
        while (i >= 0 && isRegionalIndicator(codePointAt(str, i))) {
          countBefore++;
          i -= 2;
        }
        if (countBefore % 2 == 0) break;
        else pos += 2;
      } else {
        break;
      }
    }
    return pos;
  }
  function prevClusterBreak(str, pos, includeExtending) {
    while (pos > 0) {
      let found = nextClusterBreak(str, pos - 2, includeExtending);
      if (found < pos) return found;
      pos--;
    }
    return 0;
  }
  function codePointAt(str, pos) {
    let code0 = str.charCodeAt(pos);
    if (!surrogateHigh(code0) || pos + 1 == str.length) return code0;
    let code1 = str.charCodeAt(pos + 1);
    if (!surrogateLow(code1)) return code0;
    return (code0 - 55296 << 10) + (code1 - 56320) + 65536;
  }
  function surrogateLow(ch) {
    return ch >= 56320 && ch < 57344;
  }
  function surrogateHigh(ch) {
    return ch >= 55296 && ch < 56320;
  }
  function codePointSize(code) {
    return code < 65536 ? 1 : 2;
  }

  // node_modules/@codemirror/state/dist/index.js
  var Text = class _Text {
    /**
    Get the line description around the given position.
    */
    lineAt(pos) {
      if (pos < 0 || pos > this.length)
        throw new RangeError(`Invalid position ${pos} in document of length ${this.length}`);
      return this.lineInner(pos, false, 1, 0);
    }
    /**
    Get the description for the given (1-based) line number.
    */
    line(n) {
      if (n < 1 || n > this.lines)
        throw new RangeError(`Invalid line number ${n} in ${this.lines}-line document`);
      return this.lineInner(n, true, 1, 0);
    }
    /**
    Replace a range of the text with the given content.
    */
    replace(from, to, text) {
      [from, to] = clip(this, from, to);
      let parts = [];
      this.decompose(
        0,
        from,
        parts,
        2
        /* Open.To */
      );
      if (text.length)
        text.decompose(
          0,
          text.length,
          parts,
          1 | 2
          /* Open.To */
        );
      this.decompose(
        to,
        this.length,
        parts,
        1
        /* Open.From */
      );
      return TextNode.from(parts, this.length - (to - from) + text.length);
    }
    /**
    Append another document to this one.
    */
    append(other) {
      return this.replace(this.length, this.length, other);
    }
    /**
    Retrieve the text between the given points.
    */
    slice(from, to = this.length) {
      [from, to] = clip(this, from, to);
      let parts = [];
      this.decompose(from, to, parts, 0);
      return TextNode.from(parts, to - from);
    }
    /**
    Test whether this text is equal to another instance.
    */
    eq(other) {
      if (other == this)
        return true;
      if (other.length != this.length || other.lines != this.lines)
        return false;
      let start = this.scanIdentical(other, 1), end = this.length - this.scanIdentical(other, -1);
      let a = new RawTextCursor(this), b = new RawTextCursor(other);
      for (let skip = start, pos = start; ; ) {
        a.next(skip);
        b.next(skip);
        skip = 0;
        if (a.lineBreak != b.lineBreak || a.done != b.done || a.value != b.value)
          return false;
        pos += a.value.length;
        if (a.done || pos >= end)
          return true;
      }
    }
    /**
    Iterate over the text. When `dir` is `-1`, iteration happens
    from end to start. This will return lines and the breaks between
    them as separate strings.
    */
    iter(dir = 1) {
      return new RawTextCursor(this, dir);
    }
    /**
    Iterate over a range of the text. When `from` > `to`, the
    iterator will run in reverse.
    */
    iterRange(from, to = this.length) {
      return new PartialTextCursor(this, from, to);
    }
    /**
    Return a cursor that iterates over the given range of lines,
    _without_ returning the line breaks between, and yielding empty
    strings for empty lines.
    
    When `from` and `to` are given, they should be 1-based line numbers.
    */
    iterLines(from, to) {
      let inner;
      if (from == null) {
        inner = this.iter();
      } else {
        if (to == null)
          to = this.lines + 1;
        let start = this.line(from).from;
        inner = this.iterRange(start, Math.max(start, to == this.lines + 1 ? this.length : to <= 1 ? 0 : this.line(to - 1).to));
      }
      return new LineCursor(inner);
    }
    /**
    Return the document as a string, using newline characters to
    separate lines.
    */
    toString() {
      return this.sliceString(0);
    }
    /**
    Convert the document to an array of lines (which can be
    deserialized again via [`Text.of`](https://codemirror.net/6/docs/ref/#state.Text^of)).
    */
    toJSON() {
      let lines = [];
      this.flatten(lines);
      return lines;
    }
    /**
    @internal
    */
    constructor() {
    }
    /**
    Create a `Text` instance for the given array of lines.
    */
    static of(text) {
      if (text.length == 0)
        throw new RangeError("A document must have at least one line");
      if (text.length == 1 && !text[0])
        return _Text.empty;
      return text.length <= 32 ? new TextLeaf(text) : TextNode.from(TextLeaf.split(text, []));
    }
  };
  var TextLeaf = class _TextLeaf extends Text {
    constructor(text, length = textLength(text)) {
      super();
      this.text = text;
      this.length = length;
    }
    get lines() {
      return this.text.length;
    }
    get children() {
      return null;
    }
    lineInner(target, isLine, line, offset) {
      for (let i = 0; ; i++) {
        let string2 = this.text[i], end = offset + string2.length;
        if ((isLine ? line : end) >= target)
          return new Line(offset, end, line, string2);
        offset = end + 1;
        line++;
      }
    }
    decompose(from, to, target, open) {
      let text = from <= 0 && to >= this.length ? this : new _TextLeaf(sliceText(this.text, from, to), Math.min(to, this.length) - Math.max(0, from));
      if (open & 1) {
        let prev = target.pop();
        let joined = appendText(text.text, prev.text.slice(), 0, text.length);
        if (joined.length <= 32) {
          target.push(new _TextLeaf(joined, prev.length + text.length));
        } else {
          let mid = joined.length >> 1;
          target.push(new _TextLeaf(joined.slice(0, mid)), new _TextLeaf(joined.slice(mid)));
        }
      } else {
        target.push(text);
      }
    }
    replace(from, to, text) {
      if (!(text instanceof _TextLeaf))
        return super.replace(from, to, text);
      [from, to] = clip(this, from, to);
      let lines = appendText(this.text, appendText(text.text, sliceText(this.text, 0, from)), to);
      let newLen = this.length + text.length - (to - from);
      if (lines.length <= 32)
        return new _TextLeaf(lines, newLen);
      return TextNode.from(_TextLeaf.split(lines, []), newLen);
    }
    sliceString(from, to = this.length, lineSep = "\n") {
      [from, to] = clip(this, from, to);
      let result = "";
      for (let pos = 0, i = 0; pos <= to && i < this.text.length; i++) {
        let line = this.text[i], end = pos + line.length;
        if (pos > from && i)
          result += lineSep;
        if (from < end && to > pos)
          result += line.slice(Math.max(0, from - pos), to - pos);
        pos = end + 1;
      }
      return result;
    }
    flatten(target) {
      for (let line of this.text)
        target.push(line);
    }
    scanIdentical() {
      return 0;
    }
    static split(text, target) {
      let part = [], len = -1;
      for (let line of text) {
        part.push(line);
        len += line.length + 1;
        if (part.length == 32) {
          target.push(new _TextLeaf(part, len));
          part = [];
          len = -1;
        }
      }
      if (len > -1)
        target.push(new _TextLeaf(part, len));
      return target;
    }
  };
  var TextNode = class _TextNode extends Text {
    constructor(children, length) {
      super();
      this.children = children;
      this.length = length;
      this.lines = 0;
      for (let child of children)
        this.lines += child.lines;
    }
    lineInner(target, isLine, line, offset) {
      for (let i = 0; ; i++) {
        let child = this.children[i], end = offset + child.length, endLine = line + child.lines - 1;
        if ((isLine ? endLine : end) >= target)
          return child.lineInner(target, isLine, line, offset);
        offset = end + 1;
        line = endLine + 1;
      }
    }
    decompose(from, to, target, open) {
      for (let i = 0, pos = 0; pos <= to && i < this.children.length; i++) {
        let child = this.children[i], end = pos + child.length;
        if (from <= end && to >= pos) {
          let childOpen = open & ((pos <= from ? 1 : 0) | (end >= to ? 2 : 0));
          if (pos >= from && end <= to && !childOpen)
            target.push(child);
          else
            child.decompose(from - pos, to - pos, target, childOpen);
        }
        pos = end + 1;
      }
    }
    replace(from, to, text) {
      [from, to] = clip(this, from, to);
      if (text.lines < this.lines)
        for (let i = 0, pos = 0; i < this.children.length; i++) {
          let child = this.children[i], end = pos + child.length;
          if (from >= pos && to <= end) {
            let updated = child.replace(from - pos, to - pos, text);
            let totalLines = this.lines - child.lines + updated.lines;
            if (updated.lines < totalLines >> 5 - 1 && updated.lines > totalLines >> 5 + 1) {
              let copy = this.children.slice();
              copy[i] = updated;
              return new _TextNode(copy, this.length - (to - from) + text.length);
            }
            return super.replace(pos, end, updated);
          }
          pos = end + 1;
        }
      return super.replace(from, to, text);
    }
    sliceString(from, to = this.length, lineSep = "\n") {
      [from, to] = clip(this, from, to);
      let result = "";
      for (let i = 0, pos = 0; i < this.children.length && pos <= to; i++) {
        let child = this.children[i], end = pos + child.length;
        if (pos > from && i)
          result += lineSep;
        if (from < end && to > pos)
          result += child.sliceString(from - pos, to - pos, lineSep);
        pos = end + 1;
      }
      return result;
    }
    flatten(target) {
      for (let child of this.children)
        child.flatten(target);
    }
    scanIdentical(other, dir) {
      if (!(other instanceof _TextNode))
        return 0;
      let length = 0;
      let [iA, iB, eA, eB] = dir > 0 ? [0, 0, this.children.length, other.children.length] : [this.children.length - 1, other.children.length - 1, -1, -1];
      for (; ; iA += dir, iB += dir) {
        if (iA == eA || iB == eB)
          return length;
        let chA = this.children[iA], chB = other.children[iB];
        if (chA != chB)
          return length + chA.scanIdentical(chB, dir);
        length += chA.length + 1;
      }
    }
    static from(children, length = children.reduce((l, ch) => l + ch.length + 1, -1)) {
      let lines = 0;
      for (let ch of children)
        lines += ch.lines;
      if (lines < 32) {
        let flat = [];
        for (let ch of children)
          ch.flatten(flat);
        return new TextLeaf(flat, length);
      }
      let chunk = Math.max(
        32,
        lines >> 5
        /* Tree.BranchShift */
      ), maxChunk = chunk << 1, minChunk = chunk >> 1;
      let chunked = [], currentLines = 0, currentLen = -1, currentChunk = [];
      function add2(child) {
        let last;
        if (child.lines > maxChunk && child instanceof _TextNode) {
          for (let node of child.children)
            add2(node);
        } else if (child.lines > minChunk && (currentLines > minChunk || !currentLines)) {
          flush();
          chunked.push(child);
        } else if (child instanceof TextLeaf && currentLines && (last = currentChunk[currentChunk.length - 1]) instanceof TextLeaf && child.lines + last.lines <= 32) {
          currentLines += child.lines;
          currentLen += child.length + 1;
          currentChunk[currentChunk.length - 1] = new TextLeaf(last.text.concat(child.text), last.length + 1 + child.length);
        } else {
          if (currentLines + child.lines > chunk)
            flush();
          currentLines += child.lines;
          currentLen += child.length + 1;
          currentChunk.push(child);
        }
      }
      function flush() {
        if (currentLines == 0)
          return;
        chunked.push(currentChunk.length == 1 ? currentChunk[0] : _TextNode.from(currentChunk, currentLen));
        currentLen = -1;
        currentLines = currentChunk.length = 0;
      }
      for (let child of children)
        add2(child);
      flush();
      return chunked.length == 1 ? chunked[0] : new _TextNode(chunked, length);
    }
  };
  Text.empty = /* @__PURE__ */ new TextLeaf([""], 0);
  function textLength(text) {
    let length = -1;
    for (let line of text)
      length += line.length + 1;
    return length;
  }
  function appendText(text, target, from = 0, to = 1e9) {
    for (let pos = 0, i = 0, first = true; i < text.length && pos <= to; i++) {
      let line = text[i], end = pos + line.length;
      if (end >= from) {
        if (end > to)
          line = line.slice(0, to - pos);
        if (pos < from)
          line = line.slice(from - pos);
        if (first) {
          target[target.length - 1] += line;
          first = false;
        } else
          target.push(line);
      }
      pos = end + 1;
    }
    return target;
  }
  function sliceText(text, from, to) {
    return appendText(text, [""], from, to);
  }
  var RawTextCursor = class {
    constructor(text, dir = 1) {
      this.dir = dir;
      this.done = false;
      this.lineBreak = false;
      this.value = "";
      this.nodes = [text];
      this.offsets = [dir > 0 ? 1 : (text instanceof TextLeaf ? text.text.length : text.children.length) << 1];
    }
    nextInner(skip, dir) {
      this.done = this.lineBreak = false;
      for (; ; ) {
        let last = this.nodes.length - 1;
        let top2 = this.nodes[last], offsetValue = this.offsets[last], offset = offsetValue >> 1;
        let size = top2 instanceof TextLeaf ? top2.text.length : top2.children.length;
        if (offset == (dir > 0 ? size : 0)) {
          if (last == 0) {
            this.done = true;
            this.value = "";
            return this;
          }
          if (dir > 0)
            this.offsets[last - 1]++;
          this.nodes.pop();
          this.offsets.pop();
        } else if ((offsetValue & 1) == (dir > 0 ? 0 : 1)) {
          this.offsets[last] += dir;
          if (skip == 0) {
            this.lineBreak = true;
            this.value = "\n";
            return this;
          }
          skip--;
        } else if (top2 instanceof TextLeaf) {
          let next = top2.text[offset + (dir < 0 ? -1 : 0)];
          this.offsets[last] += dir;
          if (next.length > Math.max(0, skip)) {
            this.value = skip == 0 ? next : dir > 0 ? next.slice(skip) : next.slice(0, next.length - skip);
            return this;
          }
          skip -= next.length;
        } else {
          let next = top2.children[offset + (dir < 0 ? -1 : 0)];
          if (skip > next.length) {
            skip -= next.length;
            this.offsets[last] += dir;
          } else {
            if (dir < 0)
              this.offsets[last]--;
            this.nodes.push(next);
            this.offsets.push(dir > 0 ? 1 : (next instanceof TextLeaf ? next.text.length : next.children.length) << 1);
          }
        }
      }
    }
    next(skip = 0) {
      if (skip < 0) {
        this.nextInner(-skip, -this.dir);
        skip = this.value.length;
      }
      return this.nextInner(skip, this.dir);
    }
  };
  var PartialTextCursor = class {
    constructor(text, start, end) {
      this.value = "";
      this.done = false;
      this.cursor = new RawTextCursor(text, start > end ? -1 : 1);
      this.pos = start > end ? text.length : 0;
      this.from = Math.min(start, end);
      this.to = Math.max(start, end);
    }
    nextInner(skip, dir) {
      if (dir < 0 ? this.pos <= this.from : this.pos >= this.to) {
        this.value = "";
        this.done = true;
        return this;
      }
      skip += Math.max(0, dir < 0 ? this.pos - this.to : this.from - this.pos);
      let limit = dir < 0 ? this.pos - this.from : this.to - this.pos;
      if (skip > limit)
        skip = limit;
      limit -= skip;
      let { value } = this.cursor.next(skip);
      this.pos += (value.length + skip) * dir;
      this.value = value.length <= limit ? value : dir < 0 ? value.slice(value.length - limit) : value.slice(0, limit);
      this.done = !this.value;
      return this;
    }
    next(skip = 0) {
      if (skip < 0)
        skip = Math.max(skip, this.from - this.pos);
      else if (skip > 0)
        skip = Math.min(skip, this.to - this.pos);
      return this.nextInner(skip, this.cursor.dir);
    }
    get lineBreak() {
      return this.cursor.lineBreak && this.value != "";
    }
  };
  var LineCursor = class {
    constructor(inner) {
      this.inner = inner;
      this.afterBreak = true;
      this.value = "";
      this.done = false;
    }
    next(skip = 0) {
      let { done, lineBreak, value } = this.inner.next(skip);
      if (done && this.afterBreak) {
        this.value = "";
        this.afterBreak = false;
      } else if (done) {
        this.done = true;
        this.value = "";
      } else if (lineBreak) {
        if (this.afterBreak) {
          this.value = "";
        } else {
          this.afterBreak = true;
          this.next();
        }
      } else {
        this.value = value;
        this.afterBreak = false;
      }
      return this;
    }
    get lineBreak() {
      return false;
    }
  };
  if (typeof Symbol != "undefined") {
    Text.prototype[Symbol.iterator] = function() {
      return this.iter();
    };
    RawTextCursor.prototype[Symbol.iterator] = PartialTextCursor.prototype[Symbol.iterator] = LineCursor.prototype[Symbol.iterator] = function() {
      return this;
    };
  }
  var Line = class {
    /**
    @internal
    */
    constructor(from, to, number2, text) {
      this.from = from;
      this.to = to;
      this.number = number2;
      this.text = text;
    }
    /**
    The length of the line (not including any line break after it).
    */
    get length() {
      return this.to - this.from;
    }
  };
  function clip(text, from, to) {
    from = Math.max(0, Math.min(text.length, from));
    return [from, Math.max(from, Math.min(text.length, to))];
  }
  function findClusterBreak2(str, pos, forward = true, includeExtending = true) {
    return findClusterBreak(str, pos, forward, includeExtending);
  }
  function surrogateLow2(ch) {
    return ch >= 56320 && ch < 57344;
  }
  function surrogateHigh2(ch) {
    return ch >= 55296 && ch < 56320;
  }
  function codePointAt2(str, pos) {
    let code0 = str.charCodeAt(pos);
    if (!surrogateHigh2(code0) || pos + 1 == str.length)
      return code0;
    let code1 = str.charCodeAt(pos + 1);
    if (!surrogateLow2(code1))
      return code0;
    return (code0 - 55296 << 10) + (code1 - 56320) + 65536;
  }
  function fromCodePoint(code) {
    if (code <= 65535)
      return String.fromCharCode(code);
    code -= 65536;
    return String.fromCharCode((code >> 10) + 55296, (code & 1023) + 56320);
  }
  function codePointSize2(code) {
    return code < 65536 ? 1 : 2;
  }
  var DefaultSplit = /\r\n?|\n/;
  var MapMode = /* @__PURE__ */ (function(MapMode2) {
    MapMode2[MapMode2["Simple"] = 0] = "Simple";
    MapMode2[MapMode2["TrackDel"] = 1] = "TrackDel";
    MapMode2[MapMode2["TrackBefore"] = 2] = "TrackBefore";
    MapMode2[MapMode2["TrackAfter"] = 3] = "TrackAfter";
    return MapMode2;
  })(MapMode || (MapMode = {}));
  var ChangeDesc = class _ChangeDesc {
    // Sections are encoded as pairs of integers. The first is the
    // length in the current document, and the second is -1 for
    // unaffected sections, and the length of the replacement content
    // otherwise. So an insertion would be (0, n>0), a deletion (n>0,
    // 0), and a replacement two positive numbers.
    /**
    @internal
    */
    constructor(sections) {
      this.sections = sections;
    }
    /**
    The length of the document before the change.
    */
    get length() {
      let result = 0;
      for (let i = 0; i < this.sections.length; i += 2)
        result += this.sections[i];
      return result;
    }
    /**
    The length of the document after the change.
    */
    get newLength() {
      let result = 0;
      for (let i = 0; i < this.sections.length; i += 2) {
        let ins = this.sections[i + 1];
        result += ins < 0 ? this.sections[i] : ins;
      }
      return result;
    }
    /**
    False when there are actual changes in this set.
    */
    get empty() {
      return this.sections.length == 0 || this.sections.length == 2 && this.sections[1] < 0;
    }
    /**
    Iterate over the unchanged parts left by these changes. `posA`
    provides the position of the range in the old document, `posB`
    the new position in the changed document.
    */
    iterGaps(f) {
      for (let i = 0, posA = 0, posB = 0; i < this.sections.length; ) {
        let len = this.sections[i++], ins = this.sections[i++];
        if (ins < 0) {
          f(posA, posB, len);
          posB += len;
        } else {
          posB += ins;
        }
        posA += len;
      }
    }
    /**
    Iterate over the ranges changed by these changes. (See
    [`ChangeSet.iterChanges`](https://codemirror.net/6/docs/ref/#state.ChangeSet.iterChanges) for a
    variant that also provides you with the inserted text.)
    `fromA`/`toA` provides the extent of the change in the starting
    document, `fromB`/`toB` the extent of the replacement in the
    changed document.
    
    When `individual` is true, adjacent changes (which are kept
    separate for [position mapping](https://codemirror.net/6/docs/ref/#state.ChangeDesc.mapPos)) are
    reported separately.
    */
    iterChangedRanges(f, individual = false) {
      iterChanges(this, f, individual);
    }
    /**
    Get a description of the inverted form of these changes.
    */
    get invertedDesc() {
      let sections = [];
      for (let i = 0; i < this.sections.length; ) {
        let len = this.sections[i++], ins = this.sections[i++];
        if (ins < 0)
          sections.push(len, ins);
        else
          sections.push(ins, len);
      }
      return new _ChangeDesc(sections);
    }
    /**
    Compute the combined effect of applying another set of changes
    after this one. The length of the document after this set should
    match the length before `other`.
    */
    composeDesc(other) {
      return this.empty ? other : other.empty ? this : composeSets(this, other);
    }
    /**
    Map this description, which should start with the same document
    as `other`, over another set of changes, so that it can be
    applied after it. When `before` is true, map as if the changes
    in `this` happened before the ones in `other`.
    */
    mapDesc(other, before = false) {
      return other.empty ? this : mapSet(this, other, before);
    }
    mapPos(pos, assoc = -1, mode = MapMode.Simple) {
      let posA = 0, posB = 0;
      for (let i = 0; i < this.sections.length; ) {
        let len = this.sections[i++], ins = this.sections[i++], endA = posA + len;
        if (ins < 0) {
          if (endA > pos)
            return posB + (pos - posA);
          posB += len;
        } else {
          if (mode != MapMode.Simple && endA >= pos && (mode == MapMode.TrackDel && posA < pos && endA > pos || mode == MapMode.TrackBefore && posA < pos || mode == MapMode.TrackAfter && endA > pos))
            return null;
          if (endA > pos || endA == pos && assoc < 0 && !len)
            return pos == posA || assoc < 0 ? posB : posB + ins;
          posB += ins;
        }
        posA = endA;
      }
      if (pos > posA)
        throw new RangeError(`Position ${pos} is out of range for changeset of length ${posA}`);
      return posB;
    }
    /**
    Check whether these changes touch a given range. When one of the
    changes entirely covers the range, the string `"cover"` is
    returned.
    */
    touchesRange(from, to = from) {
      for (let i = 0, pos = 0; i < this.sections.length && pos <= to; ) {
        let len = this.sections[i++], ins = this.sections[i++], end = pos + len;
        if (ins >= 0 && pos <= to && end >= from)
          return pos < from && end > to ? "cover" : true;
        pos = end;
      }
      return false;
    }
    /**
    @internal
    */
    toString() {
      let result = "";
      for (let i = 0; i < this.sections.length; ) {
        let len = this.sections[i++], ins = this.sections[i++];
        result += (result ? " " : "") + len + (ins >= 0 ? ":" + ins : "");
      }
      return result;
    }
    /**
    Serialize this change desc to a JSON-representable value.
    */
    toJSON() {
      return this.sections;
    }
    /**
    Create a change desc from its JSON representation (as produced
    by [`toJSON`](https://codemirror.net/6/docs/ref/#state.ChangeDesc.toJSON).
    */
    static fromJSON(json) {
      if (!Array.isArray(json) || json.length % 2 || json.some((a) => typeof a != "number"))
        throw new RangeError("Invalid JSON representation of ChangeDesc");
      return new _ChangeDesc(json);
    }
    /**
    @internal
    */
    static create(sections) {
      return new _ChangeDesc(sections);
    }
  };
  var ChangeSet = class _ChangeSet extends ChangeDesc {
    constructor(sections, inserted) {
      super(sections);
      this.inserted = inserted;
    }
    /**
    Apply the changes to a document, returning the modified
    document.
    */
    apply(doc2) {
      if (this.length != doc2.length)
        throw new RangeError("Applying change set to a document with the wrong length");
      iterChanges(this, (fromA, toA, fromB, _toB, text) => doc2 = doc2.replace(fromB, fromB + (toA - fromA), text), false);
      return doc2;
    }
    mapDesc(other, before = false) {
      return mapSet(this, other, before, true);
    }
    /**
    Given the document as it existed _before_ the changes, return a
    change set that represents the inverse of this set, which could
    be used to go from the document created by the changes back to
    the document as it existed before the changes.
    */
    invert(doc2) {
      let sections = this.sections.slice(), inserted = [];
      for (let i = 0, pos = 0; i < sections.length; i += 2) {
        let len = sections[i], ins = sections[i + 1];
        if (ins >= 0) {
          sections[i] = ins;
          sections[i + 1] = len;
          let index = i >> 1;
          while (inserted.length < index)
            inserted.push(Text.empty);
          inserted.push(len ? doc2.slice(pos, pos + len) : Text.empty);
        }
        pos += len;
      }
      return new _ChangeSet(sections, inserted);
    }
    /**
    Combine two subsequent change sets into a single set. `other`
    must start in the document produced by `this`. If `this` goes
    `docA` → `docB` and `other` represents `docB` → `docC`, the
    returned value will represent the change `docA` → `docC`.
    */
    compose(other) {
      return this.empty ? other : other.empty ? this : composeSets(this, other, true);
    }
    /**
    Given another change set starting in the same document, maps this
    change set over the other, producing a new change set that can be
    applied to the document produced by applying `other`. When
    `before` is `true`, order changes as if `this` comes before
    `other`, otherwise (the default) treat `other` as coming first.
    
    Given two changes `A` and `B`, `A.compose(B.map(A))` and
    `B.compose(A.map(B, true))` will produce the same document. This
    provides a basic form of [operational
    transformation](https://en.wikipedia.org/wiki/Operational_transformation),
    and can be used for collaborative editing.
    */
    map(other, before = false) {
      return other.empty ? this : mapSet(this, other, before, true);
    }
    /**
    Iterate over the changed ranges in the document, calling `f` for
    each, with the range in the original document (`fromA`-`toA`)
    and the range that replaces it in the new document
    (`fromB`-`toB`).
    
    When `individual` is true, adjacent changes are reported
    separately.
    */
    iterChanges(f, individual = false) {
      iterChanges(this, f, individual);
    }
    /**
    Get a [change description](https://codemirror.net/6/docs/ref/#state.ChangeDesc) for this change
    set.
    */
    get desc() {
      return ChangeDesc.create(this.sections);
    }
    /**
    @internal
    */
    filter(ranges) {
      let resultSections = [], resultInserted = [], filteredSections = [];
      let iter = new SectionIter(this);
      done: for (let i = 0, pos = 0; ; ) {
        let next = i == ranges.length ? 1e9 : ranges[i++];
        while (pos < next || pos == next && iter.len == 0) {
          if (iter.done)
            break done;
          let len = Math.min(iter.len, next - pos);
          addSection(filteredSections, len, -1);
          let ins = iter.ins == -1 ? -1 : iter.off == 0 ? iter.ins : 0;
          addSection(resultSections, len, ins);
          if (ins > 0)
            addInsert(resultInserted, resultSections, iter.text);
          iter.forward(len);
          pos += len;
        }
        let end = ranges[i++];
        while (pos < end) {
          if (iter.done)
            break done;
          let len = Math.min(iter.len, end - pos);
          addSection(resultSections, len, -1);
          addSection(filteredSections, len, iter.ins == -1 ? -1 : iter.off == 0 ? iter.ins : 0);
          iter.forward(len);
          pos += len;
        }
      }
      return {
        changes: new _ChangeSet(resultSections, resultInserted),
        filtered: ChangeDesc.create(filteredSections)
      };
    }
    /**
    Serialize this change set to a JSON-representable value.
    */
    toJSON() {
      let parts = [];
      for (let i = 0; i < this.sections.length; i += 2) {
        let len = this.sections[i], ins = this.sections[i + 1];
        if (ins < 0)
          parts.push(len);
        else if (ins == 0)
          parts.push([len]);
        else
          parts.push([len].concat(this.inserted[i >> 1].toJSON()));
      }
      return parts;
    }
    /**
    Create a change set for the given changes, for a document of the
    given length, using `lineSep` as line separator.
    */
    static of(changes, length, lineSep) {
      let sections = [], inserted = [], pos = 0;
      let total = null;
      function flush(force = false) {
        if (!force && !sections.length)
          return;
        if (pos < length)
          addSection(sections, length - pos, -1);
        let set = new _ChangeSet(sections, inserted);
        total = total ? total.compose(set.map(total)) : set;
        sections = [];
        inserted = [];
        pos = 0;
      }
      function process(spec) {
        if (Array.isArray(spec)) {
          for (let sub of spec)
            process(sub);
        } else if (spec instanceof _ChangeSet) {
          if (spec.length != length)
            throw new RangeError(`Mismatched change set length (got ${spec.length}, expected ${length})`);
          flush();
          total = total ? total.compose(spec.map(total)) : spec;
        } else {
          let { from, to = from, insert: insert2 } = spec;
          if (from > to || from < 0 || to > length)
            throw new RangeError(`Invalid change range ${from} to ${to} (in doc of length ${length})`);
          let insText = !insert2 ? Text.empty : typeof insert2 == "string" ? Text.of(insert2.split(lineSep || DefaultSplit)) : insert2;
          let insLen = insText.length;
          if (from == to && insLen == 0)
            return;
          if (from < pos)
            flush();
          if (from > pos)
            addSection(sections, from - pos, -1);
          addSection(sections, to - from, insLen);
          addInsert(inserted, sections, insText);
          pos = to;
        }
      }
      process(changes);
      flush(!total);
      return total;
    }
    /**
    Create an empty changeset of the given length.
    */
    static empty(length) {
      return new _ChangeSet(length ? [length, -1] : [], []);
    }
    /**
    Create a changeset from its JSON representation (as produced by
    [`toJSON`](https://codemirror.net/6/docs/ref/#state.ChangeSet.toJSON).
    */
    static fromJSON(json) {
      if (!Array.isArray(json))
        throw new RangeError("Invalid JSON representation of ChangeSet");
      let sections = [], inserted = [];
      for (let i = 0; i < json.length; i++) {
        let part = json[i];
        if (typeof part == "number") {
          sections.push(part, -1);
        } else if (!Array.isArray(part) || typeof part[0] != "number" || part.some((e, i2) => i2 && typeof e != "string")) {
          throw new RangeError("Invalid JSON representation of ChangeSet");
        } else if (part.length == 1) {
          sections.push(part[0], 0);
        } else {
          while (inserted.length < i)
            inserted.push(Text.empty);
          inserted[i] = Text.of(part.slice(1));
          sections.push(part[0], inserted[i].length);
        }
      }
      return new _ChangeSet(sections, inserted);
    }
    /**
    @internal
    */
    static createSet(sections, inserted) {
      return new _ChangeSet(sections, inserted);
    }
  };
  function addSection(sections, len, ins, forceJoin = false) {
    if (len == 0 && ins <= 0)
      return;
    let last = sections.length - 2;
    if (last >= 0 && ins <= 0 && ins == sections[last + 1])
      sections[last] += len;
    else if (last >= 0 && len == 0 && sections[last] == 0)
      sections[last + 1] += ins;
    else if (forceJoin) {
      sections[last] += len;
      sections[last + 1] += ins;
    } else
      sections.push(len, ins);
  }
  function addInsert(values, sections, value) {
    if (value.length == 0)
      return;
    let index = sections.length - 2 >> 1;
    if (index < values.length) {
      values[values.length - 1] = values[values.length - 1].append(value);
    } else {
      while (values.length < index)
        values.push(Text.empty);
      values.push(value);
    }
  }
  function iterChanges(desc, f, individual) {
    let inserted = desc.inserted;
    for (let posA = 0, posB = 0, i = 0; i < desc.sections.length; ) {
      let len = desc.sections[i++], ins = desc.sections[i++];
      if (ins < 0) {
        posA += len;
        posB += len;
      } else {
        let endA = posA, endB = posB, text = Text.empty;
        for (; ; ) {
          endA += len;
          endB += ins;
          if (ins && inserted)
            text = text.append(inserted[i - 2 >> 1]);
          if (individual || i == desc.sections.length || desc.sections[i + 1] < 0)
            break;
          len = desc.sections[i++];
          ins = desc.sections[i++];
        }
        f(posA, endA, posB, endB, text);
        posA = endA;
        posB = endB;
      }
    }
  }
  function mapSet(setA, setB, before, mkSet = false) {
    let sections = [], insert2 = mkSet ? [] : null;
    let a = new SectionIter(setA), b = new SectionIter(setB);
    for (let inserted = -1; ; ) {
      if (a.done && b.len || b.done && a.len) {
        throw new Error("Mismatched change set lengths");
      } else if (a.ins == -1 && b.ins == -1) {
        let len = Math.min(a.len, b.len);
        addSection(sections, len, -1);
        a.forward(len);
        b.forward(len);
      } else if (b.ins >= 0 && (a.ins < 0 || inserted == a.i || a.off == 0 && (b.len < a.len || b.len == a.len && !before))) {
        let len = b.len;
        addSection(sections, b.ins, -1);
        while (len) {
          let piece = Math.min(a.len, len);
          if (a.ins >= 0 && inserted < a.i && a.len <= piece) {
            addSection(sections, 0, a.ins);
            if (insert2)
              addInsert(insert2, sections, a.text);
            inserted = a.i;
          }
          a.forward(piece);
          len -= piece;
        }
        b.next();
      } else if (a.ins >= 0) {
        let len = 0, left = a.len;
        while (left) {
          if (b.ins == -1) {
            let piece = Math.min(left, b.len);
            len += piece;
            left -= piece;
            b.forward(piece);
          } else if (b.ins == 0 && b.len < left) {
            left -= b.len;
            b.next();
          } else {
            break;
          }
        }
        addSection(sections, len, inserted < a.i ? a.ins : 0);
        if (insert2 && inserted < a.i)
          addInsert(insert2, sections, a.text);
        inserted = a.i;
        a.forward(a.len - left);
      } else if (a.done && b.done) {
        return insert2 ? ChangeSet.createSet(sections, insert2) : ChangeDesc.create(sections);
      } else {
        throw new Error("Mismatched change set lengths");
      }
    }
  }
  function composeSets(setA, setB, mkSet = false) {
    let sections = [];
    let insert2 = mkSet ? [] : null;
    let a = new SectionIter(setA), b = new SectionIter(setB);
    for (let open = false; ; ) {
      if (a.done && b.done) {
        return insert2 ? ChangeSet.createSet(sections, insert2) : ChangeDesc.create(sections);
      } else if (a.ins == 0) {
        addSection(sections, a.len, 0, open);
        a.next();
      } else if (b.len == 0 && !b.done) {
        addSection(sections, 0, b.ins, open);
        if (insert2)
          addInsert(insert2, sections, b.text);
        b.next();
      } else if (a.done || b.done) {
        throw new Error("Mismatched change set lengths");
      } else {
        let len = Math.min(a.len2, b.len), sectionLen = sections.length;
        if (a.ins == -1) {
          let insB = b.ins == -1 ? -1 : b.off ? 0 : b.ins;
          addSection(sections, len, insB, open);
          if (insert2 && insB)
            addInsert(insert2, sections, b.text);
        } else if (b.ins == -1) {
          addSection(sections, a.off ? 0 : a.len, len, open);
          if (insert2)
            addInsert(insert2, sections, a.textBit(len));
        } else {
          addSection(sections, a.off ? 0 : a.len, b.off ? 0 : b.ins, open);
          if (insert2 && !b.off)
            addInsert(insert2, sections, b.text);
        }
        open = (a.ins > len || b.ins >= 0 && b.len > len) && (open || sections.length > sectionLen);
        a.forward2(len);
        b.forward(len);
      }
    }
  }
  var SectionIter = class {
    constructor(set) {
      this.set = set;
      this.i = 0;
      this.next();
    }
    next() {
      let { sections } = this.set;
      if (this.i < sections.length) {
        this.len = sections[this.i++];
        this.ins = sections[this.i++];
      } else {
        this.len = 0;
        this.ins = -2;
      }
      this.off = 0;
    }
    get done() {
      return this.ins == -2;
    }
    get len2() {
      return this.ins < 0 ? this.len : this.ins;
    }
    get text() {
      let { inserted } = this.set, index = this.i - 2 >> 1;
      return index >= inserted.length ? Text.empty : inserted[index];
    }
    textBit(len) {
      let { inserted } = this.set, index = this.i - 2 >> 1;
      return index >= inserted.length && !len ? Text.empty : inserted[index].slice(this.off, len == null ? void 0 : this.off + len);
    }
    forward(len) {
      if (len == this.len)
        this.next();
      else {
        this.len -= len;
        this.off += len;
      }
    }
    forward2(len) {
      if (this.ins == -1)
        this.forward(len);
      else if (len == this.ins)
        this.next();
      else {
        this.ins -= len;
        this.off += len;
      }
    }
  };
  var SelectionRange = class _SelectionRange {
    constructor(from, to, flags) {
      this.from = from;
      this.to = to;
      this.flags = flags;
    }
    /**
    The anchor of the range—the side that doesn't move when you
    extend it.
    */
    get anchor() {
      return this.flags & 32 ? this.to : this.from;
    }
    /**
    The head of the range, which is moved when the range is
    [extended](https://codemirror.net/6/docs/ref/#state.SelectionRange.extend).
    */
    get head() {
      return this.flags & 32 ? this.from : this.to;
    }
    /**
    True when `anchor` and `head` are at the same position.
    */
    get empty() {
      return this.from == this.to;
    }
    /**
    If this is a cursor that is explicitly associated with the
    character on one of its sides, this returns the side. -1 means
    the character before its position, 1 the character after, and 0
    means no association.
    */
    get assoc() {
      return this.flags & 8 ? -1 : this.flags & 16 ? 1 : 0;
    }
    /**
    The bidirectional text level associated with this cursor, if
    any.
    */
    get bidiLevel() {
      let level = this.flags & 7;
      return level == 7 ? null : level;
    }
    /**
    The goal column (stored vertical offset) associated with a
    cursor. This is used to preserve the vertical position when
    [moving](https://codemirror.net/6/docs/ref/#view.EditorView.moveVertically) across
    lines of different length.
    */
    get goalColumn() {
      let value = this.flags >> 6;
      return value == 16777215 ? void 0 : value;
    }
    /**
    Map this range through a change, producing a valid range in the
    updated document.
    */
    map(change, assoc = -1) {
      let from, to;
      if (this.empty) {
        from = to = change.mapPos(this.from, assoc);
      } else {
        from = change.mapPos(this.from, 1);
        to = change.mapPos(this.to, -1);
      }
      return from == this.from && to == this.to ? this : new _SelectionRange(from, to, this.flags);
    }
    /**
    Extend this range to cover at least `from` to `to`.
    */
    extend(from, to = from) {
      if (from <= this.anchor && to >= this.anchor)
        return EditorSelection.range(from, to);
      let head = Math.abs(from - this.anchor) > Math.abs(to - this.anchor) ? from : to;
      return EditorSelection.range(this.anchor, head);
    }
    /**
    Compare this range to another range.
    */
    eq(other, includeAssoc = false) {
      return this.anchor == other.anchor && this.head == other.head && this.goalColumn == other.goalColumn && (!includeAssoc || !this.empty || this.assoc == other.assoc);
    }
    /**
    Return a JSON-serializable object representing the range.
    */
    toJSON() {
      return { anchor: this.anchor, head: this.head };
    }
    /**
    Convert a JSON representation of a range to a `SelectionRange`
    instance.
    */
    static fromJSON(json) {
      if (!json || typeof json.anchor != "number" || typeof json.head != "number")
        throw new RangeError("Invalid JSON representation for SelectionRange");
      return EditorSelection.range(json.anchor, json.head);
    }
    /**
    @internal
    */
    static create(from, to, flags) {
      return new _SelectionRange(from, to, flags);
    }
  };
  var EditorSelection = class _EditorSelection {
    constructor(ranges, mainIndex) {
      this.ranges = ranges;
      this.mainIndex = mainIndex;
    }
    /**
    Map a selection through a change. Used to adjust the selection
    position for changes.
    */
    map(change, assoc = -1) {
      if (change.empty)
        return this;
      return _EditorSelection.create(this.ranges.map((r) => r.map(change, assoc)), this.mainIndex);
    }
    /**
    Compare this selection to another selection. By default, ranges
    are compared only by position. When `includeAssoc` is true,
    cursor ranges must also have the same
    [`assoc`](https://codemirror.net/6/docs/ref/#state.SelectionRange.assoc) value.
    */
    eq(other, includeAssoc = false) {
      if (this.ranges.length != other.ranges.length || this.mainIndex != other.mainIndex)
        return false;
      for (let i = 0; i < this.ranges.length; i++)
        if (!this.ranges[i].eq(other.ranges[i], includeAssoc))
          return false;
      return true;
    }
    /**
    Get the primary selection range. Usually, you should make sure
    your code applies to _all_ ranges, by using methods like
    [`changeByRange`](https://codemirror.net/6/docs/ref/#state.EditorState.changeByRange).
    */
    get main() {
      return this.ranges[this.mainIndex];
    }
    /**
    Make sure the selection only has one range. Returns a selection
    holding only the main range from this selection.
    */
    asSingle() {
      return this.ranges.length == 1 ? this : new _EditorSelection([this.main], 0);
    }
    /**
    Extend this selection with an extra range.
    */
    addRange(range, main = true) {
      return _EditorSelection.create([range].concat(this.ranges), main ? 0 : this.mainIndex + 1);
    }
    /**
    Replace a given range with another range, and then normalize the
    selection to merge and sort ranges if necessary.
    */
    replaceRange(range, which = this.mainIndex) {
      let ranges = this.ranges.slice();
      ranges[which] = range;
      return _EditorSelection.create(ranges, this.mainIndex);
    }
    /**
    Convert this selection to an object that can be serialized to
    JSON.
    */
    toJSON() {
      return { ranges: this.ranges.map((r) => r.toJSON()), main: this.mainIndex };
    }
    /**
    Create a selection from a JSON representation.
    */
    static fromJSON(json) {
      if (!json || !Array.isArray(json.ranges) || typeof json.main != "number" || json.main >= json.ranges.length)
        throw new RangeError("Invalid JSON representation for EditorSelection");
      return new _EditorSelection(json.ranges.map((r) => SelectionRange.fromJSON(r)), json.main);
    }
    /**
    Create a selection holding a single range.
    */
    static single(anchor, head = anchor) {
      return new _EditorSelection([_EditorSelection.range(anchor, head)], 0);
    }
    /**
    Sort and merge the given set of ranges, creating a valid
    selection.
    */
    static create(ranges, mainIndex = 0) {
      if (ranges.length == 0)
        throw new RangeError("A selection needs at least one range");
      for (let pos = 0, i = 0; i < ranges.length; i++) {
        let range = ranges[i];
        if (range.empty ? range.from <= pos : range.from < pos)
          return _EditorSelection.normalized(ranges.slice(), mainIndex);
        pos = range.to;
      }
      return new _EditorSelection(ranges, mainIndex);
    }
    /**
    Create a cursor selection range at the given position. You can
    safely ignore the optional arguments in most situations.
    */
    static cursor(pos, assoc = 0, bidiLevel, goalColumn) {
      return SelectionRange.create(pos, pos, (assoc == 0 ? 0 : assoc < 0 ? 8 : 16) | (bidiLevel == null ? 7 : Math.min(6, bidiLevel)) | (goalColumn !== null && goalColumn !== void 0 ? goalColumn : 16777215) << 6);
    }
    /**
    Create a selection range.
    */
    static range(anchor, head, goalColumn, bidiLevel) {
      let flags = (goalColumn !== null && goalColumn !== void 0 ? goalColumn : 16777215) << 6 | (bidiLevel == null ? 7 : Math.min(6, bidiLevel));
      return head < anchor ? SelectionRange.create(head, anchor, 32 | 16 | flags) : SelectionRange.create(anchor, head, (head > anchor ? 8 : 0) | flags);
    }
    /**
    @internal
    */
    static normalized(ranges, mainIndex = 0) {
      let main = ranges[mainIndex];
      ranges.sort((a, b) => a.from - b.from);
      mainIndex = ranges.indexOf(main);
      for (let i = 1; i < ranges.length; i++) {
        let range = ranges[i], prev = ranges[i - 1];
        if (range.empty ? range.from <= prev.to : range.from < prev.to) {
          let from = prev.from, to = Math.max(range.to, prev.to);
          if (i <= mainIndex)
            mainIndex--;
          ranges.splice(--i, 2, range.anchor > range.head ? _EditorSelection.range(to, from) : _EditorSelection.range(from, to));
        }
      }
      return new _EditorSelection(ranges, mainIndex);
    }
  };
  function checkSelection(selection, docLength) {
    for (let range of selection.ranges)
      if (range.to > docLength)
        throw new RangeError("Selection points outside of document");
  }
  var nextID = 0;
  var Facet = class _Facet {
    constructor(combine, compareInput, compare2, isStatic, enables) {
      this.combine = combine;
      this.compareInput = compareInput;
      this.compare = compare2;
      this.isStatic = isStatic;
      this.id = nextID++;
      this.default = combine([]);
      this.extensions = typeof enables == "function" ? enables(this) : enables;
    }
    /**
    Returns a facet reader for this facet, which can be used to
    [read](https://codemirror.net/6/docs/ref/#state.EditorState.facet) it but not to define values for it.
    */
    get reader() {
      return this;
    }
    /**
    Define a new facet.
    */
    static define(config = {}) {
      return new _Facet(config.combine || ((a) => a), config.compareInput || ((a, b) => a === b), config.compare || (!config.combine ? sameArray : (a, b) => a === b), !!config.static, config.enables);
    }
    /**
    Returns an extension that adds the given value to this facet.
    */
    of(value) {
      return new FacetProvider([], this, 0, value);
    }
    /**
    Create an extension that computes a value for the facet from a
    state. You must take care to declare the parts of the state that
    this value depends on, since your function is only called again
    for a new state when one of those parts changed.
    
    In cases where your value depends only on a single field, you'll
    want to use the [`from`](https://codemirror.net/6/docs/ref/#state.Facet.from) method instead.
    */
    compute(deps, get) {
      if (this.isStatic)
        throw new Error("Can't compute a static facet");
      return new FacetProvider(deps, this, 1, get);
    }
    /**
    Create an extension that computes zero or more values for this
    facet from a state.
    */
    computeN(deps, get) {
      if (this.isStatic)
        throw new Error("Can't compute a static facet");
      return new FacetProvider(deps, this, 2, get);
    }
    from(field, get) {
      if (!get)
        get = (x) => x;
      return this.compute([field], (state) => get(state.field(field)));
    }
  };
  function sameArray(a, b) {
    return a == b || a.length == b.length && a.every((e, i) => e === b[i]);
  }
  var FacetProvider = class {
    constructor(dependencies, facet, type, value) {
      this.dependencies = dependencies;
      this.facet = facet;
      this.type = type;
      this.value = value;
      this.id = nextID++;
    }
    dynamicSlot(addresses) {
      var _a2;
      let getter = this.value;
      let compare2 = this.facet.compareInput;
      let id = this.id, idx = addresses[id] >> 1, multi = this.type == 2;
      let depDoc = false, depSel = false, depAddrs = [];
      for (let dep of this.dependencies) {
        if (dep == "doc")
          depDoc = true;
        else if (dep == "selection")
          depSel = true;
        else if ((((_a2 = addresses[dep.id]) !== null && _a2 !== void 0 ? _a2 : 1) & 1) == 0)
          depAddrs.push(addresses[dep.id]);
      }
      return {
        create(state) {
          state.values[idx] = getter(state);
          return 1;
        },
        update(state, tr) {
          if (depDoc && tr.docChanged || depSel && (tr.docChanged || tr.selection) || ensureAll(state, depAddrs)) {
            let newVal = getter(state);
            if (multi ? !compareArray(newVal, state.values[idx], compare2) : !compare2(newVal, state.values[idx])) {
              state.values[idx] = newVal;
              return 1;
            }
          }
          return 0;
        },
        reconfigure: (state, oldState) => {
          let newVal, oldAddr = oldState.config.address[id];
          if (oldAddr != null) {
            let oldVal = getAddr(oldState, oldAddr);
            if (this.dependencies.every((dep) => {
              return dep instanceof Facet ? oldState.facet(dep) === state.facet(dep) : dep instanceof StateField ? oldState.field(dep, false) == state.field(dep, false) : true;
            }) || (multi ? compareArray(newVal = getter(state), oldVal, compare2) : compare2(newVal = getter(state), oldVal))) {
              state.values[idx] = oldVal;
              return 0;
            }
          } else {
            newVal = getter(state);
          }
          state.values[idx] = newVal;
          return 1;
        }
      };
    }
  };
  function compareArray(a, b, compare2) {
    if (a.length != b.length)
      return false;
    for (let i = 0; i < a.length; i++)
      if (!compare2(a[i], b[i]))
        return false;
    return true;
  }
  function ensureAll(state, addrs) {
    let changed = false;
    for (let addr of addrs)
      if (ensureAddr(state, addr) & 1)
        changed = true;
    return changed;
  }
  function dynamicFacetSlot(addresses, facet, providers) {
    let providerAddrs = providers.map((p) => addresses[p.id]);
    let providerTypes = providers.map((p) => p.type);
    let dynamic = providerAddrs.filter((p) => !(p & 1));
    let idx = addresses[facet.id] >> 1;
    function get(state) {
      let values = [];
      for (let i = 0; i < providerAddrs.length; i++) {
        let value = getAddr(state, providerAddrs[i]);
        if (providerTypes[i] == 2)
          for (let val of value)
            values.push(val);
        else
          values.push(value);
      }
      return facet.combine(values);
    }
    return {
      create(state) {
        for (let addr of providerAddrs)
          ensureAddr(state, addr);
        state.values[idx] = get(state);
        return 1;
      },
      update(state, tr) {
        if (!ensureAll(state, dynamic))
          return 0;
        let value = get(state);
        if (facet.compare(value, state.values[idx]))
          return 0;
        state.values[idx] = value;
        return 1;
      },
      reconfigure(state, oldState) {
        let depChanged = ensureAll(state, providerAddrs);
        let oldProviders = oldState.config.facets[facet.id], oldValue = oldState.facet(facet);
        if (oldProviders && !depChanged && sameArray(providers, oldProviders)) {
          state.values[idx] = oldValue;
          return 0;
        }
        let value = get(state);
        if (facet.compare(value, oldValue)) {
          state.values[idx] = oldValue;
          return 0;
        }
        state.values[idx] = value;
        return 1;
      }
    };
  }
  var initField = /* @__PURE__ */ Facet.define({ static: true });
  var StateField = class _StateField {
    constructor(id, createF, updateF, compareF, spec) {
      this.id = id;
      this.createF = createF;
      this.updateF = updateF;
      this.compareF = compareF;
      this.spec = spec;
      this.provides = void 0;
    }
    /**
    Define a state field.
    */
    static define(config) {
      let field = new _StateField(nextID++, config.create, config.update, config.compare || ((a, b) => a === b), config);
      if (config.provide)
        field.provides = config.provide(field);
      return field;
    }
    create(state) {
      let init = state.facet(initField).find((i) => i.field == this);
      return ((init === null || init === void 0 ? void 0 : init.create) || this.createF)(state);
    }
    /**
    @internal
    */
    slot(addresses) {
      let idx = addresses[this.id] >> 1;
      return {
        create: (state) => {
          state.values[idx] = this.create(state);
          return 1;
        },
        update: (state, tr) => {
          let oldVal = state.values[idx];
          let value = this.updateF(oldVal, tr);
          if (this.compareF(oldVal, value))
            return 0;
          state.values[idx] = value;
          return 1;
        },
        reconfigure: (state, oldState) => {
          let init = state.facet(initField), oldInit = oldState.facet(initField), reInit;
          if ((reInit = init.find((i) => i.field == this)) && reInit != oldInit.find((i) => i.field == this)) {
            state.values[idx] = reInit.create(state);
            return 1;
          }
          if (oldState.config.address[this.id] != null) {
            state.values[idx] = oldState.field(this);
            return 0;
          }
          state.values[idx] = this.create(state);
          return 1;
        }
      };
    }
    /**
    Returns an extension that enables this field and overrides the
    way it is initialized. Can be useful when you need to provide a
    non-default starting value for the field.
    */
    init(create) {
      return [this, initField.of({ field: this, create })];
    }
    /**
    State field instances can be used as
    [`Extension`](https://codemirror.net/6/docs/ref/#state.Extension) values to enable the field in a
    given state.
    */
    get extension() {
      return this;
    }
  };
  var Prec_ = { lowest: 4, low: 3, default: 2, high: 1, highest: 0 };
  function prec(value) {
    return (ext) => new PrecExtension(ext, value);
  }
  var Prec = {
    /**
    The highest precedence level, for extensions that should end up
    near the start of the precedence ordering.
    */
    highest: /* @__PURE__ */ prec(Prec_.highest),
    /**
    A higher-than-default precedence, for extensions that should
    come before those with default precedence.
    */
    high: /* @__PURE__ */ prec(Prec_.high),
    /**
    The default precedence, which is also used for extensions
    without an explicit precedence.
    */
    default: /* @__PURE__ */ prec(Prec_.default),
    /**
    A lower-than-default precedence.
    */
    low: /* @__PURE__ */ prec(Prec_.low),
    /**
    The lowest precedence level. Meant for things that should end up
    near the end of the extension order.
    */
    lowest: /* @__PURE__ */ prec(Prec_.lowest)
  };
  var PrecExtension = class {
    constructor(inner, prec2) {
      this.inner = inner;
      this.prec = prec2;
    }
  };
  var Compartment = class _Compartment {
    /**
    Create an instance of this compartment to add to your [state
    configuration](https://codemirror.net/6/docs/ref/#state.EditorStateConfig.extensions).
    */
    of(ext) {
      return new CompartmentInstance(this, ext);
    }
    /**
    Create an [effect](https://codemirror.net/6/docs/ref/#state.TransactionSpec.effects) that
    reconfigures this compartment.
    */
    reconfigure(content2) {
      return _Compartment.reconfigure.of({ compartment: this, extension: content2 });
    }
    /**
    Get the current content of the compartment in the state, or
    `undefined` if it isn't present.
    */
    get(state) {
      return state.config.compartments.get(this);
    }
  };
  var CompartmentInstance = class {
    constructor(compartment, inner) {
      this.compartment = compartment;
      this.inner = inner;
    }
  };
  var Configuration = class _Configuration {
    constructor(base2, compartments, dynamicSlots, address, staticValues, facets) {
      this.base = base2;
      this.compartments = compartments;
      this.dynamicSlots = dynamicSlots;
      this.address = address;
      this.staticValues = staticValues;
      this.facets = facets;
      this.statusTemplate = [];
      while (this.statusTemplate.length < dynamicSlots.length)
        this.statusTemplate.push(
          0
          /* SlotStatus.Unresolved */
        );
    }
    staticFacet(facet) {
      let addr = this.address[facet.id];
      return addr == null ? facet.default : this.staticValues[addr >> 1];
    }
    static resolve(base2, compartments, oldState) {
      let fields = [];
      let facets = /* @__PURE__ */ Object.create(null);
      let newCompartments = /* @__PURE__ */ new Map();
      for (let ext of flatten(base2, compartments, newCompartments)) {
        if (ext instanceof StateField)
          fields.push(ext);
        else
          (facets[ext.facet.id] || (facets[ext.facet.id] = [])).push(ext);
      }
      let address = /* @__PURE__ */ Object.create(null);
      let staticValues = [];
      let dynamicSlots = [];
      for (let field of fields) {
        address[field.id] = dynamicSlots.length << 1;
        dynamicSlots.push((a) => field.slot(a));
      }
      let oldFacets = oldState === null || oldState === void 0 ? void 0 : oldState.config.facets;
      for (let id in facets) {
        let providers = facets[id], facet = providers[0].facet;
        let oldProviders = oldFacets && oldFacets[id] || [];
        if (providers.every(
          (p) => p.type == 0
          /* Provider.Static */
        )) {
          address[facet.id] = staticValues.length << 1 | 1;
          if (sameArray(oldProviders, providers)) {
            staticValues.push(oldState.facet(facet));
          } else {
            let value = facet.combine(providers.map((p) => p.value));
            staticValues.push(oldState && facet.compare(value, oldState.facet(facet)) ? oldState.facet(facet) : value);
          }
        } else {
          for (let p of providers) {
            if (p.type == 0) {
              address[p.id] = staticValues.length << 1 | 1;
              staticValues.push(p.value);
            } else {
              address[p.id] = dynamicSlots.length << 1;
              dynamicSlots.push((a) => p.dynamicSlot(a));
            }
          }
          address[facet.id] = dynamicSlots.length << 1;
          dynamicSlots.push((a) => dynamicFacetSlot(a, facet, providers));
        }
      }
      let dynamic = dynamicSlots.map((f) => f(address));
      return new _Configuration(base2, newCompartments, dynamic, address, staticValues, facets);
    }
  };
  function flatten(extension, compartments, newCompartments) {
    let result = [[], [], [], [], []];
    let seen = /* @__PURE__ */ new Map();
    function inner(ext, prec2) {
      let known = seen.get(ext);
      if (known != null) {
        if (known <= prec2)
          return;
        let found = result[known].indexOf(ext);
        if (found > -1)
          result[known].splice(found, 1);
        if (ext instanceof CompartmentInstance)
          newCompartments.delete(ext.compartment);
      }
      seen.set(ext, prec2);
      if (Array.isArray(ext)) {
        for (let e of ext)
          inner(e, prec2);
      } else if (ext instanceof CompartmentInstance) {
        if (newCompartments.has(ext.compartment))
          throw new RangeError(`Duplicate use of compartment in extensions`);
        let content2 = compartments.get(ext.compartment) || ext.inner;
        newCompartments.set(ext.compartment, content2);
        inner(content2, prec2);
      } else if (ext instanceof PrecExtension) {
        inner(ext.inner, ext.prec);
      } else if (ext instanceof StateField) {
        result[prec2].push(ext);
        if (ext.provides)
          inner(ext.provides, prec2);
      } else if (ext instanceof FacetProvider) {
        result[prec2].push(ext);
        if (ext.facet.extensions)
          inner(ext.facet.extensions, Prec_.default);
      } else {
        let content2 = ext.extension;
        if (!content2)
          throw new Error(`Unrecognized extension value in extension set (${ext}). This sometimes happens because multiple instances of @codemirror/state are loaded, breaking instanceof checks.`);
        inner(content2, prec2);
      }
    }
    inner(extension, Prec_.default);
    return result.reduce((a, b) => a.concat(b));
  }
  function ensureAddr(state, addr) {
    if (addr & 1)
      return 2;
    let idx = addr >> 1;
    let status = state.status[idx];
    if (status == 4)
      throw new Error("Cyclic dependency between fields and/or facets");
    if (status & 2)
      return status;
    state.status[idx] = 4;
    let changed = state.computeSlot(state, state.config.dynamicSlots[idx]);
    return state.status[idx] = 2 | changed;
  }
  function getAddr(state, addr) {
    return addr & 1 ? state.config.staticValues[addr >> 1] : state.values[addr >> 1];
  }
  var languageData = /* @__PURE__ */ Facet.define();
  var allowMultipleSelections = /* @__PURE__ */ Facet.define({
    combine: (values) => values.some((v) => v),
    static: true
  });
  var lineSeparator = /* @__PURE__ */ Facet.define({
    combine: (values) => values.length ? values[0] : void 0,
    static: true
  });
  var changeFilter = /* @__PURE__ */ Facet.define();
  var transactionFilter = /* @__PURE__ */ Facet.define();
  var transactionExtender = /* @__PURE__ */ Facet.define();
  var readOnly = /* @__PURE__ */ Facet.define({
    combine: (values) => values.length ? values[0] : false
  });
  var Annotation = class {
    /**
    @internal
    */
    constructor(type, value) {
      this.type = type;
      this.value = value;
    }
    /**
    Define a new type of annotation.
    */
    static define() {
      return new AnnotationType();
    }
  };
  var AnnotationType = class {
    /**
    Create an instance of this annotation.
    */
    of(value) {
      return new Annotation(this, value);
    }
  };
  var StateEffectType = class {
    /**
    @internal
    */
    constructor(map) {
      this.map = map;
    }
    /**
    Create a [state effect](https://codemirror.net/6/docs/ref/#state.StateEffect) instance of this
    type.
    */
    of(value) {
      return new StateEffect(this, value);
    }
  };
  var StateEffect = class _StateEffect {
    /**
    @internal
    */
    constructor(type, value) {
      this.type = type;
      this.value = value;
    }
    /**
    Map this effect through a position mapping. Will return
    `undefined` when that ends up deleting the effect.
    */
    map(mapping) {
      let mapped = this.type.map(this.value, mapping);
      return mapped === void 0 ? void 0 : mapped == this.value ? this : new _StateEffect(this.type, mapped);
    }
    /**
    Tells you whether this effect object is of a given
    [type](https://codemirror.net/6/docs/ref/#state.StateEffectType).
    */
    is(type) {
      return this.type == type;
    }
    /**
    Define a new effect type. The type parameter indicates the type
    of values that his effect holds. It should be a type that
    doesn't include `undefined`, since that is used in
    [mapping](https://codemirror.net/6/docs/ref/#state.StateEffect.map) to indicate that an effect is
    removed.
    */
    static define(spec = {}) {
      return new StateEffectType(spec.map || ((v) => v));
    }
    /**
    Map an array of effects through a change set.
    */
    static mapEffects(effects, mapping) {
      if (!effects.length)
        return effects;
      let result = [];
      for (let effect of effects) {
        let mapped = effect.map(mapping);
        if (mapped)
          result.push(mapped);
      }
      return result;
    }
  };
  StateEffect.reconfigure = /* @__PURE__ */ StateEffect.define();
  StateEffect.appendConfig = /* @__PURE__ */ StateEffect.define();
  var Transaction = class _Transaction {
    constructor(startState, changes, selection, effects, annotations, scrollIntoView3) {
      this.startState = startState;
      this.changes = changes;
      this.selection = selection;
      this.effects = effects;
      this.annotations = annotations;
      this.scrollIntoView = scrollIntoView3;
      this._doc = null;
      this._state = null;
      if (selection)
        checkSelection(selection, changes.newLength);
      if (!annotations.some((a) => a.type == _Transaction.time))
        this.annotations = annotations.concat(_Transaction.time.of(Date.now()));
    }
    /**
    @internal
    */
    static create(startState, changes, selection, effects, annotations, scrollIntoView3) {
      return new _Transaction(startState, changes, selection, effects, annotations, scrollIntoView3);
    }
    /**
    The new document produced by the transaction. Contrary to
    [`.state`](https://codemirror.net/6/docs/ref/#state.Transaction.state)`.doc`, accessing this won't
    force the entire new state to be computed right away, so it is
    recommended that [transaction
    filters](https://codemirror.net/6/docs/ref/#state.EditorState^transactionFilter) use this getter
    when they need to look at the new document.
    */
    get newDoc() {
      return this._doc || (this._doc = this.changes.apply(this.startState.doc));
    }
    /**
    The new selection produced by the transaction. If
    [`this.selection`](https://codemirror.net/6/docs/ref/#state.Transaction.selection) is undefined,
    this will [map](https://codemirror.net/6/docs/ref/#state.EditorSelection.map) the start state's
    current selection through the changes made by the transaction.
    */
    get newSelection() {
      return this.selection || this.startState.selection.map(this.changes);
    }
    /**
    The new state created by the transaction. Computed on demand
    (but retained for subsequent access), so it is recommended not to
    access it in [transaction
    filters](https://codemirror.net/6/docs/ref/#state.EditorState^transactionFilter) when possible.
    */
    get state() {
      if (!this._state)
        this.startState.applyTransaction(this);
      return this._state;
    }
    /**
    Get the value of the given annotation type, if any.
    */
    annotation(type) {
      for (let ann of this.annotations)
        if (ann.type == type)
          return ann.value;
      return void 0;
    }
    /**
    Indicates whether the transaction changed the document.
    */
    get docChanged() {
      return !this.changes.empty;
    }
    /**
    Indicates whether this transaction reconfigures the state
    (through a [configuration compartment](https://codemirror.net/6/docs/ref/#state.Compartment) or
    with a top-level configuration
    [effect](https://codemirror.net/6/docs/ref/#state.StateEffect^reconfigure).
    */
    get reconfigured() {
      return this.startState.config != this.state.config;
    }
    /**
    Returns true if the transaction has a [user
    event](https://codemirror.net/6/docs/ref/#state.Transaction^userEvent) annotation that is equal to
    or more specific than `event`. For example, if the transaction
    has `"select.pointer"` as user event, `"select"` and
    `"select.pointer"` will match it.
    */
    isUserEvent(event) {
      let e = this.annotation(_Transaction.userEvent);
      return !!(e && (e == event || e.length > event.length && e.slice(0, event.length) == event && e[event.length] == "."));
    }
  };
  Transaction.time = /* @__PURE__ */ Annotation.define();
  Transaction.userEvent = /* @__PURE__ */ Annotation.define();
  Transaction.addToHistory = /* @__PURE__ */ Annotation.define();
  Transaction.remote = /* @__PURE__ */ Annotation.define();
  function joinRanges(a, b) {
    let result = [];
    for (let iA = 0, iB = 0; ; ) {
      let from, to;
      if (iA < a.length && (iB == b.length || b[iB] >= a[iA])) {
        from = a[iA++];
        to = a[iA++];
      } else if (iB < b.length) {
        from = b[iB++];
        to = b[iB++];
      } else
        return result;
      if (!result.length || result[result.length - 1] < from)
        result.push(from, to);
      else if (result[result.length - 1] < to)
        result[result.length - 1] = to;
    }
  }
  function mergeTransaction(a, b, sequential) {
    var _a2;
    let mapForA, mapForB, changes;
    if (sequential) {
      mapForA = b.changes;
      mapForB = ChangeSet.empty(b.changes.length);
      changes = a.changes.compose(b.changes);
    } else {
      mapForA = b.changes.map(a.changes);
      mapForB = a.changes.mapDesc(b.changes, true);
      changes = a.changes.compose(mapForA);
    }
    return {
      changes,
      selection: b.selection ? b.selection.map(mapForB) : (_a2 = a.selection) === null || _a2 === void 0 ? void 0 : _a2.map(mapForA),
      effects: StateEffect.mapEffects(a.effects, mapForA).concat(StateEffect.mapEffects(b.effects, mapForB)),
      annotations: a.annotations.length ? a.annotations.concat(b.annotations) : b.annotations,
      scrollIntoView: a.scrollIntoView || b.scrollIntoView
    };
  }
  function resolveTransactionInner(state, spec, docSize) {
    let sel = spec.selection, annotations = asArray(spec.annotations);
    if (spec.userEvent)
      annotations = annotations.concat(Transaction.userEvent.of(spec.userEvent));
    return {
      changes: spec.changes instanceof ChangeSet ? spec.changes : ChangeSet.of(spec.changes || [], docSize, state.facet(lineSeparator)),
      selection: sel && (sel instanceof EditorSelection ? sel : EditorSelection.single(sel.anchor, sel.head)),
      effects: asArray(spec.effects),
      annotations,
      scrollIntoView: !!spec.scrollIntoView
    };
  }
  function resolveTransaction(state, specs, filter) {
    let s = resolveTransactionInner(state, specs.length ? specs[0] : {}, state.doc.length);
    if (specs.length && specs[0].filter === false)
      filter = false;
    for (let i = 1; i < specs.length; i++) {
      if (specs[i].filter === false)
        filter = false;
      let seq = !!specs[i].sequential;
      s = mergeTransaction(s, resolveTransactionInner(state, specs[i], seq ? s.changes.newLength : state.doc.length), seq);
    }
    let tr = Transaction.create(state, s.changes, s.selection, s.effects, s.annotations, s.scrollIntoView);
    return extendTransaction(filter ? filterTransaction(tr) : tr);
  }
  function filterTransaction(tr) {
    let state = tr.startState;
    let result = true;
    for (let filter of state.facet(changeFilter)) {
      let value = filter(tr);
      if (value === false) {
        result = false;
        break;
      }
      if (Array.isArray(value))
        result = result === true ? value : joinRanges(result, value);
    }
    if (result !== true) {
      let changes, back;
      if (result === false) {
        back = tr.changes.invertedDesc;
        changes = ChangeSet.empty(state.doc.length);
      } else {
        let filtered = tr.changes.filter(result);
        changes = filtered.changes;
        back = filtered.filtered.mapDesc(filtered.changes).invertedDesc;
      }
      tr = Transaction.create(state, changes, tr.selection && tr.selection.map(back), StateEffect.mapEffects(tr.effects, back), tr.annotations, tr.scrollIntoView);
    }
    let filters = state.facet(transactionFilter);
    for (let i = filters.length - 1; i >= 0; i--) {
      let filtered = filters[i](tr);
      if (filtered instanceof Transaction)
        tr = filtered;
      else if (Array.isArray(filtered) && filtered.length == 1 && filtered[0] instanceof Transaction)
        tr = filtered[0];
      else
        tr = resolveTransaction(state, asArray(filtered), false);
    }
    return tr;
  }
  function extendTransaction(tr) {
    let state = tr.startState, extenders = state.facet(transactionExtender), spec = tr;
    for (let i = extenders.length - 1; i >= 0; i--) {
      let extension = extenders[i](tr);
      if (extension && Object.keys(extension).length)
        spec = mergeTransaction(spec, resolveTransactionInner(state, extension, tr.changes.newLength), true);
    }
    return spec == tr ? tr : Transaction.create(state, tr.changes, tr.selection, spec.effects, spec.annotations, spec.scrollIntoView);
  }
  var none = [];
  function asArray(value) {
    return value == null ? none : Array.isArray(value) ? value : [value];
  }
  var CharCategory = /* @__PURE__ */ (function(CharCategory2) {
    CharCategory2[CharCategory2["Word"] = 0] = "Word";
    CharCategory2[CharCategory2["Space"] = 1] = "Space";
    CharCategory2[CharCategory2["Other"] = 2] = "Other";
    return CharCategory2;
  })(CharCategory || (CharCategory = {}));
  var nonASCIISingleCaseWordChar = /[\u00df\u0587\u0590-\u05f4\u0600-\u06ff\u3040-\u309f\u30a0-\u30ff\u3400-\u4db5\u4e00-\u9fcc\uac00-\ud7af]/;
  var wordChar;
  try {
    wordChar = /* @__PURE__ */ new RegExp("[\\p{Alphabetic}\\p{Number}_]", "u");
  } catch (_) {
  }
  function hasWordChar(str) {
    if (wordChar)
      return wordChar.test(str);
    for (let i = 0; i < str.length; i++) {
      let ch = str[i];
      if (/\w/.test(ch) || ch > "\x80" && (ch.toUpperCase() != ch.toLowerCase() || nonASCIISingleCaseWordChar.test(ch)))
        return true;
    }
    return false;
  }
  function makeCategorizer(wordChars) {
    return (char) => {
      if (!/\S/.test(char))
        return CharCategory.Space;
      if (hasWordChar(char))
        return CharCategory.Word;
      for (let i = 0; i < wordChars.length; i++)
        if (char.indexOf(wordChars[i]) > -1)
          return CharCategory.Word;
      return CharCategory.Other;
    };
  }
  var EditorState = class _EditorState {
    constructor(config, doc2, selection, values, computeSlot, tr) {
      this.config = config;
      this.doc = doc2;
      this.selection = selection;
      this.values = values;
      this.status = config.statusTemplate.slice();
      this.computeSlot = computeSlot;
      if (tr)
        tr._state = this;
      for (let i = 0; i < this.config.dynamicSlots.length; i++)
        ensureAddr(this, i << 1);
      this.computeSlot = null;
    }
    field(field, require2 = true) {
      let addr = this.config.address[field.id];
      if (addr == null) {
        if (require2)
          throw new RangeError("Field is not present in this state");
        return void 0;
      }
      ensureAddr(this, addr);
      return getAddr(this, addr);
    }
    /**
    Create a [transaction](https://codemirror.net/6/docs/ref/#state.Transaction) that updates this
    state. Any number of [transaction specs](https://codemirror.net/6/docs/ref/#state.TransactionSpec)
    can be passed. Unless
    [`sequential`](https://codemirror.net/6/docs/ref/#state.TransactionSpec.sequential) is set, the
    [changes](https://codemirror.net/6/docs/ref/#state.TransactionSpec.changes) (if any) of each spec
    are assumed to start in the _current_ document (not the document
    produced by previous specs), and its
    [selection](https://codemirror.net/6/docs/ref/#state.TransactionSpec.selection) and
    [effects](https://codemirror.net/6/docs/ref/#state.TransactionSpec.effects) are assumed to refer
    to the document created by its _own_ changes. The resulting
    transaction contains the combined effect of all the different
    specs. For [selection](https://codemirror.net/6/docs/ref/#state.TransactionSpec.selection), later
    specs take precedence over earlier ones.
    */
    update(...specs) {
      return resolveTransaction(this, specs, true);
    }
    /**
    @internal
    */
    applyTransaction(tr) {
      let conf = this.config, { base: base2, compartments } = conf;
      for (let effect of tr.effects) {
        if (effect.is(Compartment.reconfigure)) {
          if (conf) {
            compartments = /* @__PURE__ */ new Map();
            conf.compartments.forEach((val, key) => compartments.set(key, val));
            conf = null;
          }
          compartments.set(effect.value.compartment, effect.value.extension);
        } else if (effect.is(StateEffect.reconfigure)) {
          conf = null;
          base2 = effect.value;
        } else if (effect.is(StateEffect.appendConfig)) {
          conf = null;
          base2 = asArray(base2).concat(effect.value);
        }
      }
      let startValues;
      if (!conf) {
        conf = Configuration.resolve(base2, compartments, this);
        let intermediateState = new _EditorState(conf, this.doc, this.selection, conf.dynamicSlots.map(() => null), (state, slot) => slot.reconfigure(state, this), null);
        startValues = intermediateState.values;
      } else {
        startValues = tr.startState.values.slice();
      }
      let selection = tr.startState.facet(allowMultipleSelections) ? tr.newSelection : tr.newSelection.asSingle();
      new _EditorState(conf, tr.newDoc, selection, startValues, (state, slot) => slot.update(state, tr), tr);
    }
    /**
    Create a [transaction spec](https://codemirror.net/6/docs/ref/#state.TransactionSpec) that
    replaces every selection range with the given content.
    */
    replaceSelection(text) {
      if (typeof text == "string")
        text = this.toText(text);
      return this.changeByRange((range) => ({
        changes: { from: range.from, to: range.to, insert: text },
        range: EditorSelection.cursor(range.from + text.length)
      }));
    }
    /**
    Create a set of changes and a new selection by running the given
    function for each range in the active selection. The function
    can return an optional set of changes (in the coordinate space
    of the start document), plus an updated range (in the coordinate
    space of the document produced by the call's own changes). This
    method will merge all the changes and ranges into a single
    changeset and selection, and return it as a [transaction
    spec](https://codemirror.net/6/docs/ref/#state.TransactionSpec), which can be passed to
    [`update`](https://codemirror.net/6/docs/ref/#state.EditorState.update).
    */
    changeByRange(f) {
      let sel = this.selection;
      let result1 = f(sel.ranges[0]);
      let changes = this.changes(result1.changes), ranges = [result1.range];
      let effects = asArray(result1.effects);
      for (let i = 1; i < sel.ranges.length; i++) {
        let result = f(sel.ranges[i]);
        let newChanges = this.changes(result.changes), newMapped = newChanges.map(changes);
        for (let j = 0; j < i; j++)
          ranges[j] = ranges[j].map(newMapped);
        let mapBy = changes.mapDesc(newChanges, true);
        ranges.push(result.range.map(mapBy));
        changes = changes.compose(newMapped);
        effects = StateEffect.mapEffects(effects, newMapped).concat(StateEffect.mapEffects(asArray(result.effects), mapBy));
      }
      return {
        changes,
        selection: EditorSelection.create(ranges, sel.mainIndex),
        effects
      };
    }
    /**
    Create a [change set](https://codemirror.net/6/docs/ref/#state.ChangeSet) from the given change
    description, taking the state's document length and line
    separator into account.
    */
    changes(spec = []) {
      if (spec instanceof ChangeSet)
        return spec;
      return ChangeSet.of(spec, this.doc.length, this.facet(_EditorState.lineSeparator));
    }
    /**
    Using the state's [line
    separator](https://codemirror.net/6/docs/ref/#state.EditorState^lineSeparator), create a
    [`Text`](https://codemirror.net/6/docs/ref/#state.Text) instance from the given string.
    */
    toText(string2) {
      return Text.of(string2.split(this.facet(_EditorState.lineSeparator) || DefaultSplit));
    }
    /**
    Return the given range of the document as a string.
    */
    sliceDoc(from = 0, to = this.doc.length) {
      return this.doc.sliceString(from, to, this.lineBreak);
    }
    /**
    Get the value of a state [facet](https://codemirror.net/6/docs/ref/#state.Facet).
    */
    facet(facet) {
      let addr = this.config.address[facet.id];
      if (addr == null)
        return facet.default;
      ensureAddr(this, addr);
      return getAddr(this, addr);
    }
    /**
    Convert this state to a JSON-serializable object. When custom
    fields should be serialized, you can pass them in as an object
    mapping property names (in the resulting object, which should
    not use `doc` or `selection`) to fields.
    */
    toJSON(fields) {
      let result = {
        doc: this.sliceDoc(),
        selection: this.selection.toJSON()
      };
      if (fields)
        for (let prop in fields) {
          let value = fields[prop];
          if (value instanceof StateField && this.config.address[value.id] != null)
            result[prop] = value.spec.toJSON(this.field(fields[prop]), this);
        }
      return result;
    }
    /**
    Deserialize a state from its JSON representation. When custom
    fields should be deserialized, pass the same object you passed
    to [`toJSON`](https://codemirror.net/6/docs/ref/#state.EditorState.toJSON) when serializing as
    third argument.
    */
    static fromJSON(json, config = {}, fields) {
      if (!json || typeof json.doc != "string")
        throw new RangeError("Invalid JSON representation for EditorState");
      let fieldInit = [];
      if (fields)
        for (let prop in fields) {
          if (Object.prototype.hasOwnProperty.call(json, prop)) {
            let field = fields[prop], value = json[prop];
            fieldInit.push(field.init((state) => field.spec.fromJSON(value, state)));
          }
        }
      return _EditorState.create({
        doc: json.doc,
        selection: EditorSelection.fromJSON(json.selection),
        extensions: config.extensions ? fieldInit.concat([config.extensions]) : fieldInit
      });
    }
    /**
    Create a new state. You'll usually only need this when
    initializing an editor—updated states are created by applying
    transactions.
    */
    static create(config = {}) {
      let configuration = Configuration.resolve(config.extensions || [], /* @__PURE__ */ new Map());
      let doc2 = config.doc instanceof Text ? config.doc : Text.of((config.doc || "").split(configuration.staticFacet(_EditorState.lineSeparator) || DefaultSplit));
      let selection = !config.selection ? EditorSelection.single(0) : config.selection instanceof EditorSelection ? config.selection : EditorSelection.single(config.selection.anchor, config.selection.head);
      checkSelection(selection, doc2.length);
      if (!configuration.staticFacet(allowMultipleSelections))
        selection = selection.asSingle();
      return new _EditorState(configuration, doc2, selection, configuration.dynamicSlots.map(() => null), (state, slot) => slot.create(state), null);
    }
    /**
    The size (in columns) of a tab in the document, determined by
    the [`tabSize`](https://codemirror.net/6/docs/ref/#state.EditorState^tabSize) facet.
    */
    get tabSize() {
      return this.facet(_EditorState.tabSize);
    }
    /**
    Get the proper [line-break](https://codemirror.net/6/docs/ref/#state.EditorState^lineSeparator)
    string for this state.
    */
    get lineBreak() {
      return this.facet(_EditorState.lineSeparator) || "\n";
    }
    /**
    Returns true when the editor is
    [configured](https://codemirror.net/6/docs/ref/#state.EditorState^readOnly) to be read-only.
    */
    get readOnly() {
      return this.facet(readOnly);
    }
    /**
    Look up a translation for the given phrase (via the
    [`phrases`](https://codemirror.net/6/docs/ref/#state.EditorState^phrases) facet), or return the
    original string if no translation is found.
    
    If additional arguments are passed, they will be inserted in
    place of markers like `$1` (for the first value) and `$2`, etc.
    A single `$` is equivalent to `$1`, and `$$` will produce a
    literal dollar sign.
    */
    phrase(phrase2, ...insert2) {
      for (let map of this.facet(_EditorState.phrases))
        if (Object.prototype.hasOwnProperty.call(map, phrase2)) {
          phrase2 = map[phrase2];
          break;
        }
      if (insert2.length)
        phrase2 = phrase2.replace(/\$(\$|\d*)/g, (m, i) => {
          if (i == "$")
            return "$";
          let n = +(i || 1);
          return !n || n > insert2.length ? m : insert2[n - 1];
        });
      return phrase2;
    }
    /**
    Find the values for a given language data field, provided by the
    the [`languageData`](https://codemirror.net/6/docs/ref/#state.EditorState^languageData) facet.
    
    Examples of language data fields are...
    
    - [`"commentTokens"`](https://codemirror.net/6/docs/ref/#commands.CommentTokens) for specifying
      comment syntax.
    - [`"autocomplete"`](https://codemirror.net/6/docs/ref/#autocomplete.autocompletion^config.override)
      for providing language-specific completion sources.
    - [`"wordChars"`](https://codemirror.net/6/docs/ref/#state.EditorState.charCategorizer) for adding
      characters that should be considered part of words in this
      language.
    - [`"closeBrackets"`](https://codemirror.net/6/docs/ref/#autocomplete.CloseBracketConfig) controls
      bracket closing behavior.
    */
    languageDataAt(name2, pos, side = -1) {
      let values = [];
      for (let provider of this.facet(languageData)) {
        for (let result of provider(this, pos, side)) {
          if (Object.prototype.hasOwnProperty.call(result, name2))
            values.push(result[name2]);
        }
      }
      return values;
    }
    /**
    Return a function that can categorize strings (expected to
    represent a single [grapheme cluster](https://codemirror.net/6/docs/ref/#state.findClusterBreak))
    into one of:
    
     - Word (contains an alphanumeric character or a character
       explicitly listed in the local language's `"wordChars"`
       language data, which should be a string)
     - Space (contains only whitespace)
     - Other (anything else)
    */
    charCategorizer(at) {
      let chars = this.languageDataAt("wordChars", at);
      return makeCategorizer(chars.length ? chars[0] : "");
    }
    /**
    Find the word at the given position, meaning the range
    containing all [word](https://codemirror.net/6/docs/ref/#state.CharCategory.Word) characters
    around it. If no word characters are adjacent to the position,
    this returns null.
    */
    wordAt(pos) {
      let { text, from, length } = this.doc.lineAt(pos);
      let cat = this.charCategorizer(pos);
      let start = pos - from, end = pos - from;
      while (start > 0) {
        let prev = findClusterBreak2(text, start, false);
        if (cat(text.slice(prev, start)) != CharCategory.Word)
          break;
        start = prev;
      }
      while (end < length) {
        let next = findClusterBreak2(text, end);
        if (cat(text.slice(end, next)) != CharCategory.Word)
          break;
        end = next;
      }
      return start == end ? null : EditorSelection.range(start + from, end + from);
    }
  };
  EditorState.allowMultipleSelections = allowMultipleSelections;
  EditorState.tabSize = /* @__PURE__ */ Facet.define({
    combine: (values) => values.length ? values[0] : 4
  });
  EditorState.lineSeparator = lineSeparator;
  EditorState.readOnly = readOnly;
  EditorState.phrases = /* @__PURE__ */ Facet.define({
    compare(a, b) {
      let kA = Object.keys(a), kB = Object.keys(b);
      return kA.length == kB.length && kA.every((k) => a[k] == b[k]);
    }
  });
  EditorState.languageData = languageData;
  EditorState.changeFilter = changeFilter;
  EditorState.transactionFilter = transactionFilter;
  EditorState.transactionExtender = transactionExtender;
  Compartment.reconfigure = /* @__PURE__ */ StateEffect.define();
  function combineConfig(configs, defaults, combine = {}) {
    let result = {};
    for (let config of configs)
      for (let key of Object.keys(config)) {
        let value = config[key], current = result[key];
        if (current === void 0)
          result[key] = value;
        else if (current === value || value === void 0) ;
        else if (Object.hasOwnProperty.call(combine, key))
          result[key] = combine[key](current, value);
        else
          throw new Error("Config merge conflict for field " + key);
      }
    for (let key in defaults)
      if (result[key] === void 0)
        result[key] = defaults[key];
    return result;
  }
  var RangeValue = class {
    /**
    Compare this value with another value. Used when comparing
    rangesets. The default implementation compares by identity.
    Unless you are only creating a fixed number of unique instances
    of your value type, it is a good idea to implement this
    properly.
    */
    eq(other) {
      return this == other;
    }
    /**
    Create a [range](https://codemirror.net/6/docs/ref/#state.Range) with this value.
    */
    range(from, to = from) {
      return Range.create(from, to, this);
    }
  };
  RangeValue.prototype.startSide = RangeValue.prototype.endSide = 0;
  RangeValue.prototype.point = false;
  RangeValue.prototype.mapMode = MapMode.TrackDel;
  function cmpVal(a, b) {
    return a == b || a.constructor == b.constructor && a.eq(b);
  }
  var Range = class _Range {
    constructor(from, to, value) {
      this.from = from;
      this.to = to;
      this.value = value;
    }
    /**
    @internal
    */
    static create(from, to, value) {
      return new _Range(from, to, value);
    }
  };
  function cmpRange(a, b) {
    return a.from - b.from || a.value.startSide - b.value.startSide;
  }
  var Chunk = class _Chunk {
    constructor(from, to, value, maxPoint) {
      this.from = from;
      this.to = to;
      this.value = value;
      this.maxPoint = maxPoint;
    }
    get length() {
      return this.to[this.to.length - 1];
    }
    // Find the index of the given position and side. Use the ranges'
    // `from` pos when `end == false`, `to` when `end == true`.
    findIndex(pos, side, end, startAt = 0) {
      let arr = end ? this.to : this.from;
      for (let lo = startAt, hi = arr.length; ; ) {
        if (lo == hi)
          return lo;
        let mid = lo + hi >> 1;
        let diff = arr[mid] - pos || (end ? this.value[mid].endSide : this.value[mid].startSide) - side;
        if (mid == lo)
          return diff >= 0 ? lo : hi;
        if (diff >= 0)
          hi = mid;
        else
          lo = mid + 1;
      }
    }
    between(offset, from, to, f) {
      for (let i = this.findIndex(from, -1e9, true), e = this.findIndex(to, 1e9, false, i); i < e; i++)
        if (f(this.from[i] + offset, this.to[i] + offset, this.value[i]) === false)
          return false;
    }
    map(offset, changes) {
      let value = [], from = [], to = [], newPos = -1, maxPoint = -1;
      for (let i = 0; i < this.value.length; i++) {
        let val = this.value[i], curFrom = this.from[i] + offset, curTo = this.to[i] + offset, newFrom, newTo;
        if (curFrom == curTo) {
          let mapped = changes.mapPos(curFrom, val.startSide, val.mapMode);
          if (mapped == null)
            continue;
          newFrom = newTo = mapped;
          if (val.startSide != val.endSide) {
            newTo = changes.mapPos(curFrom, val.endSide);
            if (newTo < newFrom)
              continue;
          }
        } else {
          newFrom = changes.mapPos(curFrom, val.startSide);
          newTo = changes.mapPos(curTo, val.endSide);
          if (newFrom > newTo || newFrom == newTo && val.startSide > 0 && val.endSide <= 0)
            continue;
        }
        if ((newTo - newFrom || val.endSide - val.startSide) < 0)
          continue;
        if (newPos < 0)
          newPos = newFrom;
        if (val.point)
          maxPoint = Math.max(maxPoint, newTo - newFrom);
        value.push(val);
        from.push(newFrom - newPos);
        to.push(newTo - newPos);
      }
      return { mapped: value.length ? new _Chunk(from, to, value, maxPoint) : null, pos: newPos };
    }
  };
  var RangeSet = class _RangeSet {
    constructor(chunkPos, chunk, nextLayer, maxPoint) {
      this.chunkPos = chunkPos;
      this.chunk = chunk;
      this.nextLayer = nextLayer;
      this.maxPoint = maxPoint;
    }
    /**
    @internal
    */
    static create(chunkPos, chunk, nextLayer, maxPoint) {
      return new _RangeSet(chunkPos, chunk, nextLayer, maxPoint);
    }
    /**
    @internal
    */
    get length() {
      let last = this.chunk.length - 1;
      return last < 0 ? 0 : Math.max(this.chunkEnd(last), this.nextLayer.length);
    }
    /**
    The number of ranges in the set.
    */
    get size() {
      if (this.isEmpty)
        return 0;
      let size = this.nextLayer.size;
      for (let chunk of this.chunk)
        size += chunk.value.length;
      return size;
    }
    /**
    @internal
    */
    chunkEnd(index) {
      return this.chunkPos[index] + this.chunk[index].length;
    }
    /**
    Update the range set, optionally adding new ranges or filtering
    out existing ones.
    
    (Note: The type parameter is just there as a kludge to work
    around TypeScript variance issues that prevented `RangeSet<X>`
    from being a subtype of `RangeSet<Y>` when `X` is a subtype of
    `Y`.)
    */
    update(updateSpec) {
      let { add: add2 = [], sort = false, filterFrom = 0, filterTo = this.length } = updateSpec;
      let filter = updateSpec.filter;
      if (add2.length == 0 && !filter)
        return this;
      if (sort)
        add2 = add2.slice().sort(cmpRange);
      if (this.isEmpty)
        return add2.length ? _RangeSet.of(add2) : this;
      let cur2 = new LayerCursor(this, null, -1).goto(0), i = 0, spill = [];
      let builder = new RangeSetBuilder();
      while (cur2.value || i < add2.length) {
        if (i < add2.length && (cur2.from - add2[i].from || cur2.startSide - add2[i].value.startSide) >= 0) {
          let range = add2[i++];
          if (!builder.addInner(range.from, range.to, range.value))
            spill.push(range);
        } else if (cur2.rangeIndex == 1 && cur2.chunkIndex < this.chunk.length && (i == add2.length || this.chunkEnd(cur2.chunkIndex) < add2[i].from) && (!filter || filterFrom > this.chunkEnd(cur2.chunkIndex) || filterTo < this.chunkPos[cur2.chunkIndex]) && builder.addChunk(this.chunkPos[cur2.chunkIndex], this.chunk[cur2.chunkIndex])) {
          cur2.nextChunk();
        } else {
          if (!filter || filterFrom > cur2.to || filterTo < cur2.from || filter(cur2.from, cur2.to, cur2.value)) {
            if (!builder.addInner(cur2.from, cur2.to, cur2.value))
              spill.push(Range.create(cur2.from, cur2.to, cur2.value));
          }
          cur2.next();
        }
      }
      return builder.finishInner(this.nextLayer.isEmpty && !spill.length ? _RangeSet.empty : this.nextLayer.update({ add: spill, filter, filterFrom, filterTo }));
    }
    /**
    Map this range set through a set of changes, return the new set.
    */
    map(changes) {
      if (changes.empty || this.isEmpty)
        return this;
      let chunks = [], chunkPos = [], maxPoint = -1;
      for (let i = 0; i < this.chunk.length; i++) {
        let start = this.chunkPos[i], chunk = this.chunk[i];
        let touch = changes.touchesRange(start, start + chunk.length);
        if (touch === false) {
          maxPoint = Math.max(maxPoint, chunk.maxPoint);
          chunks.push(chunk);
          chunkPos.push(changes.mapPos(start));
        } else if (touch === true) {
          let { mapped, pos } = chunk.map(start, changes);
          if (mapped) {
            maxPoint = Math.max(maxPoint, mapped.maxPoint);
            chunks.push(mapped);
            chunkPos.push(pos);
          }
        }
      }
      let next = this.nextLayer.map(changes);
      return chunks.length == 0 ? next : new _RangeSet(chunkPos, chunks, next || _RangeSet.empty, maxPoint);
    }
    /**
    Iterate over the ranges that touch the region `from` to `to`,
    calling `f` for each. There is no guarantee that the ranges will
    be reported in any specific order. When the callback returns
    `false`, iteration stops.
    */
    between(from, to, f) {
      if (this.isEmpty)
        return;
      for (let i = 0; i < this.chunk.length; i++) {
        let start = this.chunkPos[i], chunk = this.chunk[i];
        if (to >= start && from <= start + chunk.length && chunk.between(start, from - start, to - start, f) === false)
          return;
      }
      this.nextLayer.between(from, to, f);
    }
    /**
    Iterate over the ranges in this set, in order, including all
    ranges that end at or after `from`.
    */
    iter(from = 0) {
      return HeapCursor.from([this]).goto(from);
    }
    /**
    @internal
    */
    get isEmpty() {
      return this.nextLayer == this;
    }
    /**
    Iterate over the ranges in a collection of sets, in order,
    starting from `from`.
    */
    static iter(sets, from = 0) {
      return HeapCursor.from(sets).goto(from);
    }
    /**
    Iterate over two groups of sets, calling methods on `comparator`
    to notify it of possible differences.
    */
    static compare(oldSets, newSets, textDiff, comparator, minPointSize = -1) {
      let a = oldSets.filter((set) => set.maxPoint > 0 || !set.isEmpty && set.maxPoint >= minPointSize);
      let b = newSets.filter((set) => set.maxPoint > 0 || !set.isEmpty && set.maxPoint >= minPointSize);
      let sharedChunks = findSharedChunks(a, b, textDiff);
      let sideA = new SpanCursor(a, sharedChunks, minPointSize);
      let sideB = new SpanCursor(b, sharedChunks, minPointSize);
      textDiff.iterGaps((fromA, fromB, length) => compare(sideA, fromA, sideB, fromB, length, comparator));
      if (textDiff.empty && textDiff.length == 0)
        compare(sideA, 0, sideB, 0, 0, comparator);
    }
    /**
    Compare the contents of two groups of range sets, returning true
    if they are equivalent in the given range.
    */
    static eq(oldSets, newSets, from = 0, to) {
      if (to == null)
        to = 1e9 - 1;
      let a = oldSets.filter((set) => !set.isEmpty && newSets.indexOf(set) < 0);
      let b = newSets.filter((set) => !set.isEmpty && oldSets.indexOf(set) < 0);
      if (a.length != b.length)
        return false;
      if (!a.length)
        return true;
      let sharedChunks = findSharedChunks(a, b);
      let sideA = new SpanCursor(a, sharedChunks, 0).goto(from), sideB = new SpanCursor(b, sharedChunks, 0).goto(from);
      for (; ; ) {
        if (sideA.to != sideB.to || !sameValues(sideA.active, sideB.active) || sideA.point && (!sideB.point || !cmpVal(sideA.point, sideB.point)))
          return false;
        if (sideA.to > to)
          return true;
        sideA.next();
        sideB.next();
      }
    }
    /**
    Iterate over a group of range sets at the same time, notifying
    the iterator about the ranges covering every given piece of
    content. Returns the open count (see
    [`SpanIterator.span`](https://codemirror.net/6/docs/ref/#state.SpanIterator.span)) at the end
    of the iteration.
    */
    static spans(sets, from, to, iterator, minPointSize = -1) {
      let cursor = new SpanCursor(sets, null, minPointSize).goto(from), pos = from;
      let openRanges = cursor.openStart;
      for (; ; ) {
        let curTo = Math.min(cursor.to, to);
        if (cursor.point) {
          let active = cursor.activeForPoint(cursor.to);
          let openCount = cursor.pointFrom < from ? active.length + 1 : cursor.point.startSide < 0 ? active.length : Math.min(active.length, openRanges);
          iterator.point(pos, curTo, cursor.point, active, openCount, cursor.pointRank);
          openRanges = Math.min(cursor.openEnd(curTo), active.length);
        } else if (curTo > pos) {
          iterator.span(pos, curTo, cursor.active, openRanges);
          openRanges = cursor.openEnd(curTo);
        }
        if (cursor.to > to)
          return openRanges + (cursor.point && cursor.to > to ? 1 : 0);
        pos = cursor.to;
        cursor.next();
      }
    }
    /**
    Create a range set for the given range or array of ranges. By
    default, this expects the ranges to be _sorted_ (by start
    position and, if two start at the same position,
    `value.startSide`). You can pass `true` as second argument to
    cause the method to sort them.
    */
    static of(ranges, sort = false) {
      let build = new RangeSetBuilder();
      for (let range of ranges instanceof Range ? [ranges] : sort ? lazySort(ranges) : ranges)
        build.add(range.from, range.to, range.value);
      return build.finish();
    }
    /**
    Join an array of range sets into a single set.
    */
    static join(sets) {
      if (!sets.length)
        return _RangeSet.empty;
      let result = sets[sets.length - 1];
      for (let i = sets.length - 2; i >= 0; i--) {
        for (let layer2 = sets[i]; layer2 != _RangeSet.empty; layer2 = layer2.nextLayer)
          result = new _RangeSet(layer2.chunkPos, layer2.chunk, result, Math.max(layer2.maxPoint, result.maxPoint));
      }
      return result;
    }
  };
  RangeSet.empty = /* @__PURE__ */ new RangeSet([], [], null, -1);
  function lazySort(ranges) {
    if (ranges.length > 1)
      for (let prev = ranges[0], i = 1; i < ranges.length; i++) {
        let cur2 = ranges[i];
        if (cmpRange(prev, cur2) > 0)
          return ranges.slice().sort(cmpRange);
        prev = cur2;
      }
    return ranges;
  }
  RangeSet.empty.nextLayer = RangeSet.empty;
  var RangeSetBuilder = class _RangeSetBuilder {
    finishChunk(newArrays) {
      this.chunks.push(new Chunk(this.from, this.to, this.value, this.maxPoint));
      this.chunkPos.push(this.chunkStart);
      this.chunkStart = -1;
      this.setMaxPoint = Math.max(this.setMaxPoint, this.maxPoint);
      this.maxPoint = -1;
      if (newArrays) {
        this.from = [];
        this.to = [];
        this.value = [];
      }
    }
    /**
    Create an empty builder.
    */
    constructor() {
      this.chunks = [];
      this.chunkPos = [];
      this.chunkStart = -1;
      this.last = null;
      this.lastFrom = -1e9;
      this.lastTo = -1e9;
      this.from = [];
      this.to = [];
      this.value = [];
      this.maxPoint = -1;
      this.setMaxPoint = -1;
      this.nextLayer = null;
    }
    /**
    Add a range. Ranges should be added in sorted (by `from` and
    `value.startSide`) order.
    */
    add(from, to, value) {
      if (!this.addInner(from, to, value))
        (this.nextLayer || (this.nextLayer = new _RangeSetBuilder())).add(from, to, value);
    }
    /**
    @internal
    */
    addInner(from, to, value) {
      let diff = from - this.lastTo || value.startSide - this.last.endSide;
      if (diff <= 0 && (from - this.lastFrom || value.startSide - this.last.startSide) < 0)
        throw new Error("Ranges must be added sorted by `from` position and `startSide`");
      if (diff < 0)
        return false;
      if (this.from.length == 250)
        this.finishChunk(true);
      if (this.chunkStart < 0)
        this.chunkStart = from;
      this.from.push(from - this.chunkStart);
      this.to.push(to - this.chunkStart);
      this.last = value;
      this.lastFrom = from;
      this.lastTo = to;
      this.value.push(value);
      if (value.point)
        this.maxPoint = Math.max(this.maxPoint, to - from);
      return true;
    }
    /**
    @internal
    */
    addChunk(from, chunk) {
      if ((from - this.lastTo || chunk.value[0].startSide - this.last.endSide) < 0)
        return false;
      if (this.from.length)
        this.finishChunk(true);
      this.setMaxPoint = Math.max(this.setMaxPoint, chunk.maxPoint);
      this.chunks.push(chunk);
      this.chunkPos.push(from);
      let last = chunk.value.length - 1;
      this.last = chunk.value[last];
      this.lastFrom = chunk.from[last] + from;
      this.lastTo = chunk.to[last] + from;
      return true;
    }
    /**
    Finish the range set. Returns the new set. The builder can't be
    used anymore after this has been called.
    */
    finish() {
      return this.finishInner(RangeSet.empty);
    }
    /**
    @internal
    */
    finishInner(next) {
      if (this.from.length)
        this.finishChunk(false);
      if (this.chunks.length == 0)
        return next;
      let result = RangeSet.create(this.chunkPos, this.chunks, this.nextLayer ? this.nextLayer.finishInner(next) : next, this.setMaxPoint);
      this.from = null;
      return result;
    }
  };
  function findSharedChunks(a, b, textDiff) {
    let inA = /* @__PURE__ */ new Map();
    for (let set of a)
      for (let i = 0; i < set.chunk.length; i++)
        if (set.chunk[i].maxPoint <= 0)
          inA.set(set.chunk[i], set.chunkPos[i]);
    let shared = /* @__PURE__ */ new Set();
    for (let set of b)
      for (let i = 0; i < set.chunk.length; i++) {
        let known = inA.get(set.chunk[i]);
        if (known != null && (textDiff ? textDiff.mapPos(known) : known) == set.chunkPos[i] && !(textDiff === null || textDiff === void 0 ? void 0 : textDiff.touchesRange(known, known + set.chunk[i].length)))
          shared.add(set.chunk[i]);
      }
    return shared;
  }
  var LayerCursor = class {
    constructor(layer2, skip, minPoint, rank = 0) {
      this.layer = layer2;
      this.skip = skip;
      this.minPoint = minPoint;
      this.rank = rank;
    }
    get startSide() {
      return this.value ? this.value.startSide : 0;
    }
    get endSide() {
      return this.value ? this.value.endSide : 0;
    }
    goto(pos, side = -1e9) {
      this.chunkIndex = this.rangeIndex = 0;
      this.gotoInner(pos, side, false);
      return this;
    }
    gotoInner(pos, side, forward) {
      while (this.chunkIndex < this.layer.chunk.length) {
        let next = this.layer.chunk[this.chunkIndex];
        if (!(this.skip && this.skip.has(next) || this.layer.chunkEnd(this.chunkIndex) < pos || next.maxPoint < this.minPoint))
          break;
        this.chunkIndex++;
        forward = false;
      }
      if (this.chunkIndex < this.layer.chunk.length) {
        let rangeIndex = this.layer.chunk[this.chunkIndex].findIndex(pos - this.layer.chunkPos[this.chunkIndex], side, true);
        if (!forward || this.rangeIndex < rangeIndex)
          this.setRangeIndex(rangeIndex);
      }
      this.next();
    }
    forward(pos, side) {
      if ((this.to - pos || this.endSide - side) < 0)
        this.gotoInner(pos, side, true);
    }
    next() {
      for (; ; ) {
        if (this.chunkIndex == this.layer.chunk.length) {
          this.from = this.to = 1e9;
          this.value = null;
          break;
        } else {
          let chunkPos = this.layer.chunkPos[this.chunkIndex], chunk = this.layer.chunk[this.chunkIndex];
          let from = chunkPos + chunk.from[this.rangeIndex];
          this.from = from;
          this.to = chunkPos + chunk.to[this.rangeIndex];
          this.value = chunk.value[this.rangeIndex];
          this.setRangeIndex(this.rangeIndex + 1);
          if (this.minPoint < 0 || this.value.point && this.to - this.from >= this.minPoint)
            break;
        }
      }
    }
    setRangeIndex(index) {
      if (index == this.layer.chunk[this.chunkIndex].value.length) {
        this.chunkIndex++;
        if (this.skip) {
          while (this.chunkIndex < this.layer.chunk.length && this.skip.has(this.layer.chunk[this.chunkIndex]))
            this.chunkIndex++;
        }
        this.rangeIndex = 0;
      } else {
        this.rangeIndex = index;
      }
    }
    nextChunk() {
      this.chunkIndex++;
      this.rangeIndex = 0;
      this.next();
    }
    compare(other) {
      return this.from - other.from || this.startSide - other.startSide || this.rank - other.rank || this.to - other.to || this.endSide - other.endSide;
    }
  };
  var HeapCursor = class _HeapCursor {
    constructor(heap) {
      this.heap = heap;
    }
    static from(sets, skip = null, minPoint = -1) {
      let heap = [];
      for (let i = 0; i < sets.length; i++) {
        for (let cur2 = sets[i]; !cur2.isEmpty; cur2 = cur2.nextLayer) {
          if (cur2.maxPoint >= minPoint)
            heap.push(new LayerCursor(cur2, skip, minPoint, i));
        }
      }
      return heap.length == 1 ? heap[0] : new _HeapCursor(heap);
    }
    get startSide() {
      return this.value ? this.value.startSide : 0;
    }
    goto(pos, side = -1e9) {
      for (let cur2 of this.heap)
        cur2.goto(pos, side);
      for (let i = this.heap.length >> 1; i >= 0; i--)
        heapBubble(this.heap, i);
      this.next();
      return this;
    }
    forward(pos, side) {
      for (let cur2 of this.heap)
        cur2.forward(pos, side);
      for (let i = this.heap.length >> 1; i >= 0; i--)
        heapBubble(this.heap, i);
      if ((this.to - pos || this.value.endSide - side) < 0)
        this.next();
    }
    next() {
      if (this.heap.length == 0) {
        this.from = this.to = 1e9;
        this.value = null;
        this.rank = -1;
      } else {
        let top2 = this.heap[0];
        this.from = top2.from;
        this.to = top2.to;
        this.value = top2.value;
        this.rank = top2.rank;
        if (top2.value)
          top2.next();
        heapBubble(this.heap, 0);
      }
    }
  };
  function heapBubble(heap, index) {
    for (let cur2 = heap[index]; ; ) {
      let childIndex = (index << 1) + 1;
      if (childIndex >= heap.length)
        break;
      let child = heap[childIndex];
      if (childIndex + 1 < heap.length && child.compare(heap[childIndex + 1]) >= 0) {
        child = heap[childIndex + 1];
        childIndex++;
      }
      if (cur2.compare(child) < 0)
        break;
      heap[childIndex] = cur2;
      heap[index] = child;
      index = childIndex;
    }
  }
  var SpanCursor = class {
    constructor(sets, skip, minPoint) {
      this.minPoint = minPoint;
      this.active = [];
      this.activeTo = [];
      this.activeRank = [];
      this.minActive = -1;
      this.point = null;
      this.pointFrom = 0;
      this.pointRank = 0;
      this.to = -1e9;
      this.endSide = 0;
      this.openStart = -1;
      this.cursor = HeapCursor.from(sets, skip, minPoint);
    }
    goto(pos, side = -1e9) {
      this.cursor.goto(pos, side);
      this.active.length = this.activeTo.length = this.activeRank.length = 0;
      this.minActive = -1;
      this.to = pos;
      this.endSide = side;
      this.openStart = -1;
      this.next();
      return this;
    }
    forward(pos, side) {
      while (this.minActive > -1 && (this.activeTo[this.minActive] - pos || this.active[this.minActive].endSide - side) < 0)
        this.removeActive(this.minActive);
      this.cursor.forward(pos, side);
    }
    removeActive(index) {
      remove(this.active, index);
      remove(this.activeTo, index);
      remove(this.activeRank, index);
      this.minActive = findMinIndex(this.active, this.activeTo);
    }
    addActive(trackOpen) {
      let i = 0, { value, to, rank } = this.cursor;
      while (i < this.activeRank.length && (rank - this.activeRank[i] || to - this.activeTo[i]) > 0)
        i++;
      insert(this.active, i, value);
      insert(this.activeTo, i, to);
      insert(this.activeRank, i, rank);
      if (trackOpen)
        insert(trackOpen, i, this.cursor.from);
      this.minActive = findMinIndex(this.active, this.activeTo);
    }
    // After calling this, if `this.point` != null, the next range is a
    // point. Otherwise, it's a regular range, covered by `this.active`.
    next() {
      let from = this.to, wasPoint = this.point;
      this.point = null;
      let trackOpen = this.openStart < 0 ? [] : null;
      for (; ; ) {
        let a = this.minActive;
        if (a > -1 && (this.activeTo[a] - this.cursor.from || this.active[a].endSide - this.cursor.startSide) < 0) {
          if (this.activeTo[a] > from) {
            this.to = this.activeTo[a];
            this.endSide = this.active[a].endSide;
            break;
          }
          this.removeActive(a);
          if (trackOpen)
            remove(trackOpen, a);
        } else if (!this.cursor.value) {
          this.to = this.endSide = 1e9;
          break;
        } else if (this.cursor.from > from) {
          this.to = this.cursor.from;
          this.endSide = this.cursor.startSide;
          break;
        } else {
          let nextVal = this.cursor.value;
          if (!nextVal.point) {
            this.addActive(trackOpen);
            this.cursor.next();
          } else if (wasPoint && this.cursor.to == this.to && this.cursor.from < this.cursor.to) {
            this.cursor.next();
          } else {
            this.point = nextVal;
            this.pointFrom = this.cursor.from;
            this.pointRank = this.cursor.rank;
            this.to = this.cursor.to;
            this.endSide = nextVal.endSide;
            this.cursor.next();
            this.forward(this.to, this.endSide);
            break;
          }
        }
      }
      if (trackOpen) {
        this.openStart = 0;
        for (let i = trackOpen.length - 1; i >= 0 && trackOpen[i] < from; i--)
          this.openStart++;
      }
    }
    activeForPoint(to) {
      if (!this.active.length)
        return this.active;
      let active = [];
      for (let i = this.active.length - 1; i >= 0; i--) {
        if (this.activeRank[i] < this.pointRank)
          break;
        if (this.activeTo[i] > to || this.activeTo[i] == to && this.active[i].endSide >= this.point.endSide)
          active.push(this.active[i]);
      }
      return active.reverse();
    }
    openEnd(to) {
      let open = 0;
      for (let i = this.activeTo.length - 1; i >= 0 && this.activeTo[i] > to; i--)
        open++;
      return open;
    }
  };
  function compare(a, startA, b, startB, length, comparator) {
    a.goto(startA);
    b.goto(startB);
    let endB = startB + length;
    let pos = startB, dPos = startB - startA;
    let bounds = !!comparator.boundChange;
    for (let boundChange = false; ; ) {
      let dEnd = a.to + dPos - b.to, diff = dEnd || a.endSide - b.endSide;
      let end = diff < 0 ? a.to + dPos : b.to, clipEnd = Math.min(end, endB);
      let point = a.point || b.point;
      if (point) {
        if (!(a.point && b.point && cmpVal(a.point, b.point) && sameValues(a.activeForPoint(a.to), b.activeForPoint(b.to))))
          comparator.comparePoint(pos, clipEnd, a.point, b.point);
        boundChange = false;
      } else {
        if (boundChange)
          comparator.boundChange(pos);
        if (clipEnd > pos && !sameValues(a.active, b.active))
          comparator.compareRange(pos, clipEnd, a.active, b.active);
        if (bounds && clipEnd < endB && (dEnd || a.openEnd(end) != b.openEnd(end)))
          boundChange = true;
      }
      if (end > endB)
        break;
      pos = end;
      if (diff <= 0)
        a.next();
      if (diff >= 0)
        b.next();
    }
  }
  function sameValues(a, b) {
    if (a.length != b.length)
      return false;
    for (let i = 0; i < a.length; i++)
      if (a[i] != b[i] && !cmpVal(a[i], b[i]))
        return false;
    return true;
  }
  function remove(array, index) {
    for (let i = index, e = array.length - 1; i < e; i++)
      array[i] = array[i + 1];
    array.pop();
  }
  function insert(array, index, value) {
    for (let i = array.length - 1; i >= index; i--)
      array[i + 1] = array[i];
    array[index] = value;
  }
  function findMinIndex(value, array) {
    let found = -1, foundPos = 1e9;
    for (let i = 0; i < array.length; i++)
      if ((array[i] - foundPos || value[i].endSide - value[found].endSide) < 0) {
        found = i;
        foundPos = array[i];
      }
    return found;
  }
  function countColumn(string2, tabSize, to = string2.length) {
    let n = 0;
    for (let i = 0; i < to && i < string2.length; ) {
      if (string2.charCodeAt(i) == 9) {
        n += tabSize - n % tabSize;
        i++;
      } else {
        n++;
        i = findClusterBreak2(string2, i);
      }
    }
    return n;
  }
  function findColumn(string2, col, tabSize, strict) {
    for (let i = 0, n = 0; ; ) {
      if (n >= col)
        return i;
      if (i == string2.length)
        break;
      n += string2.charCodeAt(i) == 9 ? tabSize - n % tabSize : 1;
      i = findClusterBreak2(string2, i);
    }
    return strict === true ? -1 : string2.length;
  }

  // node_modules/style-mod/src/style-mod.js
  var C = "\u037C";
  var COUNT = typeof Symbol == "undefined" ? "__" + C : Symbol.for(C);
  var SET = typeof Symbol == "undefined" ? "__styleSet" + Math.floor(Math.random() * 1e8) : /* @__PURE__ */ Symbol("styleSet");
  var top = typeof globalThis != "undefined" ? globalThis : typeof window != "undefined" ? window : {};
  var StyleModule = class {
    // :: (Object<Style>, ?{finish: ?(string) → string})
    // Create a style module from the given spec.
    //
    // When `finish` is given, it is called on regular (non-`@`)
    // selectors (after `&` expansion) to compute the final selector.
    constructor(spec, options) {
      this.rules = [];
      let { finish } = options || {};
      function splitSelector(selector) {
        return /^@/.test(selector) ? [selector] : selector.split(/,\s*/);
      }
      function render(selectors, spec2, target, isKeyframes) {
        let local = [], isAt = /^@(\w+)\b/.exec(selectors[0]), keyframes = isAt && isAt[1] == "keyframes";
        if (isAt && spec2 == null) return target.push(selectors[0] + ";");
        for (let prop in spec2) {
          let value = spec2[prop];
          if (/&/.test(prop)) {
            render(
              prop.split(/,\s*/).map((part) => selectors.map((sel) => part.replace(/&/, sel))).reduce((a, b) => a.concat(b)),
              value,
              target
            );
          } else if (value && typeof value == "object") {
            if (!isAt) throw new RangeError("The value of a property (" + prop + ") should be a primitive value.");
            render(splitSelector(prop), value, local, keyframes);
          } else if (value != null) {
            local.push(prop.replace(/_.*/, "").replace(/[A-Z]/g, (l) => "-" + l.toLowerCase()) + ": " + value + ";");
          }
        }
        if (local.length || keyframes) {
          target.push((finish && !isAt && !isKeyframes ? selectors.map(finish) : selectors).join(", ") + " {" + local.join(" ") + "}");
        }
      }
      for (let prop in spec) render(splitSelector(prop), spec[prop], this.rules);
    }
    // :: () → string
    // Returns a string containing the module's CSS rules.
    getRules() {
      return this.rules.join("\n");
    }
    // :: () → string
    // Generate a new unique CSS class name.
    static newName() {
      let id = top[COUNT] || 1;
      top[COUNT] = id + 1;
      return C + id.toString(36);
    }
    // :: (union<Document, ShadowRoot>, union<[StyleModule], StyleModule>, ?{nonce: ?string})
    //
    // Mount the given set of modules in the given DOM root, which ensures
    // that the CSS rules defined by the module are available in that
    // context.
    //
    // Rules are only added to the document once per root.
    //
    // Rule order will follow the order of the modules, so that rules from
    // modules later in the array take precedence of those from earlier
    // modules. If you call this function multiple times for the same root
    // in a way that changes the order of already mounted modules, the old
    // order will be changed.
    //
    // If a Content Security Policy nonce is provided, it is added to
    // the `<style>` tag generated by the library.
    static mount(root, modules, options) {
      let set = root[SET], nonce = options && options.nonce;
      if (!set) set = new StyleSet(root, nonce);
      else if (nonce) set.setNonce(nonce);
      set.mount(Array.isArray(modules) ? modules : [modules], root);
    }
  };
  var adoptedSet = /* @__PURE__ */ new Map();
  var StyleSet = class {
    constructor(root, nonce) {
      let doc2 = root.ownerDocument || root, win = doc2.defaultView;
      if (!root.head && root.adoptedStyleSheets && win.CSSStyleSheet) {
        let adopted = adoptedSet.get(doc2);
        if (adopted) return root[SET] = adopted;
        this.sheet = new win.CSSStyleSheet();
        adoptedSet.set(doc2, this);
      } else {
        this.styleTag = doc2.createElement("style");
        if (nonce) this.styleTag.setAttribute("nonce", nonce);
      }
      this.modules = [];
      root[SET] = this;
    }
    mount(modules, root) {
      let sheet = this.sheet;
      let pos = 0, j = 0;
      for (let i = 0; i < modules.length; i++) {
        let mod = modules[i], index = this.modules.indexOf(mod);
        if (index < j && index > -1) {
          this.modules.splice(index, 1);
          j--;
          index = -1;
        }
        if (index == -1) {
          this.modules.splice(j++, 0, mod);
          if (sheet) for (let k = 0; k < mod.rules.length; k++)
            sheet.insertRule(mod.rules[k], pos++);
        } else {
          while (j < index) pos += this.modules[j++].rules.length;
          pos += mod.rules.length;
          j++;
        }
      }
      if (sheet) {
        if (root.adoptedStyleSheets.indexOf(this.sheet) < 0)
          root.adoptedStyleSheets = [this.sheet, ...root.adoptedStyleSheets];
      } else {
        let text = "";
        for (let i = 0; i < this.modules.length; i++)
          text += this.modules[i].getRules() + "\n";
        this.styleTag.textContent = text;
        let target = root.head || root;
        if (this.styleTag.parentNode != target)
          target.insertBefore(this.styleTag, target.firstChild);
      }
    }
    setNonce(nonce) {
      if (this.styleTag && this.styleTag.getAttribute("nonce") != nonce)
        this.styleTag.setAttribute("nonce", nonce);
    }
  };

  // node_modules/w3c-keyname/index.js
  var base = {
    8: "Backspace",
    9: "Tab",
    10: "Enter",
    12: "NumLock",
    13: "Enter",
    16: "Shift",
    17: "Control",
    18: "Alt",
    20: "CapsLock",
    27: "Escape",
    32: " ",
    33: "PageUp",
    34: "PageDown",
    35: "End",
    36: "Home",
    37: "ArrowLeft",
    38: "ArrowUp",
    39: "ArrowRight",
    40: "ArrowDown",
    44: "PrintScreen",
    45: "Insert",
    46: "Delete",
    59: ";",
    61: "=",
    91: "Meta",
    92: "Meta",
    106: "*",
    107: "+",
    108: ",",
    109: "-",
    110: ".",
    111: "/",
    144: "NumLock",
    145: "ScrollLock",
    160: "Shift",
    161: "Shift",
    162: "Control",
    163: "Control",
    164: "Alt",
    165: "Alt",
    173: "-",
    186: ";",
    187: "=",
    188: ",",
    189: "-",
    190: ".",
    191: "/",
    192: "`",
    219: "[",
    220: "\\",
    221: "]",
    222: "'"
  };
  var shift = {
    48: ")",
    49: "!",
    50: "@",
    51: "#",
    52: "$",
    53: "%",
    54: "^",
    55: "&",
    56: "*",
    57: "(",
    59: ":",
    61: "+",
    173: "_",
    186: ":",
    187: "+",
    188: "<",
    189: "_",
    190: ">",
    191: "?",
    192: "~",
    219: "{",
    220: "|",
    221: "}",
    222: '"'
  };
  var mac = typeof navigator != "undefined" && /Mac/.test(navigator.platform);
  var ie = typeof navigator != "undefined" && /MSIE \d|Trident\/(?:[7-9]|\d{2,})\..*rv:(\d+)/.exec(navigator.userAgent);
  for (i = 0; i < 10; i++) base[48 + i] = base[96 + i] = String(i);
  var i;
  for (i = 1; i <= 24; i++) base[i + 111] = "F" + i;
  var i;
  for (i = 65; i <= 90; i++) {
    base[i] = String.fromCharCode(i + 32);
    shift[i] = String.fromCharCode(i);
  }
  var i;
  for (code in base) if (!shift.hasOwnProperty(code)) shift[code] = base[code];
  var code;
  function keyName(event) {
    var ignoreKey = mac && event.metaKey && event.shiftKey && !event.ctrlKey && !event.altKey || ie && event.shiftKey && event.key && event.key.length == 1 || event.key == "Unidentified";
    var name2 = !ignoreKey && event.key || (event.shiftKey ? shift : base)[event.keyCode] || event.key || "Unidentified";
    if (name2 == "Esc") name2 = "Escape";
    if (name2 == "Del") name2 = "Delete";
    if (name2 == "Left") name2 = "ArrowLeft";
    if (name2 == "Up") name2 = "ArrowUp";
    if (name2 == "Right") name2 = "ArrowRight";
    if (name2 == "Down") name2 = "ArrowDown";
    return name2;
  }

  // node_modules/crelt/index.js
  function crelt() {
    var elt = arguments[0];
    if (typeof elt == "string") elt = document.createElement(elt);
    var i = 1, next = arguments[1];
    if (next && typeof next == "object" && next.nodeType == null && !Array.isArray(next)) {
      for (var name2 in next) if (Object.prototype.hasOwnProperty.call(next, name2)) {
        var value = next[name2];
        if (typeof value == "string") elt.setAttribute(name2, value);
        else if (value != null) elt[name2] = value;
      }
      i++;
    }
    for (; i < arguments.length; i++) add(elt, arguments[i]);
    return elt;
  }
  function add(elt, child) {
    if (typeof child == "string") {
      elt.appendChild(document.createTextNode(child));
    } else if (child == null) {
    } else if (child.nodeType != null) {
      elt.appendChild(child);
    } else if (Array.isArray(child)) {
      for (var i = 0; i < child.length; i++) add(elt, child[i]);
    } else {
      throw new RangeError("Unsupported child node: " + child);
    }
  }

  // node_modules/@codemirror/view/dist/index.js
  var nav = typeof navigator != "undefined" ? navigator : { userAgent: "", vendor: "", platform: "" };
  var doc = typeof document != "undefined" ? document : { documentElement: { style: {} } };
  var ie_edge = /* @__PURE__ */ /Edge\/(\d+)/.exec(nav.userAgent);
  var ie_upto10 = /* @__PURE__ */ /MSIE \d/.test(nav.userAgent);
  var ie_11up = /* @__PURE__ */ /Trident\/(?:[7-9]|\d{2,})\..*rv:(\d+)/.exec(nav.userAgent);
  var ie2 = !!(ie_upto10 || ie_11up || ie_edge);
  var gecko = !ie2 && /* @__PURE__ */ /gecko\/(\d+)/i.test(nav.userAgent);
  var chrome = !ie2 && /* @__PURE__ */ /Chrome\/(\d+)/.exec(nav.userAgent);
  var webkit = "webkitFontSmoothing" in doc.documentElement.style;
  var safari = !ie2 && /* @__PURE__ */ /Apple Computer/.test(nav.vendor);
  var ios = safari && (/* @__PURE__ */ /Mobile\/\w+/.test(nav.userAgent) || nav.maxTouchPoints > 2);
  var browser = {
    mac: ios || /* @__PURE__ */ /Mac/.test(nav.platform),
    windows: /* @__PURE__ */ /Win/.test(nav.platform),
    linux: /* @__PURE__ */ /Linux|X11/.test(nav.platform),
    ie: ie2,
    ie_version: ie_upto10 ? doc.documentMode || 6 : ie_11up ? +ie_11up[1] : ie_edge ? +ie_edge[1] : 0,
    gecko,
    gecko_version: gecko ? +(/* @__PURE__ */ /Firefox\/(\d+)/.exec(nav.userAgent) || [0, 0])[1] : 0,
    chrome: !!chrome,
    chrome_version: chrome ? +chrome[1] : 0,
    ios,
    android: /* @__PURE__ */ /Android\b/.test(nav.userAgent),
    webkit,
    webkit_version: webkit ? +(/* @__PURE__ */ /\bAppleWebKit\/(\d+)/.exec(nav.userAgent) || [0, 0])[1] : 0,
    safari,
    safari_version: safari ? +(/* @__PURE__ */ /\bVersion\/(\d+(\.\d+)?)/.exec(nav.userAgent) || [0, 0])[1] : 0,
    tabSize: doc.documentElement.style.tabSize != null ? "tab-size" : "-moz-tab-size"
  };
  function combineAttrs(source, target) {
    for (let name2 in source) {
      if (name2 == "class" && target.class)
        target.class += " " + source.class;
      else if (name2 == "style" && target.style)
        target.style += ";" + source.style;
      else
        target[name2] = source[name2];
    }
    return target;
  }
  var noAttrs = /* @__PURE__ */ Object.create(null);
  function attrsEq(a, b, ignore) {
    if (a == b)
      return true;
    if (!a)
      a = noAttrs;
    if (!b)
      b = noAttrs;
    let keysA = Object.keys(a), keysB = Object.keys(b);
    if (keysA.length - (ignore && keysA.indexOf(ignore) > -1 ? 1 : 0) != keysB.length - (ignore && keysB.indexOf(ignore) > -1 ? 1 : 0))
      return false;
    for (let key of keysA) {
      if (key != ignore && (keysB.indexOf(key) == -1 || a[key] !== b[key]))
        return false;
    }
    return true;
  }
  function setAttrs(dom, attrs) {
    for (let i = dom.attributes.length - 1; i >= 0; i--) {
      let name2 = dom.attributes[i].name;
      if (attrs[name2] == null)
        dom.removeAttribute(name2);
    }
    for (let name2 in attrs) {
      let value = attrs[name2];
      if (name2 == "style")
        dom.style.cssText = value;
      else if (dom.getAttribute(name2) != value)
        dom.setAttribute(name2, value);
    }
  }
  function updateAttrs(dom, prev, attrs) {
    let changed = false;
    if (prev) {
      for (let name2 in prev)
        if (!(attrs && name2 in attrs)) {
          changed = true;
          if (name2 == "style")
            dom.style.cssText = "";
          else
            dom.removeAttribute(name2);
        }
    }
    if (attrs) {
      for (let name2 in attrs)
        if (!(prev && prev[name2] == attrs[name2])) {
          changed = true;
          if (name2 == "style")
            dom.style.cssText = attrs[name2];
          else
            dom.setAttribute(name2, attrs[name2]);
        }
    }
    return changed;
  }
  function getAttrs(dom) {
    let attrs = /* @__PURE__ */ Object.create(null);
    for (let i = 0; i < dom.attributes.length; i++) {
      let attr = dom.attributes[i];
      attrs[attr.name] = attr.value;
    }
    return attrs;
  }
  var WidgetType = class {
    /**
    Compare this instance to another instance of the same type.
    (TypeScript can't express this, but only instances of the same
    specific class will be passed to this method.) This is used to
    avoid redrawing widgets when they are replaced by a new
    decoration of the same type. The default implementation just
    returns `false`, which will cause new instances of the widget to
    always be redrawn.
    */
    eq(widget) {
      return false;
    }
    /**
    Update a DOM element created by a widget of the same type (but
    different, non-`eq` content) to reflect this widget. May return
    true to indicate that it could update, false to indicate it
    couldn't (in which case the widget will be redrawn). The default
    implementation just returns false.
    */
    updateDOM(dom, view) {
      return false;
    }
    /**
    @internal
    */
    compare(other) {
      return this == other || this.constructor == other.constructor && this.eq(other);
    }
    /**
    The estimated height this widget will have, to be used when
    estimating the height of content that hasn't been drawn. May
    return -1 to indicate you don't know. The default implementation
    returns -1.
    */
    get estimatedHeight() {
      return -1;
    }
    /**
    For inline widgets that are displayed inline (as opposed to
    `inline-block`) and introduce line breaks (through `<br>` tags
    or textual newlines), this must indicate the amount of line
    breaks they introduce. Defaults to 0.
    */
    get lineBreaks() {
      return 0;
    }
    /**
    Can be used to configure which kinds of events inside the widget
    should be ignored by the editor. The default is to ignore all
    events.
    */
    ignoreEvent(event) {
      return true;
    }
    /**
    Override the way screen coordinates for positions at/in the
    widget are found. `pos` will be the offset into the widget, and
    `side` the side of the position that is being queried—less than
    zero for before, greater than zero for after, and zero for
    directly at that position.
    */
    coordsAt(dom, pos, side) {
      return null;
    }
    /**
    @internal
    */
    get isHidden() {
      return false;
    }
    /**
    @internal
    */
    get editable() {
      return false;
    }
    /**
    This is called when the an instance of the widget is removed
    from the editor view.
    */
    destroy(dom) {
    }
  };
  var BlockType = /* @__PURE__ */ (function(BlockType2) {
    BlockType2[BlockType2["Text"] = 0] = "Text";
    BlockType2[BlockType2["WidgetBefore"] = 1] = "WidgetBefore";
    BlockType2[BlockType2["WidgetAfter"] = 2] = "WidgetAfter";
    BlockType2[BlockType2["WidgetRange"] = 3] = "WidgetRange";
    return BlockType2;
  })(BlockType || (BlockType = {}));
  var Decoration = class extends RangeValue {
    constructor(startSide, endSide, widget, spec) {
      super();
      this.startSide = startSide;
      this.endSide = endSide;
      this.widget = widget;
      this.spec = spec;
    }
    /**
    @internal
    */
    get heightRelevant() {
      return false;
    }
    /**
    Create a mark decoration, which influences the styling of the
    content in its range. Nested mark decorations will cause nested
    DOM elements to be created. Nesting order is determined by
    precedence of the [facet](https://codemirror.net/6/docs/ref/#view.EditorView^decorations), with
    the higher-precedence decorations creating the inner DOM nodes.
    Such elements are split on line boundaries and on the boundaries
    of lower-precedence decorations.
    */
    static mark(spec) {
      return new MarkDecoration(spec);
    }
    /**
    Create a widget decoration, which displays a DOM element at the
    given position.
    */
    static widget(spec) {
      let side = Math.max(-1e4, Math.min(1e4, spec.side || 0)), block = !!spec.block;
      side += block && !spec.inlineOrder ? side > 0 ? 3e8 : -4e8 : side > 0 ? 1e8 : -1e8;
      return new PointDecoration(spec, side, side, block, spec.widget || null, false);
    }
    /**
    Create a replace decoration which replaces the given range with
    a widget, or simply hides it.
    */
    static replace(spec) {
      let block = !!spec.block, startSide, endSide;
      if (spec.isBlockGap) {
        startSide = -5e8;
        endSide = 4e8;
      } else {
        let { start, end } = getInclusive(spec, block);
        startSide = (start ? block ? -3e8 : -1 : 5e8) - 1;
        endSide = (end ? block ? 2e8 : 1 : -6e8) + 1;
      }
      return new PointDecoration(spec, startSide, endSide, block, spec.widget || null, true);
    }
    /**
    Create a line decoration, which can add DOM attributes to the
    line starting at the given position.
    */
    static line(spec) {
      return new LineDecoration(spec);
    }
    /**
    Build a [`DecorationSet`](https://codemirror.net/6/docs/ref/#view.DecorationSet) from the given
    decorated range or ranges. If the ranges aren't already sorted,
    pass `true` for `sort` to make the library sort them for you.
    */
    static set(of, sort = false) {
      return RangeSet.of(of, sort);
    }
    /**
    @internal
    */
    hasHeight() {
      return this.widget ? this.widget.estimatedHeight > -1 : false;
    }
  };
  Decoration.none = RangeSet.empty;
  var MarkDecoration = class _MarkDecoration extends Decoration {
    constructor(spec) {
      let { start, end } = getInclusive(spec);
      super(start ? -1 : 5e8, end ? 1 : -6e8, null, spec);
      this.tagName = spec.tagName || "span";
      this.attrs = spec.class && spec.attributes ? combineAttrs(spec.attributes, { class: spec.class }) : spec.class ? { class: spec.class } : spec.attributes || noAttrs;
    }
    eq(other) {
      return this == other || other instanceof _MarkDecoration && this.tagName == other.tagName && attrsEq(this.attrs, other.attrs);
    }
    range(from, to = from) {
      if (from >= to)
        throw new RangeError("Mark decorations may not be empty");
      return super.range(from, to);
    }
  };
  MarkDecoration.prototype.point = false;
  var LineDecoration = class _LineDecoration extends Decoration {
    constructor(spec) {
      super(-2e8, -2e8, null, spec);
    }
    eq(other) {
      return other instanceof _LineDecoration && this.spec.class == other.spec.class && attrsEq(this.spec.attributes, other.spec.attributes);
    }
    range(from, to = from) {
      if (to != from)
        throw new RangeError("Line decoration ranges must be zero-length");
      return super.range(from, to);
    }
  };
  LineDecoration.prototype.mapMode = MapMode.TrackBefore;
  LineDecoration.prototype.point = true;
  var PointDecoration = class _PointDecoration extends Decoration {
    constructor(spec, startSide, endSide, block, widget, isReplace) {
      super(startSide, endSide, widget, spec);
      this.block = block;
      this.isReplace = isReplace;
      this.mapMode = !block ? MapMode.TrackDel : startSide <= 0 ? MapMode.TrackBefore : MapMode.TrackAfter;
    }
    // Only relevant when this.block == true
    get type() {
      return this.startSide != this.endSide ? BlockType.WidgetRange : this.startSide <= 0 ? BlockType.WidgetBefore : BlockType.WidgetAfter;
    }
    get heightRelevant() {
      return this.block || !!this.widget && (this.widget.estimatedHeight >= 5 || this.widget.lineBreaks > 0);
    }
    eq(other) {
      return other instanceof _PointDecoration && widgetsEq(this.widget, other.widget) && this.block == other.block && this.startSide == other.startSide && this.endSide == other.endSide;
    }
    range(from, to = from) {
      if (this.isReplace && (from > to || from == to && this.startSide > 0 && this.endSide <= 0))
        throw new RangeError("Invalid range for replacement decoration");
      if (!this.isReplace && to != from)
        throw new RangeError("Widget decorations can only have zero-length ranges");
      return super.range(from, to);
    }
  };
  PointDecoration.prototype.point = true;
  function getInclusive(spec, block = false) {
    let { inclusiveStart: start, inclusiveEnd: end } = spec;
    if (start == null)
      start = spec.inclusive;
    if (end == null)
      end = spec.inclusive;
    return { start: start !== null && start !== void 0 ? start : block, end: end !== null && end !== void 0 ? end : block };
  }
  function widgetsEq(a, b) {
    return a == b || !!(a && b && a.compare(b));
  }
  function addRange(from, to, ranges, margin = 0) {
    let last = ranges.length - 1;
    if (last >= 0 && ranges[last] + margin >= from)
      ranges[last] = Math.max(ranges[last], to);
    else
      ranges.push(from, to);
  }
  var BlockWrapper = class _BlockWrapper extends RangeValue {
    constructor(tagName, attributes) {
      super();
      this.tagName = tagName;
      this.attributes = attributes;
    }
    eq(other) {
      return other == this || other instanceof _BlockWrapper && this.tagName == other.tagName && attrsEq(this.attributes, other.attributes);
    }
    /**
    Create a block wrapper object with the given tag name and
    attributes.
    */
    static create(spec) {
      return new _BlockWrapper(spec.tagName, spec.attributes || noAttrs);
    }
    /**
    Create a range set from the given block wrapper ranges.
    */
    static set(of, sort = false) {
      return RangeSet.of(of, sort);
    }
  };
  BlockWrapper.prototype.startSide = BlockWrapper.prototype.endSide = -1;
  function getSelection(root) {
    let target;
    if (root.nodeType == 11) {
      target = root.getSelection ? root : root.ownerDocument;
    } else {
      target = root;
    }
    return target.getSelection();
  }
  function contains(dom, node) {
    return node ? dom == node || dom.contains(node.nodeType != 1 ? node.parentNode : node) : false;
  }
  function hasSelection(dom, selection) {
    if (!selection.anchorNode)
      return false;
    try {
      return contains(dom, selection.anchorNode);
    } catch (_) {
      return false;
    }
  }
  function clientRectsFor(dom) {
    if (dom.nodeType == 3)
      return textRange(dom, 0, dom.nodeValue.length).getClientRects();
    else if (dom.nodeType == 1)
      return dom.getClientRects();
    else
      return [];
  }
  function isEquivalentPosition(node, off, targetNode, targetOff) {
    return targetNode ? scanFor(node, off, targetNode, targetOff, -1) || scanFor(node, off, targetNode, targetOff, 1) : false;
  }
  function domIndex(node) {
    for (var index = 0; ; index++) {
      node = node.previousSibling;
      if (!node)
        return index;
    }
  }
  function isBlockElement(node) {
    return node.nodeType == 1 && /^(DIV|P|LI|UL|OL|BLOCKQUOTE|DD|DT|H\d|SECTION|PRE)$/.test(node.nodeName);
  }
  function scanFor(node, off, targetNode, targetOff, dir) {
    for (; ; ) {
      if (node == targetNode && off == targetOff)
        return true;
      if (off == (dir < 0 ? 0 : maxOffset(node))) {
        if (node.nodeName == "DIV")
          return false;
        let parent = node.parentNode;
        if (!parent || parent.nodeType != 1)
          return false;
        off = domIndex(node) + (dir < 0 ? 0 : 1);
        node = parent;
      } else if (node.nodeType == 1) {
        node = node.childNodes[off + (dir < 0 ? -1 : 0)];
        if (node.nodeType == 1 && node.contentEditable == "false")
          return false;
        off = dir < 0 ? maxOffset(node) : 0;
      } else {
        return false;
      }
    }
  }
  function maxOffset(node) {
    return node.nodeType == 3 ? node.nodeValue.length : node.childNodes.length;
  }
  function flattenRect(rect, left) {
    let x = left ? rect.left : rect.right;
    return { left: x, right: x, top: rect.top, bottom: rect.bottom };
  }
  function windowRect(win) {
    let vp = win.visualViewport;
    if (vp)
      return {
        left: 0,
        right: vp.width,
        top: 0,
        bottom: vp.height
      };
    return {
      left: 0,
      right: win.innerWidth,
      top: 0,
      bottom: win.innerHeight
    };
  }
  function getScale(elt, rect) {
    let scaleX = rect.width / elt.offsetWidth;
    let scaleY = rect.height / elt.offsetHeight;
    if (scaleX > 0.995 && scaleX < 1.005 || !isFinite(scaleX) || Math.abs(rect.width - elt.offsetWidth) < 1)
      scaleX = 1;
    if (scaleY > 0.995 && scaleY < 1.005 || !isFinite(scaleY) || Math.abs(rect.height - elt.offsetHeight) < 1)
      scaleY = 1;
    return { scaleX, scaleY };
  }
  function scrollRectIntoView(dom, rect, side, x, y, xMargin, yMargin, ltr) {
    let doc2 = dom.ownerDocument, win = doc2.defaultView || window;
    for (let cur2 = dom, stop = false; cur2 && !stop; ) {
      if (cur2.nodeType == 1) {
        let bounding, top2 = cur2 == doc2.body;
        let scaleX = 1, scaleY = 1;
        if (top2) {
          bounding = windowRect(win);
        } else {
          if (/^(fixed|sticky)$/.test(getComputedStyle(cur2).position))
            stop = true;
          if (cur2.scrollHeight <= cur2.clientHeight && cur2.scrollWidth <= cur2.clientWidth) {
            cur2 = cur2.assignedSlot || cur2.parentNode;
            continue;
          }
          let rect2 = cur2.getBoundingClientRect();
          ({ scaleX, scaleY } = getScale(cur2, rect2));
          bounding = {
            left: rect2.left,
            right: rect2.left + cur2.clientWidth * scaleX,
            top: rect2.top,
            bottom: rect2.top + cur2.clientHeight * scaleY
          };
        }
        let moveX = 0, moveY = 0;
        if (y == "nearest") {
          if (rect.top < bounding.top) {
            moveY = rect.top - (bounding.top + yMargin);
            if (side > 0 && rect.bottom > bounding.bottom + moveY)
              moveY = rect.bottom - bounding.bottom + yMargin;
          } else if (rect.bottom > bounding.bottom) {
            moveY = rect.bottom - bounding.bottom + yMargin;
            if (side < 0 && rect.top - moveY < bounding.top)
              moveY = rect.top - (bounding.top + yMargin);
          }
        } else {
          let rectHeight = rect.bottom - rect.top, boundingHeight = bounding.bottom - bounding.top;
          let targetTop = y == "center" && rectHeight <= boundingHeight ? rect.top + rectHeight / 2 - boundingHeight / 2 : y == "start" || y == "center" && side < 0 ? rect.top - yMargin : rect.bottom - boundingHeight + yMargin;
          moveY = targetTop - bounding.top;
        }
        if (x == "nearest") {
          if (rect.left < bounding.left) {
            moveX = rect.left - (bounding.left + xMargin);
            if (side > 0 && rect.right > bounding.right + moveX)
              moveX = rect.right - bounding.right + xMargin;
          } else if (rect.right > bounding.right) {
            moveX = rect.right - bounding.right + xMargin;
            if (side < 0 && rect.left < bounding.left + moveX)
              moveX = rect.left - (bounding.left + xMargin);
          }
        } else {
          let targetLeft = x == "center" ? rect.left + (rect.right - rect.left) / 2 - (bounding.right - bounding.left) / 2 : x == "start" == ltr ? rect.left - xMargin : rect.right - (bounding.right - bounding.left) + xMargin;
          moveX = targetLeft - bounding.left;
        }
        if (moveX || moveY) {
          if (top2) {
            win.scrollBy(moveX, moveY);
          } else {
            let movedX = 0, movedY = 0;
            if (moveY) {
              let start = cur2.scrollTop;
              cur2.scrollTop += moveY / scaleY;
              movedY = (cur2.scrollTop - start) * scaleY;
            }
            if (moveX) {
              let start = cur2.scrollLeft;
              cur2.scrollLeft += moveX / scaleX;
              movedX = (cur2.scrollLeft - start) * scaleX;
            }
            rect = {
              left: rect.left - movedX,
              top: rect.top - movedY,
              right: rect.right - movedX,
              bottom: rect.bottom - movedY
            };
            if (movedX && Math.abs(movedX - moveX) < 1)
              x = "nearest";
            if (movedY && Math.abs(movedY - moveY) < 1)
              y = "nearest";
          }
        }
        if (top2)
          break;
        if (rect.top < bounding.top || rect.bottom > bounding.bottom || rect.left < bounding.left || rect.right > bounding.right)
          rect = {
            left: Math.max(rect.left, bounding.left),
            right: Math.min(rect.right, bounding.right),
            top: Math.max(rect.top, bounding.top),
            bottom: Math.min(rect.bottom, bounding.bottom)
          };
        cur2 = cur2.assignedSlot || cur2.parentNode;
      } else if (cur2.nodeType == 11) {
        cur2 = cur2.host;
      } else {
        break;
      }
    }
  }
  function scrollableParents(dom, getX = true) {
    let doc2 = dom.ownerDocument, x = null, y = null;
    for (let cur2 = dom.parentNode; cur2; ) {
      if (cur2 == doc2.body || (!getX || x) && y) {
        break;
      } else if (cur2.nodeType == 1) {
        if (!y && cur2.scrollHeight > cur2.clientHeight)
          y = cur2;
        if (getX && !x && cur2.scrollWidth > cur2.clientWidth)
          x = cur2;
        cur2 = cur2.assignedSlot || cur2.parentNode;
      } else if (cur2.nodeType == 11) {
        cur2 = cur2.host;
      } else {
        break;
      }
    }
    return { x, y };
  }
  var DOMSelectionState = class {
    constructor() {
      this.anchorNode = null;
      this.anchorOffset = 0;
      this.focusNode = null;
      this.focusOffset = 0;
    }
    eq(domSel) {
      return this.anchorNode == domSel.anchorNode && this.anchorOffset == domSel.anchorOffset && this.focusNode == domSel.focusNode && this.focusOffset == domSel.focusOffset;
    }
    setRange(range) {
      let { anchorNode, focusNode } = range;
      this.set(anchorNode, Math.min(range.anchorOffset, anchorNode ? maxOffset(anchorNode) : 0), focusNode, Math.min(range.focusOffset, focusNode ? maxOffset(focusNode) : 0));
    }
    set(anchorNode, anchorOffset, focusNode, focusOffset) {
      this.anchorNode = anchorNode;
      this.anchorOffset = anchorOffset;
      this.focusNode = focusNode;
      this.focusOffset = focusOffset;
    }
  };
  var preventScrollSupported = null;
  if (browser.safari && browser.safari_version >= 26)
    preventScrollSupported = false;
  function focusPreventScroll(dom) {
    if (dom.setActive)
      return dom.setActive();
    if (preventScrollSupported)
      return dom.focus(preventScrollSupported);
    let stack = [];
    for (let cur2 = dom; cur2; cur2 = cur2.parentNode) {
      stack.push(cur2, cur2.scrollTop, cur2.scrollLeft);
      if (cur2 == cur2.ownerDocument)
        break;
    }
    dom.focus(preventScrollSupported == null ? {
      get preventScroll() {
        preventScrollSupported = { preventScroll: true };
        return true;
      }
    } : void 0);
    if (!preventScrollSupported) {
      preventScrollSupported = false;
      for (let i = 0; i < stack.length; ) {
        let elt = stack[i++], top2 = stack[i++], left = stack[i++];
        if (elt.scrollTop != top2)
          elt.scrollTop = top2;
        if (elt.scrollLeft != left)
          elt.scrollLeft = left;
      }
    }
  }
  var scratchRange;
  function textRange(node, from, to = from) {
    let range = scratchRange || (scratchRange = document.createRange());
    range.setEnd(node, to);
    range.setStart(node, from);
    return range;
  }
  function dispatchKey(elt, name2, code, mods) {
    let options = { key: name2, code: name2, keyCode: code, which: code, cancelable: true };
    if (mods)
      ({ altKey: options.altKey, ctrlKey: options.ctrlKey, shiftKey: options.shiftKey, metaKey: options.metaKey } = mods);
    let down = new KeyboardEvent("keydown", options);
    down.synthetic = true;
    elt.dispatchEvent(down);
    let up = new KeyboardEvent("keyup", options);
    up.synthetic = true;
    elt.dispatchEvent(up);
    return down.defaultPrevented || up.defaultPrevented;
  }
  function getRoot(node) {
    while (node) {
      if (node && (node.nodeType == 9 || node.nodeType == 11 && node.host))
        return node;
      node = node.assignedSlot || node.parentNode;
    }
    return null;
  }
  function atElementStart(doc2, selection) {
    let node = selection.focusNode, offset = selection.focusOffset;
    if (!node || selection.anchorNode != node || selection.anchorOffset != offset)
      return false;
    offset = Math.min(offset, maxOffset(node));
    for (; ; ) {
      if (offset) {
        if (node.nodeType != 1)
          return false;
        let prev = node.childNodes[offset - 1];
        if (prev.contentEditable == "false")
          offset--;
        else {
          node = prev;
          offset = maxOffset(node);
        }
      } else if (node == doc2) {
        return true;
      } else {
        offset = domIndex(node);
        node = node.parentNode;
      }
    }
  }
  function isScrolledToBottom(elt) {
    if (elt instanceof Window)
      return elt.pageYOffset > Math.max(0, elt.document.documentElement.scrollHeight - elt.innerHeight - 4);
    return elt.scrollTop > Math.max(1, elt.scrollHeight - elt.clientHeight - 4);
  }
  function textNodeBefore(startNode, startOffset) {
    for (let node = startNode, offset = startOffset; ; ) {
      if (node.nodeType == 3 && offset > 0) {
        return { node, offset };
      } else if (node.nodeType == 1 && offset > 0) {
        if (node.contentEditable == "false")
          return null;
        node = node.childNodes[offset - 1];
        offset = maxOffset(node);
      } else if (node.parentNode && !isBlockElement(node)) {
        offset = domIndex(node);
        node = node.parentNode;
      } else {
        return null;
      }
    }
  }
  function textNodeAfter(startNode, startOffset) {
    for (let node = startNode, offset = startOffset; ; ) {
      if (node.nodeType == 3 && offset < node.nodeValue.length) {
        return { node, offset };
      } else if (node.nodeType == 1 && offset < node.childNodes.length) {
        if (node.contentEditable == "false")
          return null;
        node = node.childNodes[offset];
        offset = 0;
      } else if (node.parentNode && !isBlockElement(node)) {
        offset = domIndex(node) + 1;
        node = node.parentNode;
      } else {
        return null;
      }
    }
  }
  var DOMPos = class _DOMPos {
    constructor(node, offset, precise = true) {
      this.node = node;
      this.offset = offset;
      this.precise = precise;
    }
    static before(dom, precise) {
      return new _DOMPos(dom.parentNode, domIndex(dom), precise);
    }
    static after(dom, precise) {
      return new _DOMPos(dom.parentNode, domIndex(dom) + 1, precise);
    }
  };
  var Direction = /* @__PURE__ */ (function(Direction2) {
    Direction2[Direction2["LTR"] = 0] = "LTR";
    Direction2[Direction2["RTL"] = 1] = "RTL";
    return Direction2;
  })(Direction || (Direction = {}));
  var LTR = Direction.LTR;
  var RTL = Direction.RTL;
  function dec(str) {
    let result = [];
    for (let i = 0; i < str.length; i++)
      result.push(1 << +str[i]);
    return result;
  }
  var LowTypes = /* @__PURE__ */ dec("88888888888888888888888888888888888666888888787833333333337888888000000000000000000000000008888880000000000000000000000000088888888888888888888888888888888888887866668888088888663380888308888800000000000000000000000800000000000000000000000000000008");
  var ArabicTypes = /* @__PURE__ */ dec("4444448826627288999999999992222222222222222222222222222222222222222222222229999999999999999999994444444444644222822222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222999999949999999229989999223333333333");
  var Brackets = /* @__PURE__ */ Object.create(null);
  var BracketStack = [];
  for (let p of ["()", "[]", "{}"]) {
    let l = /* @__PURE__ */ p.charCodeAt(0), r = /* @__PURE__ */ p.charCodeAt(1);
    Brackets[l] = r;
    Brackets[r] = -l;
  }
  function charType(ch) {
    return ch <= 247 ? LowTypes[ch] : 1424 <= ch && ch <= 1524 ? 2 : 1536 <= ch && ch <= 1785 ? ArabicTypes[ch - 1536] : 1774 <= ch && ch <= 2220 ? 4 : 8192 <= ch && ch <= 8204 ? 256 : 64336 <= ch && ch <= 65023 ? 4 : 1;
  }
  var BidiRE = /[\u0590-\u05f4\u0600-\u06ff\u0700-\u08ac\ufb50-\ufdff]/;
  var BidiSpan = class {
    /**
    The direction of this span.
    */
    get dir() {
      return this.level % 2 ? RTL : LTR;
    }
    /**
    @internal
    */
    constructor(from, to, level) {
      this.from = from;
      this.to = to;
      this.level = level;
    }
    /**
    @internal
    */
    side(end, dir) {
      return this.dir == dir == end ? this.to : this.from;
    }
    /**
    @internal
    */
    forward(forward, dir) {
      return forward == (this.dir == dir);
    }
    /**
    @internal
    */
    static find(order, index, level, assoc) {
      let maybe = -1;
      for (let i = 0; i < order.length; i++) {
        let span = order[i];
        if (span.from <= index && span.to >= index) {
          if (span.level == level)
            return i;
          if (maybe < 0 || (assoc != 0 ? assoc < 0 ? span.from < index : span.to > index : order[maybe].level > span.level))
            maybe = i;
        }
      }
      if (maybe < 0)
        throw new RangeError("Index out of range");
      return maybe;
    }
  };
  function isolatesEq(a, b) {
    if (a.length != b.length)
      return false;
    for (let i = 0; i < a.length; i++) {
      let iA = a[i], iB = b[i];
      if (iA.from != iB.from || iA.to != iB.to || iA.direction != iB.direction || !isolatesEq(iA.inner, iB.inner))
        return false;
    }
    return true;
  }
  var types = [];
  function computeCharTypes(line, rFrom, rTo, isolates, outerType) {
    for (let iI = 0; iI <= isolates.length; iI++) {
      let from = iI ? isolates[iI - 1].to : rFrom, to = iI < isolates.length ? isolates[iI].from : rTo;
      let prevType = iI ? 256 : outerType;
      for (let i = from, prev = prevType, prevStrong = prevType; i < to; i++) {
        let type = charType(line.charCodeAt(i));
        if (type == 512)
          type = prev;
        else if (type == 8 && prevStrong == 4)
          type = 16;
        types[i] = type == 4 ? 2 : type;
        if (type & 7)
          prevStrong = type;
        prev = type;
      }
      for (let i = from, prev = prevType, prevStrong = prevType; i < to; i++) {
        let type = types[i];
        if (type == 128) {
          if (i < to - 1 && prev == types[i + 1] && prev & 24)
            type = types[i] = prev;
          else
            types[i] = 256;
        } else if (type == 64) {
          let end = i + 1;
          while (end < to && types[end] == 64)
            end++;
          let replace2 = i && prev == 8 || end < rTo && types[end] == 8 ? prevStrong == 1 ? 1 : 8 : 256;
          for (let j = i; j < end; j++)
            types[j] = replace2;
          i = end - 1;
        } else if (type == 8 && prevStrong == 1) {
          types[i] = 1;
        }
        prev = type;
        if (type & 7)
          prevStrong = type;
      }
    }
  }
  function processBracketPairs(line, rFrom, rTo, isolates, outerType) {
    let oppositeType = outerType == 1 ? 2 : 1;
    for (let iI = 0, sI = 0, context = 0; iI <= isolates.length; iI++) {
      let from = iI ? isolates[iI - 1].to : rFrom, to = iI < isolates.length ? isolates[iI].from : rTo;
      for (let i = from, ch, br, type; i < to; i++) {
        if (br = Brackets[ch = line.charCodeAt(i)]) {
          if (br < 0) {
            for (let sJ = sI - 3; sJ >= 0; sJ -= 3) {
              if (BracketStack[sJ + 1] == -br) {
                let flags = BracketStack[sJ + 2];
                let type2 = flags & 2 ? outerType : !(flags & 4) ? 0 : flags & 1 ? oppositeType : outerType;
                if (type2)
                  types[i] = types[BracketStack[sJ]] = type2;
                sI = sJ;
                break;
              }
            }
          } else if (BracketStack.length == 189) {
            break;
          } else {
            BracketStack[sI++] = i;
            BracketStack[sI++] = ch;
            BracketStack[sI++] = context;
          }
        } else if ((type = types[i]) == 2 || type == 1) {
          let embed = type == outerType;
          context = embed ? 0 : 1;
          for (let sJ = sI - 3; sJ >= 0; sJ -= 3) {
            let cur2 = BracketStack[sJ + 2];
            if (cur2 & 2)
              break;
            if (embed) {
              BracketStack[sJ + 2] |= 2;
            } else {
              if (cur2 & 4)
                break;
              BracketStack[sJ + 2] |= 4;
            }
          }
        }
      }
    }
  }
  function processNeutrals(rFrom, rTo, isolates, outerType) {
    for (let iI = 0, prev = outerType; iI <= isolates.length; iI++) {
      let from = iI ? isolates[iI - 1].to : rFrom, to = iI < isolates.length ? isolates[iI].from : rTo;
      for (let i = from; i < to; ) {
        let type = types[i];
        if (type == 256) {
          let end = i + 1;
          for (; ; ) {
            if (end == to) {
              if (iI == isolates.length)
                break;
              end = isolates[iI++].to;
              to = iI < isolates.length ? isolates[iI].from : rTo;
            } else if (types[end] == 256) {
              end++;
            } else {
              break;
            }
          }
          let beforeL = prev == 1;
          let afterL = (end < rTo ? types[end] : outerType) == 1;
          let replace2 = beforeL == afterL ? beforeL ? 1 : 2 : outerType;
          for (let j = end, jI = iI, fromJ = jI ? isolates[jI - 1].to : rFrom; j > i; ) {
            if (j == fromJ) {
              j = isolates[--jI].from;
              fromJ = jI ? isolates[jI - 1].to : rFrom;
            }
            types[--j] = replace2;
          }
          i = end;
        } else {
          prev = type;
          i++;
        }
      }
    }
  }
  function emitSpans(line, from, to, level, baseLevel, isolates, order) {
    let ourType = level % 2 ? 2 : 1;
    if (level % 2 == baseLevel % 2) {
      for (let iCh = from, iI = 0; iCh < to; ) {
        let sameDir = true, isNum = false;
        if (iI == isolates.length || iCh < isolates[iI].from) {
          let next = types[iCh];
          if (next != ourType) {
            sameDir = false;
            isNum = next == 16;
          }
        }
        let recurse = !sameDir && ourType == 1 ? [] : null;
        let localLevel = sameDir ? level : level + 1;
        let iScan = iCh;
        run: for (; ; ) {
          if (iI < isolates.length && iScan == isolates[iI].from) {
            if (isNum)
              break run;
            let iso = isolates[iI];
            if (!sameDir)
              for (let upto = iso.to, jI = iI + 1; ; ) {
                if (upto == to)
                  break run;
                if (jI < isolates.length && isolates[jI].from == upto)
                  upto = isolates[jI++].to;
                else if (types[upto] == ourType)
                  break run;
                else
                  break;
              }
            iI++;
            if (recurse) {
              recurse.push(iso);
            } else {
              if (iso.from > iCh)
                order.push(new BidiSpan(iCh, iso.from, localLevel));
              let dirSwap = iso.direction == LTR != !(localLevel % 2);
              computeSectionOrder(line, dirSwap ? level + 1 : level, baseLevel, iso.inner, iso.from, iso.to, order);
              iCh = iso.to;
            }
            iScan = iso.to;
          } else if (iScan == to || (sameDir ? types[iScan] != ourType : types[iScan] == ourType)) {
            break;
          } else {
            iScan++;
          }
        }
        if (recurse)
          emitSpans(line, iCh, iScan, level + 1, baseLevel, recurse, order);
        else if (iCh < iScan)
          order.push(new BidiSpan(iCh, iScan, localLevel));
        iCh = iScan;
      }
    } else {
      for (let iCh = to, iI = isolates.length; iCh > from; ) {
        let sameDir = true, isNum = false;
        if (!iI || iCh > isolates[iI - 1].to) {
          let next = types[iCh - 1];
          if (next != ourType) {
            sameDir = false;
            isNum = next == 16;
          }
        }
        let recurse = !sameDir && ourType == 1 ? [] : null;
        let localLevel = sameDir ? level : level + 1;
        let iScan = iCh;
        run: for (; ; ) {
          if (iI && iScan == isolates[iI - 1].to) {
            if (isNum)
              break run;
            let iso = isolates[--iI];
            if (!sameDir)
              for (let upto = iso.from, jI = iI; ; ) {
                if (upto == from)
                  break run;
                if (jI && isolates[jI - 1].to == upto)
                  upto = isolates[--jI].from;
                else if (types[upto - 1] == ourType)
                  break run;
                else
                  break;
              }
            if (recurse) {
              recurse.push(iso);
            } else {
              if (iso.to < iCh)
                order.push(new BidiSpan(iso.to, iCh, localLevel));
              let dirSwap = iso.direction == LTR != !(localLevel % 2);
              computeSectionOrder(line, dirSwap ? level + 1 : level, baseLevel, iso.inner, iso.from, iso.to, order);
              iCh = iso.from;
            }
            iScan = iso.from;
          } else if (iScan == from || (sameDir ? types[iScan - 1] != ourType : types[iScan - 1] == ourType)) {
            break;
          } else {
            iScan--;
          }
        }
        if (recurse)
          emitSpans(line, iScan, iCh, level + 1, baseLevel, recurse, order);
        else if (iScan < iCh)
          order.push(new BidiSpan(iScan, iCh, localLevel));
        iCh = iScan;
      }
    }
  }
  function computeSectionOrder(line, level, baseLevel, isolates, from, to, order) {
    let outerType = level % 2 ? 2 : 1;
    computeCharTypes(line, from, to, isolates, outerType);
    processBracketPairs(line, from, to, isolates, outerType);
    processNeutrals(from, to, isolates, outerType);
    emitSpans(line, from, to, level, baseLevel, isolates, order);
  }
  function computeOrder(line, direction, isolates) {
    if (!line)
      return [new BidiSpan(0, 0, direction == RTL ? 1 : 0)];
    if (direction == LTR && !isolates.length && !BidiRE.test(line))
      return trivialOrder(line.length);
    if (isolates.length)
      while (line.length > types.length)
        types[types.length] = 256;
    let order = [], level = direction == LTR ? 0 : 1;
    computeSectionOrder(line, level, level, isolates, 0, line.length, order);
    return order;
  }
  function trivialOrder(length) {
    return [new BidiSpan(0, length, 0)];
  }
  var movedOver = "";
  function moveVisually(line, order, dir, start, forward) {
    var _a2;
    let startIndex = start.head - line.from;
    let spanI = BidiSpan.find(order, startIndex, (_a2 = start.bidiLevel) !== null && _a2 !== void 0 ? _a2 : -1, start.assoc);
    let span = order[spanI], spanEnd = span.side(forward, dir);
    if (startIndex == spanEnd) {
      let nextI = spanI += forward ? 1 : -1;
      if (nextI < 0 || nextI >= order.length)
        return null;
      span = order[spanI = nextI];
      startIndex = span.side(!forward, dir);
      spanEnd = span.side(forward, dir);
    }
    let nextIndex = findClusterBreak2(line.text, startIndex, span.forward(forward, dir));
    if (nextIndex < span.from || nextIndex > span.to)
      nextIndex = spanEnd;
    movedOver = line.text.slice(Math.min(startIndex, nextIndex), Math.max(startIndex, nextIndex));
    let nextSpan = spanI == (forward ? order.length - 1 : 0) ? null : order[spanI + (forward ? 1 : -1)];
    if (nextSpan && nextIndex == spanEnd && nextSpan.level + (forward ? 0 : 1) < span.level)
      return EditorSelection.cursor(nextSpan.side(!forward, dir) + line.from, nextSpan.forward(forward, dir) ? 1 : -1, nextSpan.level);
    return EditorSelection.cursor(nextIndex + line.from, span.forward(forward, dir) ? -1 : 1, span.level);
  }
  function autoDirection(text, from, to) {
    for (let i = from; i < to; i++) {
      let type = charType(text.charCodeAt(i));
      if (type == 1)
        return LTR;
      if (type == 2 || type == 4)
        return RTL;
    }
    return LTR;
  }
  var clickAddsSelectionRange = /* @__PURE__ */ Facet.define();
  var dragMovesSelection$1 = /* @__PURE__ */ Facet.define();
  var mouseSelectionStyle = /* @__PURE__ */ Facet.define();
  var exceptionSink = /* @__PURE__ */ Facet.define();
  var updateListener = /* @__PURE__ */ Facet.define();
  var inputHandler = /* @__PURE__ */ Facet.define();
  var focusChangeEffect = /* @__PURE__ */ Facet.define();
  var clipboardInputFilter = /* @__PURE__ */ Facet.define();
  var clipboardOutputFilter = /* @__PURE__ */ Facet.define();
  var perLineTextDirection = /* @__PURE__ */ Facet.define({
    combine: (values) => values.some((x) => x)
  });
  var nativeSelectionHidden = /* @__PURE__ */ Facet.define({
    combine: (values) => values.some((x) => x)
  });
  var scrollHandler = /* @__PURE__ */ Facet.define();
  var ScrollTarget = class _ScrollTarget {
    constructor(range, y = "nearest", x = "nearest", yMargin = 5, xMargin = 5, isSnapshot = false) {
      this.range = range;
      this.y = y;
      this.x = x;
      this.yMargin = yMargin;
      this.xMargin = xMargin;
      this.isSnapshot = isSnapshot;
    }
    map(changes) {
      return changes.empty ? this : new _ScrollTarget(this.range.map(changes), this.y, this.x, this.yMargin, this.xMargin, this.isSnapshot);
    }
    clip(state) {
      return this.range.to <= state.doc.length ? this : new _ScrollTarget(EditorSelection.cursor(state.doc.length), this.y, this.x, this.yMargin, this.xMargin, this.isSnapshot);
    }
  };
  var scrollIntoView = /* @__PURE__ */ StateEffect.define({ map: (t2, ch) => t2.map(ch) });
  var setEditContextFormatting = /* @__PURE__ */ StateEffect.define();
  function logException(state, exception, context) {
    let handler = state.facet(exceptionSink);
    if (handler.length)
      handler[0](exception);
    else if (window.onerror && window.onerror(String(exception), context, void 0, void 0, exception)) ;
    else if (context)
      console.error(context + ":", exception);
    else
      console.error(exception);
  }
  var editable = /* @__PURE__ */ Facet.define({ combine: (values) => values.length ? values[0] : true });
  var nextPluginID = 0;
  var viewPlugin = /* @__PURE__ */ Facet.define({
    combine(plugins) {
      return plugins.filter((p, i) => {
        for (let j = 0; j < i; j++)
          if (plugins[j].plugin == p.plugin)
            return false;
        return true;
      });
    }
  });
  var ViewPlugin = class _ViewPlugin {
    constructor(id, create, domEventHandlers, domEventObservers, buildExtensions) {
      this.id = id;
      this.create = create;
      this.domEventHandlers = domEventHandlers;
      this.domEventObservers = domEventObservers;
      this.baseExtensions = buildExtensions(this);
      this.extension = this.baseExtensions.concat(viewPlugin.of({ plugin: this, arg: void 0 }));
    }
    /**
    Create an extension for this plugin with the given argument.
    */
    of(arg) {
      return this.baseExtensions.concat(viewPlugin.of({ plugin: this, arg }));
    }
    /**
    Define a plugin from a constructor function that creates the
    plugin's value, given an editor view.
    */
    static define(create, spec) {
      const { eventHandlers, eventObservers, provide, decorations: deco } = spec || {};
      return new _ViewPlugin(nextPluginID++, create, eventHandlers, eventObservers, (plugin) => {
        let ext = [];
        if (deco)
          ext.push(decorations.of((view) => {
            let pluginInst = view.plugin(plugin);
            return pluginInst ? deco(pluginInst) : Decoration.none;
          }));
        if (provide)
          ext.push(provide(plugin));
        return ext;
      });
    }
    /**
    Create a plugin for a class whose constructor takes a single
    editor view as argument.
    */
    static fromClass(cls, spec) {
      return _ViewPlugin.define((view, arg) => new cls(view, arg), spec);
    }
  };
  var PluginInstance = class {
    constructor(spec) {
      this.spec = spec;
      this.mustUpdate = null;
      this.value = null;
    }
    get plugin() {
      return this.spec && this.spec.plugin;
    }
    update(view) {
      if (!this.value) {
        if (this.spec) {
          try {
            this.value = this.spec.plugin.create(view, this.spec.arg);
          } catch (e) {
            logException(view.state, e, "CodeMirror plugin crashed");
            this.deactivate();
          }
        }
      } else if (this.mustUpdate) {
        let update = this.mustUpdate;
        this.mustUpdate = null;
        if (this.value.update) {
          try {
            this.value.update(update);
          } catch (e) {
            logException(update.state, e, "CodeMirror plugin crashed");
            if (this.value.destroy)
              try {
                this.value.destroy();
              } catch (_) {
              }
            this.deactivate();
          }
        }
      }
      return this;
    }
    destroy(view) {
      var _a2;
      if ((_a2 = this.value) === null || _a2 === void 0 ? void 0 : _a2.destroy) {
        try {
          this.value.destroy();
        } catch (e) {
          logException(view.state, e, "CodeMirror plugin crashed");
        }
      }
    }
    deactivate() {
      this.spec = this.value = null;
    }
  };
  var editorAttributes = /* @__PURE__ */ Facet.define();
  var contentAttributes = /* @__PURE__ */ Facet.define();
  var decorations = /* @__PURE__ */ Facet.define();
  var blockWrappers = /* @__PURE__ */ Facet.define();
  var outerDecorations = /* @__PURE__ */ Facet.define();
  var atomicRanges = /* @__PURE__ */ Facet.define();
  var bidiIsolatedRanges = /* @__PURE__ */ Facet.define();
  function getIsolatedRanges(view, line) {
    let isolates = view.state.facet(bidiIsolatedRanges);
    if (!isolates.length)
      return isolates;
    let sets = isolates.map((i) => i instanceof Function ? i(view) : i);
    let result = [];
    RangeSet.spans(sets, line.from, line.to, {
      point() {
      },
      span(fromDoc, toDoc, active, open) {
        let from = fromDoc - line.from, to = toDoc - line.from;
        let level = result;
        for (let i = active.length - 1; i >= 0; i--, open--) {
          let direction = active[i].spec.bidiIsolate, update;
          if (direction == null)
            direction = autoDirection(line.text, from, to);
          if (open > 0 && level.length && (update = level[level.length - 1]).to == from && update.direction == direction) {
            update.to = to;
            level = update.inner;
          } else {
            let add2 = { from, to, direction, inner: [] };
            level.push(add2);
            level = add2.inner;
          }
        }
      }
    });
    return result;
  }
  var scrollMargins = /* @__PURE__ */ Facet.define();
  function getScrollMargins(view) {
    let left = 0, right = 0, top2 = 0, bottom = 0;
    for (let source of view.state.facet(scrollMargins)) {
      let m = source(view);
      if (m) {
        if (m.left != null)
          left = Math.max(left, m.left);
        if (m.right != null)
          right = Math.max(right, m.right);
        if (m.top != null)
          top2 = Math.max(top2, m.top);
        if (m.bottom != null)
          bottom = Math.max(bottom, m.bottom);
      }
    }
    return { left, right, top: top2, bottom };
  }
  var styleModule = /* @__PURE__ */ Facet.define();
  var ChangedRange = class _ChangedRange {
    constructor(fromA, toA, fromB, toB) {
      this.fromA = fromA;
      this.toA = toA;
      this.fromB = fromB;
      this.toB = toB;
    }
    join(other) {
      return new _ChangedRange(Math.min(this.fromA, other.fromA), Math.max(this.toA, other.toA), Math.min(this.fromB, other.fromB), Math.max(this.toB, other.toB));
    }
    addToSet(set) {
      let i = set.length, me = this;
      for (; i > 0; i--) {
        let range = set[i - 1];
        if (range.fromA > me.toA)
          continue;
        if (range.toA < me.fromA)
          break;
        me = me.join(range);
        set.splice(i - 1, 1);
      }
      set.splice(i, 0, me);
      return set;
    }
    // Extend a set to cover all the content in `ranges`, which is a
    // flat array with each pair of numbers representing fromB/toB
    // positions. These pairs are generated in unchanged ranges, so the
    // offset between doc A and doc B is the same for their start and
    // end points.
    static extendWithRanges(diff, ranges) {
      if (ranges.length == 0)
        return diff;
      let result = [];
      for (let dI = 0, rI = 0, off = 0; ; ) {
        let nextD = dI < diff.length ? diff[dI].fromB : 1e9;
        let nextR = rI < ranges.length ? ranges[rI] : 1e9;
        let fromB = Math.min(nextD, nextR);
        if (fromB == 1e9)
          break;
        let fromA = fromB + off, toB = fromB, toA = fromA;
        for (; ; ) {
          if (rI < ranges.length && ranges[rI] <= toB) {
            let end = ranges[rI + 1];
            rI += 2;
            toB = Math.max(toB, end);
            for (let i = dI; i < diff.length && diff[i].fromB <= toB; i++)
              off = diff[i].toA - diff[i].toB;
            toA = Math.max(toA, end + off);
          } else if (dI < diff.length && diff[dI].fromB <= toB) {
            let next = diff[dI++];
            toB = Math.max(toB, next.toB);
            toA = Math.max(toA, next.toA);
            off = next.toA - next.toB;
          } else {
            break;
          }
        }
        result.push(new _ChangedRange(fromA, toA, fromB, toB));
      }
      return result;
    }
  };
  var ViewUpdate = class _ViewUpdate {
    constructor(view, state, transactions) {
      this.view = view;
      this.state = state;
      this.transactions = transactions;
      this.flags = 0;
      this.startState = view.state;
      this.changes = ChangeSet.empty(this.startState.doc.length);
      for (let tr of transactions)
        this.changes = this.changes.compose(tr.changes);
      let changedRanges = [];
      this.changes.iterChangedRanges((fromA, toA, fromB, toB) => changedRanges.push(new ChangedRange(fromA, toA, fromB, toB)));
      this.changedRanges = changedRanges;
    }
    /**
    @internal
    */
    static create(view, state, transactions) {
      return new _ViewUpdate(view, state, transactions);
    }
    /**
    Tells you whether the [viewport](https://codemirror.net/6/docs/ref/#view.EditorView.viewport) or
    [visible ranges](https://codemirror.net/6/docs/ref/#view.EditorView.visibleRanges) changed in this
    update.
    */
    get viewportChanged() {
      return (this.flags & 4) > 0;
    }
    /**
    Returns true when
    [`viewportChanged`](https://codemirror.net/6/docs/ref/#view.ViewUpdate.viewportChanged) is true
    and the viewport change is not just the result of mapping it in
    response to document changes.
    */
    get viewportMoved() {
      return (this.flags & 8) > 0;
    }
    /**
    Indicates whether the height of a block element in the editor
    changed in this update.
    */
    get heightChanged() {
      return (this.flags & 2) > 0;
    }
    /**
    Returns true when the document was modified or the size of the
    editor, or elements within the editor, changed.
    */
    get geometryChanged() {
      return this.docChanged || (this.flags & (16 | 2)) > 0;
    }
    /**
    True when this update indicates a focus change.
    */
    get focusChanged() {
      return (this.flags & 1) > 0;
    }
    /**
    Whether the document changed in this update.
    */
    get docChanged() {
      return !this.changes.empty;
    }
    /**
    Whether the selection was explicitly set in this update.
    */
    get selectionSet() {
      return this.transactions.some((tr) => tr.selection);
    }
    /**
    @internal
    */
    get empty() {
      return this.flags == 0 && this.transactions.length == 0;
    }
  };
  var noChildren = [];
  var Tile = class {
    constructor(dom, length, flags = 0) {
      this.dom = dom;
      this.length = length;
      this.flags = flags;
      this.parent = null;
      dom.cmTile = this;
    }
    get breakAfter() {
      return this.flags & 1;
    }
    get children() {
      return noChildren;
    }
    isWidget() {
      return false;
    }
    get isHidden() {
      return false;
    }
    isComposite() {
      return false;
    }
    isLine() {
      return false;
    }
    isText() {
      return false;
    }
    isBlock() {
      return false;
    }
    get domAttrs() {
      return null;
    }
    sync(track) {
      this.flags |= 2;
      if (this.flags & 4) {
        this.flags &= ~4;
        let attrs = this.domAttrs;
        if (attrs)
          setAttrs(this.dom, attrs);
      }
    }
    toString() {
      return this.constructor.name + (this.children.length ? `(${this.children})` : "") + (this.breakAfter ? "#" : "");
    }
    destroy() {
      this.parent = null;
    }
    setDOM(dom) {
      this.dom = dom;
      dom.cmTile = this;
    }
    get posAtStart() {
      return this.parent ? this.parent.posBefore(this) : 0;
    }
    get posAtEnd() {
      return this.posAtStart + this.length;
    }
    posBefore(tile, start = this.posAtStart) {
      let pos = start;
      for (let child of this.children) {
        if (child == tile)
          return pos;
        pos += child.length + child.breakAfter;
      }
      throw new RangeError("Invalid child in posBefore");
    }
    posAfter(tile) {
      return this.posBefore(tile) + tile.length;
    }
    covers(side) {
      return true;
    }
    coordsIn(pos, side) {
      return null;
    }
    domPosFor(off, side) {
      let index = domIndex(this.dom);
      let after = this.length ? off > 0 : side > 0;
      return new DOMPos(this.parent.dom, index + (after ? 1 : 0), off == 0 || off == this.length);
    }
    markDirty(attrs) {
      this.flags &= ~2;
      if (attrs)
        this.flags |= 4;
      if (this.parent && this.parent.flags & 2)
        this.parent.markDirty(false);
    }
    get overrideDOMText() {
      return null;
    }
    get root() {
      for (let t2 = this; t2; t2 = t2.parent)
        if (t2 instanceof DocTile)
          return t2;
      return null;
    }
    static get(dom) {
      return dom.cmTile;
    }
  };
  var CompositeTile = class extends Tile {
    constructor(dom) {
      super(dom, 0);
      this._children = [];
    }
    isComposite() {
      return true;
    }
    get children() {
      return this._children;
    }
    get lastChild() {
      return this.children.length ? this.children[this.children.length - 1] : null;
    }
    append(child) {
      this.children.push(child);
      child.parent = this;
    }
    sync(track) {
      if (this.flags & 2)
        return;
      super.sync(track);
      let parent = this.dom, prev = null, next;
      let tracking = (track === null || track === void 0 ? void 0 : track.node) == parent ? track : null;
      let length = 0;
      for (let child of this.children) {
        child.sync(track);
        length += child.length + child.breakAfter;
        next = prev ? prev.nextSibling : parent.firstChild;
        if (tracking && next != child.dom)
          tracking.written = true;
        if (child.dom.parentNode == parent) {
          while (next && next != child.dom)
            next = rm$1(next);
        } else {
          parent.insertBefore(child.dom, next);
        }
        prev = child.dom;
      }
      next = prev ? prev.nextSibling : parent.firstChild;
      if (tracking && next)
        tracking.written = true;
      while (next)
        next = rm$1(next);
      this.length = length;
    }
  };
  function rm$1(dom) {
    let next = dom.nextSibling;
    dom.parentNode.removeChild(dom);
    return next;
  }
  var DocTile = class extends CompositeTile {
    constructor(view, dom) {
      super(dom);
      this.view = view;
    }
    owns(tile) {
      for (; tile; tile = tile.parent)
        if (tile == this)
          return true;
      return false;
    }
    isBlock() {
      return true;
    }
    nearest(dom) {
      for (; ; ) {
        if (!dom)
          return null;
        let tile = Tile.get(dom);
        if (tile && this.owns(tile))
          return tile;
        dom = dom.parentNode;
      }
    }
    blockTiles(f) {
      for (let stack = [], cur2 = this, i = 0, pos = 0; ; ) {
        if (i == cur2.children.length) {
          if (!stack.length)
            return;
          cur2 = cur2.parent;
          if (cur2.breakAfter)
            pos++;
          i = stack.pop();
        } else {
          let next = cur2.children[i++];
          if (next instanceof BlockWrapperTile) {
            stack.push(i);
            cur2 = next;
            i = 0;
          } else {
            let end = pos + next.length;
            let result = f(next, pos);
            if (result !== void 0)
              return result;
            pos = end + next.breakAfter;
          }
        }
      }
    }
    // Find the block at the given position. If side < -1, make sure to
    // stay before block widgets at that position, if side > 1, after
    // such widgets (used for selection drawing, which needs to be able
    // to get coordinates for positions that aren't valid cursor positions).
    resolveBlock(pos, side) {
      let before, beforeOff = -1, after, afterOff = -1;
      this.blockTiles((tile, off) => {
        let end = off + tile.length;
        if (pos >= off && pos <= end) {
          if (tile.isWidget() && side >= -1 && side <= 1) {
            if (tile.flags & 32)
              return true;
            if (tile.flags & 16)
              before = void 0;
          }
          if ((off < pos || pos == end && (side < -1 ? tile.length : tile.covers(1))) && (!before || !tile.isWidget() && before.isWidget())) {
            before = tile;
            beforeOff = pos - off;
          }
          if ((end > pos || pos == off && (side > 1 ? tile.length : tile.covers(-1))) && (!after || !tile.isWidget() && after.isWidget())) {
            after = tile;
            afterOff = pos - off;
          }
        }
      });
      if (!before && !after)
        throw new Error("No tile at position " + pos);
      return before && side < 0 || !after ? { tile: before, offset: beforeOff } : { tile: after, offset: afterOff };
    }
  };
  var BlockWrapperTile = class _BlockWrapperTile extends CompositeTile {
    constructor(dom, wrapper) {
      super(dom);
      this.wrapper = wrapper;
    }
    isBlock() {
      return true;
    }
    covers(side) {
      if (!this.children.length)
        return false;
      return side < 0 ? this.children[0].covers(-1) : this.lastChild.covers(1);
    }
    get domAttrs() {
      return this.wrapper.attributes;
    }
    static of(wrapper, dom) {
      let tile = new _BlockWrapperTile(dom || document.createElement(wrapper.tagName), wrapper);
      if (!dom)
        tile.flags |= 4;
      return tile;
    }
  };
  var LineTile = class _LineTile extends CompositeTile {
    constructor(dom, attrs) {
      super(dom);
      this.attrs = attrs;
    }
    isLine() {
      return true;
    }
    static start(attrs, dom, keepAttrs) {
      let line = new _LineTile(dom || document.createElement("div"), attrs);
      if (!dom || !keepAttrs)
        line.flags |= 4;
      return line;
    }
    get domAttrs() {
      return this.attrs;
    }
    // Find the tile associated with a given position in this line.
    resolveInline(pos, side, forCoords) {
      let before = null, beforeOff = -1, after = null, afterOff = -1;
      function scan(tile, pos2) {
        for (let i = 0, off = 0; i < tile.children.length && off <= pos2; i++) {
          let child = tile.children[i], end = off + child.length;
          if (end >= pos2) {
            if (child.isComposite()) {
              scan(child, pos2 - off);
            } else if ((!after || after.isHidden && (side > 0 || forCoords && onSameLine(after, child))) && (end > pos2 || child.flags & 32)) {
              after = child;
              afterOff = pos2 - off;
            } else if (off < pos2 || child.flags & 16 && !child.isHidden) {
              before = child;
              beforeOff = pos2 - off;
            }
          }
          off = end;
        }
      }
      scan(this, pos);
      let target = (side < 0 ? before : after) || before || after;
      return target ? { tile: target, offset: target == before ? beforeOff : afterOff } : null;
    }
    coordsIn(pos, side) {
      let found = this.resolveInline(pos, side, true);
      if (!found)
        return fallbackRect(this);
      return found.tile.coordsIn(Math.max(0, found.offset), side);
    }
    domIn(pos, side) {
      let found = this.resolveInline(pos, side);
      if (found) {
        let { tile, offset } = found;
        if (this.dom.contains(tile.dom)) {
          if (tile.isText())
            return new DOMPos(tile.dom, Math.min(tile.dom.nodeValue.length, offset));
          return tile.domPosFor(offset, tile.flags & 16 ? 1 : tile.flags & 32 ? -1 : side);
        }
        let parent = found.tile.parent, saw = false;
        for (let ch of parent.children) {
          if (saw)
            return new DOMPos(ch.dom, 0);
          if (ch == found.tile) {
            saw = true;
          }
        }
      }
      return new DOMPos(this.dom, 0);
    }
  };
  function fallbackRect(tile) {
    let last = tile.dom.lastChild;
    if (!last)
      return tile.dom.getBoundingClientRect();
    let rects = clientRectsFor(last);
    return rects[rects.length - 1] || null;
  }
  function onSameLine(a, b) {
    let posA = a.coordsIn(0, 1), posB = b.coordsIn(0, 1);
    return posA && posB && posB.top < posA.bottom;
  }
  var MarkTile = class _MarkTile extends CompositeTile {
    constructor(dom, mark) {
      super(dom);
      this.mark = mark;
    }
    get domAttrs() {
      return this.mark.attrs;
    }
    static of(mark, dom) {
      let tile = new _MarkTile(dom || document.createElement(mark.tagName), mark);
      if (!dom)
        tile.flags |= 4;
      return tile;
    }
  };
  var TextTile = class _TextTile extends Tile {
    constructor(dom, text) {
      super(dom, text.length);
      this.text = text;
    }
    sync(track) {
      if (this.flags & 2)
        return;
      super.sync(track);
      if (this.dom.nodeValue != this.text) {
        if (track && track.node == this.dom)
          track.written = true;
        this.dom.nodeValue = this.text;
      }
    }
    isText() {
      return true;
    }
    toString() {
      return JSON.stringify(this.text);
    }
    coordsIn(pos, side) {
      let length = this.dom.nodeValue.length;
      if (pos > length)
        pos = length;
      let from = pos, to = pos, flatten2 = 0;
      if (pos == 0 && side < 0 || pos == length && side >= 0) {
        if (!(browser.chrome || browser.gecko)) {
          if (pos) {
            from--;
            flatten2 = 1;
          } else if (to < length) {
            to++;
            flatten2 = -1;
          }
        }
      } else {
        if (side < 0)
          from--;
        else if (to < length)
          to++;
      }
      let rects = textRange(this.dom, from, to).getClientRects();
      if (!rects.length)
        return null;
      let rect = rects[(flatten2 ? flatten2 < 0 : side >= 0) ? 0 : rects.length - 1];
      if (browser.safari && !flatten2 && rect.width == 0)
        rect = Array.prototype.find.call(rects, (r) => r.width) || rect;
      return flatten2 ? flattenRect(rect, flatten2 < 0) : rect || null;
    }
    static of(text, dom) {
      let tile = new _TextTile(dom || document.createTextNode(text), text);
      if (!dom)
        tile.flags |= 2;
      return tile;
    }
  };
  var WidgetTile = class _WidgetTile extends Tile {
    constructor(dom, length, widget, flags) {
      super(dom, length, flags);
      this.widget = widget;
    }
    isWidget() {
      return true;
    }
    get isHidden() {
      return this.widget.isHidden;
    }
    covers(side) {
      if (this.flags & 48)
        return false;
      return (this.flags & (side < 0 ? 64 : 128)) > 0;
    }
    coordsIn(pos, side) {
      return this.coordsInWidget(pos, side, false);
    }
    coordsInWidget(pos, side, block) {
      let custom = this.widget.coordsAt(this.dom, pos, side);
      if (custom)
        return custom;
      if (block) {
        return flattenRect(this.dom.getBoundingClientRect(), this.length ? pos == 0 : side <= 0);
      } else {
        let rects = this.dom.getClientRects(), rect = null;
        if (!rects.length)
          return null;
        let fromBack = this.flags & 16 ? true : this.flags & 32 ? false : pos > 0;
        for (let i = fromBack ? rects.length - 1 : 0; ; i += fromBack ? -1 : 1) {
          rect = rects[i];
          if (pos > 0 ? i == 0 : i == rects.length - 1 || rect.top < rect.bottom)
            break;
        }
        return flattenRect(rect, !fromBack);
      }
    }
    get overrideDOMText() {
      if (!this.length)
        return Text.empty;
      let { root } = this;
      if (!root)
        return Text.empty;
      let start = this.posAtStart;
      return root.view.state.doc.slice(start, start + this.length);
    }
    destroy() {
      super.destroy();
      this.widget.destroy(this.dom);
    }
    static of(widget, view, length, flags, dom) {
      if (!dom) {
        dom = widget.toDOM(view);
        if (!widget.editable)
          dom.contentEditable = "false";
      }
      return new _WidgetTile(dom, length, widget, flags);
    }
  };
  var WidgetBufferTile = class extends Tile {
    constructor(flags) {
      let img = document.createElement("img");
      img.className = "cm-widgetBuffer";
      img.setAttribute("aria-hidden", "true");
      super(img, 0, flags);
    }
    get isHidden() {
      return true;
    }
    get overrideDOMText() {
      return Text.empty;
    }
    coordsIn(pos) {
      return this.dom.getBoundingClientRect();
    }
  };
  var TilePointer = class {
    constructor(top2) {
      this.index = 0;
      this.beforeBreak = false;
      this.parents = [];
      this.tile = top2;
    }
    // Advance by the given distance. If side is -1, stop leaving or
    // entering tiles, or skipping zero-length tiles, once the distance
    // has been traversed. When side is 1, leave, enter, or skip
    // everything at the end position.
    advance(dist2, side, walker) {
      let { tile, index, beforeBreak, parents } = this;
      while (dist2 || side > 0) {
        if (!tile.isComposite()) {
          if (index == tile.length) {
            beforeBreak = !!tile.breakAfter;
            ({ tile, index } = parents.pop());
            index++;
          } else if (!dist2) {
            break;
          } else {
            let take = Math.min(dist2, tile.length - index);
            if (walker)
              walker.skip(tile, index, index + take);
            dist2 -= take;
            index += take;
          }
        } else if (beforeBreak) {
          if (!dist2)
            break;
          if (walker)
            walker.break();
          dist2--;
          beforeBreak = false;
        } else if (index == tile.children.length) {
          if (!dist2 && !parents.length)
            break;
          if (walker)
            walker.leave(tile);
          beforeBreak = !!tile.breakAfter;
          ({ tile, index } = parents.pop());
          index++;
        } else {
          let next = tile.children[index], brk = next.breakAfter;
          if ((side > 0 ? next.length <= dist2 : next.length < dist2) && (!walker || walker.skip(next, 0, next.length) !== false || !next.isComposite)) {
            beforeBreak = !!brk;
            index++;
            dist2 -= next.length;
          } else {
            parents.push({ tile, index });
            tile = next;
            index = 0;
            if (walker && next.isComposite())
              walker.enter(next);
          }
        }
      }
      this.tile = tile;
      this.index = index;
      this.beforeBreak = beforeBreak;
      return this;
    }
    get root() {
      return this.parents.length ? this.parents[0].tile : this.tile;
    }
  };
  var OpenWrapper = class {
    constructor(from, to, wrapper, rank) {
      this.from = from;
      this.to = to;
      this.wrapper = wrapper;
      this.rank = rank;
    }
  };
  var TileBuilder = class {
    constructor(cache, root, blockWrappers2) {
      this.cache = cache;
      this.root = root;
      this.blockWrappers = blockWrappers2;
      this.curLine = null;
      this.lastBlock = null;
      this.afterWidget = null;
      this.pos = 0;
      this.wrappers = [];
      this.wrapperPos = 0;
    }
    addText(text, marks2, openStart, tile) {
      var _a2;
      this.flushBuffer();
      let parent = this.ensureMarks(marks2, openStart);
      let prev = parent.lastChild;
      if (prev && prev.isText() && !(prev.flags & 8) && prev.length + text.length < 512) {
        this.cache.reused.set(
          prev,
          2
          /* Reused.DOM */
        );
        let tile2 = parent.children[parent.children.length - 1] = new TextTile(prev.dom, prev.text + text);
        tile2.parent = parent;
      } else {
        parent.append(tile || TextTile.of(text, (_a2 = this.cache.find(TextTile)) === null || _a2 === void 0 ? void 0 : _a2.dom));
      }
      this.pos += text.length;
      this.afterWidget = null;
    }
    addComposition(composition, context) {
      let line = this.curLine;
      if (line.dom != context.line.dom) {
        line.setDOM(this.cache.reused.has(context.line) ? freeNode(context.line.dom) : context.line.dom);
        this.cache.reused.set(
          context.line,
          2
          /* Reused.DOM */
        );
      }
      let head = line;
      for (let i = context.marks.length - 1; i >= 0; i--) {
        let mark = context.marks[i];
        let last = head.lastChild;
        if (last instanceof MarkTile && last.mark.eq(mark.mark)) {
          if (last.dom != mark.dom)
            last.setDOM(freeNode(mark.dom));
          head = last;
        } else {
          if (this.cache.reused.get(mark)) {
            let tile = Tile.get(mark.dom);
            if (tile)
              tile.setDOM(freeNode(mark.dom));
          }
          let nw = MarkTile.of(mark.mark, mark.dom);
          head.append(nw);
          head = nw;
        }
        this.cache.reused.set(
          mark,
          2
          /* Reused.DOM */
        );
      }
      let oldTile = Tile.get(composition.text);
      if (oldTile)
        this.cache.reused.set(
          oldTile,
          2
          /* Reused.DOM */
        );
      let text = new TextTile(composition.text, composition.text.nodeValue);
      text.flags |= 8;
      head.append(text);
    }
    addInlineWidget(widget, marks2, openStart) {
      let noSpace = this.afterWidget && widget.flags & 48 && (this.afterWidget.flags & 48) == (widget.flags & 48);
      if (!noSpace)
        this.flushBuffer();
      let parent = this.ensureMarks(marks2, openStart);
      if (!noSpace && !(widget.flags & 16))
        parent.append(this.getBuffer(1));
      parent.append(widget);
      this.pos += widget.length;
      this.afterWidget = widget;
    }
    addMark(tile, marks2, openStart) {
      this.flushBuffer();
      let parent = this.ensureMarks(marks2, openStart);
      parent.append(tile);
      this.pos += tile.length;
      this.afterWidget = null;
    }
    addBlockWidget(widget) {
      this.getBlockPos().append(widget);
      this.pos += widget.length;
      this.lastBlock = widget;
      this.endLine();
    }
    continueWidget(length) {
      let widget = this.afterWidget || this.lastBlock;
      widget.length += length;
      this.pos += length;
    }
    addLineStart(attrs, dom) {
      var _a2;
      if (!attrs)
        attrs = lineBaseAttrs;
      let tile = LineTile.start(attrs, dom || ((_a2 = this.cache.find(LineTile)) === null || _a2 === void 0 ? void 0 : _a2.dom), !!dom);
      this.getBlockPos().append(this.lastBlock = this.curLine = tile);
    }
    addLine(tile) {
      this.getBlockPos().append(tile);
      this.pos += tile.length;
      this.lastBlock = tile;
      this.endLine();
    }
    addBreak() {
      this.lastBlock.flags |= 1;
      this.endLine();
      this.pos++;
    }
    addLineStartIfNotCovered(attrs) {
      if (!this.blockPosCovered())
        this.addLineStart(attrs);
    }
    ensureLine(attrs) {
      if (!this.curLine)
        this.addLineStart(attrs);
    }
    ensureMarks(marks2, openStart) {
      var _a2;
      let parent = this.curLine;
      for (let i = marks2.length - 1; i >= 0; i--) {
        let mark = marks2[i], last;
        if (openStart > 0 && (last = parent.lastChild) && last instanceof MarkTile && last.mark.eq(mark)) {
          parent = last;
          openStart--;
        } else {
          let tile = MarkTile.of(mark, (_a2 = this.cache.find(MarkTile, (m) => m.mark.eq(mark))) === null || _a2 === void 0 ? void 0 : _a2.dom);
          parent.append(tile);
          parent = tile;
          openStart = 0;
        }
      }
      return parent;
    }
    endLine() {
      if (this.curLine) {
        this.flushBuffer();
        let last = this.curLine.lastChild;
        if (!last || !hasContent(this.curLine, false) || last.dom.nodeName != "BR" && last.isWidget() && !(browser.ios && hasContent(this.curLine, true)))
          this.curLine.append(this.cache.findWidget(
            BreakWidget,
            0,
            32
            /* TileFlag.After */
          ) || new WidgetTile(
            BreakWidget.toDOM(),
            0,
            BreakWidget,
            32
            /* TileFlag.After */
          ));
        this.curLine = this.afterWidget = null;
      }
    }
    updateBlockWrappers() {
      if (this.wrapperPos > this.pos + 1e4) {
        this.blockWrappers.goto(this.pos);
        this.wrappers.length = 0;
      }
      for (let i = this.wrappers.length - 1; i >= 0; i--)
        if (this.wrappers[i].to < this.pos)
          this.wrappers.splice(i, 1);
      for (let cur2 = this.blockWrappers; cur2.value && cur2.from <= this.pos; cur2.next())
        if (cur2.to >= this.pos) {
          let wrap = new OpenWrapper(cur2.from, cur2.to, cur2.value, cur2.rank), i = this.wrappers.length;
          while (i > 0 && (this.wrappers[i - 1].rank - wrap.rank || this.wrappers[i - 1].to - wrap.to) < 0)
            i--;
          this.wrappers.splice(i, 0, wrap);
        }
      this.wrapperPos = this.pos;
    }
    getBlockPos() {
      var _a2;
      this.updateBlockWrappers();
      let parent = this.root;
      for (let wrap of this.wrappers) {
        let last = parent.lastChild;
        if (wrap.from < this.pos && last instanceof BlockWrapperTile && last.wrapper.eq(wrap.wrapper)) {
          parent = last;
        } else {
          let tile = BlockWrapperTile.of(wrap.wrapper, (_a2 = this.cache.find(BlockWrapperTile, (t2) => t2.wrapper.eq(wrap.wrapper))) === null || _a2 === void 0 ? void 0 : _a2.dom);
          parent.append(tile);
          parent = tile;
        }
      }
      return parent;
    }
    blockPosCovered() {
      let last = this.lastBlock;
      return last != null && !last.breakAfter && (!last.isWidget() || (last.flags & (32 | 128)) > 0);
    }
    getBuffer(side) {
      let flags = 2 | (side < 0 ? 16 : 32);
      let found = this.cache.find(
        WidgetBufferTile,
        void 0,
        1
        /* Reused.Full */
      );
      if (found)
        found.flags = flags;
      return found || new WidgetBufferTile(flags);
    }
    flushBuffer() {
      if (this.afterWidget && !(this.afterWidget.flags & 32)) {
        this.afterWidget.parent.append(this.getBuffer(-1));
        this.afterWidget = null;
      }
    }
  };
  var TextStream = class {
    constructor(doc2) {
      this.skipCount = 0;
      this.text = "";
      this.textOff = 0;
      this.cursor = doc2.iter();
    }
    skip(len) {
      if (this.textOff + len <= this.text.length) {
        this.textOff += len;
      } else {
        this.skipCount += len - (this.text.length - this.textOff);
        this.text = "";
        this.textOff = 0;
      }
    }
    next(maxLen) {
      if (this.textOff == this.text.length) {
        let { value, lineBreak, done } = this.cursor.next(this.skipCount);
        this.skipCount = 0;
        if (done)
          throw new Error("Ran out of text content when drawing inline views");
        this.text = value;
        let len = this.textOff = Math.min(maxLen, value.length);
        return lineBreak ? null : value.slice(0, len);
      }
      let end = Math.min(this.text.length, this.textOff + maxLen);
      let chars = this.text.slice(this.textOff, end);
      this.textOff = end;
      return chars;
    }
  };
  var buckets = [WidgetTile, LineTile, TextTile, MarkTile, WidgetBufferTile, BlockWrapperTile, DocTile];
  for (let i = 0; i < buckets.length; i++)
    buckets[i].bucket = i;
  var TileCache = class {
    constructor(view) {
      this.view = view;
      this.buckets = buckets.map(() => []);
      this.index = buckets.map(() => 0);
      this.reused = /* @__PURE__ */ new Map();
    }
    // Put a tile in the cache.
    add(tile) {
      let i = tile.constructor.bucket, bucket = this.buckets[i];
      if (bucket.length < 6)
        bucket.push(tile);
      else
        bucket[
          this.index[i] = (this.index[i] + 1) % 6
          /* C.Bucket */
        ] = tile;
    }
    find(cls, test, type = 2) {
      let i = cls.bucket;
      let bucket = this.buckets[i], off = this.index[i];
      for (let j = bucket.length - 1; j >= 0; j--) {
        let index = (j + off) % bucket.length, tile = bucket[index];
        if ((!test || test(tile)) && !this.reused.has(tile)) {
          bucket.splice(index, 1);
          if (index < off)
            this.index[i]--;
          this.reused.set(tile, type);
          return tile;
        }
      }
      return null;
    }
    findWidget(widget, length, flags) {
      let widgets = this.buckets[0];
      if (widgets.length)
        for (let i = 0, pass = 0; ; i++) {
          if (i == widgets.length) {
            if (pass)
              return null;
            pass = 1;
            i = 0;
          }
          let tile = widgets[i];
          if (!this.reused.has(tile) && (pass == 0 ? tile.widget.compare(widget) : tile.widget.constructor == widget.constructor && widget.updateDOM(tile.dom, this.view))) {
            widgets.splice(i, 1);
            if (i < this.index[0])
              this.index[0]--;
            if (tile.widget == widget && tile.length == length && (tile.flags & (496 | 1)) == flags) {
              this.reused.set(
                tile,
                1
                /* Reused.Full */
              );
              return tile;
            } else {
              this.reused.set(
                tile,
                2
                /* Reused.DOM */
              );
              return new WidgetTile(tile.dom, length, widget, tile.flags & ~(496 | 1) | flags);
            }
          }
        }
    }
    reuse(tile) {
      this.reused.set(
        tile,
        1
        /* Reused.Full */
      );
      return tile;
    }
    maybeReuse(tile, type = 2) {
      if (this.reused.has(tile))
        return void 0;
      this.reused.set(tile, type);
      return tile.dom;
    }
    clear() {
      for (let i = 0; i < this.buckets.length; i++)
        this.buckets[i].length = this.index[i] = 0;
    }
  };
  var TileUpdate = class {
    constructor(view, old, blockWrappers2, decorations2, disallowBlockEffectsFor) {
      this.view = view;
      this.decorations = decorations2;
      this.disallowBlockEffectsFor = disallowBlockEffectsFor;
      this.openWidget = false;
      this.openMarks = 0;
      this.cache = new TileCache(view);
      this.text = new TextStream(view.state.doc);
      this.builder = new TileBuilder(this.cache, new DocTile(view, view.contentDOM), RangeSet.iter(blockWrappers2));
      this.cache.reused.set(
        old,
        2
        /* Reused.DOM */
      );
      this.old = new TilePointer(old);
      this.reuseWalker = {
        skip: (tile, from, to) => {
          this.cache.add(tile);
          if (tile.isComposite())
            return false;
        },
        enter: (tile) => this.cache.add(tile),
        leave: () => {
        },
        break: () => {
        }
      };
    }
    run(changes, composition) {
      let compositionContext = composition && this.getCompositionContext(composition.text);
      for (let posA = 0, posB = 0, i = 0; ; ) {
        let next = i < changes.length ? changes[i++] : null;
        let skipA = next ? next.fromA : this.old.root.length;
        if (skipA > posA) {
          let len = skipA - posA;
          this.preserve(len, !i, !next);
          posA = skipA;
          posB += len;
        }
        if (!next)
          break;
        if (composition && next.fromA <= composition.range.fromA && next.toA >= composition.range.toA) {
          this.forward(next.fromA, composition.range.fromA, composition.range.fromA < composition.range.toA ? 1 : -1);
          this.emit(posB, composition.range.fromB);
          this.cache.clear();
          this.builder.addComposition(composition, compositionContext);
          this.text.skip(composition.range.toB - composition.range.fromB);
          this.forward(composition.range.fromA, next.toA);
          this.emit(composition.range.toB, next.toB);
        } else {
          this.forward(next.fromA, next.toA);
          this.emit(posB, next.toB);
        }
        posB = next.toB;
        posA = next.toA;
      }
      if (this.builder.curLine)
        this.builder.endLine();
      return this.builder.root;
    }
    preserve(length, incStart, incEnd) {
      let activeMarks = getMarks(this.old), openMarks = this.openMarks;
      this.old.advance(length, incEnd ? 1 : -1, {
        skip: (tile, from, to) => {
          if (tile.isWidget()) {
            if (this.openWidget) {
              this.builder.continueWidget(to - from);
            } else {
              let widget = to > 0 || from < tile.length ? WidgetTile.of(tile.widget, this.view, to - from, tile.flags & 496, this.cache.maybeReuse(tile)) : this.cache.reuse(tile);
              if (widget.flags & 256) {
                widget.flags &= ~1;
                this.builder.addBlockWidget(widget);
              } else {
                this.builder.ensureLine(null);
                this.builder.addInlineWidget(widget, activeMarks, openMarks);
                openMarks = activeMarks.length;
              }
            }
          } else if (tile.isText()) {
            this.builder.ensureLine(null);
            if (!from && to == tile.length && !this.cache.reused.has(tile)) {
              this.builder.addText(tile.text, activeMarks, openMarks, this.cache.reuse(tile));
            } else {
              this.cache.add(tile);
              this.builder.addText(tile.text.slice(from, to), activeMarks, openMarks);
            }
            openMarks = activeMarks.length;
          } else if (tile.isLine()) {
            tile.flags &= ~1;
            this.cache.reused.set(
              tile,
              1
              /* Reused.Full */
            );
            this.builder.addLine(tile);
          } else if (tile instanceof WidgetBufferTile) {
            this.cache.add(tile);
          } else if (tile instanceof MarkTile) {
            this.builder.ensureLine(null);
            this.builder.addMark(tile, activeMarks, openMarks);
            this.cache.reused.set(
              tile,
              1
              /* Reused.Full */
            );
            openMarks = activeMarks.length;
          } else {
            return false;
          }
          this.openWidget = false;
        },
        enter: (tile) => {
          if (tile.isLine()) {
            this.builder.addLineStart(tile.attrs, this.cache.maybeReuse(tile));
          } else {
            this.cache.add(tile);
            if (tile instanceof MarkTile)
              activeMarks.unshift(tile.mark);
          }
          this.openWidget = false;
        },
        leave: (tile) => {
          if (tile.isLine()) {
            if (activeMarks.length)
              activeMarks.length = openMarks = 0;
          } else if (tile instanceof MarkTile) {
            activeMarks.shift();
            openMarks = Math.min(openMarks, activeMarks.length);
          }
        },
        break: () => {
          this.builder.addBreak();
          this.openWidget = false;
        }
      });
      this.text.skip(length);
    }
    emit(from, to) {
      let pendingLineAttrs = null;
      let b = this.builder, markCount = 0;
      let openEnd = RangeSet.spans(this.decorations, from, to, {
        point: (from2, to2, deco, active, openStart, index) => {
          if (deco instanceof PointDecoration) {
            if (this.disallowBlockEffectsFor[index]) {
              if (deco.block)
                throw new RangeError("Block decorations may not be specified via plugins");
              if (to2 > this.view.state.doc.lineAt(from2).to)
                throw new RangeError("Decorations that replace line breaks may not be specified via plugins");
            }
            markCount = active.length;
            if (openStart > active.length) {
              b.continueWidget(to2 - from2);
            } else {
              let widget = deco.widget || (deco.block ? NullWidget.block : NullWidget.inline);
              let flags = widgetFlags(deco);
              let tile = this.cache.findWidget(widget, to2 - from2, flags) || WidgetTile.of(widget, this.view, to2 - from2, flags);
              if (deco.block) {
                if (deco.startSide > 0)
                  b.addLineStartIfNotCovered(pendingLineAttrs);
                b.addBlockWidget(tile);
              } else {
                b.ensureLine(pendingLineAttrs);
                b.addInlineWidget(tile, active, openStart);
              }
            }
            pendingLineAttrs = null;
          } else {
            pendingLineAttrs = addLineDeco(pendingLineAttrs, deco);
          }
          if (to2 > from2)
            this.text.skip(to2 - from2);
        },
        span: (from2, to2, active, openStart) => {
          for (let pos = from2; pos < to2; ) {
            let chars = this.text.next(Math.min(512, to2 - pos));
            if (chars == null) {
              b.addLineStartIfNotCovered(pendingLineAttrs);
              b.addBreak();
              pos++;
            } else {
              b.ensureLine(pendingLineAttrs);
              b.addText(chars, active, pos == from2 ? openStart : active.length);
              pos += chars.length;
            }
            pendingLineAttrs = null;
          }
        }
      });
      b.addLineStartIfNotCovered(pendingLineAttrs);
      this.openWidget = openEnd > markCount;
      this.openMarks = openEnd;
    }
    forward(from, to, side = 1) {
      if (to - from <= 10) {
        this.old.advance(to - from, side, this.reuseWalker);
      } else {
        this.old.advance(5, -1, this.reuseWalker);
        this.old.advance(to - from - 10, -1);
        this.old.advance(5, side, this.reuseWalker);
      }
    }
    getCompositionContext(text) {
      let marks2 = [], line = null;
      for (let parent = text.parentNode; ; parent = parent.parentNode) {
        let tile = Tile.get(parent);
        if (parent == this.view.contentDOM)
          break;
        if (tile instanceof MarkTile)
          marks2.push(tile);
        else if (tile === null || tile === void 0 ? void 0 : tile.isLine())
          line = tile;
        else if (tile instanceof BlockWrapperTile) ;
        else if (parent.nodeName == "DIV" && !line && parent != this.view.contentDOM)
          line = new LineTile(parent, lineBaseAttrs);
        else if (!line)
          marks2.push(MarkTile.of(new MarkDecoration({ tagName: parent.nodeName.toLowerCase(), attributes: getAttrs(parent) }), parent));
      }
      return { line, marks: marks2 };
    }
  };
  function hasContent(tile, requireText) {
    let scan = (tile2) => {
      for (let ch of tile2.children)
        if ((requireText ? ch.isText() : ch.length) || scan(ch))
          return true;
      return false;
    };
    return scan(tile);
  }
  function widgetFlags(deco) {
    let flags = deco.isReplace ? (deco.startSide < 0 ? 64 : 0) | (deco.endSide > 0 ? 128 : 0) : deco.startSide > 0 ? 32 : 16;
    if (deco.block)
      flags |= 256;
    return flags;
  }
  var lineBaseAttrs = { class: "cm-line" };
  function addLineDeco(value, deco) {
    let attrs = deco.spec.attributes, cls = deco.spec.class;
    if (!attrs && !cls)
      return value;
    if (!value)
      value = { class: "cm-line" };
    if (attrs)
      combineAttrs(attrs, value);
    if (cls)
      value.class += " " + cls;
    return value;
  }
  function getMarks(ptr) {
    let found = [];
    for (let i = ptr.parents.length; i > 1; i--) {
      let tile = i == ptr.parents.length ? ptr.tile : ptr.parents[i].tile;
      if (tile instanceof MarkTile)
        found.push(tile.mark);
    }
    return found;
  }
  function freeNode(node) {
    let tile = Tile.get(node);
    if (tile)
      tile.setDOM(node.cloneNode());
    return node;
  }
  var NullWidget = class extends WidgetType {
    constructor(tag) {
      super();
      this.tag = tag;
    }
    eq(other) {
      return other.tag == this.tag;
    }
    toDOM() {
      return document.createElement(this.tag);
    }
    updateDOM(elt) {
      return elt.nodeName.toLowerCase() == this.tag;
    }
    get isHidden() {
      return true;
    }
  };
  NullWidget.inline = /* @__PURE__ */ new NullWidget("span");
  NullWidget.block = /* @__PURE__ */ new NullWidget("div");
  var BreakWidget = /* @__PURE__ */ new class extends WidgetType {
    toDOM() {
      return document.createElement("br");
    }
    get isHidden() {
      return true;
    }
    get editable() {
      return true;
    }
  }();
  var DocView = class {
    constructor(view) {
      this.view = view;
      this.decorations = [];
      this.blockWrappers = [];
      this.dynamicDecorationMap = [false];
      this.domChanged = null;
      this.hasComposition = null;
      this.editContextFormatting = Decoration.none;
      this.lastCompositionAfterCursor = false;
      this.minWidth = 0;
      this.minWidthFrom = 0;
      this.minWidthTo = 0;
      this.impreciseAnchor = null;
      this.impreciseHead = null;
      this.forceSelection = false;
      this.lastUpdate = Date.now();
      this.updateDeco();
      this.tile = new DocTile(view, view.contentDOM);
      this.updateInner([new ChangedRange(0, 0, 0, view.state.doc.length)], null);
    }
    // Update the document view to a given state.
    update(update) {
      var _a2;
      let changedRanges = update.changedRanges;
      if (this.minWidth > 0 && changedRanges.length) {
        if (!changedRanges.every(({ fromA, toA }) => toA < this.minWidthFrom || fromA > this.minWidthTo)) {
          this.minWidth = this.minWidthFrom = this.minWidthTo = 0;
        } else {
          this.minWidthFrom = update.changes.mapPos(this.minWidthFrom, 1);
          this.minWidthTo = update.changes.mapPos(this.minWidthTo, 1);
        }
      }
      this.updateEditContextFormatting(update);
      let readCompositionAt = -1;
      if (this.view.inputState.composing >= 0 && !this.view.observer.editContext) {
        if ((_a2 = this.domChanged) === null || _a2 === void 0 ? void 0 : _a2.newSel)
          readCompositionAt = this.domChanged.newSel.head;
        else if (!touchesComposition(update.changes, this.hasComposition) && !update.selectionSet)
          readCompositionAt = update.state.selection.main.head;
      }
      let composition = readCompositionAt > -1 ? findCompositionRange(this.view, update.changes, readCompositionAt) : null;
      this.domChanged = null;
      if (this.hasComposition) {
        let { from, to } = this.hasComposition;
        changedRanges = new ChangedRange(from, to, update.changes.mapPos(from, -1), update.changes.mapPos(to, 1)).addToSet(changedRanges.slice());
      }
      this.hasComposition = composition ? { from: composition.range.fromB, to: composition.range.toB } : null;
      if ((browser.ie || browser.chrome) && !composition && update && update.state.doc.lines != update.startState.doc.lines)
        this.forceSelection = true;
      let prevDeco = this.decorations, prevWrappers = this.blockWrappers;
      this.updateDeco();
      let decoDiff = findChangedDeco(prevDeco, this.decorations, update.changes);
      if (decoDiff.length)
        changedRanges = ChangedRange.extendWithRanges(changedRanges, decoDiff);
      let blockDiff = findChangedWrappers(prevWrappers, this.blockWrappers, update.changes);
      if (blockDiff.length)
        changedRanges = ChangedRange.extendWithRanges(changedRanges, blockDiff);
      if (composition && !changedRanges.some((r) => r.fromA <= composition.range.fromA && r.toA >= composition.range.toA))
        changedRanges = composition.range.addToSet(changedRanges.slice());
      if (this.tile.flags & 2 && changedRanges.length == 0) {
        return false;
      } else {
        this.updateInner(changedRanges, composition);
        if (update.transactions.length)
          this.lastUpdate = Date.now();
        return true;
      }
    }
    // Used by update and the constructor do perform the actual DOM
    // update
    updateInner(changes, composition) {
      this.view.viewState.mustMeasureContent = true;
      let { observer } = this.view;
      observer.ignore(() => {
        if (composition || changes.length) {
          let oldTile = this.tile;
          let builder = new TileUpdate(this.view, oldTile, this.blockWrappers, this.decorations, this.dynamicDecorationMap);
          if (composition && Tile.get(composition.text))
            builder.cache.reused.set(
              Tile.get(composition.text),
              2
              /* Reused.DOM */
            );
          this.tile = builder.run(changes, composition);
          destroyDropped(oldTile, builder.cache.reused);
        }
        this.tile.dom.style.height = this.view.viewState.contentHeight / this.view.scaleY + "px";
        this.tile.dom.style.flexBasis = this.minWidth ? this.minWidth + "px" : "";
        let track = browser.chrome || browser.ios ? { node: observer.selectionRange.focusNode, written: false } : void 0;
        this.tile.sync(track);
        if (track && (track.written || observer.selectionRange.focusNode != track.node || !this.tile.dom.contains(track.node)))
          this.forceSelection = true;
        this.tile.dom.style.height = "";
      });
      let gaps = [];
      if (this.view.viewport.from || this.view.viewport.to < this.view.state.doc.length) {
        for (let child of this.tile.children)
          if (child.isWidget() && child.widget instanceof BlockGapWidget)
            gaps.push(child.dom);
      }
      observer.updateGaps(gaps);
    }
    updateEditContextFormatting(update) {
      this.editContextFormatting = this.editContextFormatting.map(update.changes);
      for (let tr of update.transactions)
        for (let effect of tr.effects)
          if (effect.is(setEditContextFormatting)) {
            this.editContextFormatting = effect.value;
          }
    }
    // Sync the DOM selection to this.state.selection
    updateSelection(mustRead = false, fromPointer = false) {
      if (mustRead || !this.view.observer.selectionRange.focusNode)
        this.view.observer.readSelectionRange();
      let { dom } = this.tile;
      let activeElt = this.view.root.activeElement, focused = activeElt == dom;
      let selectionNotFocus = !focused && !(this.view.state.facet(editable) || dom.tabIndex > -1) && hasSelection(dom, this.view.observer.selectionRange) && !(activeElt && dom.contains(activeElt));
      if (!(focused || fromPointer || selectionNotFocus))
        return;
      let force = this.forceSelection;
      this.forceSelection = false;
      let main = this.view.state.selection.main, anchor, head;
      if (main.empty) {
        head = anchor = this.inlineDOMNearPos(main.anchor, main.assoc || 1);
      } else {
        head = this.inlineDOMNearPos(main.head, main.head == main.from ? 1 : -1);
        anchor = this.inlineDOMNearPos(main.anchor, main.anchor == main.from ? 1 : -1);
      }
      if (browser.gecko && main.empty && !this.hasComposition && betweenUneditable(anchor)) {
        let dummy = document.createTextNode("");
        this.view.observer.ignore(() => anchor.node.insertBefore(dummy, anchor.node.childNodes[anchor.offset] || null));
        anchor = head = new DOMPos(dummy, 0);
        force = true;
      }
      let domSel = this.view.observer.selectionRange;
      if (force || !domSel.focusNode || (!isEquivalentPosition(anchor.node, anchor.offset, domSel.anchorNode, domSel.anchorOffset) || !isEquivalentPosition(head.node, head.offset, domSel.focusNode, domSel.focusOffset)) && !this.suppressWidgetCursorChange(domSel, main)) {
        this.view.observer.ignore(() => {
          if (browser.android && browser.chrome && dom.contains(domSel.focusNode) && inUneditable(domSel.focusNode, dom)) {
            dom.blur();
            dom.focus({ preventScroll: true });
          }
          let rawSel = getSelection(this.view.root);
          if (!rawSel) ;
          else if (main.empty) {
            if (browser.gecko) {
              let nextTo = nextToUneditable(anchor.node, anchor.offset);
              if (nextTo && nextTo != (1 | 2)) {
                let text = (nextTo == 1 ? textNodeBefore : textNodeAfter)(anchor.node, anchor.offset);
                if (text)
                  anchor = new DOMPos(text.node, text.offset);
              }
            }
            rawSel.collapse(anchor.node, anchor.offset);
            if (main.bidiLevel != null && rawSel.caretBidiLevel !== void 0)
              rawSel.caretBidiLevel = main.bidiLevel;
          } else if (rawSel.extend) {
            rawSel.collapse(anchor.node, anchor.offset);
            try {
              rawSel.extend(head.node, head.offset);
            } catch (_) {
            }
          } else {
            let range = document.createRange();
            if (main.anchor > main.head)
              [anchor, head] = [head, anchor];
            range.setEnd(head.node, head.offset);
            range.setStart(anchor.node, anchor.offset);
            rawSel.removeAllRanges();
            rawSel.addRange(range);
          }
          if (selectionNotFocus && this.view.root.activeElement == dom) {
            dom.blur();
            if (activeElt)
              activeElt.focus();
          }
        });
        this.view.observer.setSelectionRange(anchor, head);
      }
      this.impreciseAnchor = anchor.precise ? null : new DOMPos(domSel.anchorNode, domSel.anchorOffset);
      this.impreciseHead = head.precise ? null : new DOMPos(domSel.focusNode, domSel.focusOffset);
    }
    // If a zero-length widget is inserted next to the cursor during
    // composition, avoid moving it across it and disrupting the
    // composition.
    suppressWidgetCursorChange(sel, cursor) {
      return this.hasComposition && cursor.empty && isEquivalentPosition(sel.focusNode, sel.focusOffset, sel.anchorNode, sel.anchorOffset) && this.posFromDOM(sel.focusNode, sel.focusOffset) == cursor.head;
    }
    enforceCursorAssoc() {
      if (this.hasComposition)
        return;
      let { view } = this, cursor = view.state.selection.main;
      let sel = getSelection(view.root);
      let { anchorNode, anchorOffset } = view.observer.selectionRange;
      if (!sel || !cursor.empty || !cursor.assoc || !sel.modify)
        return;
      let line = this.lineAt(cursor.head, cursor.assoc);
      if (!line)
        return;
      let lineStart = line.posAtStart;
      if (cursor.head == lineStart || cursor.head == lineStart + line.length)
        return;
      let before = this.coordsAt(cursor.head, -1), after = this.coordsAt(cursor.head, 1);
      if (!before || !after || before.bottom > after.top)
        return;
      let dom = this.domAtPos(cursor.head + cursor.assoc, cursor.assoc);
      sel.collapse(dom.node, dom.offset);
      sel.modify("move", cursor.assoc < 0 ? "forward" : "backward", "lineboundary");
      view.observer.readSelectionRange();
      let newRange = view.observer.selectionRange;
      if (view.docView.posFromDOM(newRange.anchorNode, newRange.anchorOffset) != cursor.from)
        sel.collapse(anchorNode, anchorOffset);
    }
    posFromDOM(node, offset) {
      let tile = this.tile.nearest(node);
      if (!tile)
        return this.tile.dom.compareDocumentPosition(node) & 2 ? 0 : this.view.state.doc.length;
      let start = tile.posAtStart;
      if (tile.isComposite()) {
        let after;
        if (node == tile.dom) {
          after = tile.dom.childNodes[offset];
        } else {
          let bias = maxOffset(node) == 0 ? 0 : offset == 0 ? -1 : 1;
          for (; ; ) {
            let parent = node.parentNode;
            if (parent == tile.dom)
              break;
            if (bias == 0 && parent.firstChild != parent.lastChild) {
              if (node == parent.firstChild)
                bias = -1;
              else
                bias = 1;
            }
            node = parent;
          }
          if (bias < 0)
            after = node;
          else
            after = node.nextSibling;
        }
        if (after == tile.dom.firstChild)
          return start;
        while (after && !Tile.get(after))
          after = after.nextSibling;
        if (!after)
          return start + tile.length;
        for (let i = 0, pos = start; ; i++) {
          let child = tile.children[i];
          if (child.dom == after)
            return pos;
          pos += child.length + child.breakAfter;
        }
      } else if (tile.isText()) {
        return node == tile.dom ? start + offset : start + (offset ? tile.length : 0);
      } else {
        return start;
      }
    }
    domAtPos(pos, side) {
      let { tile, offset } = this.tile.resolveBlock(pos, side);
      if (tile.isWidget())
        return tile.domPosFor(pos, side);
      return tile.domIn(offset, side);
    }
    inlineDOMNearPos(pos, side) {
      let before, beforeOff = -1, beforeBad = false;
      let after, afterOff = -1, afterBad = false;
      this.tile.blockTiles((tile, off) => {
        if (tile.isWidget()) {
          if (tile.flags & 32 && off >= pos)
            return true;
          if (tile.flags & 16)
            beforeBad = true;
        } else {
          let end = off + tile.length;
          if (off <= pos) {
            before = tile;
            beforeOff = pos - off;
            beforeBad = end < pos;
          }
          if (end >= pos && !after) {
            after = tile;
            afterOff = pos - off;
            afterBad = off > pos;
          }
          if (off > pos && after)
            return true;
        }
      });
      if (!before && !after)
        return this.domAtPos(pos, side);
      if (beforeBad && after)
        before = null;
      else if (afterBad && before)
        after = null;
      return before && side < 0 || !after ? before.domIn(beforeOff, side) : after.domIn(afterOff, side);
    }
    coordsAt(pos, side) {
      let { tile, offset } = this.tile.resolveBlock(pos, side);
      if (tile.isWidget()) {
        if (tile.widget instanceof BlockGapWidget)
          return null;
        return tile.coordsInWidget(offset, side, true);
      }
      return tile.coordsIn(offset, side);
    }
    lineAt(pos, side) {
      let { tile } = this.tile.resolveBlock(pos, side);
      return tile.isLine() ? tile : null;
    }
    coordsForChar(pos) {
      let { tile, offset } = this.tile.resolveBlock(pos, 1);
      if (!tile.isLine())
        return null;
      function scan(tile2, offset2) {
        if (tile2.isComposite()) {
          for (let ch of tile2.children) {
            if (ch.length >= offset2) {
              let found = scan(ch, offset2);
              if (found)
                return found;
            }
            offset2 -= ch.length;
            if (offset2 < 0)
              break;
          }
        } else if (tile2.isText() && offset2 < tile2.length) {
          let end = findClusterBreak2(tile2.text, offset2);
          if (end == offset2)
            return null;
          let rects = textRange(tile2.dom, offset2, end).getClientRects();
          for (let i = 0; i < rects.length; i++) {
            let rect = rects[i];
            if (i == rects.length - 1 || rect.top < rect.bottom && rect.left < rect.right)
              return rect;
          }
        }
        return null;
      }
      return scan(tile, offset);
    }
    measureVisibleLineHeights(viewport) {
      let result = [], { from, to } = viewport;
      let contentWidth = this.view.contentDOM.clientWidth;
      let isWider = contentWidth > Math.max(this.view.scrollDOM.clientWidth, this.minWidth) + 1;
      let widest = -1, ltr = this.view.textDirection == Direction.LTR;
      let spaceAbove = 0;
      let scan = (tile, pos, measureBounds) => {
        for (let i = 0; i < tile.children.length; i++) {
          if (pos > to)
            break;
          let child = tile.children[i], end = pos + child.length;
          let childRect = child.dom.getBoundingClientRect(), { height } = childRect;
          if (measureBounds && !i)
            spaceAbove += childRect.top - measureBounds.top;
          if (child instanceof BlockWrapperTile) {
            if (end > from)
              scan(child, pos, childRect);
          } else if (pos >= from) {
            if (spaceAbove > 0)
              result.push(-spaceAbove);
            result.push(height + spaceAbove);
            spaceAbove = 0;
            if (isWider) {
              let last = child.dom.lastChild;
              let rects = last ? clientRectsFor(last) : [];
              if (rects.length) {
                let rect = rects[rects.length - 1];
                let width = ltr ? rect.right - childRect.left : childRect.right - rect.left;
                if (width > widest) {
                  widest = width;
                  this.minWidth = contentWidth;
                  this.minWidthFrom = pos;
                  this.minWidthTo = end;
                }
              }
            }
          }
          if (measureBounds && i == tile.children.length - 1)
            spaceAbove += measureBounds.bottom - childRect.bottom;
          pos = end + child.breakAfter;
        }
      };
      scan(this.tile, 0, null);
      return result;
    }
    textDirectionAt(pos) {
      let { tile } = this.tile.resolveBlock(pos, 1);
      return getComputedStyle(tile.dom).direction == "rtl" ? Direction.RTL : Direction.LTR;
    }
    measureTextSize() {
      let lineMeasure = this.tile.blockTiles((tile) => {
        if (tile.isLine() && tile.children.length && tile.length <= 20) {
          let totalWidth = 0, textHeight2;
          for (let child of tile.children) {
            if (!child.isText() || /[^ -~]/.test(child.text))
              return void 0;
            let rects = clientRectsFor(child.dom);
            if (rects.length != 1)
              return void 0;
            totalWidth += rects[0].width;
            textHeight2 = rects[0].height;
          }
          if (totalWidth)
            return {
              lineHeight: tile.dom.getBoundingClientRect().height,
              charWidth: totalWidth / tile.length,
              textHeight: textHeight2
            };
        }
      });
      if (lineMeasure)
        return lineMeasure;
      let dummy = document.createElement("div"), lineHeight, charWidth, textHeight;
      dummy.className = "cm-line";
      dummy.style.width = "99999px";
      dummy.style.position = "absolute";
      dummy.textContent = "abc def ghi jkl mno pqr stu";
      this.view.observer.ignore(() => {
        this.tile.dom.appendChild(dummy);
        let rect = clientRectsFor(dummy.firstChild)[0];
        lineHeight = dummy.getBoundingClientRect().height;
        charWidth = rect && rect.width ? rect.width / 27 : 7;
        textHeight = rect && rect.height ? rect.height : lineHeight;
        dummy.remove();
      });
      return { lineHeight, charWidth, textHeight };
    }
    computeBlockGapDeco() {
      let deco = [], vs = this.view.viewState;
      for (let pos = 0, i = 0; ; i++) {
        let next = i == vs.viewports.length ? null : vs.viewports[i];
        let end = next ? next.from - 1 : this.view.state.doc.length;
        if (end > pos) {
          let height = (vs.lineBlockAt(end).bottom - vs.lineBlockAt(pos).top) / this.view.scaleY;
          deco.push(Decoration.replace({
            widget: new BlockGapWidget(height),
            block: true,
            inclusive: true,
            isBlockGap: true
          }).range(pos, end));
        }
        if (!next)
          break;
        pos = next.to + 1;
      }
      return Decoration.set(deco);
    }
    updateDeco() {
      let i = 1;
      let allDeco = this.view.state.facet(decorations).map((d) => {
        let dynamic = this.dynamicDecorationMap[i++] = typeof d == "function";
        return dynamic ? d(this.view) : d;
      });
      let dynamicOuter = false, outerDeco = this.view.state.facet(outerDecorations).map((d, i2) => {
        let dynamic = typeof d == "function";
        if (dynamic)
          dynamicOuter = true;
        return dynamic ? d(this.view) : d;
      });
      if (outerDeco.length) {
        this.dynamicDecorationMap[i++] = dynamicOuter;
        allDeco.push(RangeSet.join(outerDeco));
      }
      this.decorations = [
        this.editContextFormatting,
        ...allDeco,
        this.computeBlockGapDeco(),
        this.view.viewState.lineGapDeco
      ];
      while (i < this.decorations.length)
        this.dynamicDecorationMap[i++] = false;
      this.blockWrappers = this.view.state.facet(blockWrappers).map((v) => typeof v == "function" ? v(this.view) : v);
    }
    scrollIntoView(target) {
      if (target.isSnapshot) {
        let ref = this.view.viewState.lineBlockAt(target.range.head);
        this.view.scrollDOM.scrollTop = ref.top - target.yMargin;
        this.view.scrollDOM.scrollLeft = target.xMargin;
        return;
      }
      for (let handler of this.view.state.facet(scrollHandler)) {
        try {
          if (handler(this.view, target.range, target))
            return true;
        } catch (e) {
          logException(this.view.state, e, "scroll handler");
        }
      }
      let { range } = target;
      let rect = this.coordsAt(range.head, range.empty ? range.assoc : range.head > range.anchor ? -1 : 1), other;
      if (!rect)
        return;
      if (!range.empty && (other = this.coordsAt(range.anchor, range.anchor > range.head ? -1 : 1)))
        rect = {
          left: Math.min(rect.left, other.left),
          top: Math.min(rect.top, other.top),
          right: Math.max(rect.right, other.right),
          bottom: Math.max(rect.bottom, other.bottom)
        };
      let margins = getScrollMargins(this.view);
      let targetRect = {
        left: rect.left - margins.left,
        top: rect.top - margins.top,
        right: rect.right + margins.right,
        bottom: rect.bottom + margins.bottom
      };
      let { offsetWidth, offsetHeight } = this.view.scrollDOM;
      scrollRectIntoView(this.view.scrollDOM, targetRect, range.head < range.anchor ? -1 : 1, target.x, target.y, Math.max(Math.min(target.xMargin, offsetWidth), -offsetWidth), Math.max(Math.min(target.yMargin, offsetHeight), -offsetHeight), this.view.textDirection == Direction.LTR);
      if (window.visualViewport && window.innerHeight - window.visualViewport.height > 1 && (rect.top > window.pageYOffset + window.visualViewport.offsetTop + window.visualViewport.height || rect.bottom < window.pageYOffset + window.visualViewport.offsetTop)) {
        let line = this.view.docView.lineAt(range.head, 1);
        if (line)
          line.dom.scrollIntoView({ block: "nearest" });
      }
    }
    lineHasWidget(pos) {
      let scan = (child) => child.isWidget() || child.children.some(scan);
      return scan(this.tile.resolveBlock(pos, 1).tile);
    }
    destroy() {
      destroyDropped(this.tile);
    }
  };
  function destroyDropped(tile, reused) {
    let r = reused === null || reused === void 0 ? void 0 : reused.get(tile);
    if (r != 1) {
      if (r == null)
        tile.destroy();
      for (let ch of tile.children)
        destroyDropped(ch, reused);
    }
  }
  function betweenUneditable(pos) {
    return pos.node.nodeType == 1 && pos.node.firstChild && (pos.offset == 0 || pos.node.childNodes[pos.offset - 1].contentEditable == "false") && (pos.offset == pos.node.childNodes.length || pos.node.childNodes[pos.offset].contentEditable == "false");
  }
  function findCompositionNode(view, headPos) {
    let sel = view.observer.selectionRange;
    if (!sel.focusNode)
      return null;
    let textBefore = textNodeBefore(sel.focusNode, sel.focusOffset);
    let textAfter = textNodeAfter(sel.focusNode, sel.focusOffset);
    let textNode = textBefore || textAfter;
    if (textAfter && textBefore && textAfter.node != textBefore.node) {
      let tileAfter = Tile.get(textAfter.node);
      if (!tileAfter || tileAfter.isText() && tileAfter.text != textAfter.node.nodeValue) {
        textNode = textAfter;
      } else if (view.docView.lastCompositionAfterCursor) {
        let tileBefore = Tile.get(textBefore.node);
        if (!(!tileBefore || tileBefore.isText() && tileBefore.text != textBefore.node.nodeValue))
          textNode = textAfter;
      }
    }
    view.docView.lastCompositionAfterCursor = textNode != textBefore;
    if (!textNode)
      return null;
    let from = headPos - textNode.offset;
    return { from, to: from + textNode.node.nodeValue.length, node: textNode.node };
  }
  function findCompositionRange(view, changes, headPos) {
    let found = findCompositionNode(view, headPos);
    if (!found)
      return null;
    let { node: textNode, from, to } = found, text = textNode.nodeValue;
    if (/[\n\r]/.test(text))
      return null;
    if (view.state.doc.sliceString(found.from, found.to) != text)
      return null;
    let inv = changes.invertedDesc;
    return { range: new ChangedRange(inv.mapPos(from), inv.mapPos(to), from, to), text: textNode };
  }
  function nextToUneditable(node, offset) {
    if (node.nodeType != 1)
      return 0;
    return (offset && node.childNodes[offset - 1].contentEditable == "false" ? 1 : 0) | (offset < node.childNodes.length && node.childNodes[offset].contentEditable == "false" ? 2 : 0);
  }
  var DecorationComparator$1 = class DecorationComparator {
    constructor() {
      this.changes = [];
    }
    compareRange(from, to) {
      addRange(from, to, this.changes);
    }
    comparePoint(from, to) {
      addRange(from, to, this.changes);
    }
    boundChange(pos) {
      addRange(pos, pos, this.changes);
    }
  };
  function findChangedDeco(a, b, diff) {
    let comp = new DecorationComparator$1();
    RangeSet.compare(a, b, diff, comp);
    return comp.changes;
  }
  var WrapperComparator = class {
    constructor() {
      this.changes = [];
    }
    compareRange(from, to) {
      addRange(from, to, this.changes);
    }
    comparePoint() {
    }
    boundChange(pos) {
      addRange(pos, pos, this.changes);
    }
  };
  function findChangedWrappers(a, b, diff) {
    let comp = new WrapperComparator();
    RangeSet.compare(a, b, diff, comp);
    return comp.changes;
  }
  function inUneditable(node, inside) {
    for (let cur2 = node; cur2 && cur2 != inside; cur2 = cur2.assignedSlot || cur2.parentNode) {
      if (cur2.nodeType == 1 && cur2.contentEditable == "false") {
        return true;
      }
    }
    return false;
  }
  function touchesComposition(changes, composition) {
    let touched = false;
    if (composition)
      changes.iterChangedRanges((from, to) => {
        if (from < composition.to && to > composition.from)
          touched = true;
      });
    return touched;
  }
  var BlockGapWidget = class extends WidgetType {
    constructor(height) {
      super();
      this.height = height;
    }
    toDOM() {
      let elt = document.createElement("div");
      elt.className = "cm-gap";
      this.updateDOM(elt);
      return elt;
    }
    eq(other) {
      return other.height == this.height;
    }
    updateDOM(elt) {
      elt.style.height = this.height + "px";
      return true;
    }
    get editable() {
      return true;
    }
    get estimatedHeight() {
      return this.height;
    }
    ignoreEvent() {
      return false;
    }
  };
  function groupAt(state, pos, bias = 1) {
    let categorize = state.charCategorizer(pos);
    let line = state.doc.lineAt(pos), linePos = pos - line.from;
    if (line.length == 0)
      return EditorSelection.cursor(pos);
    if (linePos == 0)
      bias = 1;
    else if (linePos == line.length)
      bias = -1;
    let from = linePos, to = linePos;
    if (bias < 0)
      from = findClusterBreak2(line.text, linePos, false);
    else
      to = findClusterBreak2(line.text, linePos);
    let cat = categorize(line.text.slice(from, to));
    while (from > 0) {
      let prev = findClusterBreak2(line.text, from, false);
      if (categorize(line.text.slice(prev, from)) != cat)
        break;
      from = prev;
    }
    while (to < line.length) {
      let next = findClusterBreak2(line.text, to);
      if (categorize(line.text.slice(to, next)) != cat)
        break;
      to = next;
    }
    return EditorSelection.range(from + line.from, to + line.from);
  }
  function posAtCoordsImprecise(view, contentRect, block, x, y) {
    let into = Math.round((x - contentRect.left) * view.defaultCharacterWidth);
    if (view.lineWrapping && block.height > view.defaultLineHeight * 1.5) {
      let textHeight = view.viewState.heightOracle.textHeight;
      let line = Math.floor((y - block.top - (view.defaultLineHeight - textHeight) * 0.5) / textHeight);
      into += line * view.viewState.heightOracle.lineLength;
    }
    let content2 = view.state.sliceDoc(block.from, block.to);
    return block.from + findColumn(content2, into, view.state.tabSize);
  }
  function blockAt(view, pos, side) {
    let line = view.lineBlockAt(pos);
    if (Array.isArray(line.type)) {
      let best;
      for (let l of line.type) {
        if (l.from > pos)
          break;
        if (l.to < pos)
          continue;
        if (l.from < pos && l.to > pos)
          return l;
        if (!best || l.type == BlockType.Text && (best.type != l.type || (side < 0 ? l.from < pos : l.to > pos)))
          best = l;
      }
      return best || line;
    }
    return line;
  }
  function moveToLineBoundary(view, start, forward, includeWrap) {
    let line = blockAt(view, start.head, start.assoc || -1);
    let coords = !includeWrap || line.type != BlockType.Text || !(view.lineWrapping || line.widgetLineBreaks) ? null : view.coordsAtPos(start.assoc < 0 && start.head > line.from ? start.head - 1 : start.head);
    if (coords) {
      let editorRect = view.dom.getBoundingClientRect();
      let direction = view.textDirectionAt(line.from);
      let pos = view.posAtCoords({
        x: forward == (direction == Direction.LTR) ? editorRect.right - 1 : editorRect.left + 1,
        y: (coords.top + coords.bottom) / 2
      });
      if (pos != null)
        return EditorSelection.cursor(pos, forward ? -1 : 1);
    }
    return EditorSelection.cursor(forward ? line.to : line.from, forward ? -1 : 1);
  }
  function moveByChar(view, start, forward, by) {
    let line = view.state.doc.lineAt(start.head), spans = view.bidiSpans(line);
    let direction = view.textDirectionAt(line.from);
    for (let cur2 = start, check = null; ; ) {
      let next = moveVisually(line, spans, direction, cur2, forward), char = movedOver;
      if (!next) {
        if (line.number == (forward ? view.state.doc.lines : 1))
          return cur2;
        char = "\n";
        line = view.state.doc.line(line.number + (forward ? 1 : -1));
        spans = view.bidiSpans(line);
        next = view.visualLineSide(line, !forward);
      }
      if (!check) {
        if (!by)
          return next;
        check = by(char);
      } else if (!check(char)) {
        return cur2;
      }
      cur2 = next;
    }
  }
  function byGroup(view, pos, start) {
    let categorize = view.state.charCategorizer(pos);
    let cat = categorize(start);
    return (next) => {
      let nextCat = categorize(next);
      if (cat == CharCategory.Space)
        cat = nextCat;
      return cat == nextCat;
    };
  }
  function moveVertically(view, start, forward, distance) {
    let startPos = start.head, dir = forward ? 1 : -1;
    if (startPos == (forward ? view.state.doc.length : 0))
      return EditorSelection.cursor(startPos, start.assoc);
    let goal = start.goalColumn, startY;
    let rect = view.contentDOM.getBoundingClientRect();
    let startCoords = view.coordsAtPos(startPos, (start.empty ? start.assoc : 0) || (forward ? 1 : -1)), docTop = view.documentTop;
    if (startCoords) {
      if (goal == null)
        goal = startCoords.left - rect.left;
      startY = dir < 0 ? startCoords.top : startCoords.bottom;
    } else {
      let line = view.viewState.lineBlockAt(startPos);
      if (goal == null)
        goal = Math.min(rect.right - rect.left, view.defaultCharacterWidth * (startPos - line.from));
      startY = (dir < 0 ? line.top : line.bottom) + docTop;
    }
    let resolvedGoal = rect.left + goal;
    let dist2 = distance !== null && distance !== void 0 ? distance : view.viewState.heightOracle.textHeight >> 1;
    let pos = posAtCoords(view, { x: resolvedGoal, y: startY + dist2 * dir }, false, dir);
    return EditorSelection.cursor(pos.pos, pos.assoc, void 0, goal);
  }
  function skipAtomicRanges(atoms, pos, bias) {
    for (; ; ) {
      let moved = 0;
      for (let set of atoms) {
        set.between(pos - 1, pos + 1, (from, to, value) => {
          if (pos > from && pos < to) {
            let side = moved || bias || (pos - from < to - pos ? -1 : 1);
            pos = side < 0 ? from : to;
            moved = side;
          }
        });
      }
      if (!moved)
        return pos;
    }
  }
  function skipAtomsForSelection(atoms, sel) {
    let ranges = null;
    for (let i = 0; i < sel.ranges.length; i++) {
      let range = sel.ranges[i], updated = null;
      if (range.empty) {
        let pos = skipAtomicRanges(atoms, range.from, 0);
        if (pos != range.from)
          updated = EditorSelection.cursor(pos, -1);
      } else {
        let from = skipAtomicRanges(atoms, range.from, -1);
        let to = skipAtomicRanges(atoms, range.to, 1);
        if (from != range.from || to != range.to)
          updated = EditorSelection.range(range.from == range.anchor ? from : to, range.from == range.head ? from : to);
      }
      if (updated) {
        if (!ranges)
          ranges = sel.ranges.slice();
        ranges[i] = updated;
      }
    }
    return ranges ? EditorSelection.create(ranges, sel.mainIndex) : sel;
  }
  function skipAtoms(view, oldPos, pos) {
    let newPos = skipAtomicRanges(view.state.facet(atomicRanges).map((f) => f(view)), pos.from, oldPos.head > pos.from ? -1 : 1);
    return newPos == pos.from ? pos : EditorSelection.cursor(newPos, newPos < pos.from ? 1 : -1);
  }
  var PosAssoc = class {
    constructor(pos, assoc) {
      this.pos = pos;
      this.assoc = assoc;
    }
  };
  function posAtCoords(view, coords, precise, scanY) {
    let content2 = view.contentDOM.getBoundingClientRect(), docTop = content2.top + view.viewState.paddingTop;
    let { x, y } = coords, yOffset = y - docTop, block;
    for (; ; ) {
      if (yOffset < 0)
        return new PosAssoc(0, 1);
      if (yOffset > view.viewState.docHeight)
        return new PosAssoc(view.state.doc.length, -1);
      block = view.elementAtHeight(yOffset);
      if (scanY == null)
        break;
      if (block.type == BlockType.Text) {
        if (scanY < 0 ? block.to < view.viewport.from : block.from > view.viewport.to)
          break;
        let rect = view.docView.coordsAt(scanY < 0 ? block.from : block.to, scanY > 0 ? -1 : 1);
        if (rect && (scanY < 0 ? rect.top <= yOffset + docTop : rect.bottom >= yOffset + docTop))
          break;
      }
      let halfLine = view.viewState.heightOracle.textHeight / 2;
      yOffset = scanY > 0 ? block.bottom + halfLine : block.top - halfLine;
    }
    if (view.viewport.from >= block.to || view.viewport.to <= block.from) {
      if (precise)
        return null;
      if (block.type == BlockType.Text) {
        let pos = posAtCoordsImprecise(view, content2, block, x, y);
        return new PosAssoc(pos, pos == block.from ? 1 : -1);
      }
    }
    if (block.type != BlockType.Text)
      return yOffset < (block.top + block.bottom) / 2 ? new PosAssoc(block.from, 1) : new PosAssoc(block.to, -1);
    let line = view.docView.lineAt(block.from, 2);
    if (!line || line.length != block.length)
      line = view.docView.lineAt(block.from, -2);
    return new InlineCoordsScan(view, x, y, view.textDirectionAt(block.from)).scanTile(line, block.from);
  }
  var InlineCoordsScan = class {
    constructor(view, x, y, baseDir) {
      this.view = view;
      this.x = x;
      this.y = y;
      this.baseDir = baseDir;
      this.line = null;
      this.spans = null;
    }
    bidiSpansAt(pos) {
      if (!this.line || this.line.from > pos || this.line.to < pos) {
        this.line = this.view.state.doc.lineAt(pos);
        this.spans = this.view.bidiSpans(this.line);
      }
      return this;
    }
    baseDirAt(pos, side) {
      let { line, spans } = this.bidiSpansAt(pos);
      let level = spans[BidiSpan.find(spans, pos - line.from, -1, side)].level;
      return level == this.baseDir;
    }
    dirAt(pos, side) {
      let { line, spans } = this.bidiSpansAt(pos);
      return spans[BidiSpan.find(spans, pos - line.from, -1, side)].dir;
    }
    // Used to short-circuit bidi tests for content with a uniform direction
    bidiIn(from, to) {
      let { spans, line } = this.bidiSpansAt(from);
      return spans.length > 1 || spans.length && (spans[0].level != this.baseDir || spans[0].to + line.from < to);
    }
    // Scan through the rectangles for the content of a tile with inline
    // content, looking for one that overlaps the queried position
    // vertically andis
    // closest horizontally. The caller is responsible for dividing its
    // content into N pieces, and pass an array with N+1 positions
    // (including the position after the last piece). For a text tile,
    // these will be character clusters, for a composite tile, these
    // will be child tiles.
    scan(positions, getRects) {
      let lo = 0, hi = positions.length - 1, seen = /* @__PURE__ */ new Set();
      let bidi = this.bidiIn(positions[0], positions[hi]);
      let above, below;
      let closestI = -1, closestDx = 1e9, closestRect;
      search: while (lo < hi) {
        let dist2 = hi - lo, mid = lo + hi >> 1;
        adjust: if (seen.has(mid)) {
          let scan = lo + Math.floor(Math.random() * dist2);
          for (let i = 0; i < dist2; i++) {
            if (!seen.has(scan)) {
              mid = scan;
              break adjust;
            }
            scan++;
            if (scan == hi)
              scan = lo;
          }
          break search;
        }
        seen.add(mid);
        let rects = getRects(mid);
        if (rects)
          for (let i = 0; i < rects.length; i++) {
            let rect = rects[i], side = 0;
            if (rect.bottom < this.y) {
              if (!above || above.bottom < rect.bottom)
                above = rect;
              side = 1;
            } else if (rect.top > this.y) {
              if (!below || below.top > rect.top)
                below = rect;
              side = -1;
            } else {
              let off = rect.left > this.x ? this.x - rect.left : rect.right < this.x ? this.x - rect.right : 0;
              let dx = Math.abs(off);
              if (dx < closestDx) {
                closestI = mid;
                closestDx = dx;
                closestRect = rect;
              }
              if (off)
                side = off < 0 == (this.baseDir == Direction.LTR) ? -1 : 1;
            }
            if (side == -1 && (!bidi || this.baseDirAt(positions[mid], 1)))
              hi = mid;
            else if (side == 1 && (!bidi || this.baseDirAt(positions[mid + 1], -1)))
              lo = mid + 1;
          }
      }
      if (!closestRect) {
        let side = above && (!below || this.y - above.bottom < below.top - this.y) ? above : below;
        this.y = (side.top + side.bottom) / 2;
        return this.scan(positions, getRects);
      }
      let ltr = (bidi ? this.dirAt(positions[closestI], 1) : this.baseDir) == Direction.LTR;
      return {
        i: closestI,
        // Test whether x is closes to the start or end of this element
        after: this.x > (closestRect.left + closestRect.right) / 2 == ltr
      };
    }
    scanText(tile, offset) {
      let positions = [];
      for (let i = 0; i < tile.length; i = findClusterBreak2(tile.text, i))
        positions.push(offset + i);
      positions.push(offset + tile.length);
      let scan = this.scan(positions, (i) => {
        let off = positions[i] - offset, end = positions[i + 1] - offset;
        return textRange(tile.dom, off, end).getClientRects();
      });
      return scan.after ? new PosAssoc(positions[scan.i + 1], -1) : new PosAssoc(positions[scan.i], 1);
    }
    scanTile(tile, offset) {
      if (!tile.length)
        return new PosAssoc(offset, 1);
      if (tile.children.length == 1) {
        let child2 = tile.children[0];
        if (child2.isText())
          return this.scanText(child2, offset);
        else if (child2.isComposite())
          return this.scanTile(child2, offset);
      }
      let positions = [offset];
      for (let i = 0, pos2 = offset; i < tile.children.length; i++)
        positions.push(pos2 += tile.children[i].length);
      let scan = this.scan(positions, (i) => {
        let child2 = tile.children[i];
        if (child2.flags & 48)
          return null;
        return (child2.dom.nodeType == 1 ? child2.dom : textRange(child2.dom, 0, child2.length)).getClientRects();
      });
      let child = tile.children[scan.i], pos = positions[scan.i];
      if (child.isText())
        return this.scanText(child, pos);
      if (child.isComposite())
        return this.scanTile(child, pos);
      return scan.after ? new PosAssoc(positions[scan.i + 1], -1) : new PosAssoc(pos, 1);
    }
  };
  var LineBreakPlaceholder = "\uFFFF";
  var DOMReader = class {
    constructor(points, view) {
      this.points = points;
      this.view = view;
      this.text = "";
      this.lineSeparator = view.state.facet(EditorState.lineSeparator);
    }
    append(text) {
      this.text += text;
    }
    lineBreak() {
      this.text += LineBreakPlaceholder;
    }
    readRange(start, end) {
      if (!start)
        return this;
      let parent = start.parentNode;
      for (let cur2 = start; ; ) {
        this.findPointBefore(parent, cur2);
        let oldLen = this.text.length;
        this.readNode(cur2);
        let tile = Tile.get(cur2), next = cur2.nextSibling;
        if (next == end) {
          if ((tile === null || tile === void 0 ? void 0 : tile.breakAfter) && !next && parent != this.view.contentDOM)
            this.lineBreak();
          break;
        }
        let nextTile = Tile.get(next);
        if ((tile && nextTile ? tile.breakAfter : (tile ? tile.breakAfter : isBlockElement(cur2)) || isBlockElement(next) && (cur2.nodeName != "BR" || (tile === null || tile === void 0 ? void 0 : tile.isWidget())) && this.text.length > oldLen) && !isEmptyToEnd(next, end))
          this.lineBreak();
        cur2 = next;
      }
      this.findPointBefore(parent, end);
      return this;
    }
    readTextNode(node) {
      let text = node.nodeValue;
      for (let point of this.points)
        if (point.node == node)
          point.pos = this.text.length + Math.min(point.offset, text.length);
      for (let off = 0, re = this.lineSeparator ? null : /\r\n?|\n/g; ; ) {
        let nextBreak = -1, breakSize = 1, m;
        if (this.lineSeparator) {
          nextBreak = text.indexOf(this.lineSeparator, off);
          breakSize = this.lineSeparator.length;
        } else if (m = re.exec(text)) {
          nextBreak = m.index;
          breakSize = m[0].length;
        }
        this.append(text.slice(off, nextBreak < 0 ? text.length : nextBreak));
        if (nextBreak < 0)
          break;
        this.lineBreak();
        if (breakSize > 1) {
          for (let point of this.points)
            if (point.node == node && point.pos > this.text.length)
              point.pos -= breakSize - 1;
        }
        off = nextBreak + breakSize;
      }
    }
    readNode(node) {
      let tile = Tile.get(node);
      let fromView = tile && tile.overrideDOMText;
      if (fromView != null) {
        this.findPointInside(node, fromView.length);
        for (let i = fromView.iter(); !i.next().done; ) {
          if (i.lineBreak)
            this.lineBreak();
          else
            this.append(i.value);
        }
      } else if (node.nodeType == 3) {
        this.readTextNode(node);
      } else if (node.nodeName == "BR") {
        if (node.nextSibling)
          this.lineBreak();
      } else if (node.nodeType == 1) {
        this.readRange(node.firstChild, null);
      }
    }
    findPointBefore(node, next) {
      for (let point of this.points)
        if (point.node == node && node.childNodes[point.offset] == next)
          point.pos = this.text.length;
    }
    findPointInside(node, length) {
      for (let point of this.points)
        if (node.nodeType == 3 ? point.node == node : node.contains(point.node))
          point.pos = this.text.length + (isAtEnd(node, point.node, point.offset) ? length : 0);
    }
  };
  function isAtEnd(parent, node, offset) {
    for (; ; ) {
      if (!node || offset < maxOffset(node))
        return false;
      if (node == parent)
        return true;
      offset = domIndex(node) + 1;
      node = node.parentNode;
    }
  }
  function isEmptyToEnd(node, end) {
    let widgets;
    for (; ; node = node.nextSibling) {
      if (node == end || !node)
        break;
      let view = Tile.get(node);
      if (!(view === null || view === void 0 ? void 0 : view.isWidget()))
        return false;
      if (view)
        (widgets || (widgets = [])).push(view);
    }
    if (widgets)
      for (let w of widgets) {
        let override = w.overrideDOMText;
        if (override === null || override === void 0 ? void 0 : override.length)
          return false;
      }
    return true;
  }
  var DOMPoint = class {
    constructor(node, offset) {
      this.node = node;
      this.offset = offset;
      this.pos = -1;
    }
  };
  var DOMChange = class {
    constructor(view, start, end, typeOver) {
      this.typeOver = typeOver;
      this.bounds = null;
      this.text = "";
      this.domChanged = start > -1;
      let { impreciseHead: iHead, impreciseAnchor: iAnchor } = view.docView;
      if (view.state.readOnly && start > -1) {
        this.newSel = null;
      } else if (start > -1 && (this.bounds = domBoundsAround(view.docView.tile, start, end, 0))) {
        let selPoints = iHead || iAnchor ? [] : selectionPoints(view);
        let reader = new DOMReader(selPoints, view);
        reader.readRange(this.bounds.startDOM, this.bounds.endDOM);
        this.text = reader.text;
        this.newSel = selectionFromPoints(selPoints, this.bounds.from);
      } else {
        let domSel = view.observer.selectionRange;
        let head = iHead && iHead.node == domSel.focusNode && iHead.offset == domSel.focusOffset || !contains(view.contentDOM, domSel.focusNode) ? view.state.selection.main.head : view.docView.posFromDOM(domSel.focusNode, domSel.focusOffset);
        let anchor = iAnchor && iAnchor.node == domSel.anchorNode && iAnchor.offset == domSel.anchorOffset || !contains(view.contentDOM, domSel.anchorNode) ? view.state.selection.main.anchor : view.docView.posFromDOM(domSel.anchorNode, domSel.anchorOffset);
        let vp = view.viewport;
        if ((browser.ios || browser.chrome) && view.state.selection.main.empty && head != anchor && (vp.from > 0 || vp.to < view.state.doc.length)) {
          let from = Math.min(head, anchor), to = Math.max(head, anchor);
          let offFrom = vp.from - from, offTo = vp.to - to;
          if ((offFrom == 0 || offFrom == 1 || from == 0) && (offTo == 0 || offTo == -1 || to == view.state.doc.length)) {
            head = 0;
            anchor = view.state.doc.length;
          }
        }
        if (view.inputState.composing > -1 && view.state.selection.ranges.length > 1)
          this.newSel = view.state.selection.replaceRange(EditorSelection.range(anchor, head));
        else
          this.newSel = EditorSelection.single(anchor, head);
      }
    }
  };
  function domBoundsAround(tile, from, to, offset) {
    if (tile.isComposite()) {
      let fromI = -1, fromStart = -1, toI = -1, toEnd = -1;
      for (let i = 0, pos = offset, prevEnd = offset; i < tile.children.length; i++) {
        let child = tile.children[i], end = pos + child.length;
        if (pos < from && end > to)
          return domBoundsAround(child, from, to, pos);
        if (end >= from && fromI == -1) {
          fromI = i;
          fromStart = pos;
        }
        if (pos > to && child.dom.parentNode == tile.dom) {
          toI = i;
          toEnd = prevEnd;
          break;
        }
        prevEnd = end;
        pos = end + child.breakAfter;
      }
      return {
        from: fromStart,
        to: toEnd < 0 ? offset + tile.length : toEnd,
        startDOM: (fromI ? tile.children[fromI - 1].dom.nextSibling : null) || tile.dom.firstChild,
        endDOM: toI < tile.children.length && toI >= 0 ? tile.children[toI].dom : null
      };
    } else if (tile.isText()) {
      return { from: offset, to: offset + tile.length, startDOM: tile.dom, endDOM: tile.dom.nextSibling };
    } else {
      return null;
    }
  }
  function applyDOMChange(view, domChange) {
    let change;
    let { newSel } = domChange, { state } = view, sel = state.selection.main;
    let lastKey = view.inputState.lastKeyTime > Date.now() - 100 ? view.inputState.lastKeyCode : -1;
    if (domChange.bounds) {
      let { from, to } = domChange.bounds;
      let preferredPos = sel.from, preferredSide = null;
      if (lastKey === 8 || browser.android && domChange.text.length < to - from) {
        preferredPos = sel.to;
        preferredSide = "end";
      }
      let cmp = state.doc.sliceString(from, to, LineBreakPlaceholder), selEnd, diff;
      if (!sel.empty && sel.from >= from && sel.to <= to && (domChange.typeOver || cmp != domChange.text) && cmp.slice(0, sel.from - from) == domChange.text.slice(0, sel.from - from) && cmp.slice(sel.to - from) == domChange.text.slice(selEnd = domChange.text.length - (cmp.length - (sel.to - from)))) {
        change = {
          from: sel.from,
          to: sel.to,
          insert: Text.of(domChange.text.slice(sel.from - from, selEnd).split(LineBreakPlaceholder))
        };
      } else if (diff = findDiff(cmp, domChange.text, preferredPos - from, preferredSide)) {
        if (browser.chrome && lastKey == 13 && diff.toB == diff.from + 2 && domChange.text.slice(diff.from, diff.toB) == LineBreakPlaceholder + LineBreakPlaceholder)
          diff.toB--;
        change = {
          from: from + diff.from,
          to: from + diff.toA,
          insert: Text.of(domChange.text.slice(diff.from, diff.toB).split(LineBreakPlaceholder))
        };
      }
    } else if (newSel && (!view.hasFocus && state.facet(editable) || sameSelPos(newSel, sel))) {
      newSel = null;
    }
    if (!change && !newSel)
      return false;
    if ((browser.mac || browser.android) && change && change.from == change.to && change.from == sel.head - 1 && /^\. ?$/.test(change.insert.toString()) && view.contentDOM.getAttribute("autocorrect") == "off") {
      if (newSel && change.insert.length == 2)
        newSel = EditorSelection.single(newSel.main.anchor - 1, newSel.main.head - 1);
      change = { from: change.from, to: change.to, insert: Text.of([change.insert.toString().replace(".", " ")]) };
    } else if (state.doc.lineAt(sel.from).to < sel.to && view.docView.lineHasWidget(sel.to) && view.inputState.insertingTextAt > Date.now() - 50) {
      change = {
        from: sel.from,
        to: sel.to,
        insert: state.toText(view.inputState.insertingText)
      };
    } else if (browser.chrome && change && change.from == change.to && change.from == sel.head && change.insert.toString() == "\n " && view.lineWrapping) {
      if (newSel)
        newSel = EditorSelection.single(newSel.main.anchor - 1, newSel.main.head - 1);
      change = { from: sel.from, to: sel.to, insert: Text.of([" "]) };
    }
    if (change) {
      return applyDOMChangeInner(view, change, newSel, lastKey);
    } else if (newSel && !sameSelPos(newSel, sel)) {
      let scrollIntoView3 = false, userEvent = "select";
      if (view.inputState.lastSelectionTime > Date.now() - 50) {
        if (view.inputState.lastSelectionOrigin == "select")
          scrollIntoView3 = true;
        userEvent = view.inputState.lastSelectionOrigin;
        if (userEvent == "select.pointer")
          newSel = skipAtomsForSelection(state.facet(atomicRanges).map((f) => f(view)), newSel);
      }
      view.dispatch({ selection: newSel, scrollIntoView: scrollIntoView3, userEvent });
      return true;
    } else {
      return false;
    }
  }
  function applyDOMChangeInner(view, change, newSel, lastKey = -1) {
    if (browser.ios && view.inputState.flushIOSKey(change))
      return true;
    let sel = view.state.selection.main;
    if (browser.android && (change.to == sel.to && // GBoard will sometimes remove a space it just inserted
    // after a completion when you press enter
    (change.from == sel.from || change.from == sel.from - 1 && view.state.sliceDoc(change.from, sel.from) == " ") && change.insert.length == 1 && change.insert.lines == 2 && dispatchKey(view.contentDOM, "Enter", 13) || (change.from == sel.from - 1 && change.to == sel.to && change.insert.length == 0 || lastKey == 8 && change.insert.length < change.to - change.from && change.to > sel.head) && dispatchKey(view.contentDOM, "Backspace", 8) || change.from == sel.from && change.to == sel.to + 1 && change.insert.length == 0 && dispatchKey(view.contentDOM, "Delete", 46)))
      return true;
    let text = change.insert.toString();
    if (view.inputState.composing >= 0)
      view.inputState.composing++;
    let defaultTr;
    let defaultInsert = () => defaultTr || (defaultTr = applyDefaultInsert(view, change, newSel));
    if (!view.state.facet(inputHandler).some((h) => h(view, change.from, change.to, text, defaultInsert)))
      view.dispatch(defaultInsert());
    return true;
  }
  function applyDefaultInsert(view, change, newSel) {
    let tr, startState = view.state, sel = startState.selection.main, inAtomic = -1;
    if (change.from == change.to && change.from < sel.from || change.from > sel.to) {
      let side = change.from < sel.from ? -1 : 1, pos = side < 0 ? sel.from : sel.to;
      let moved = skipAtomicRanges(startState.facet(atomicRanges).map((f) => f(view)), pos, side);
      if (change.from == moved)
        inAtomic = moved;
    }
    if (inAtomic > -1) {
      tr = {
        changes: change,
        selection: EditorSelection.cursor(change.from + change.insert.length, -1)
      };
    } else if (change.from >= sel.from && change.to <= sel.to && change.to - change.from >= (sel.to - sel.from) / 3 && (!newSel || newSel.main.empty && newSel.main.from == change.from + change.insert.length) && view.inputState.composing < 0) {
      let before = sel.from < change.from ? startState.sliceDoc(sel.from, change.from) : "";
      let after = sel.to > change.to ? startState.sliceDoc(change.to, sel.to) : "";
      tr = startState.replaceSelection(view.state.toText(before + change.insert.sliceString(0, void 0, view.state.lineBreak) + after));
    } else {
      let changes = startState.changes(change);
      let mainSel = newSel && newSel.main.to <= changes.newLength ? newSel.main : void 0;
      if (startState.selection.ranges.length > 1 && (view.inputState.composing >= 0 || view.inputState.compositionPendingChange) && change.to <= sel.to + 10 && change.to >= sel.to - 10) {
        let replaced = view.state.sliceDoc(change.from, change.to);
        let compositionRange, composition = newSel && findCompositionNode(view, newSel.main.head);
        if (composition) {
          let dLen = change.insert.length - (change.to - change.from);
          compositionRange = { from: composition.from, to: composition.to - dLen };
        } else {
          compositionRange = view.state.doc.lineAt(sel.head);
        }
        let offset = sel.to - change.to;
        tr = startState.changeByRange((range) => {
          if (range.from == sel.from && range.to == sel.to)
            return { changes, range: mainSel || range.map(changes) };
          let to = range.to - offset, from = to - replaced.length;
          if (view.state.sliceDoc(from, to) != replaced || // Unfortunately, there's no way to make multiple
          // changes in the same node work without aborting
          // composition, so cursors in the composition range are
          // ignored.
          to >= compositionRange.from && from <= compositionRange.to)
            return { range };
          let rangeChanges = startState.changes({ from, to, insert: change.insert }), selOff = range.to - sel.to;
          return {
            changes: rangeChanges,
            range: !mainSel ? range.map(rangeChanges) : EditorSelection.range(Math.max(0, mainSel.anchor + selOff), Math.max(0, mainSel.head + selOff))
          };
        });
      } else {
        tr = {
          changes,
          selection: mainSel && startState.selection.replaceRange(mainSel)
        };
      }
    }
    let userEvent = "input.type";
    if (view.composing || view.inputState.compositionPendingChange && view.inputState.compositionEndedAt > Date.now() - 50) {
      view.inputState.compositionPendingChange = false;
      userEvent += ".compose";
      if (view.inputState.compositionFirstChange) {
        userEvent += ".start";
        view.inputState.compositionFirstChange = false;
      }
    }
    return startState.update(tr, { userEvent, scrollIntoView: true });
  }
  function findDiff(a, b, preferredPos, preferredSide) {
    let minLen = Math.min(a.length, b.length);
    let from = 0;
    while (from < minLen && a.charCodeAt(from) == b.charCodeAt(from))
      from++;
    if (from == minLen && a.length == b.length)
      return null;
    let toA = a.length, toB = b.length;
    while (toA > 0 && toB > 0 && a.charCodeAt(toA - 1) == b.charCodeAt(toB - 1)) {
      toA--;
      toB--;
    }
    if (preferredSide == "end") {
      let adjust = Math.max(0, from - Math.min(toA, toB));
      preferredPos -= toA + adjust - from;
    }
    if (toA < from && a.length < b.length) {
      let move = preferredPos <= from && preferredPos >= toA ? from - preferredPos : 0;
      from -= move;
      toB = from + (toB - toA);
      toA = from;
    } else if (toB < from) {
      let move = preferredPos <= from && preferredPos >= toB ? from - preferredPos : 0;
      from -= move;
      toA = from + (toA - toB);
      toB = from;
    }
    return { from, toA, toB };
  }
  function selectionPoints(view) {
    let result = [];
    if (view.root.activeElement != view.contentDOM)
      return result;
    let { anchorNode, anchorOffset, focusNode, focusOffset } = view.observer.selectionRange;
    if (anchorNode) {
      result.push(new DOMPoint(anchorNode, anchorOffset));
      if (focusNode != anchorNode || focusOffset != anchorOffset)
        result.push(new DOMPoint(focusNode, focusOffset));
    }
    return result;
  }
  function selectionFromPoints(points, base2) {
    if (points.length == 0)
      return null;
    let anchor = points[0].pos, head = points.length == 2 ? points[1].pos : anchor;
    return anchor > -1 && head > -1 ? EditorSelection.single(anchor + base2, head + base2) : null;
  }
  function sameSelPos(selection, range) {
    return range.head == selection.main.head && range.anchor == selection.main.anchor;
  }
  var InputState = class {
    setSelectionOrigin(origin) {
      this.lastSelectionOrigin = origin;
      this.lastSelectionTime = Date.now();
    }
    constructor(view) {
      this.view = view;
      this.lastKeyCode = 0;
      this.lastKeyTime = 0;
      this.lastTouchTime = 0;
      this.lastFocusTime = 0;
      this.lastScrollTop = 0;
      this.lastScrollLeft = 0;
      this.lastWheelEvent = 0;
      this.pendingIOSKey = void 0;
      this.tabFocusMode = -1;
      this.lastSelectionOrigin = null;
      this.lastSelectionTime = 0;
      this.lastContextMenu = 0;
      this.scrollHandlers = [];
      this.handlers = /* @__PURE__ */ Object.create(null);
      this.composing = -1;
      this.compositionFirstChange = null;
      this.compositionEndedAt = 0;
      this.compositionPendingKey = false;
      this.compositionPendingChange = false;
      this.insertingText = "";
      this.insertingTextAt = 0;
      this.mouseSelection = null;
      this.draggedContent = null;
      this.handleEvent = this.handleEvent.bind(this);
      this.notifiedFocused = view.hasFocus;
      if (browser.safari)
        view.contentDOM.addEventListener("input", () => null);
      if (browser.gecko)
        firefoxCopyCutHack(view.contentDOM.ownerDocument);
    }
    handleEvent(event) {
      if (!eventBelongsToEditor(this.view, event) || this.ignoreDuringComposition(event))
        return;
      if (event.type == "keydown" && this.keydown(event))
        return;
      if (this.view.updateState != 0)
        Promise.resolve().then(() => this.runHandlers(event.type, event));
      else
        this.runHandlers(event.type, event);
    }
    runHandlers(type, event) {
      let handlers2 = this.handlers[type];
      if (handlers2) {
        for (let observer of handlers2.observers)
          observer(this.view, event);
        for (let handler of handlers2.handlers) {
          if (event.defaultPrevented)
            break;
          if (handler(this.view, event)) {
            event.preventDefault();
            break;
          }
        }
      }
    }
    ensureHandlers(plugins) {
      let handlers2 = computeHandlers(plugins), prev = this.handlers, dom = this.view.contentDOM;
      for (let type in handlers2)
        if (type != "scroll") {
          let passive = !handlers2[type].handlers.length;
          let exists = prev[type];
          if (exists && passive != !exists.handlers.length) {
            dom.removeEventListener(type, this.handleEvent);
            exists = null;
          }
          if (!exists)
            dom.addEventListener(type, this.handleEvent, { passive });
        }
      for (let type in prev)
        if (type != "scroll" && !handlers2[type])
          dom.removeEventListener(type, this.handleEvent);
      this.handlers = handlers2;
    }
    keydown(event) {
      this.lastKeyCode = event.keyCode;
      this.lastKeyTime = Date.now();
      if (event.keyCode == 9 && this.tabFocusMode > -1 && (!this.tabFocusMode || Date.now() <= this.tabFocusMode))
        return true;
      if (this.tabFocusMode > 0 && event.keyCode != 27 && modifierCodes.indexOf(event.keyCode) < 0)
        this.tabFocusMode = -1;
      if (browser.android && browser.chrome && !event.synthetic && (event.keyCode == 13 || event.keyCode == 8)) {
        this.view.observer.delayAndroidKey(event.key, event.keyCode);
        return true;
      }
      let pending;
      if (browser.ios && !event.synthetic && !event.altKey && !event.metaKey && ((pending = PendingKeys.find((key) => key.keyCode == event.keyCode)) && !event.ctrlKey || EmacsyPendingKeys.indexOf(event.key) > -1 && event.ctrlKey && !event.shiftKey)) {
        this.pendingIOSKey = pending || event;
        setTimeout(() => this.flushIOSKey(), 250);
        return true;
      }
      if (event.keyCode != 229)
        this.view.observer.forceFlush();
      return false;
    }
    flushIOSKey(change) {
      let key = this.pendingIOSKey;
      if (!key)
        return false;
      if (key.key == "Enter" && change && change.from < change.to && /^\S+$/.test(change.insert.toString()))
        return false;
      this.pendingIOSKey = void 0;
      return dispatchKey(this.view.contentDOM, key.key, key.keyCode, key instanceof KeyboardEvent ? key : void 0);
    }
    ignoreDuringComposition(event) {
      if (!/^key/.test(event.type) || event.synthetic)
        return false;
      if (this.composing > 0)
        return true;
      if (browser.safari && !browser.ios && this.compositionPendingKey && Date.now() - this.compositionEndedAt < 100) {
        this.compositionPendingKey = false;
        return true;
      }
      return false;
    }
    startMouseSelection(mouseSelection) {
      if (this.mouseSelection)
        this.mouseSelection.destroy();
      this.mouseSelection = mouseSelection;
    }
    update(update) {
      this.view.observer.update(update);
      if (this.mouseSelection)
        this.mouseSelection.update(update);
      if (this.draggedContent && update.docChanged)
        this.draggedContent = this.draggedContent.map(update.changes);
      if (update.transactions.length)
        this.lastKeyCode = this.lastSelectionTime = 0;
    }
    destroy() {
      if (this.mouseSelection)
        this.mouseSelection.destroy();
    }
  };
  function bindHandler(plugin, handler) {
    return (view, event) => {
      try {
        return handler.call(plugin, event, view);
      } catch (e) {
        logException(view.state, e);
      }
    };
  }
  function computeHandlers(plugins) {
    let result = /* @__PURE__ */ Object.create(null);
    function record(type) {
      return result[type] || (result[type] = { observers: [], handlers: [] });
    }
    for (let plugin of plugins) {
      let spec = plugin.spec, handlers2 = spec && spec.plugin.domEventHandlers, observers2 = spec && spec.plugin.domEventObservers;
      if (handlers2)
        for (let type in handlers2) {
          let f = handlers2[type];
          if (f)
            record(type).handlers.push(bindHandler(plugin.value, f));
        }
      if (observers2)
        for (let type in observers2) {
          let f = observers2[type];
          if (f)
            record(type).observers.push(bindHandler(plugin.value, f));
        }
    }
    for (let type in handlers)
      record(type).handlers.push(handlers[type]);
    for (let type in observers)
      record(type).observers.push(observers[type]);
    return result;
  }
  var PendingKeys = [
    { key: "Backspace", keyCode: 8, inputType: "deleteContentBackward" },
    { key: "Enter", keyCode: 13, inputType: "insertParagraph" },
    { key: "Enter", keyCode: 13, inputType: "insertLineBreak" },
    { key: "Delete", keyCode: 46, inputType: "deleteContentForward" }
  ];
  var EmacsyPendingKeys = "dthko";
  var modifierCodes = [16, 17, 18, 20, 91, 92, 224, 225];
  var dragScrollMargin = 6;
  function dragScrollSpeed(dist2) {
    return Math.max(0, dist2) * 0.7 + 8;
  }
  function dist(a, b) {
    return Math.max(Math.abs(a.clientX - b.clientX), Math.abs(a.clientY - b.clientY));
  }
  var MouseSelection = class {
    constructor(view, startEvent, style, mustSelect) {
      this.view = view;
      this.startEvent = startEvent;
      this.style = style;
      this.mustSelect = mustSelect;
      this.scrollSpeed = { x: 0, y: 0 };
      this.scrolling = -1;
      this.lastEvent = startEvent;
      this.scrollParents = scrollableParents(view.contentDOM);
      this.atoms = view.state.facet(atomicRanges).map((f) => f(view));
      let doc2 = view.contentDOM.ownerDocument;
      doc2.addEventListener("mousemove", this.move = this.move.bind(this));
      doc2.addEventListener("mouseup", this.up = this.up.bind(this));
      this.extend = startEvent.shiftKey;
      this.multiple = view.state.facet(EditorState.allowMultipleSelections) && addsSelectionRange(view, startEvent);
      this.dragging = isInPrimarySelection(view, startEvent) && getClickType(startEvent) == 1 ? null : false;
    }
    start(event) {
      if (this.dragging === false)
        this.select(event);
    }
    move(event) {
      if (event.buttons == 0)
        return this.destroy();
      if (this.dragging || this.dragging == null && dist(this.startEvent, event) < 10)
        return;
      this.select(this.lastEvent = event);
      let sx = 0, sy = 0;
      let left = 0, top2 = 0, right = this.view.win.innerWidth, bottom = this.view.win.innerHeight;
      if (this.scrollParents.x)
        ({ left, right } = this.scrollParents.x.getBoundingClientRect());
      if (this.scrollParents.y)
        ({ top: top2, bottom } = this.scrollParents.y.getBoundingClientRect());
      let margins = getScrollMargins(this.view);
      if (event.clientX - margins.left <= left + dragScrollMargin)
        sx = -dragScrollSpeed(left - event.clientX);
      else if (event.clientX + margins.right >= right - dragScrollMargin)
        sx = dragScrollSpeed(event.clientX - right);
      if (event.clientY - margins.top <= top2 + dragScrollMargin)
        sy = -dragScrollSpeed(top2 - event.clientY);
      else if (event.clientY + margins.bottom >= bottom - dragScrollMargin)
        sy = dragScrollSpeed(event.clientY - bottom);
      this.setScrollSpeed(sx, sy);
    }
    up(event) {
      if (this.dragging == null)
        this.select(this.lastEvent);
      if (!this.dragging)
        event.preventDefault();
      this.destroy();
    }
    destroy() {
      this.setScrollSpeed(0, 0);
      let doc2 = this.view.contentDOM.ownerDocument;
      doc2.removeEventListener("mousemove", this.move);
      doc2.removeEventListener("mouseup", this.up);
      this.view.inputState.mouseSelection = this.view.inputState.draggedContent = null;
    }
    setScrollSpeed(sx, sy) {
      this.scrollSpeed = { x: sx, y: sy };
      if (sx || sy) {
        if (this.scrolling < 0)
          this.scrolling = setInterval(() => this.scroll(), 50);
      } else if (this.scrolling > -1) {
        clearInterval(this.scrolling);
        this.scrolling = -1;
      }
    }
    scroll() {
      let { x, y } = this.scrollSpeed;
      if (x && this.scrollParents.x) {
        this.scrollParents.x.scrollLeft += x;
        x = 0;
      }
      if (y && this.scrollParents.y) {
        this.scrollParents.y.scrollTop += y;
        y = 0;
      }
      if (x || y)
        this.view.win.scrollBy(x, y);
      if (this.dragging === false)
        this.select(this.lastEvent);
    }
    select(event) {
      let { view } = this, selection = skipAtomsForSelection(this.atoms, this.style.get(event, this.extend, this.multiple));
      if (this.mustSelect || !selection.eq(view.state.selection, this.dragging === false))
        this.view.dispatch({
          selection,
          userEvent: "select.pointer"
        });
      this.mustSelect = false;
    }
    update(update) {
      if (update.transactions.some((tr) => tr.isUserEvent("input.type")))
        this.destroy();
      else if (this.style.update(update))
        setTimeout(() => this.select(this.lastEvent), 20);
    }
  };
  function addsSelectionRange(view, event) {
    let facet = view.state.facet(clickAddsSelectionRange);
    return facet.length ? facet[0](event) : browser.mac ? event.metaKey : event.ctrlKey;
  }
  function dragMovesSelection(view, event) {
    let facet = view.state.facet(dragMovesSelection$1);
    return facet.length ? facet[0](event) : browser.mac ? !event.altKey : !event.ctrlKey;
  }
  function isInPrimarySelection(view, event) {
    let { main } = view.state.selection;
    if (main.empty)
      return false;
    let sel = getSelection(view.root);
    if (!sel || sel.rangeCount == 0)
      return true;
    let rects = sel.getRangeAt(0).getClientRects();
    for (let i = 0; i < rects.length; i++) {
      let rect = rects[i];
      if (rect.left <= event.clientX && rect.right >= event.clientX && rect.top <= event.clientY && rect.bottom >= event.clientY)
        return true;
    }
    return false;
  }
  function eventBelongsToEditor(view, event) {
    if (!event.bubbles)
      return true;
    if (event.defaultPrevented)
      return false;
    for (let node = event.target, tile; node != view.contentDOM; node = node.parentNode)
      if (!node || node.nodeType == 11 || (tile = Tile.get(node)) && tile.isWidget() && !tile.isHidden && tile.widget.ignoreEvent(event))
        return false;
    return true;
  }
  var handlers = /* @__PURE__ */ Object.create(null);
  var observers = /* @__PURE__ */ Object.create(null);
  var brokenClipboardAPI = browser.ie && browser.ie_version < 15 || browser.ios && browser.webkit_version < 604;
  function capturePaste(view) {
    let parent = view.dom.parentNode;
    if (!parent)
      return;
    let target = parent.appendChild(document.createElement("textarea"));
    target.style.cssText = "position: fixed; left: -10000px; top: 10px";
    target.focus();
    setTimeout(() => {
      view.focus();
      target.remove();
      doPaste(view, target.value);
    }, 50);
  }
  function textFilter(state, facet, text) {
    for (let filter of state.facet(facet))
      text = filter(text, state);
    return text;
  }
  function doPaste(view, input) {
    input = textFilter(view.state, clipboardInputFilter, input);
    let { state } = view, changes, i = 1, text = state.toText(input);
    let byLine = text.lines == state.selection.ranges.length;
    let linewise = lastLinewiseCopy != null && state.selection.ranges.every((r) => r.empty) && lastLinewiseCopy == text.toString();
    if (linewise) {
      let lastLine = -1;
      changes = state.changeByRange((range) => {
        let line = state.doc.lineAt(range.from);
        if (line.from == lastLine)
          return { range };
        lastLine = line.from;
        let insert2 = state.toText((byLine ? text.line(i++).text : input) + state.lineBreak);
        return {
          changes: { from: line.from, insert: insert2 },
          range: EditorSelection.cursor(range.from + insert2.length)
        };
      });
    } else if (byLine) {
      changes = state.changeByRange((range) => {
        let line = text.line(i++);
        return {
          changes: { from: range.from, to: range.to, insert: line.text },
          range: EditorSelection.cursor(range.from + line.length)
        };
      });
    } else {
      changes = state.replaceSelection(text);
    }
    view.dispatch(changes, {
      userEvent: "input.paste",
      scrollIntoView: true
    });
  }
  observers.scroll = (view) => {
    view.inputState.lastScrollTop = view.scrollDOM.scrollTop;
    view.inputState.lastScrollLeft = view.scrollDOM.scrollLeft;
  };
  observers.wheel = observers.mousewheel = (view) => {
    view.inputState.lastWheelEvent = Date.now();
  };
  handlers.keydown = (view, event) => {
    view.inputState.setSelectionOrigin("select");
    if (event.keyCode == 27 && view.inputState.tabFocusMode != 0)
      view.inputState.tabFocusMode = Date.now() + 2e3;
    return false;
  };
  observers.touchstart = (view, e) => {
    view.inputState.lastTouchTime = Date.now();
    view.inputState.setSelectionOrigin("select.pointer");
  };
  observers.touchmove = (view) => {
    view.inputState.setSelectionOrigin("select.pointer");
  };
  handlers.mousedown = (view, event) => {
    view.observer.flush();
    if (view.inputState.lastTouchTime > Date.now() - 2e3)
      return false;
    let style = null;
    for (let makeStyle of view.state.facet(mouseSelectionStyle)) {
      style = makeStyle(view, event);
      if (style)
        break;
    }
    if (!style && event.button == 0)
      style = basicMouseSelection(view, event);
    if (style) {
      let mustFocus = !view.hasFocus;
      view.inputState.startMouseSelection(new MouseSelection(view, event, style, mustFocus));
      if (mustFocus)
        view.observer.ignore(() => {
          focusPreventScroll(view.contentDOM);
          let active = view.root.activeElement;
          if (active && !active.contains(view.contentDOM))
            active.blur();
        });
      let mouseSel = view.inputState.mouseSelection;
      if (mouseSel) {
        mouseSel.start(event);
        return mouseSel.dragging === false;
      }
    } else {
      view.inputState.setSelectionOrigin("select.pointer");
    }
    return false;
  };
  function rangeForClick(view, pos, bias, type) {
    if (type == 1) {
      return EditorSelection.cursor(pos, bias);
    } else if (type == 2) {
      return groupAt(view.state, pos, bias);
    } else {
      let visual = view.docView.lineAt(pos, bias), line = view.state.doc.lineAt(visual ? visual.posAtEnd : pos);
      let from = visual ? visual.posAtStart : line.from, to = visual ? visual.posAtEnd : line.to;
      if (to < view.state.doc.length && to == line.to)
        to++;
      return EditorSelection.range(from, to);
    }
  }
  var BadMouseDetail = browser.ie && browser.ie_version <= 11;
  var lastMouseDown = null;
  var lastMouseDownCount = 0;
  var lastMouseDownTime = 0;
  function getClickType(event) {
    if (!BadMouseDetail)
      return event.detail;
    let last = lastMouseDown, lastTime = lastMouseDownTime;
    lastMouseDown = event;
    lastMouseDownTime = Date.now();
    return lastMouseDownCount = !last || lastTime > Date.now() - 400 && Math.abs(last.clientX - event.clientX) < 2 && Math.abs(last.clientY - event.clientY) < 2 ? (lastMouseDownCount + 1) % 3 : 1;
  }
  function basicMouseSelection(view, event) {
    let start = view.posAndSideAtCoords({ x: event.clientX, y: event.clientY }, false), type = getClickType(event);
    let startSel = view.state.selection;
    return {
      update(update) {
        if (update.docChanged) {
          start.pos = update.changes.mapPos(start.pos);
          startSel = startSel.map(update.changes);
        }
      },
      get(event2, extend, multiple) {
        let cur2 = view.posAndSideAtCoords({ x: event2.clientX, y: event2.clientY }, false), removed;
        let range = rangeForClick(view, cur2.pos, cur2.assoc, type);
        if (start.pos != cur2.pos && !extend) {
          let startRange = rangeForClick(view, start.pos, start.assoc, type);
          let from = Math.min(startRange.from, range.from), to = Math.max(startRange.to, range.to);
          range = from < range.from ? EditorSelection.range(from, to) : EditorSelection.range(to, from);
        }
        if (extend)
          return startSel.replaceRange(startSel.main.extend(range.from, range.to));
        else if (multiple && type == 1 && startSel.ranges.length > 1 && (removed = removeRangeAround(startSel, cur2.pos)))
          return removed;
        else if (multiple)
          return startSel.addRange(range);
        else
          return EditorSelection.create([range]);
      }
    };
  }
  function removeRangeAround(sel, pos) {
    for (let i = 0; i < sel.ranges.length; i++) {
      let { from, to } = sel.ranges[i];
      if (from <= pos && to >= pos)
        return EditorSelection.create(sel.ranges.slice(0, i).concat(sel.ranges.slice(i + 1)), sel.mainIndex == i ? 0 : sel.mainIndex - (sel.mainIndex > i ? 1 : 0));
    }
    return null;
  }
  handlers.dragstart = (view, event) => {
    let { selection: { main: range } } = view.state;
    if (event.target.draggable) {
      let tile = view.docView.tile.nearest(event.target);
      if (tile && tile.isWidget()) {
        let from = tile.posAtStart, to = from + tile.length;
        if (from >= range.to || to <= range.from)
          range = EditorSelection.range(from, to);
      }
    }
    let { inputState } = view;
    if (inputState.mouseSelection)
      inputState.mouseSelection.dragging = true;
    inputState.draggedContent = range;
    if (event.dataTransfer) {
      event.dataTransfer.setData("Text", textFilter(view.state, clipboardOutputFilter, view.state.sliceDoc(range.from, range.to)));
      event.dataTransfer.effectAllowed = "copyMove";
    }
    return false;
  };
  handlers.dragend = (view) => {
    view.inputState.draggedContent = null;
    return false;
  };
  function dropText(view, event, text, direct) {
    text = textFilter(view.state, clipboardInputFilter, text);
    if (!text)
      return;
    let dropPos = view.posAtCoords({ x: event.clientX, y: event.clientY }, false);
    let { draggedContent } = view.inputState;
    let del = direct && draggedContent && dragMovesSelection(view, event) ? { from: draggedContent.from, to: draggedContent.to } : null;
    let ins = { from: dropPos, insert: text };
    let changes = view.state.changes(del ? [del, ins] : ins);
    view.focus();
    view.dispatch({
      changes,
      selection: { anchor: changes.mapPos(dropPos, -1), head: changes.mapPos(dropPos, 1) },
      userEvent: del ? "move.drop" : "input.drop"
    });
    view.inputState.draggedContent = null;
  }
  handlers.drop = (view, event) => {
    if (!event.dataTransfer)
      return false;
    if (view.state.readOnly)
      return true;
    let files = event.dataTransfer.files;
    if (files && files.length) {
      let text = Array(files.length), read = 0;
      let finishFile = () => {
        if (++read == files.length)
          dropText(view, event, text.filter((s) => s != null).join(view.state.lineBreak), false);
      };
      for (let i = 0; i < files.length; i++) {
        let reader = new FileReader();
        reader.onerror = finishFile;
        reader.onload = () => {
          if (!/[\x00-\x08\x0e-\x1f]{2}/.test(reader.result))
            text[i] = reader.result;
          finishFile();
        };
        reader.readAsText(files[i]);
      }
      return true;
    } else {
      let text = event.dataTransfer.getData("Text");
      if (text) {
        dropText(view, event, text, true);
        return true;
      }
    }
    return false;
  };
  handlers.paste = (view, event) => {
    if (view.state.readOnly)
      return true;
    view.observer.flush();
    let data = brokenClipboardAPI ? null : event.clipboardData;
    if (data) {
      doPaste(view, data.getData("text/plain") || data.getData("text/uri-list"));
      return true;
    } else {
      capturePaste(view);
      return false;
    }
  };
  function captureCopy(view, text) {
    let parent = view.dom.parentNode;
    if (!parent)
      return;
    let target = parent.appendChild(document.createElement("textarea"));
    target.style.cssText = "position: fixed; left: -10000px; top: 10px";
    target.value = text;
    target.focus();
    target.selectionEnd = text.length;
    target.selectionStart = 0;
    setTimeout(() => {
      target.remove();
      view.focus();
    }, 50);
  }
  function copiedRange(state) {
    let content2 = [], ranges = [], linewise = false;
    for (let range of state.selection.ranges)
      if (!range.empty) {
        content2.push(state.sliceDoc(range.from, range.to));
        ranges.push(range);
      }
    if (!content2.length) {
      let upto = -1;
      for (let { from } of state.selection.ranges) {
        let line = state.doc.lineAt(from);
        if (line.number > upto) {
          content2.push(line.text);
          ranges.push({ from: line.from, to: Math.min(state.doc.length, line.to + 1) });
        }
        upto = line.number;
      }
      linewise = true;
    }
    return { text: textFilter(state, clipboardOutputFilter, content2.join(state.lineBreak)), ranges, linewise };
  }
  var lastLinewiseCopy = null;
  handlers.copy = handlers.cut = (view, event) => {
    if (!hasSelection(view.contentDOM, view.observer.selectionRange))
      return false;
    let { text, ranges, linewise } = copiedRange(view.state);
    if (!text && !linewise)
      return false;
    lastLinewiseCopy = linewise ? text : null;
    if (event.type == "cut" && !view.state.readOnly)
      view.dispatch({
        changes: ranges,
        scrollIntoView: true,
        userEvent: "delete.cut"
      });
    let data = brokenClipboardAPI ? null : event.clipboardData;
    if (data) {
      data.clearData();
      data.setData("text/plain", text);
      return true;
    } else {
      captureCopy(view, text);
      return false;
    }
  };
  var isFocusChange = /* @__PURE__ */ Annotation.define();
  function focusChangeTransaction(state, focus) {
    let effects = [];
    for (let getEffect of state.facet(focusChangeEffect)) {
      let effect = getEffect(state, focus);
      if (effect)
        effects.push(effect);
    }
    return effects.length ? state.update({ effects, annotations: isFocusChange.of(true) }) : null;
  }
  function updateForFocusChange(view) {
    setTimeout(() => {
      let focus = view.hasFocus;
      if (focus != view.inputState.notifiedFocused) {
        let tr = focusChangeTransaction(view.state, focus);
        if (tr)
          view.dispatch(tr);
        else
          view.update([]);
      }
    }, 10);
  }
  observers.focus = (view) => {
    view.inputState.lastFocusTime = Date.now();
    if (!view.scrollDOM.scrollTop && (view.inputState.lastScrollTop || view.inputState.lastScrollLeft)) {
      view.scrollDOM.scrollTop = view.inputState.lastScrollTop;
      view.scrollDOM.scrollLeft = view.inputState.lastScrollLeft;
    }
    updateForFocusChange(view);
  };
  observers.blur = (view) => {
    view.observer.clearSelectionRange();
    updateForFocusChange(view);
  };
  observers.compositionstart = observers.compositionupdate = (view) => {
    if (view.observer.editContext)
      return;
    if (view.inputState.compositionFirstChange == null)
      view.inputState.compositionFirstChange = true;
    if (view.inputState.composing < 0) {
      view.inputState.composing = 0;
    }
  };
  observers.compositionend = (view) => {
    if (view.observer.editContext)
      return;
    view.inputState.composing = -1;
    view.inputState.compositionEndedAt = Date.now();
    view.inputState.compositionPendingKey = true;
    view.inputState.compositionPendingChange = view.observer.pendingRecords().length > 0;
    view.inputState.compositionFirstChange = null;
    if (browser.chrome && browser.android) {
      view.observer.flushSoon();
    } else if (view.inputState.compositionPendingChange) {
      Promise.resolve().then(() => view.observer.flush());
    } else {
      setTimeout(() => {
        if (view.inputState.composing < 0 && view.docView.hasComposition)
          view.update([]);
      }, 50);
    }
  };
  observers.contextmenu = (view) => {
    view.inputState.lastContextMenu = Date.now();
  };
  handlers.beforeinput = (view, event) => {
    var _a2, _b;
    if (event.inputType == "insertText" || event.inputType == "insertCompositionText") {
      view.inputState.insertingText = event.data;
      view.inputState.insertingTextAt = Date.now();
    }
    if (event.inputType == "insertReplacementText" && view.observer.editContext) {
      let text = (_a2 = event.dataTransfer) === null || _a2 === void 0 ? void 0 : _a2.getData("text/plain"), ranges = event.getTargetRanges();
      if (text && ranges.length) {
        let r = ranges[0];
        let from = view.posAtDOM(r.startContainer, r.startOffset), to = view.posAtDOM(r.endContainer, r.endOffset);
        applyDOMChangeInner(view, { from, to, insert: view.state.toText(text) }, null);
        return true;
      }
    }
    let pending;
    if (browser.chrome && browser.android && (pending = PendingKeys.find((key) => key.inputType == event.inputType))) {
      view.observer.delayAndroidKey(pending.key, pending.keyCode);
      if (pending.key == "Backspace" || pending.key == "Delete") {
        let startViewHeight = ((_b = window.visualViewport) === null || _b === void 0 ? void 0 : _b.height) || 0;
        setTimeout(() => {
          var _a3;
          if ((((_a3 = window.visualViewport) === null || _a3 === void 0 ? void 0 : _a3.height) || 0) > startViewHeight + 10 && view.hasFocus) {
            view.contentDOM.blur();
            view.focus();
          }
        }, 100);
      }
    }
    if (browser.ios && event.inputType == "deleteContentForward") {
      view.observer.flushSoon();
    }
    if (browser.safari && event.inputType == "insertText" && view.inputState.composing >= 0) {
      setTimeout(() => observers.compositionend(view, event), 20);
    }
    return false;
  };
  var appliedFirefoxHack = /* @__PURE__ */ new Set();
  function firefoxCopyCutHack(doc2) {
    if (!appliedFirefoxHack.has(doc2)) {
      appliedFirefoxHack.add(doc2);
      doc2.addEventListener("copy", () => {
      });
      doc2.addEventListener("cut", () => {
      });
    }
  }
  var wrappingWhiteSpace = ["pre-wrap", "normal", "pre-line", "break-spaces"];
  var heightChangeFlag = false;
  function clearHeightChangeFlag() {
    heightChangeFlag = false;
  }
  var HeightOracle = class {
    constructor(lineWrapping) {
      this.lineWrapping = lineWrapping;
      this.doc = Text.empty;
      this.heightSamples = {};
      this.lineHeight = 14;
      this.charWidth = 7;
      this.textHeight = 14;
      this.lineLength = 30;
    }
    heightForGap(from, to) {
      let lines = this.doc.lineAt(to).number - this.doc.lineAt(from).number + 1;
      if (this.lineWrapping)
        lines += Math.max(0, Math.ceil((to - from - lines * this.lineLength * 0.5) / this.lineLength));
      return this.lineHeight * lines;
    }
    heightForLine(length) {
      if (!this.lineWrapping)
        return this.lineHeight;
      let lines = 1 + Math.max(0, Math.ceil((length - this.lineLength) / Math.max(1, this.lineLength - 5)));
      return lines * this.lineHeight;
    }
    setDoc(doc2) {
      this.doc = doc2;
      return this;
    }
    mustRefreshForWrapping(whiteSpace) {
      return wrappingWhiteSpace.indexOf(whiteSpace) > -1 != this.lineWrapping;
    }
    mustRefreshForHeights(lineHeights) {
      let newHeight = false;
      for (let i = 0; i < lineHeights.length; i++) {
        let h = lineHeights[i];
        if (h < 0) {
          i++;
        } else if (!this.heightSamples[Math.floor(h * 10)]) {
          newHeight = true;
          this.heightSamples[Math.floor(h * 10)] = true;
        }
      }
      return newHeight;
    }
    refresh(whiteSpace, lineHeight, charWidth, textHeight, lineLength, knownHeights) {
      let lineWrapping = wrappingWhiteSpace.indexOf(whiteSpace) > -1;
      let changed = Math.abs(lineHeight - this.lineHeight) > 0.3 || this.lineWrapping != lineWrapping || Math.abs(charWidth - this.charWidth) > 0.1;
      this.lineWrapping = lineWrapping;
      this.lineHeight = lineHeight;
      this.charWidth = charWidth;
      this.textHeight = textHeight;
      this.lineLength = lineLength;
      if (changed) {
        this.heightSamples = {};
        for (let i = 0; i < knownHeights.length; i++) {
          let h = knownHeights[i];
          if (h < 0)
            i++;
          else
            this.heightSamples[Math.floor(h * 10)] = true;
        }
      }
      return changed;
    }
  };
  var MeasuredHeights = class {
    constructor(from, heights) {
      this.from = from;
      this.heights = heights;
      this.index = 0;
    }
    get more() {
      return this.index < this.heights.length;
    }
  };
  var BlockInfo = class _BlockInfo {
    /**
    @internal
    */
    constructor(from, length, top2, height, _content) {
      this.from = from;
      this.length = length;
      this.top = top2;
      this.height = height;
      this._content = _content;
    }
    /**
    The type of element this is. When querying lines, this may be
    an array of all the blocks that make up the line.
    */
    get type() {
      return typeof this._content == "number" ? BlockType.Text : Array.isArray(this._content) ? this._content : this._content.type;
    }
    /**
    The end of the element as a document position.
    */
    get to() {
      return this.from + this.length;
    }
    /**
    The bottom position of the element.
    */
    get bottom() {
      return this.top + this.height;
    }
    /**
    If this is a widget block, this will return the widget
    associated with it.
    */
    get widget() {
      return this._content instanceof PointDecoration ? this._content.widget : null;
    }
    /**
    If this is a textblock, this holds the number of line breaks
    that appear in widgets inside the block.
    */
    get widgetLineBreaks() {
      return typeof this._content == "number" ? this._content : 0;
    }
    /**
    @internal
    */
    join(other) {
      let content2 = (Array.isArray(this._content) ? this._content : [this]).concat(Array.isArray(other._content) ? other._content : [other]);
      return new _BlockInfo(this.from, this.length + other.length, this.top, this.height + other.height, content2);
    }
  };
  var QueryType = /* @__PURE__ */ (function(QueryType3) {
    QueryType3[QueryType3["ByPos"] = 0] = "ByPos";
    QueryType3[QueryType3["ByHeight"] = 1] = "ByHeight";
    QueryType3[QueryType3["ByPosNoHeight"] = 2] = "ByPosNoHeight";
    return QueryType3;
  })(QueryType || (QueryType = {}));
  var Epsilon = 1e-3;
  var HeightMap = class _HeightMap {
    constructor(length, height, flags = 2) {
      this.length = length;
      this.height = height;
      this.flags = flags;
    }
    get outdated() {
      return (this.flags & 2) > 0;
    }
    set outdated(value) {
      this.flags = (value ? 2 : 0) | this.flags & ~2;
    }
    setHeight(height) {
      if (this.height != height) {
        if (Math.abs(this.height - height) > Epsilon)
          heightChangeFlag = true;
        this.height = height;
      }
    }
    // Base case is to replace a leaf node, which simply builds a tree
    // from the new nodes and returns that (HeightMapBranch and
    // HeightMapGap override this to actually use from/to)
    replace(_from, _to, nodes) {
      return _HeightMap.of(nodes);
    }
    // Again, these are base cases, and are overridden for branch and gap nodes.
    decomposeLeft(_to, result) {
      result.push(this);
    }
    decomposeRight(_from, result) {
      result.push(this);
    }
    applyChanges(decorations2, oldDoc, oracle, changes) {
      let me = this, doc2 = oracle.doc;
      for (let i = changes.length - 1; i >= 0; i--) {
        let { fromA, toA, fromB, toB } = changes[i];
        let start = me.lineAt(fromA, QueryType.ByPosNoHeight, oracle.setDoc(oldDoc), 0, 0);
        let end = start.to >= toA ? start : me.lineAt(toA, QueryType.ByPosNoHeight, oracle, 0, 0);
        toB += end.to - toA;
        toA = end.to;
        while (i > 0 && start.from <= changes[i - 1].toA) {
          fromA = changes[i - 1].fromA;
          fromB = changes[i - 1].fromB;
          i--;
          if (fromA < start.from)
            start = me.lineAt(fromA, QueryType.ByPosNoHeight, oracle, 0, 0);
        }
        fromB += start.from - fromA;
        fromA = start.from;
        let nodes = NodeBuilder.build(oracle.setDoc(doc2), decorations2, fromB, toB);
        me = replace(me, me.replace(fromA, toA, nodes));
      }
      return me.updateHeight(oracle, 0);
    }
    static empty() {
      return new HeightMapText(0, 0, 0);
    }
    // nodes uses null values to indicate the position of line breaks.
    // There are never line breaks at the start or end of the array, or
    // two line breaks next to each other, and the array isn't allowed
    // to be empty (same restrictions as return value from the builder).
    static of(nodes) {
      if (nodes.length == 1)
        return nodes[0];
      let i = 0, j = nodes.length, before = 0, after = 0;
      for (; ; ) {
        if (i == j) {
          if (before > after * 2) {
            let split = nodes[i - 1];
            if (split.break)
              nodes.splice(--i, 1, split.left, null, split.right);
            else
              nodes.splice(--i, 1, split.left, split.right);
            j += 1 + split.break;
            before -= split.size;
          } else if (after > before * 2) {
            let split = nodes[j];
            if (split.break)
              nodes.splice(j, 1, split.left, null, split.right);
            else
              nodes.splice(j, 1, split.left, split.right);
            j += 2 + split.break;
            after -= split.size;
          } else {
            break;
          }
        } else if (before < after) {
          let next = nodes[i++];
          if (next)
            before += next.size;
        } else {
          let next = nodes[--j];
          if (next)
            after += next.size;
        }
      }
      let brk = 0;
      if (nodes[i - 1] == null) {
        brk = 1;
        i--;
      } else if (nodes[i] == null) {
        brk = 1;
        j++;
      }
      return new HeightMapBranch(_HeightMap.of(nodes.slice(0, i)), brk, _HeightMap.of(nodes.slice(j)));
    }
  };
  function replace(old, val) {
    if (old == val)
      return old;
    if (old.constructor != val.constructor)
      heightChangeFlag = true;
    return val;
  }
  HeightMap.prototype.size = 1;
  var SpaceDeco = /* @__PURE__ */ Decoration.replace({});
  var HeightMapBlock = class extends HeightMap {
    constructor(length, height, deco) {
      super(length, height);
      this.deco = deco;
      this.spaceAbove = 0;
    }
    mainBlock(top2, offset) {
      return new BlockInfo(offset, this.length, top2 + this.spaceAbove, this.height - this.spaceAbove, this.deco || 0);
    }
    blockAt(height, _oracle, top2, offset) {
      return this.spaceAbove && height < top2 + this.spaceAbove ? new BlockInfo(offset, 0, top2, this.spaceAbove, SpaceDeco) : this.mainBlock(top2, offset);
    }
    lineAt(_value, _type, oracle, top2, offset) {
      let main = this.mainBlock(top2, offset);
      return this.spaceAbove ? this.blockAt(0, oracle, top2, offset).join(main) : main;
    }
    forEachLine(from, to, oracle, top2, offset, f) {
      if (from <= offset + this.length && to >= offset)
        f(this.lineAt(0, QueryType.ByPos, oracle, top2, offset));
    }
    setMeasuredHeight(measured) {
      let next = measured.heights[measured.index++];
      if (next < 0) {
        this.spaceAbove = -next;
        next = measured.heights[measured.index++];
      } else {
        this.spaceAbove = 0;
      }
      this.setHeight(next);
    }
    updateHeight(oracle, offset = 0, _force = false, measured) {
      if (measured && measured.from <= offset && measured.more)
        this.setMeasuredHeight(measured);
      this.outdated = false;
      return this;
    }
    toString() {
      return `block(${this.length})`;
    }
  };
  var HeightMapText = class _HeightMapText extends HeightMapBlock {
    constructor(length, height, above) {
      super(length, height, null);
      this.collapsed = 0;
      this.widgetHeight = 0;
      this.breaks = 0;
      this.spaceAbove = above;
    }
    mainBlock(top2, offset) {
      return new BlockInfo(offset, this.length, top2 + this.spaceAbove, this.height - this.spaceAbove, this.breaks);
    }
    replace(_from, _to, nodes) {
      let node = nodes[0];
      if (nodes.length == 1 && (node instanceof _HeightMapText || node instanceof HeightMapGap && node.flags & 4) && Math.abs(this.length - node.length) < 10) {
        if (node instanceof HeightMapGap)
          node = new _HeightMapText(node.length, this.height, this.spaceAbove);
        else
          node.height = this.height;
        if (!this.outdated)
          node.outdated = false;
        return node;
      } else {
        return HeightMap.of(nodes);
      }
    }
    updateHeight(oracle, offset = 0, force = false, measured) {
      if (measured && measured.from <= offset && measured.more) {
        this.setMeasuredHeight(measured);
      } else if (force || this.outdated) {
        this.spaceAbove = 0;
        this.setHeight(Math.max(this.widgetHeight, oracle.heightForLine(this.length - this.collapsed)) + this.breaks * oracle.lineHeight);
      }
      this.outdated = false;
      return this;
    }
    toString() {
      return `line(${this.length}${this.collapsed ? -this.collapsed : ""}${this.widgetHeight ? ":" + this.widgetHeight : ""})`;
    }
  };
  var HeightMapGap = class _HeightMapGap extends HeightMap {
    constructor(length) {
      super(length, 0);
    }
    heightMetrics(oracle, offset) {
      let firstLine = oracle.doc.lineAt(offset).number, lastLine = oracle.doc.lineAt(offset + this.length).number;
      let lines = lastLine - firstLine + 1;
      let perLine, perChar = 0;
      if (oracle.lineWrapping) {
        let totalPerLine = Math.min(this.height, oracle.lineHeight * lines);
        perLine = totalPerLine / lines;
        if (this.length > lines + 1)
          perChar = (this.height - totalPerLine) / (this.length - lines - 1);
      } else {
        perLine = this.height / lines;
      }
      return { firstLine, lastLine, perLine, perChar };
    }
    blockAt(height, oracle, top2, offset) {
      let { firstLine, lastLine, perLine, perChar } = this.heightMetrics(oracle, offset);
      if (oracle.lineWrapping) {
        let guess = offset + (height < oracle.lineHeight ? 0 : Math.round(Math.max(0, Math.min(1, (height - top2) / this.height)) * this.length));
        let line = oracle.doc.lineAt(guess), lineHeight = perLine + line.length * perChar;
        let lineTop = Math.max(top2, height - lineHeight / 2);
        return new BlockInfo(line.from, line.length, lineTop, lineHeight, 0);
      } else {
        let line = Math.max(0, Math.min(lastLine - firstLine, Math.floor((height - top2) / perLine)));
        let { from, length } = oracle.doc.line(firstLine + line);
        return new BlockInfo(from, length, top2 + perLine * line, perLine, 0);
      }
    }
    lineAt(value, type, oracle, top2, offset) {
      if (type == QueryType.ByHeight)
        return this.blockAt(value, oracle, top2, offset);
      if (type == QueryType.ByPosNoHeight) {
        let { from, to } = oracle.doc.lineAt(value);
        return new BlockInfo(from, to - from, 0, 0, 0);
      }
      let { firstLine, perLine, perChar } = this.heightMetrics(oracle, offset);
      let line = oracle.doc.lineAt(value), lineHeight = perLine + line.length * perChar;
      let linesAbove = line.number - firstLine;
      let lineTop = top2 + perLine * linesAbove + perChar * (line.from - offset - linesAbove);
      return new BlockInfo(line.from, line.length, Math.max(top2, Math.min(lineTop, top2 + this.height - lineHeight)), lineHeight, 0);
    }
    forEachLine(from, to, oracle, top2, offset, f) {
      from = Math.max(from, offset);
      to = Math.min(to, offset + this.length);
      let { firstLine, perLine, perChar } = this.heightMetrics(oracle, offset);
      for (let pos = from, lineTop = top2; pos <= to; ) {
        let line = oracle.doc.lineAt(pos);
        if (pos == from) {
          let linesAbove = line.number - firstLine;
          lineTop += perLine * linesAbove + perChar * (from - offset - linesAbove);
        }
        let lineHeight = perLine + perChar * line.length;
        f(new BlockInfo(line.from, line.length, lineTop, lineHeight, 0));
        lineTop += lineHeight;
        pos = line.to + 1;
      }
    }
    replace(from, to, nodes) {
      let after = this.length - to;
      if (after > 0) {
        let last = nodes[nodes.length - 1];
        if (last instanceof _HeightMapGap)
          nodes[nodes.length - 1] = new _HeightMapGap(last.length + after);
        else
          nodes.push(null, new _HeightMapGap(after - 1));
      }
      if (from > 0) {
        let first = nodes[0];
        if (first instanceof _HeightMapGap)
          nodes[0] = new _HeightMapGap(from + first.length);
        else
          nodes.unshift(new _HeightMapGap(from - 1), null);
      }
      return HeightMap.of(nodes);
    }
    decomposeLeft(to, result) {
      result.push(new _HeightMapGap(to - 1), null);
    }
    decomposeRight(from, result) {
      result.push(null, new _HeightMapGap(this.length - from - 1));
    }
    updateHeight(oracle, offset = 0, force = false, measured) {
      let end = offset + this.length;
      if (measured && measured.from <= offset + this.length && measured.more) {
        let nodes = [], pos = Math.max(offset, measured.from), singleHeight = -1;
        if (measured.from > offset)
          nodes.push(new _HeightMapGap(measured.from - offset - 1).updateHeight(oracle, offset));
        while (pos <= end && measured.more) {
          let len = oracle.doc.lineAt(pos).length;
          if (nodes.length)
            nodes.push(null);
          let height = measured.heights[measured.index++], above = 0;
          if (height < 0) {
            above = -height;
            height = measured.heights[measured.index++];
          }
          if (singleHeight == -1)
            singleHeight = height;
          else if (Math.abs(height - singleHeight) >= Epsilon)
            singleHeight = -2;
          let line = new HeightMapText(len, height, above);
          line.outdated = false;
          nodes.push(line);
          pos += len + 1;
        }
        if (pos <= end)
          nodes.push(null, new _HeightMapGap(end - pos).updateHeight(oracle, pos));
        let result = HeightMap.of(nodes);
        if (singleHeight < 0 || Math.abs(result.height - this.height) >= Epsilon || Math.abs(singleHeight - this.heightMetrics(oracle, offset).perLine) >= Epsilon)
          heightChangeFlag = true;
        return replace(this, result);
      } else if (force || this.outdated) {
        this.setHeight(oracle.heightForGap(offset, offset + this.length));
        this.outdated = false;
      }
      return this;
    }
    toString() {
      return `gap(${this.length})`;
    }
  };
  var HeightMapBranch = class extends HeightMap {
    constructor(left, brk, right) {
      super(left.length + brk + right.length, left.height + right.height, brk | (left.outdated || right.outdated ? 2 : 0));
      this.left = left;
      this.right = right;
      this.size = left.size + right.size;
    }
    get break() {
      return this.flags & 1;
    }
    blockAt(height, oracle, top2, offset) {
      let mid = top2 + this.left.height;
      return height < mid ? this.left.blockAt(height, oracle, top2, offset) : this.right.blockAt(height, oracle, mid, offset + this.left.length + this.break);
    }
    lineAt(value, type, oracle, top2, offset) {
      let rightTop = top2 + this.left.height, rightOffset = offset + this.left.length + this.break;
      let left = type == QueryType.ByHeight ? value < rightTop : value < rightOffset;
      let base2 = left ? this.left.lineAt(value, type, oracle, top2, offset) : this.right.lineAt(value, type, oracle, rightTop, rightOffset);
      if (this.break || (left ? base2.to < rightOffset : base2.from > rightOffset))
        return base2;
      let subQuery = type == QueryType.ByPosNoHeight ? QueryType.ByPosNoHeight : QueryType.ByPos;
      if (left)
        return base2.join(this.right.lineAt(rightOffset, subQuery, oracle, rightTop, rightOffset));
      else
        return this.left.lineAt(rightOffset, subQuery, oracle, top2, offset).join(base2);
    }
    forEachLine(from, to, oracle, top2, offset, f) {
      let rightTop = top2 + this.left.height, rightOffset = offset + this.left.length + this.break;
      if (this.break) {
        if (from < rightOffset)
          this.left.forEachLine(from, to, oracle, top2, offset, f);
        if (to >= rightOffset)
          this.right.forEachLine(from, to, oracle, rightTop, rightOffset, f);
      } else {
        let mid = this.lineAt(rightOffset, QueryType.ByPos, oracle, top2, offset);
        if (from < mid.from)
          this.left.forEachLine(from, mid.from - 1, oracle, top2, offset, f);
        if (mid.to >= from && mid.from <= to)
          f(mid);
        if (to > mid.to)
          this.right.forEachLine(mid.to + 1, to, oracle, rightTop, rightOffset, f);
      }
    }
    replace(from, to, nodes) {
      let rightStart = this.left.length + this.break;
      if (to < rightStart)
        return this.balanced(this.left.replace(from, to, nodes), this.right);
      if (from > this.left.length)
        return this.balanced(this.left, this.right.replace(from - rightStart, to - rightStart, nodes));
      let result = [];
      if (from > 0)
        this.decomposeLeft(from, result);
      let left = result.length;
      for (let node of nodes)
        result.push(node);
      if (from > 0)
        mergeGaps(result, left - 1);
      if (to < this.length) {
        let right = result.length;
        this.decomposeRight(to, result);
        mergeGaps(result, right);
      }
      return HeightMap.of(result);
    }
    decomposeLeft(to, result) {
      let left = this.left.length;
      if (to <= left)
        return this.left.decomposeLeft(to, result);
      result.push(this.left);
      if (this.break) {
        left++;
        if (to >= left)
          result.push(null);
      }
      if (to > left)
        this.right.decomposeLeft(to - left, result);
    }
    decomposeRight(from, result) {
      let left = this.left.length, right = left + this.break;
      if (from >= right)
        return this.right.decomposeRight(from - right, result);
      if (from < left)
        this.left.decomposeRight(from, result);
      if (this.break && from < right)
        result.push(null);
      result.push(this.right);
    }
    balanced(left, right) {
      if (left.size > 2 * right.size || right.size > 2 * left.size)
        return HeightMap.of(this.break ? [left, null, right] : [left, right]);
      this.left = replace(this.left, left);
      this.right = replace(this.right, right);
      this.setHeight(left.height + right.height);
      this.outdated = left.outdated || right.outdated;
      this.size = left.size + right.size;
      this.length = left.length + this.break + right.length;
      return this;
    }
    updateHeight(oracle, offset = 0, force = false, measured) {
      let { left, right } = this, rightStart = offset + left.length + this.break, rebalance = null;
      if (measured && measured.from <= offset + left.length && measured.more)
        rebalance = left = left.updateHeight(oracle, offset, force, measured);
      else
        left.updateHeight(oracle, offset, force);
      if (measured && measured.from <= rightStart + right.length && measured.more)
        rebalance = right = right.updateHeight(oracle, rightStart, force, measured);
      else
        right.updateHeight(oracle, rightStart, force);
      if (rebalance)
        return this.balanced(left, right);
      this.height = this.left.height + this.right.height;
      this.outdated = false;
      return this;
    }
    toString() {
      return this.left + (this.break ? " " : "-") + this.right;
    }
  };
  function mergeGaps(nodes, around) {
    let before, after;
    if (nodes[around] == null && (before = nodes[around - 1]) instanceof HeightMapGap && (after = nodes[around + 1]) instanceof HeightMapGap)
      nodes.splice(around - 1, 3, new HeightMapGap(before.length + 1 + after.length));
  }
  var relevantWidgetHeight = 5;
  var NodeBuilder = class _NodeBuilder {
    constructor(pos, oracle) {
      this.pos = pos;
      this.oracle = oracle;
      this.nodes = [];
      this.lineStart = -1;
      this.lineEnd = -1;
      this.covering = null;
      this.writtenTo = pos;
    }
    get isCovered() {
      return this.covering && this.nodes[this.nodes.length - 1] == this.covering;
    }
    span(_from, to) {
      if (this.lineStart > -1) {
        let end = Math.min(to, this.lineEnd), last = this.nodes[this.nodes.length - 1];
        if (last instanceof HeightMapText)
          last.length += end - this.pos;
        else if (end > this.pos || !this.isCovered)
          this.nodes.push(new HeightMapText(end - this.pos, -1, 0));
        this.writtenTo = end;
        if (to > end) {
          this.nodes.push(null);
          this.writtenTo++;
          this.lineStart = -1;
        }
      }
      this.pos = to;
    }
    point(from, to, deco) {
      if (from < to || deco.heightRelevant) {
        let height = deco.widget ? deco.widget.estimatedHeight : 0;
        let breaks = deco.widget ? deco.widget.lineBreaks : 0;
        if (height < 0)
          height = this.oracle.lineHeight;
        let len = to - from;
        if (deco.block) {
          this.addBlock(new HeightMapBlock(len, height, deco));
        } else if (len || breaks || height >= relevantWidgetHeight) {
          this.addLineDeco(height, breaks, len);
        }
      } else if (to > from) {
        this.span(from, to);
      }
      if (this.lineEnd > -1 && this.lineEnd < this.pos)
        this.lineEnd = this.oracle.doc.lineAt(this.pos).to;
    }
    enterLine() {
      if (this.lineStart > -1)
        return;
      let { from, to } = this.oracle.doc.lineAt(this.pos);
      this.lineStart = from;
      this.lineEnd = to;
      if (this.writtenTo < from) {
        if (this.writtenTo < from - 1 || this.nodes[this.nodes.length - 1] == null)
          this.nodes.push(this.blankContent(this.writtenTo, from - 1));
        this.nodes.push(null);
      }
      if (this.pos > from)
        this.nodes.push(new HeightMapText(this.pos - from, -1, 0));
      this.writtenTo = this.pos;
    }
    blankContent(from, to) {
      let gap = new HeightMapGap(to - from);
      if (this.oracle.doc.lineAt(from).to == to)
        gap.flags |= 4;
      return gap;
    }
    ensureLine() {
      this.enterLine();
      let last = this.nodes.length ? this.nodes[this.nodes.length - 1] : null;
      if (last instanceof HeightMapText)
        return last;
      let line = new HeightMapText(0, -1, 0);
      this.nodes.push(line);
      return line;
    }
    addBlock(block) {
      this.enterLine();
      let deco = block.deco;
      if (deco && deco.startSide > 0 && !this.isCovered)
        this.ensureLine();
      this.nodes.push(block);
      this.writtenTo = this.pos = this.pos + block.length;
      if (deco && deco.endSide > 0)
        this.covering = block;
    }
    addLineDeco(height, breaks, length) {
      let line = this.ensureLine();
      line.length += length;
      line.collapsed += length;
      line.widgetHeight = Math.max(line.widgetHeight, height);
      line.breaks += breaks;
      this.writtenTo = this.pos = this.pos + length;
    }
    finish(from) {
      let last = this.nodes.length == 0 ? null : this.nodes[this.nodes.length - 1];
      if (this.lineStart > -1 && !(last instanceof HeightMapText) && !this.isCovered)
        this.nodes.push(new HeightMapText(0, -1, 0));
      else if (this.writtenTo < this.pos || last == null)
        this.nodes.push(this.blankContent(this.writtenTo, this.pos));
      let pos = from;
      for (let node of this.nodes) {
        if (node instanceof HeightMapText)
          node.updateHeight(this.oracle, pos);
        pos += node ? node.length : 1;
      }
      return this.nodes;
    }
    // Always called with a region that on both sides either stretches
    // to a line break or the end of the document.
    // The returned array uses null to indicate line breaks, but never
    // starts or ends in a line break, or has multiple line breaks next
    // to each other.
    static build(oracle, decorations2, from, to) {
      let builder = new _NodeBuilder(from, oracle);
      RangeSet.spans(decorations2, from, to, builder, 0);
      return builder.finish(from);
    }
  };
  function heightRelevantDecoChanges(a, b, diff) {
    let comp = new DecorationComparator2();
    RangeSet.compare(a, b, diff, comp, 0);
    return comp.changes;
  }
  var DecorationComparator2 = class {
    constructor() {
      this.changes = [];
    }
    compareRange() {
    }
    comparePoint(from, to, a, b) {
      if (from < to || a && a.heightRelevant || b && b.heightRelevant)
        addRange(from, to, this.changes, 5);
    }
  };
  function visiblePixelRange(dom, paddingTop) {
    let rect = dom.getBoundingClientRect();
    let doc2 = dom.ownerDocument, win = doc2.defaultView || window;
    let left = Math.max(0, rect.left), right = Math.min(win.innerWidth, rect.right);
    let top2 = Math.max(0, rect.top), bottom = Math.min(win.innerHeight, rect.bottom);
    for (let parent = dom.parentNode; parent && parent != doc2.body; ) {
      if (parent.nodeType == 1) {
        let elt = parent;
        let style = window.getComputedStyle(elt);
        if ((elt.scrollHeight > elt.clientHeight || elt.scrollWidth > elt.clientWidth) && style.overflow != "visible") {
          let parentRect = elt.getBoundingClientRect();
          left = Math.max(left, parentRect.left);
          right = Math.min(right, parentRect.right);
          top2 = Math.max(top2, parentRect.top);
          bottom = Math.min(parent == dom.parentNode ? win.innerHeight : bottom, parentRect.bottom);
        }
        parent = style.position == "absolute" || style.position == "fixed" ? elt.offsetParent : elt.parentNode;
      } else if (parent.nodeType == 11) {
        parent = parent.host;
      } else {
        break;
      }
    }
    return {
      left: left - rect.left,
      right: Math.max(left, right) - rect.left,
      top: top2 - (rect.top + paddingTop),
      bottom: Math.max(top2, bottom) - (rect.top + paddingTop)
    };
  }
  function inWindow(elt) {
    let rect = elt.getBoundingClientRect(), win = elt.ownerDocument.defaultView || window;
    return rect.left < win.innerWidth && rect.right > 0 && rect.top < win.innerHeight && rect.bottom > 0;
  }
  function fullPixelRange(dom, paddingTop) {
    let rect = dom.getBoundingClientRect();
    return {
      left: 0,
      right: rect.right - rect.left,
      top: paddingTop,
      bottom: rect.bottom - (rect.top + paddingTop)
    };
  }
  var LineGap = class {
    constructor(from, to, size, displaySize) {
      this.from = from;
      this.to = to;
      this.size = size;
      this.displaySize = displaySize;
    }
    static same(a, b) {
      if (a.length != b.length)
        return false;
      for (let i = 0; i < a.length; i++) {
        let gA = a[i], gB = b[i];
        if (gA.from != gB.from || gA.to != gB.to || gA.size != gB.size)
          return false;
      }
      return true;
    }
    draw(viewState, wrapping) {
      return Decoration.replace({
        widget: new LineGapWidget(this.displaySize * (wrapping ? viewState.scaleY : viewState.scaleX), wrapping)
      }).range(this.from, this.to);
    }
  };
  var LineGapWidget = class extends WidgetType {
    constructor(size, vertical) {
      super();
      this.size = size;
      this.vertical = vertical;
    }
    eq(other) {
      return other.size == this.size && other.vertical == this.vertical;
    }
    toDOM() {
      let elt = document.createElement("div");
      if (this.vertical) {
        elt.style.height = this.size + "px";
      } else {
        elt.style.width = this.size + "px";
        elt.style.height = "2px";
        elt.style.display = "inline-block";
      }
      return elt;
    }
    get estimatedHeight() {
      return this.vertical ? this.size : -1;
    }
  };
  var ViewState = class {
    constructor(view, state) {
      this.view = view;
      this.state = state;
      this.pixelViewport = { left: 0, right: window.innerWidth, top: 0, bottom: 0 };
      this.inView = true;
      this.paddingTop = 0;
      this.paddingBottom = 0;
      this.contentDOMWidth = 0;
      this.contentDOMHeight = 0;
      this.editorHeight = 0;
      this.editorWidth = 0;
      this.scaleX = 1;
      this.scaleY = 1;
      this.scrollOffset = 0;
      this.scrolledToBottom = false;
      this.scrollAnchorPos = 0;
      this.scrollAnchorHeight = -1;
      this.scaler = IdScaler;
      this.scrollTarget = null;
      this.printing = false;
      this.mustMeasureContent = true;
      this.defaultTextDirection = Direction.LTR;
      this.visibleRanges = [];
      this.mustEnforceCursorAssoc = false;
      let guessWrapping = state.facet(contentAttributes).some((v) => typeof v != "function" && v.class == "cm-lineWrapping");
      this.heightOracle = new HeightOracle(guessWrapping);
      this.stateDeco = staticDeco(state);
      this.heightMap = HeightMap.empty().applyChanges(this.stateDeco, Text.empty, this.heightOracle.setDoc(state.doc), [new ChangedRange(0, 0, 0, state.doc.length)]);
      for (let i = 0; i < 2; i++) {
        this.viewport = this.getViewport(0, null);
        if (!this.updateForViewport())
          break;
      }
      this.updateViewportLines();
      this.lineGaps = this.ensureLineGaps([]);
      this.lineGapDeco = Decoration.set(this.lineGaps.map((gap) => gap.draw(this, false)));
      this.scrollParent = view.scrollDOM;
      this.computeVisibleRanges();
    }
    updateForViewport() {
      let viewports = [this.viewport], { main } = this.state.selection;
      for (let i = 0; i <= 1; i++) {
        let pos = i ? main.head : main.anchor;
        if (!viewports.some(({ from, to }) => pos >= from && pos <= to)) {
          let { from, to } = this.lineBlockAt(pos);
          viewports.push(new Viewport(from, to));
        }
      }
      this.viewports = viewports.sort((a, b) => a.from - b.from);
      return this.updateScaler();
    }
    updateScaler() {
      let scaler = this.scaler;
      this.scaler = this.heightMap.height <= 7e6 ? IdScaler : new BigScaler(this.heightOracle, this.heightMap, this.viewports);
      return scaler.eq(this.scaler) ? 0 : 2;
    }
    updateViewportLines() {
      this.viewportLines = [];
      this.heightMap.forEachLine(this.viewport.from, this.viewport.to, this.heightOracle.setDoc(this.state.doc), 0, 0, (block) => {
        this.viewportLines.push(scaleBlock(block, this.scaler));
      });
    }
    update(update, scrollTarget = null) {
      this.state = update.state;
      let prevDeco = this.stateDeco;
      this.stateDeco = staticDeco(this.state);
      let contentChanges = update.changedRanges;
      let heightChanges = ChangedRange.extendWithRanges(contentChanges, heightRelevantDecoChanges(prevDeco, this.stateDeco, update ? update.changes : ChangeSet.empty(this.state.doc.length)));
      let prevHeight = this.heightMap.height;
      let scrollAnchor = this.scrolledToBottom ? null : this.scrollAnchorAt(this.scrollOffset);
      clearHeightChangeFlag();
      this.heightMap = this.heightMap.applyChanges(this.stateDeco, update.startState.doc, this.heightOracle.setDoc(this.state.doc), heightChanges);
      if (this.heightMap.height != prevHeight || heightChangeFlag)
        update.flags |= 2;
      if (scrollAnchor) {
        this.scrollAnchorPos = update.changes.mapPos(scrollAnchor.from, -1);
        this.scrollAnchorHeight = scrollAnchor.top;
      } else {
        this.scrollAnchorPos = -1;
        this.scrollAnchorHeight = prevHeight;
      }
      let viewport = heightChanges.length ? this.mapViewport(this.viewport, update.changes) : this.viewport;
      if (scrollTarget && (scrollTarget.range.head < viewport.from || scrollTarget.range.head > viewport.to) || !this.viewportIsAppropriate(viewport))
        viewport = this.getViewport(0, scrollTarget);
      let viewportChange = viewport.from != this.viewport.from || viewport.to != this.viewport.to;
      this.viewport = viewport;
      update.flags |= this.updateForViewport();
      if (viewportChange || !update.changes.empty || update.flags & 2)
        this.updateViewportLines();
      if (this.lineGaps.length || this.viewport.to - this.viewport.from > 2e3 << 1)
        this.updateLineGaps(this.ensureLineGaps(this.mapLineGaps(this.lineGaps, update.changes)));
      update.flags |= this.computeVisibleRanges(update.changes);
      if (scrollTarget)
        this.scrollTarget = scrollTarget;
      if (!this.mustEnforceCursorAssoc && (update.selectionSet || update.focusChanged) && update.view.lineWrapping && update.state.selection.main.empty && update.state.selection.main.assoc && !update.state.facet(nativeSelectionHidden))
        this.mustEnforceCursorAssoc = true;
    }
    measure() {
      let { view } = this, dom = view.contentDOM, style = window.getComputedStyle(dom);
      let oracle = this.heightOracle;
      let whiteSpace = style.whiteSpace;
      this.defaultTextDirection = style.direction == "rtl" ? Direction.RTL : Direction.LTR;
      let refresh = this.heightOracle.mustRefreshForWrapping(whiteSpace) || this.mustMeasureContent === "refresh";
      let domRect = dom.getBoundingClientRect();
      let measureContent = refresh || this.mustMeasureContent || this.contentDOMHeight != domRect.height;
      this.contentDOMHeight = domRect.height;
      this.mustMeasureContent = false;
      let result = 0, bias = 0;
      if (domRect.width && domRect.height) {
        let { scaleX, scaleY } = getScale(dom, domRect);
        if (scaleX > 5e-3 && Math.abs(this.scaleX - scaleX) > 5e-3 || scaleY > 5e-3 && Math.abs(this.scaleY - scaleY) > 5e-3) {
          this.scaleX = scaleX;
          this.scaleY = scaleY;
          result |= 16;
          refresh = measureContent = true;
        }
      }
      let paddingTop = (parseInt(style.paddingTop) || 0) * this.scaleY;
      let paddingBottom = (parseInt(style.paddingBottom) || 0) * this.scaleY;
      if (this.paddingTop != paddingTop || this.paddingBottom != paddingBottom) {
        this.paddingTop = paddingTop;
        this.paddingBottom = paddingBottom;
        result |= 16 | 2;
      }
      if (this.editorWidth != view.scrollDOM.clientWidth) {
        if (oracle.lineWrapping)
          measureContent = true;
        this.editorWidth = view.scrollDOM.clientWidth;
        result |= 16;
      }
      let scrollParent = scrollableParents(this.view.contentDOM, false).y;
      if (scrollParent != this.scrollParent) {
        this.scrollParent = scrollParent;
        this.scrollAnchorHeight = -1;
        this.scrollOffset = 0;
      }
      let scrollOffset = this.getScrollOffset();
      if (this.scrollOffset != scrollOffset) {
        this.scrollAnchorHeight = -1;
        this.scrollOffset = scrollOffset;
      }
      this.scrolledToBottom = isScrolledToBottom(this.scrollParent || view.win);
      let pixelViewport = (this.printing ? fullPixelRange : visiblePixelRange)(dom, this.paddingTop);
      let dTop = pixelViewport.top - this.pixelViewport.top, dBottom = pixelViewport.bottom - this.pixelViewport.bottom;
      this.pixelViewport = pixelViewport;
      let inView = this.pixelViewport.bottom > this.pixelViewport.top && this.pixelViewport.right > this.pixelViewport.left;
      if (inView != this.inView) {
        this.inView = inView;
        if (inView)
          measureContent = true;
      }
      if (!this.inView && !this.scrollTarget && !inWindow(view.dom))
        return 0;
      let contentWidth = domRect.width;
      if (this.contentDOMWidth != contentWidth || this.editorHeight != view.scrollDOM.clientHeight) {
        this.contentDOMWidth = domRect.width;
        this.editorHeight = view.scrollDOM.clientHeight;
        result |= 16;
      }
      if (measureContent) {
        let lineHeights = view.docView.measureVisibleLineHeights(this.viewport);
        if (oracle.mustRefreshForHeights(lineHeights))
          refresh = true;
        if (refresh || oracle.lineWrapping && Math.abs(contentWidth - this.contentDOMWidth) > oracle.charWidth) {
          let { lineHeight, charWidth, textHeight } = view.docView.measureTextSize();
          refresh = lineHeight > 0 && oracle.refresh(whiteSpace, lineHeight, charWidth, textHeight, Math.max(5, contentWidth / charWidth), lineHeights);
          if (refresh) {
            view.docView.minWidth = 0;
            result |= 16;
          }
        }
        if (dTop > 0 && dBottom > 0)
          bias = Math.max(dTop, dBottom);
        else if (dTop < 0 && dBottom < 0)
          bias = Math.min(dTop, dBottom);
        clearHeightChangeFlag();
        for (let vp of this.viewports) {
          let heights = vp.from == this.viewport.from ? lineHeights : view.docView.measureVisibleLineHeights(vp);
          this.heightMap = (refresh ? HeightMap.empty().applyChanges(this.stateDeco, Text.empty, this.heightOracle, [new ChangedRange(0, 0, 0, view.state.doc.length)]) : this.heightMap).updateHeight(oracle, 0, refresh, new MeasuredHeights(vp.from, heights));
        }
        if (heightChangeFlag)
          result |= 2;
      }
      let viewportChange = !this.viewportIsAppropriate(this.viewport, bias) || this.scrollTarget && (this.scrollTarget.range.head < this.viewport.from || this.scrollTarget.range.head > this.viewport.to);
      if (viewportChange) {
        if (result & 2)
          result |= this.updateScaler();
        this.viewport = this.getViewport(bias, this.scrollTarget);
        result |= this.updateForViewport();
      }
      if (result & 2 || viewportChange)
        this.updateViewportLines();
      if (this.lineGaps.length || this.viewport.to - this.viewport.from > 2e3 << 1)
        this.updateLineGaps(this.ensureLineGaps(refresh ? [] : this.lineGaps, view));
      result |= this.computeVisibleRanges();
      if (this.mustEnforceCursorAssoc) {
        this.mustEnforceCursorAssoc = false;
        view.docView.enforceCursorAssoc();
      }
      return result;
    }
    get visibleTop() {
      return this.scaler.fromDOM(this.pixelViewport.top);
    }
    get visibleBottom() {
      return this.scaler.fromDOM(this.pixelViewport.bottom);
    }
    getViewport(bias, scrollTarget) {
      let marginTop = 0.5 - Math.max(-0.5, Math.min(0.5, bias / 1e3 / 2));
      let map = this.heightMap, oracle = this.heightOracle;
      let { visibleTop, visibleBottom } = this;
      let viewport = new Viewport(map.lineAt(visibleTop - marginTop * 1e3, QueryType.ByHeight, oracle, 0, 0).from, map.lineAt(visibleBottom + (1 - marginTop) * 1e3, QueryType.ByHeight, oracle, 0, 0).to);
      if (scrollTarget) {
        let { head } = scrollTarget.range;
        if (head < viewport.from || head > viewport.to) {
          let viewHeight = Math.min(this.editorHeight, this.pixelViewport.bottom - this.pixelViewport.top);
          let block = map.lineAt(head, QueryType.ByPos, oracle, 0, 0), topPos;
          if (scrollTarget.y == "center")
            topPos = (block.top + block.bottom) / 2 - viewHeight / 2;
          else if (scrollTarget.y == "start" || scrollTarget.y == "nearest" && head < viewport.from)
            topPos = block.top;
          else
            topPos = block.bottom - viewHeight;
          viewport = new Viewport(map.lineAt(topPos - 1e3 / 2, QueryType.ByHeight, oracle, 0, 0).from, map.lineAt(topPos + viewHeight + 1e3 / 2, QueryType.ByHeight, oracle, 0, 0).to);
        }
      }
      return viewport;
    }
    mapViewport(viewport, changes) {
      let from = changes.mapPos(viewport.from, -1), to = changes.mapPos(viewport.to, 1);
      return new Viewport(this.heightMap.lineAt(from, QueryType.ByPos, this.heightOracle, 0, 0).from, this.heightMap.lineAt(to, QueryType.ByPos, this.heightOracle, 0, 0).to);
    }
    // Checks if a given viewport covers the visible part of the
    // document and not too much beyond that.
    viewportIsAppropriate({ from, to }, bias = 0) {
      if (!this.inView)
        return true;
      let { top: top2 } = this.heightMap.lineAt(from, QueryType.ByPos, this.heightOracle, 0, 0);
      let { bottom } = this.heightMap.lineAt(to, QueryType.ByPos, this.heightOracle, 0, 0);
      let { visibleTop, visibleBottom } = this;
      return (from == 0 || top2 <= visibleTop - Math.max(10, Math.min(
        -bias,
        250
        /* VP.MaxCoverMargin */
      ))) && (to == this.state.doc.length || bottom >= visibleBottom + Math.max(10, Math.min(
        bias,
        250
        /* VP.MaxCoverMargin */
      ))) && (top2 > visibleTop - 2 * 1e3 && bottom < visibleBottom + 2 * 1e3);
    }
    mapLineGaps(gaps, changes) {
      if (!gaps.length || changes.empty)
        return gaps;
      let mapped = [];
      for (let gap of gaps)
        if (!changes.touchesRange(gap.from, gap.to))
          mapped.push(new LineGap(changes.mapPos(gap.from), changes.mapPos(gap.to), gap.size, gap.displaySize));
      return mapped;
    }
    // Computes positions in the viewport where the start or end of a
    // line should be hidden, trying to reuse existing line gaps when
    // appropriate to avoid unneccesary redraws.
    // Uses crude character-counting for the positioning and sizing,
    // since actual DOM coordinates aren't always available and
    // predictable. Relies on generous margins (see LG.Margin) to hide
    // the artifacts this might produce from the user.
    ensureLineGaps(current, mayMeasure) {
      let wrapping = this.heightOracle.lineWrapping;
      let margin = wrapping ? 1e4 : 2e3, halfMargin = margin >> 1, doubleMargin = margin << 1;
      if (this.defaultTextDirection != Direction.LTR && !wrapping)
        return [];
      let gaps = [];
      let addGap = (from, to, line, structure) => {
        if (to - from < halfMargin)
          return;
        let sel = this.state.selection.main, avoid = [sel.from];
        if (!sel.empty)
          avoid.push(sel.to);
        for (let pos of avoid) {
          if (pos > from && pos < to) {
            addGap(from, pos - 10, line, structure);
            addGap(pos + 10, to, line, structure);
            return;
          }
        }
        let gap = find(current, (gap2) => gap2.from >= line.from && gap2.to <= line.to && Math.abs(gap2.from - from) < halfMargin && Math.abs(gap2.to - to) < halfMargin && !avoid.some((pos) => gap2.from < pos && gap2.to > pos));
        if (!gap) {
          if (to < line.to && mayMeasure && wrapping && mayMeasure.visibleRanges.some((r) => r.from <= to && r.to >= to)) {
            let lineStart = mayMeasure.moveToLineBoundary(EditorSelection.cursor(to), false, true).head;
            if (lineStart > from)
              to = lineStart;
          }
          let size = this.gapSize(line, from, to, structure);
          let displaySize = wrapping || size < 2e6 ? size : 2e6;
          gap = new LineGap(from, to, size, displaySize);
        }
        gaps.push(gap);
      };
      let checkLine = (line) => {
        if (line.length < doubleMargin || line.type != BlockType.Text)
          return;
        let structure = lineStructure(line.from, line.to, this.stateDeco);
        if (structure.total < doubleMargin)
          return;
        let target = this.scrollTarget ? this.scrollTarget.range.head : null;
        let viewFrom, viewTo;
        if (wrapping) {
          let marginHeight = margin / this.heightOracle.lineLength * this.heightOracle.lineHeight;
          let top2, bot;
          if (target != null) {
            let targetFrac = findFraction(structure, target);
            let spaceFrac = ((this.visibleBottom - this.visibleTop) / 2 + marginHeight) / line.height;
            top2 = targetFrac - spaceFrac;
            bot = targetFrac + spaceFrac;
          } else {
            top2 = (this.visibleTop - line.top - marginHeight) / line.height;
            bot = (this.visibleBottom - line.top + marginHeight) / line.height;
          }
          viewFrom = findPosition(structure, top2);
          viewTo = findPosition(structure, bot);
        } else {
          let totalWidth = structure.total * this.heightOracle.charWidth;
          let marginWidth = margin * this.heightOracle.charWidth;
          let horizOffset = 0;
          if (totalWidth > 2e6)
            for (let old of current) {
              if (old.from >= line.from && old.from < line.to && old.size != old.displaySize && old.from * this.heightOracle.charWidth + horizOffset < this.pixelViewport.left)
                horizOffset = old.size - old.displaySize;
            }
          let pxLeft = this.pixelViewport.left + horizOffset, pxRight = this.pixelViewport.right + horizOffset;
          let left, right;
          if (target != null) {
            let targetFrac = findFraction(structure, target);
            let spaceFrac = ((pxRight - pxLeft) / 2 + marginWidth) / totalWidth;
            left = targetFrac - spaceFrac;
            right = targetFrac + spaceFrac;
          } else {
            left = (pxLeft - marginWidth) / totalWidth;
            right = (pxRight + marginWidth) / totalWidth;
          }
          viewFrom = findPosition(structure, left);
          viewTo = findPosition(structure, right);
        }
        if (viewFrom > line.from)
          addGap(line.from, viewFrom, line, structure);
        if (viewTo < line.to)
          addGap(viewTo, line.to, line, structure);
      };
      for (let line of this.viewportLines) {
        if (Array.isArray(line.type))
          line.type.forEach(checkLine);
        else
          checkLine(line);
      }
      return gaps;
    }
    gapSize(line, from, to, structure) {
      let fraction = findFraction(structure, to) - findFraction(structure, from);
      if (this.heightOracle.lineWrapping) {
        return line.height * fraction;
      } else {
        return structure.total * this.heightOracle.charWidth * fraction;
      }
    }
    updateLineGaps(gaps) {
      if (!LineGap.same(gaps, this.lineGaps)) {
        this.lineGaps = gaps;
        this.lineGapDeco = Decoration.set(gaps.map((gap) => gap.draw(this, this.heightOracle.lineWrapping)));
      }
    }
    computeVisibleRanges(changes) {
      let deco = this.stateDeco;
      if (this.lineGaps.length)
        deco = deco.concat(this.lineGapDeco);
      let ranges = [];
      RangeSet.spans(deco, this.viewport.from, this.viewport.to, {
        span(from, to) {
          ranges.push({ from, to });
        },
        point() {
        }
      }, 20);
      let changed = 0;
      if (ranges.length != this.visibleRanges.length) {
        changed = 8 | 4;
      } else {
        for (let i = 0; i < ranges.length && !(changed & 8); i++) {
          let old = this.visibleRanges[i], nw = ranges[i];
          if (old.from != nw.from || old.to != nw.to) {
            changed |= 4;
            if (!(changes && changes.mapPos(old.from, -1) == nw.from && changes.mapPos(old.to, 1) == nw.to))
              changed |= 8;
          }
        }
      }
      this.visibleRanges = ranges;
      return changed;
    }
    lineBlockAt(pos) {
      return pos >= this.viewport.from && pos <= this.viewport.to && this.viewportLines.find((b) => b.from <= pos && b.to >= pos) || scaleBlock(this.heightMap.lineAt(pos, QueryType.ByPos, this.heightOracle, 0, 0), this.scaler);
    }
    lineBlockAtHeight(height) {
      return height >= this.viewportLines[0].top && height <= this.viewportLines[this.viewportLines.length - 1].bottom && this.viewportLines.find((l) => l.top <= height && l.bottom >= height) || scaleBlock(this.heightMap.lineAt(this.scaler.fromDOM(height), QueryType.ByHeight, this.heightOracle, 0, 0), this.scaler);
    }
    getScrollOffset() {
      let base2 = this.scrollParent == this.view.scrollDOM ? this.scrollParent.scrollTop : (this.scrollParent ? this.scrollParent.getBoundingClientRect().top : 0) - this.view.contentDOM.getBoundingClientRect().top;
      return base2 * this.scaleY;
    }
    scrollAnchorAt(scrollOffset) {
      let block = this.lineBlockAtHeight(scrollOffset + 8);
      return block.from >= this.viewport.from || this.viewportLines[0].top - scrollOffset > 200 ? block : this.viewportLines[0];
    }
    elementAtHeight(height) {
      return scaleBlock(this.heightMap.blockAt(this.scaler.fromDOM(height), this.heightOracle, 0, 0), this.scaler);
    }
    get docHeight() {
      return this.scaler.toDOM(this.heightMap.height);
    }
    get contentHeight() {
      return this.docHeight + this.paddingTop + this.paddingBottom;
    }
  };
  var Viewport = class {
    constructor(from, to) {
      this.from = from;
      this.to = to;
    }
  };
  function lineStructure(from, to, stateDeco) {
    let ranges = [], pos = from, total = 0;
    RangeSet.spans(stateDeco, from, to, {
      span() {
      },
      point(from2, to2) {
        if (from2 > pos) {
          ranges.push({ from: pos, to: from2 });
          total += from2 - pos;
        }
        pos = to2;
      }
    }, 20);
    if (pos < to) {
      ranges.push({ from: pos, to });
      total += to - pos;
    }
    return { total, ranges };
  }
  function findPosition({ total, ranges }, ratio) {
    if (ratio <= 0)
      return ranges[0].from;
    if (ratio >= 1)
      return ranges[ranges.length - 1].to;
    let dist2 = Math.floor(total * ratio);
    for (let i = 0; ; i++) {
      let { from, to } = ranges[i], size = to - from;
      if (dist2 <= size)
        return from + dist2;
      dist2 -= size;
    }
  }
  function findFraction(structure, pos) {
    let counted = 0;
    for (let { from, to } of structure.ranges) {
      if (pos <= to) {
        counted += pos - from;
        break;
      }
      counted += to - from;
    }
    return counted / structure.total;
  }
  function find(array, f) {
    for (let val of array)
      if (f(val))
        return val;
    return void 0;
  }
  var IdScaler = {
    toDOM(n) {
      return n;
    },
    fromDOM(n) {
      return n;
    },
    scale: 1,
    eq(other) {
      return other == this;
    }
  };
  function staticDeco(state) {
    let deco = state.facet(decorations).filter((d) => typeof d != "function");
    let outer = state.facet(outerDecorations).filter((d) => typeof d != "function");
    if (outer.length)
      deco.push(RangeSet.join(outer));
    return deco;
  }
  var BigScaler = class _BigScaler {
    constructor(oracle, heightMap, viewports) {
      let vpHeight = 0, base2 = 0, domBase = 0;
      this.viewports = viewports.map(({ from, to }) => {
        let top2 = heightMap.lineAt(from, QueryType.ByPos, oracle, 0, 0).top;
        let bottom = heightMap.lineAt(to, QueryType.ByPos, oracle, 0, 0).bottom;
        vpHeight += bottom - top2;
        return { from, to, top: top2, bottom, domTop: 0, domBottom: 0 };
      });
      this.scale = (7e6 - vpHeight) / (heightMap.height - vpHeight);
      for (let obj of this.viewports) {
        obj.domTop = domBase + (obj.top - base2) * this.scale;
        domBase = obj.domBottom = obj.domTop + (obj.bottom - obj.top);
        base2 = obj.bottom;
      }
    }
    toDOM(n) {
      for (let i = 0, base2 = 0, domBase = 0; ; i++) {
        let vp = i < this.viewports.length ? this.viewports[i] : null;
        if (!vp || n < vp.top)
          return domBase + (n - base2) * this.scale;
        if (n <= vp.bottom)
          return vp.domTop + (n - vp.top);
        base2 = vp.bottom;
        domBase = vp.domBottom;
      }
    }
    fromDOM(n) {
      for (let i = 0, base2 = 0, domBase = 0; ; i++) {
        let vp = i < this.viewports.length ? this.viewports[i] : null;
        if (!vp || n < vp.domTop)
          return base2 + (n - domBase) / this.scale;
        if (n <= vp.domBottom)
          return vp.top + (n - vp.domTop);
        base2 = vp.bottom;
        domBase = vp.domBottom;
      }
    }
    eq(other) {
      if (!(other instanceof _BigScaler))
        return false;
      return this.scale == other.scale && this.viewports.length == other.viewports.length && this.viewports.every((vp, i) => vp.from == other.viewports[i].from && vp.to == other.viewports[i].to);
    }
  };
  function scaleBlock(block, scaler) {
    if (scaler.scale == 1)
      return block;
    let bTop = scaler.toDOM(block.top), bBottom = scaler.toDOM(block.bottom);
    return new BlockInfo(block.from, block.length, bTop, bBottom - bTop, Array.isArray(block._content) ? block._content.map((b) => scaleBlock(b, scaler)) : block._content);
  }
  var theme = /* @__PURE__ */ Facet.define({ combine: (strs) => strs.join(" ") });
  var darkTheme = /* @__PURE__ */ Facet.define({ combine: (values) => values.indexOf(true) > -1 });
  var baseThemeID = /* @__PURE__ */ StyleModule.newName();
  var baseLightID = /* @__PURE__ */ StyleModule.newName();
  var baseDarkID = /* @__PURE__ */ StyleModule.newName();
  var lightDarkIDs = { "&light": "." + baseLightID, "&dark": "." + baseDarkID };
  function buildTheme(main, spec, scopes) {
    return new StyleModule(spec, {
      finish(sel) {
        return /&/.test(sel) ? sel.replace(/&\w*/, (m) => {
          if (m == "&")
            return main;
          if (!scopes || !scopes[m])
            throw new RangeError(`Unsupported selector: ${m}`);
          return scopes[m];
        }) : main + " " + sel;
      }
    });
  }
  var baseTheme$1 = /* @__PURE__ */ buildTheme("." + baseThemeID, {
    "&": {
      position: "relative !important",
      boxSizing: "border-box",
      "&.cm-focused": {
        // Provide a simple default outline to make sure a focused
        // editor is visually distinct. Can't leave the default behavior
        // because that will apply to the content element, which is
        // inside the scrollable container and doesn't include the
        // gutters. We also can't use an 'auto' outline, since those
        // are, for some reason, drawn behind the element content, which
        // will cause things like the active line background to cover
        // the outline (#297).
        outline: "1px dotted #212121"
      },
      display: "flex !important",
      flexDirection: "column"
    },
    ".cm-scroller": {
      display: "flex !important",
      alignItems: "flex-start !important",
      fontFamily: "monospace",
      lineHeight: 1.4,
      height: "100%",
      overflowX: "auto",
      position: "relative",
      zIndex: 0,
      overflowAnchor: "none"
    },
    ".cm-content": {
      margin: 0,
      flexGrow: 2,
      flexShrink: 0,
      display: "block",
      whiteSpace: "pre",
      wordWrap: "normal",
      // https://github.com/codemirror/dev/issues/456
      boxSizing: "border-box",
      minHeight: "100%",
      padding: "4px 0",
      outline: "none",
      "&[contenteditable=true]": {
        WebkitUserModify: "read-write-plaintext-only"
      }
    },
    ".cm-lineWrapping": {
      whiteSpace_fallback: "pre-wrap",
      // For IE
      whiteSpace: "break-spaces",
      wordBreak: "break-word",
      // For Safari, which doesn't support overflow-wrap: anywhere
      overflowWrap: "anywhere",
      flexShrink: 1
    },
    "&light .cm-content": { caretColor: "black" },
    "&dark .cm-content": { caretColor: "white" },
    ".cm-line": {
      display: "block",
      padding: "0 2px 0 6px"
    },
    ".cm-layer": {
      position: "absolute",
      left: 0,
      top: 0,
      contain: "size style",
      "& > *": {
        position: "absolute"
      }
    },
    "&light .cm-selectionBackground": {
      background: "#d9d9d9"
    },
    "&dark .cm-selectionBackground": {
      background: "#222"
    },
    "&light.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground": {
      background: "#d7d4f0"
    },
    "&dark.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground": {
      background: "#233"
    },
    ".cm-cursorLayer": {
      pointerEvents: "none"
    },
    "&.cm-focused > .cm-scroller > .cm-cursorLayer": {
      animation: "steps(1) cm-blink 1.2s infinite"
    },
    // Two animations defined so that we can switch between them to
    // restart the animation without forcing another style
    // recomputation.
    "@keyframes cm-blink": { "0%": {}, "50%": { opacity: 0 }, "100%": {} },
    "@keyframes cm-blink2": { "0%": {}, "50%": { opacity: 0 }, "100%": {} },
    ".cm-cursor, .cm-dropCursor": {
      borderLeft: "1.2px solid black",
      marginLeft: "-0.6px",
      pointerEvents: "none"
    },
    ".cm-cursor": {
      display: "none"
    },
    "&dark .cm-cursor": {
      borderLeftColor: "#ddd"
    },
    ".cm-dropCursor": {
      position: "absolute"
    },
    "&.cm-focused > .cm-scroller > .cm-cursorLayer .cm-cursor": {
      display: "block"
    },
    ".cm-iso": {
      unicodeBidi: "isolate"
    },
    ".cm-announced": {
      position: "fixed",
      top: "-10000px"
    },
    "@media print": {
      ".cm-announced": { display: "none" }
    },
    "&light .cm-activeLine": { backgroundColor: "#cceeff44" },
    "&dark .cm-activeLine": { backgroundColor: "#99eeff33" },
    "&light .cm-specialChar": { color: "red" },
    "&dark .cm-specialChar": { color: "#f78" },
    ".cm-gutters": {
      flexShrink: 0,
      display: "flex",
      height: "100%",
      boxSizing: "border-box",
      zIndex: 200
    },
    ".cm-gutters-before": { insetInlineStart: 0 },
    ".cm-gutters-after": { insetInlineEnd: 0 },
    "&light .cm-gutters": {
      backgroundColor: "#f5f5f5",
      color: "#6c6c6c",
      border: "0px solid #ddd",
      "&.cm-gutters-before": { borderRightWidth: "1px" },
      "&.cm-gutters-after": { borderLeftWidth: "1px" }
    },
    "&dark .cm-gutters": {
      backgroundColor: "#333338",
      color: "#ccc"
    },
    ".cm-gutter": {
      display: "flex !important",
      // Necessary -- prevents margin collapsing
      flexDirection: "column",
      flexShrink: 0,
      boxSizing: "border-box",
      minHeight: "100%",
      overflow: "hidden"
    },
    ".cm-gutterElement": {
      boxSizing: "border-box"
    },
    ".cm-lineNumbers .cm-gutterElement": {
      padding: "0 3px 0 5px",
      minWidth: "20px",
      textAlign: "right",
      whiteSpace: "nowrap"
    },
    "&light .cm-activeLineGutter": {
      backgroundColor: "#e2f2ff"
    },
    "&dark .cm-activeLineGutter": {
      backgroundColor: "#222227"
    },
    ".cm-panels": {
      boxSizing: "border-box",
      position: "sticky",
      left: 0,
      right: 0,
      zIndex: 300
    },
    "&light .cm-panels": {
      backgroundColor: "#f5f5f5",
      color: "black"
    },
    "&light .cm-panels-top": {
      borderBottom: "1px solid #ddd"
    },
    "&light .cm-panels-bottom": {
      borderTop: "1px solid #ddd"
    },
    "&dark .cm-panels": {
      backgroundColor: "#333338",
      color: "white"
    },
    ".cm-dialog": {
      padding: "2px 19px 4px 6px",
      position: "relative",
      "& label": { fontSize: "80%" }
    },
    ".cm-dialog-close": {
      position: "absolute",
      top: "3px",
      right: "4px",
      backgroundColor: "inherit",
      border: "none",
      font: "inherit",
      fontSize: "14px",
      padding: "0"
    },
    ".cm-tab": {
      display: "inline-block",
      overflow: "hidden",
      verticalAlign: "bottom"
    },
    ".cm-widgetBuffer": {
      verticalAlign: "text-top",
      height: "1em",
      width: 0,
      display: "inline"
    },
    ".cm-placeholder": {
      color: "#888",
      display: "inline-block",
      verticalAlign: "top",
      userSelect: "none"
    },
    ".cm-highlightSpace": {
      backgroundImage: "radial-gradient(circle at 50% 55%, #aaa 20%, transparent 5%)",
      backgroundPosition: "center"
    },
    ".cm-highlightTab": {
      backgroundImage: `url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="20"><path stroke="%23888" stroke-width="1" fill="none" d="M1 10H196L190 5M190 15L196 10M197 4L197 16"/></svg>')`,
      backgroundSize: "auto 100%",
      backgroundPosition: "right 90%",
      backgroundRepeat: "no-repeat"
    },
    ".cm-trailingSpace": {
      backgroundColor: "#ff332255"
    },
    ".cm-button": {
      verticalAlign: "middle",
      color: "inherit",
      fontSize: "70%",
      padding: ".2em 1em",
      borderRadius: "1px"
    },
    "&light .cm-button": {
      backgroundImage: "linear-gradient(#eff1f5, #d9d9df)",
      border: "1px solid #888",
      "&:active": {
        backgroundImage: "linear-gradient(#b4b4b4, #d0d3d6)"
      }
    },
    "&dark .cm-button": {
      backgroundImage: "linear-gradient(#393939, #111)",
      border: "1px solid #888",
      "&:active": {
        backgroundImage: "linear-gradient(#111, #333)"
      }
    },
    ".cm-textfield": {
      verticalAlign: "middle",
      color: "inherit",
      fontSize: "70%",
      border: "1px solid silver",
      padding: ".2em .5em"
    },
    "&light .cm-textfield": {
      backgroundColor: "white"
    },
    "&dark .cm-textfield": {
      border: "1px solid #555",
      backgroundColor: "inherit"
    }
  }, lightDarkIDs);
  var observeOptions = {
    childList: true,
    characterData: true,
    subtree: true,
    attributes: true,
    characterDataOldValue: true
  };
  var useCharData = browser.ie && browser.ie_version <= 11;
  var DOMObserver = class {
    constructor(view) {
      this.view = view;
      this.active = false;
      this.editContext = null;
      this.selectionRange = new DOMSelectionState();
      this.selectionChanged = false;
      this.delayedFlush = -1;
      this.resizeTimeout = -1;
      this.queue = [];
      this.delayedAndroidKey = null;
      this.flushingAndroidKey = -1;
      this.lastChange = 0;
      this.scrollTargets = [];
      this.intersection = null;
      this.resizeScroll = null;
      this.intersecting = false;
      this.gapIntersection = null;
      this.gaps = [];
      this.printQuery = null;
      this.parentCheck = -1;
      this.dom = view.contentDOM;
      this.observer = new MutationObserver((mutations) => {
        for (let mut of mutations)
          this.queue.push(mut);
        if ((browser.ie && browser.ie_version <= 11 || browser.ios && view.composing) && mutations.some((m) => m.type == "childList" && m.removedNodes.length || m.type == "characterData" && m.oldValue.length > m.target.nodeValue.length))
          this.flushSoon();
        else
          this.flush();
      });
      if (window.EditContext && browser.android && view.constructor.EDIT_CONTEXT !== false && // Chrome <126 doesn't support inverted selections in edit context (#1392)
      !(browser.chrome && browser.chrome_version < 126)) {
        this.editContext = new EditContextManager(view);
        if (view.state.facet(editable))
          view.contentDOM.editContext = this.editContext.editContext;
      }
      if (useCharData)
        this.onCharData = (event) => {
          this.queue.push({
            target: event.target,
            type: "characterData",
            oldValue: event.prevValue
          });
          this.flushSoon();
        };
      this.onSelectionChange = this.onSelectionChange.bind(this);
      this.onResize = this.onResize.bind(this);
      this.onPrint = this.onPrint.bind(this);
      this.onScroll = this.onScroll.bind(this);
      if (window.matchMedia)
        this.printQuery = window.matchMedia("print");
      if (typeof ResizeObserver == "function") {
        this.resizeScroll = new ResizeObserver(() => {
          var _a2;
          if (((_a2 = this.view.docView) === null || _a2 === void 0 ? void 0 : _a2.lastUpdate) < Date.now() - 75)
            this.onResize();
        });
        this.resizeScroll.observe(view.scrollDOM);
      }
      this.addWindowListeners(this.win = view.win);
      this.start();
      if (typeof IntersectionObserver == "function") {
        this.intersection = new IntersectionObserver((entries) => {
          if (this.parentCheck < 0)
            this.parentCheck = setTimeout(this.listenForScroll.bind(this), 1e3);
          if (entries.length > 0 && entries[entries.length - 1].intersectionRatio > 0 != this.intersecting) {
            this.intersecting = !this.intersecting;
            if (this.intersecting != this.view.inView)
              this.onScrollChanged(document.createEvent("Event"));
          }
        }, { threshold: [0, 1e-3] });
        this.intersection.observe(this.dom);
        this.gapIntersection = new IntersectionObserver((entries) => {
          if (entries.length > 0 && entries[entries.length - 1].intersectionRatio > 0)
            this.onScrollChanged(document.createEvent("Event"));
        }, {});
      }
      this.listenForScroll();
      this.readSelectionRange();
    }
    onScrollChanged(e) {
      this.view.inputState.runHandlers("scroll", e);
      if (this.intersecting)
        this.view.measure();
    }
    onScroll(e) {
      if (this.intersecting)
        this.flush(false);
      if (this.editContext)
        this.view.requestMeasure(this.editContext.measureReq);
      this.onScrollChanged(e);
    }
    onResize() {
      if (this.resizeTimeout < 0)
        this.resizeTimeout = setTimeout(() => {
          this.resizeTimeout = -1;
          this.view.requestMeasure();
        }, 50);
    }
    onPrint(event) {
      if ((event.type == "change" || !event.type) && !event.matches)
        return;
      this.view.viewState.printing = true;
      this.view.measure();
      setTimeout(() => {
        this.view.viewState.printing = false;
        this.view.requestMeasure();
      }, 500);
    }
    updateGaps(gaps) {
      if (this.gapIntersection && (gaps.length != this.gaps.length || this.gaps.some((g, i) => g != gaps[i]))) {
        this.gapIntersection.disconnect();
        for (let gap of gaps)
          this.gapIntersection.observe(gap);
        this.gaps = gaps;
      }
    }
    onSelectionChange(event) {
      let wasChanged = this.selectionChanged;
      if (!this.readSelectionRange() || this.delayedAndroidKey)
        return;
      let { view } = this, sel = this.selectionRange;
      if (view.state.facet(editable) ? view.root.activeElement != this.dom : !hasSelection(this.dom, sel))
        return;
      let context = sel.anchorNode && view.docView.tile.nearest(sel.anchorNode);
      if (context && context.isWidget() && context.widget.ignoreEvent(event)) {
        if (!wasChanged)
          this.selectionChanged = false;
        return;
      }
      if ((browser.ie && browser.ie_version <= 11 || browser.android && browser.chrome) && !view.state.selection.main.empty && // (Selection.isCollapsed isn't reliable on IE)
      sel.focusNode && isEquivalentPosition(sel.focusNode, sel.focusOffset, sel.anchorNode, sel.anchorOffset))
        this.flushSoon();
      else
        this.flush(false);
    }
    readSelectionRange() {
      let { view } = this;
      let selection = getSelection(view.root);
      if (!selection)
        return false;
      let range = browser.safari && view.root.nodeType == 11 && view.root.activeElement == this.dom && safariSelectionRangeHack(this.view, selection) || selection;
      if (!range || this.selectionRange.eq(range))
        return false;
      let local = hasSelection(this.dom, range);
      if (local && !this.selectionChanged && view.inputState.lastFocusTime > Date.now() - 200 && view.inputState.lastTouchTime < Date.now() - 300 && atElementStart(this.dom, range)) {
        this.view.inputState.lastFocusTime = 0;
        view.docView.updateSelection();
        return false;
      }
      this.selectionRange.setRange(range);
      if (local)
        this.selectionChanged = true;
      return true;
    }
    setSelectionRange(anchor, head) {
      this.selectionRange.set(anchor.node, anchor.offset, head.node, head.offset);
      this.selectionChanged = false;
    }
    clearSelectionRange() {
      this.selectionRange.set(null, 0, null, 0);
    }
    listenForScroll() {
      this.parentCheck = -1;
      let i = 0, changed = null;
      for (let dom = this.dom; dom; ) {
        if (dom.nodeType == 1) {
          if (!changed && i < this.scrollTargets.length && this.scrollTargets[i] == dom)
            i++;
          else if (!changed)
            changed = this.scrollTargets.slice(0, i);
          if (changed)
            changed.push(dom);
          dom = dom.assignedSlot || dom.parentNode;
        } else if (dom.nodeType == 11) {
          dom = dom.host;
        } else {
          break;
        }
      }
      if (i < this.scrollTargets.length && !changed)
        changed = this.scrollTargets.slice(0, i);
      if (changed) {
        for (let dom of this.scrollTargets)
          dom.removeEventListener("scroll", this.onScroll);
        for (let dom of this.scrollTargets = changed)
          dom.addEventListener("scroll", this.onScroll);
      }
    }
    ignore(f) {
      if (!this.active)
        return f();
      try {
        this.stop();
        return f();
      } finally {
        this.start();
        this.clear();
      }
    }
    start() {
      if (this.active)
        return;
      this.observer.observe(this.dom, observeOptions);
      if (useCharData)
        this.dom.addEventListener("DOMCharacterDataModified", this.onCharData);
      this.active = true;
    }
    stop() {
      if (!this.active)
        return;
      this.active = false;
      this.observer.disconnect();
      if (useCharData)
        this.dom.removeEventListener("DOMCharacterDataModified", this.onCharData);
    }
    // Throw away any pending changes
    clear() {
      this.processRecords();
      this.queue.length = 0;
      this.selectionChanged = false;
    }
    // Chrome Android, especially in combination with GBoard, not only
    // doesn't reliably fire regular key events, but also often
    // surrounds the effect of enter or backspace with a bunch of
    // composition events that, when interrupted, cause text duplication
    // or other kinds of corruption. This hack makes the editor back off
    // from handling DOM changes for a moment when such a key is
    // detected (via beforeinput or keydown), and then tries to flush
    // them or, if that has no effect, dispatches the given key.
    delayAndroidKey(key, keyCode) {
      var _a2;
      if (!this.delayedAndroidKey) {
        let flush = () => {
          let key2 = this.delayedAndroidKey;
          if (key2) {
            this.clearDelayedAndroidKey();
            this.view.inputState.lastKeyCode = key2.keyCode;
            this.view.inputState.lastKeyTime = Date.now();
            let flushed = this.flush();
            if (!flushed && key2.force)
              dispatchKey(this.dom, key2.key, key2.keyCode);
          }
        };
        this.flushingAndroidKey = this.view.win.requestAnimationFrame(flush);
      }
      if (!this.delayedAndroidKey || key == "Enter")
        this.delayedAndroidKey = {
          key,
          keyCode,
          // Only run the key handler when no changes are detected if
          // this isn't coming right after another change, in which case
          // it is probably part of a weird chain of updates, and should
          // be ignored if it returns the DOM to its previous state.
          force: this.lastChange < Date.now() - 50 || !!((_a2 = this.delayedAndroidKey) === null || _a2 === void 0 ? void 0 : _a2.force)
        };
    }
    clearDelayedAndroidKey() {
      this.win.cancelAnimationFrame(this.flushingAndroidKey);
      this.delayedAndroidKey = null;
      this.flushingAndroidKey = -1;
    }
    flushSoon() {
      if (this.delayedFlush < 0)
        this.delayedFlush = this.view.win.requestAnimationFrame(() => {
          this.delayedFlush = -1;
          this.flush();
        });
    }
    forceFlush() {
      if (this.delayedFlush >= 0) {
        this.view.win.cancelAnimationFrame(this.delayedFlush);
        this.delayedFlush = -1;
      }
      this.flush();
    }
    pendingRecords() {
      for (let mut of this.observer.takeRecords())
        this.queue.push(mut);
      return this.queue;
    }
    processRecords() {
      let records = this.pendingRecords();
      if (records.length)
        this.queue = [];
      let from = -1, to = -1, typeOver = false;
      for (let record of records) {
        let range = this.readMutation(record);
        if (!range)
          continue;
        if (range.typeOver)
          typeOver = true;
        if (from == -1) {
          ({ from, to } = range);
        } else {
          from = Math.min(range.from, from);
          to = Math.max(range.to, to);
        }
      }
      return { from, to, typeOver };
    }
    readChange() {
      let { from, to, typeOver } = this.processRecords();
      let newSel = this.selectionChanged && hasSelection(this.dom, this.selectionRange);
      if (from < 0 && !newSel)
        return null;
      if (from > -1)
        this.lastChange = Date.now();
      this.view.inputState.lastFocusTime = 0;
      this.selectionChanged = false;
      let change = new DOMChange(this.view, from, to, typeOver);
      this.view.docView.domChanged = { newSel: change.newSel ? change.newSel.main : null };
      return change;
    }
    // Apply pending changes, if any
    flush(readSelection = true) {
      if (this.delayedFlush >= 0 || this.delayedAndroidKey)
        return false;
      if (readSelection)
        this.readSelectionRange();
      let domChange = this.readChange();
      if (!domChange) {
        this.view.requestMeasure();
        return false;
      }
      let startState = this.view.state;
      let handled = applyDOMChange(this.view, domChange);
      if (this.view.state == startState && (domChange.domChanged || domChange.newSel && !sameSelPos(this.view.state.selection, domChange.newSel.main)))
        this.view.update([]);
      return handled;
    }
    readMutation(rec) {
      let tile = this.view.docView.tile.nearest(rec.target);
      if (!tile || tile.isWidget())
        return null;
      tile.markDirty(rec.type == "attributes");
      if (rec.type == "childList") {
        let childBefore = findChild(tile, rec.previousSibling || rec.target.previousSibling, -1);
        let childAfter = findChild(tile, rec.nextSibling || rec.target.nextSibling, 1);
        return {
          from: childBefore ? tile.posAfter(childBefore) : tile.posAtStart,
          to: childAfter ? tile.posBefore(childAfter) : tile.posAtEnd,
          typeOver: false
        };
      } else if (rec.type == "characterData") {
        return { from: tile.posAtStart, to: tile.posAtEnd, typeOver: rec.target.nodeValue == rec.oldValue };
      } else {
        return null;
      }
    }
    setWindow(win) {
      if (win != this.win) {
        this.removeWindowListeners(this.win);
        this.win = win;
        this.addWindowListeners(this.win);
      }
    }
    addWindowListeners(win) {
      win.addEventListener("resize", this.onResize);
      if (this.printQuery) {
        if (this.printQuery.addEventListener)
          this.printQuery.addEventListener("change", this.onPrint);
        else
          this.printQuery.addListener(this.onPrint);
      } else
        win.addEventListener("beforeprint", this.onPrint);
      win.addEventListener("scroll", this.onScroll);
      win.document.addEventListener("selectionchange", this.onSelectionChange);
    }
    removeWindowListeners(win) {
      win.removeEventListener("scroll", this.onScroll);
      win.removeEventListener("resize", this.onResize);
      if (this.printQuery) {
        if (this.printQuery.removeEventListener)
          this.printQuery.removeEventListener("change", this.onPrint);
        else
          this.printQuery.removeListener(this.onPrint);
      } else
        win.removeEventListener("beforeprint", this.onPrint);
      win.document.removeEventListener("selectionchange", this.onSelectionChange);
    }
    update(update) {
      if (this.editContext) {
        this.editContext.update(update);
        if (update.startState.facet(editable) != update.state.facet(editable))
          update.view.contentDOM.editContext = update.state.facet(editable) ? this.editContext.editContext : null;
      }
    }
    destroy() {
      var _a2, _b, _c;
      this.stop();
      (_a2 = this.intersection) === null || _a2 === void 0 ? void 0 : _a2.disconnect();
      (_b = this.gapIntersection) === null || _b === void 0 ? void 0 : _b.disconnect();
      (_c = this.resizeScroll) === null || _c === void 0 ? void 0 : _c.disconnect();
      for (let dom of this.scrollTargets)
        dom.removeEventListener("scroll", this.onScroll);
      this.removeWindowListeners(this.win);
      clearTimeout(this.parentCheck);
      clearTimeout(this.resizeTimeout);
      this.win.cancelAnimationFrame(this.delayedFlush);
      this.win.cancelAnimationFrame(this.flushingAndroidKey);
      if (this.editContext) {
        this.view.contentDOM.editContext = null;
        this.editContext.destroy();
      }
    }
  };
  function findChild(tile, dom, dir) {
    while (dom) {
      let curTile = Tile.get(dom);
      if (curTile && curTile.parent == tile)
        return curTile;
      let parent = dom.parentNode;
      dom = parent != tile.dom ? parent : dir > 0 ? dom.nextSibling : dom.previousSibling;
    }
    return null;
  }
  function buildSelectionRangeFromRange(view, range) {
    let anchorNode = range.startContainer, anchorOffset = range.startOffset;
    let focusNode = range.endContainer, focusOffset = range.endOffset;
    let curAnchor = view.docView.domAtPos(view.state.selection.main.anchor, 1);
    if (isEquivalentPosition(curAnchor.node, curAnchor.offset, focusNode, focusOffset))
      [anchorNode, anchorOffset, focusNode, focusOffset] = [focusNode, focusOffset, anchorNode, anchorOffset];
    return { anchorNode, anchorOffset, focusNode, focusOffset };
  }
  function safariSelectionRangeHack(view, selection) {
    if (selection.getComposedRanges) {
      let range = selection.getComposedRanges(view.root)[0];
      if (range)
        return buildSelectionRangeFromRange(view, range);
    }
    let found = null;
    function read(event) {
      event.preventDefault();
      event.stopImmediatePropagation();
      found = event.getTargetRanges()[0];
    }
    view.contentDOM.addEventListener("beforeinput", read, true);
    view.dom.ownerDocument.execCommand("indent");
    view.contentDOM.removeEventListener("beforeinput", read, true);
    return found ? buildSelectionRangeFromRange(view, found) : null;
  }
  var EditContextManager = class {
    constructor(view) {
      this.from = 0;
      this.to = 0;
      this.pendingContextChange = null;
      this.handlers = /* @__PURE__ */ Object.create(null);
      this.composing = null;
      this.resetRange(view.state);
      let context = this.editContext = new window.EditContext({
        text: view.state.doc.sliceString(this.from, this.to),
        selectionStart: this.toContextPos(Math.max(this.from, Math.min(this.to, view.state.selection.main.anchor))),
        selectionEnd: this.toContextPos(view.state.selection.main.head)
      });
      this.handlers.textupdate = (e) => {
        let main = view.state.selection.main, { anchor, head } = main;
        let from = this.toEditorPos(e.updateRangeStart), to = this.toEditorPos(e.updateRangeEnd);
        if (view.inputState.composing >= 0 && !this.composing)
          this.composing = { contextBase: e.updateRangeStart, editorBase: from, drifted: false };
        let deletes = to - from > e.text.length;
        if (from == this.from && anchor < this.from)
          from = anchor;
        else if (to == this.to && anchor > this.to)
          to = anchor;
        let diff = findDiff(view.state.sliceDoc(from, to), e.text, (deletes ? main.from : main.to) - from, deletes ? "end" : null);
        if (!diff) {
          let newSel = EditorSelection.single(this.toEditorPos(e.selectionStart), this.toEditorPos(e.selectionEnd));
          if (!sameSelPos(newSel, main))
            view.dispatch({ selection: newSel, userEvent: "select" });
          return;
        }
        let change = {
          from: diff.from + from,
          to: diff.toA + from,
          insert: Text.of(e.text.slice(diff.from, diff.toB).split("\n"))
        };
        if ((browser.mac || browser.android) && change.from == head - 1 && /^\. ?$/.test(e.text) && view.contentDOM.getAttribute("autocorrect") == "off")
          change = { from, to, insert: Text.of([e.text.replace(".", " ")]) };
        this.pendingContextChange = change;
        if (!view.state.readOnly) {
          let newLen = this.to - this.from + (change.to - change.from + change.insert.length);
          applyDOMChangeInner(view, change, EditorSelection.single(this.toEditorPos(e.selectionStart, newLen), this.toEditorPos(e.selectionEnd, newLen)));
        }
        if (this.pendingContextChange) {
          this.revertPending(view.state);
          this.setSelection(view.state);
        }
        if (change.from < change.to && !change.insert.length && view.inputState.composing >= 0 && !/[\\p{Alphabetic}\\p{Number}_]/.test(context.text.slice(Math.max(0, e.updateRangeStart - 1), Math.min(context.text.length, e.updateRangeStart + 1))))
          this.handlers.compositionend(e);
      };
      this.handlers.characterboundsupdate = (e) => {
        let rects = [], prev = null;
        for (let i = this.toEditorPos(e.rangeStart), end = this.toEditorPos(e.rangeEnd); i < end; i++) {
          let rect = view.coordsForChar(i);
          prev = rect && new DOMRect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top) || prev || new DOMRect();
          rects.push(prev);
        }
        context.updateCharacterBounds(e.rangeStart, rects);
      };
      this.handlers.textformatupdate = (e) => {
        let deco = [];
        for (let format of e.getTextFormats()) {
          let lineStyle = format.underlineStyle, thickness = format.underlineThickness;
          if (!/none/i.test(lineStyle) && !/none/i.test(thickness)) {
            let from = this.toEditorPos(format.rangeStart), to = this.toEditorPos(format.rangeEnd);
            if (from < to) {
              let style = `text-decoration: underline ${/^[a-z]/.test(lineStyle) ? lineStyle + " " : lineStyle == "Dashed" ? "dashed " : lineStyle == "Squiggle" ? "wavy " : ""}${/thin/i.test(thickness) ? 1 : 2}px`;
              deco.push(Decoration.mark({ attributes: { style } }).range(from, to));
            }
          }
        }
        view.dispatch({ effects: setEditContextFormatting.of(Decoration.set(deco)) });
      };
      this.handlers.compositionstart = () => {
        if (view.inputState.composing < 0) {
          view.inputState.composing = 0;
          view.inputState.compositionFirstChange = true;
        }
      };
      this.handlers.compositionend = () => {
        view.inputState.composing = -1;
        view.inputState.compositionFirstChange = null;
        if (this.composing) {
          let { drifted } = this.composing;
          this.composing = null;
          if (drifted)
            this.reset(view.state);
        }
      };
      for (let event in this.handlers)
        context.addEventListener(event, this.handlers[event]);
      this.measureReq = { read: (view2) => {
        this.editContext.updateControlBounds(view2.contentDOM.getBoundingClientRect());
        let sel = getSelection(view2.root);
        if (sel && sel.rangeCount)
          this.editContext.updateSelectionBounds(sel.getRangeAt(0).getBoundingClientRect());
      } };
    }
    applyEdits(update) {
      let off = 0, abort = false, pending = this.pendingContextChange;
      update.changes.iterChanges((fromA, toA, _fromB, _toB, insert2) => {
        if (abort)
          return;
        let dLen = insert2.length - (toA - fromA);
        if (pending && toA >= pending.to) {
          if (pending.from == fromA && pending.to == toA && pending.insert.eq(insert2)) {
            pending = this.pendingContextChange = null;
            off += dLen;
            this.to += dLen;
            return;
          } else {
            pending = null;
            this.revertPending(update.state);
          }
        }
        fromA += off;
        toA += off;
        if (toA <= this.from) {
          this.from += dLen;
          this.to += dLen;
        } else if (fromA < this.to) {
          if (fromA < this.from || toA > this.to || this.to - this.from + insert2.length > 3e4) {
            abort = true;
            return;
          }
          this.editContext.updateText(this.toContextPos(fromA), this.toContextPos(toA), insert2.toString());
          this.to += dLen;
        }
        off += dLen;
      });
      if (pending && !abort)
        this.revertPending(update.state);
      return !abort;
    }
    update(update) {
      let reverted = this.pendingContextChange, startSel = update.startState.selection.main;
      if (this.composing && (this.composing.drifted || !update.changes.touchesRange(startSel.from, startSel.to) && update.transactions.some((tr) => !tr.isUserEvent("input.type") && tr.changes.touchesRange(this.from, this.to)))) {
        this.composing.drifted = true;
        this.composing.editorBase = update.changes.mapPos(this.composing.editorBase);
      } else if (!this.applyEdits(update) || !this.rangeIsValid(update.state)) {
        this.pendingContextChange = null;
        this.reset(update.state);
      } else if (update.docChanged || update.selectionSet || reverted) {
        this.setSelection(update.state);
      }
      if (update.geometryChanged || update.docChanged || update.selectionSet)
        update.view.requestMeasure(this.measureReq);
    }
    resetRange(state) {
      let { head } = state.selection.main;
      this.from = Math.max(
        0,
        head - 1e4
        /* CxVp.Margin */
      );
      this.to = Math.min(
        state.doc.length,
        head + 1e4
        /* CxVp.Margin */
      );
    }
    reset(state) {
      this.resetRange(state);
      this.editContext.updateText(0, this.editContext.text.length, state.doc.sliceString(this.from, this.to));
      this.setSelection(state);
    }
    revertPending(state) {
      let pending = this.pendingContextChange;
      this.pendingContextChange = null;
      this.editContext.updateText(this.toContextPos(pending.from), this.toContextPos(pending.from + pending.insert.length), state.doc.sliceString(pending.from, pending.to));
    }
    setSelection(state) {
      let { main } = state.selection;
      let start = this.toContextPos(Math.max(this.from, Math.min(this.to, main.anchor)));
      let end = this.toContextPos(main.head);
      if (this.editContext.selectionStart != start || this.editContext.selectionEnd != end)
        this.editContext.updateSelection(start, end);
    }
    rangeIsValid(state) {
      let { head } = state.selection.main;
      return !(this.from > 0 && head - this.from < 500 || this.to < state.doc.length && this.to - head < 500 || this.to - this.from > 1e4 * 3);
    }
    toEditorPos(contextPos, clipLen = this.to - this.from) {
      contextPos = Math.min(contextPos, clipLen);
      let c = this.composing;
      return c && c.drifted ? c.editorBase + (contextPos - c.contextBase) : contextPos + this.from;
    }
    toContextPos(editorPos) {
      let c = this.composing;
      return c && c.drifted ? c.contextBase + (editorPos - c.editorBase) : editorPos - this.from;
    }
    destroy() {
      for (let event in this.handlers)
        this.editContext.removeEventListener(event, this.handlers[event]);
    }
  };
  var EditorView = class _EditorView {
    /**
    The current editor state.
    */
    get state() {
      return this.viewState.state;
    }
    /**
    To be able to display large documents without consuming too much
    memory or overloading the browser, CodeMirror only draws the
    code that is visible (plus a margin around it) to the DOM. This
    property tells you the extent of the current drawn viewport, in
    document positions.
    */
    get viewport() {
      return this.viewState.viewport;
    }
    /**
    When there are, for example, large collapsed ranges in the
    viewport, its size can be a lot bigger than the actual visible
    content. Thus, if you are doing something like styling the
    content in the viewport, it is preferable to only do so for
    these ranges, which are the subset of the viewport that is
    actually drawn.
    */
    get visibleRanges() {
      return this.viewState.visibleRanges;
    }
    /**
    Returns false when the editor is entirely scrolled out of view
    or otherwise hidden.
    */
    get inView() {
      return this.viewState.inView;
    }
    /**
    Indicates whether the user is currently composing text via
    [IME](https://en.wikipedia.org/wiki/Input_method), and at least
    one change has been made in the current composition.
    */
    get composing() {
      return !!this.inputState && this.inputState.composing > 0;
    }
    /**
    Indicates whether the user is currently in composing state. Note
    that on some platforms, like Android, this will be the case a
    lot, since just putting the cursor on a word starts a
    composition there.
    */
    get compositionStarted() {
      return !!this.inputState && this.inputState.composing >= 0;
    }
    /**
    The document or shadow root that the view lives in.
    */
    get root() {
      return this._root;
    }
    /**
    @internal
    */
    get win() {
      return this.dom.ownerDocument.defaultView || window;
    }
    /**
    Construct a new view. You'll want to either provide a `parent`
    option, or put `view.dom` into your document after creating a
    view, so that the user can see the editor.
    */
    constructor(config = {}) {
      var _a2;
      this.plugins = [];
      this.pluginMap = /* @__PURE__ */ new Map();
      this.editorAttrs = {};
      this.contentAttrs = {};
      this.bidiCache = [];
      this.destroyed = false;
      this.updateState = 2;
      this.measureScheduled = -1;
      this.measureRequests = [];
      this.contentDOM = document.createElement("div");
      this.scrollDOM = document.createElement("div");
      this.scrollDOM.tabIndex = -1;
      this.scrollDOM.className = "cm-scroller";
      this.scrollDOM.appendChild(this.contentDOM);
      this.announceDOM = document.createElement("div");
      this.announceDOM.className = "cm-announced";
      this.announceDOM.setAttribute("aria-live", "polite");
      this.dom = document.createElement("div");
      this.dom.appendChild(this.announceDOM);
      this.dom.appendChild(this.scrollDOM);
      if (config.parent)
        config.parent.appendChild(this.dom);
      let { dispatch } = config;
      this.dispatchTransactions = config.dispatchTransactions || dispatch && ((trs) => trs.forEach((tr) => dispatch(tr, this))) || ((trs) => this.update(trs));
      this.dispatch = this.dispatch.bind(this);
      this._root = config.root || getRoot(config.parent) || document;
      this.viewState = new ViewState(this, config.state || EditorState.create(config));
      if (config.scrollTo && config.scrollTo.is(scrollIntoView))
        this.viewState.scrollTarget = config.scrollTo.value.clip(this.viewState.state);
      this.plugins = this.state.facet(viewPlugin).map((spec) => new PluginInstance(spec));
      for (let plugin of this.plugins)
        plugin.update(this);
      this.observer = new DOMObserver(this);
      this.inputState = new InputState(this);
      this.inputState.ensureHandlers(this.plugins);
      this.docView = new DocView(this);
      this.mountStyles();
      this.updateAttrs();
      this.updateState = 0;
      this.requestMeasure();
      if ((_a2 = document.fonts) === null || _a2 === void 0 ? void 0 : _a2.ready)
        document.fonts.ready.then(() => {
          this.viewState.mustMeasureContent = "refresh";
          this.requestMeasure();
        });
    }
    dispatch(...input) {
      let trs = input.length == 1 && input[0] instanceof Transaction ? input : input.length == 1 && Array.isArray(input[0]) ? input[0] : [this.state.update(...input)];
      this.dispatchTransactions(trs, this);
    }
    /**
    Update the view for the given array of transactions. This will
    update the visible document and selection to match the state
    produced by the transactions, and notify view plugins of the
    change. You should usually call
    [`dispatch`](https://codemirror.net/6/docs/ref/#view.EditorView.dispatch) instead, which uses this
    as a primitive.
    */
    update(transactions) {
      if (this.updateState != 0)
        throw new Error("Calls to EditorView.update are not allowed while an update is in progress");
      let redrawn = false, attrsChanged = false, update;
      let state = this.state;
      for (let tr of transactions) {
        if (tr.startState != state)
          throw new RangeError("Trying to update state with a transaction that doesn't start from the previous state.");
        state = tr.state;
      }
      if (this.destroyed) {
        this.viewState.state = state;
        return;
      }
      let focus = this.hasFocus, focusFlag = 0, dispatchFocus = null;
      if (transactions.some((tr) => tr.annotation(isFocusChange))) {
        this.inputState.notifiedFocused = focus;
        focusFlag = 1;
      } else if (focus != this.inputState.notifiedFocused) {
        this.inputState.notifiedFocused = focus;
        dispatchFocus = focusChangeTransaction(state, focus);
        if (!dispatchFocus)
          focusFlag = 1;
      }
      let pendingKey = this.observer.delayedAndroidKey, domChange = null;
      if (pendingKey) {
        this.observer.clearDelayedAndroidKey();
        domChange = this.observer.readChange();
        if (domChange && !this.state.doc.eq(state.doc) || !this.state.selection.eq(state.selection))
          domChange = null;
      } else {
        this.observer.clear();
      }
      if (state.facet(EditorState.phrases) != this.state.facet(EditorState.phrases))
        return this.setState(state);
      update = ViewUpdate.create(this, state, transactions);
      update.flags |= focusFlag;
      let scrollTarget = this.viewState.scrollTarget;
      try {
        this.updateState = 2;
        for (let tr of transactions) {
          if (scrollTarget)
            scrollTarget = scrollTarget.map(tr.changes);
          if (tr.scrollIntoView) {
            let { main } = tr.state.selection;
            scrollTarget = new ScrollTarget(main.empty ? main : EditorSelection.cursor(main.head, main.head > main.anchor ? -1 : 1));
          }
          for (let e of tr.effects)
            if (e.is(scrollIntoView))
              scrollTarget = e.value.clip(this.state);
        }
        this.viewState.update(update, scrollTarget);
        this.bidiCache = CachedOrder.update(this.bidiCache, update.changes);
        if (!update.empty) {
          this.updatePlugins(update);
          this.inputState.update(update);
        }
        redrawn = this.docView.update(update);
        if (this.state.facet(styleModule) != this.styleModules)
          this.mountStyles();
        attrsChanged = this.updateAttrs();
        this.showAnnouncements(transactions);
        this.docView.updateSelection(redrawn, transactions.some((tr) => tr.isUserEvent("select.pointer")));
      } finally {
        this.updateState = 0;
      }
      if (update.startState.facet(theme) != update.state.facet(theme))
        this.viewState.mustMeasureContent = true;
      if (redrawn || attrsChanged || scrollTarget || this.viewState.mustEnforceCursorAssoc || this.viewState.mustMeasureContent)
        this.requestMeasure();
      if (redrawn)
        this.docViewUpdate();
      if (!update.empty)
        for (let listener of this.state.facet(updateListener)) {
          try {
            listener(update);
          } catch (e) {
            logException(this.state, e, "update listener");
          }
        }
      if (dispatchFocus || domChange)
        Promise.resolve().then(() => {
          if (dispatchFocus && this.state == dispatchFocus.startState)
            this.dispatch(dispatchFocus);
          if (domChange) {
            if (!applyDOMChange(this, domChange) && pendingKey.force)
              dispatchKey(this.contentDOM, pendingKey.key, pendingKey.keyCode);
          }
        });
    }
    /**
    Reset the view to the given state. (This will cause the entire
    document to be redrawn and all view plugins to be reinitialized,
    so you should probably only use it when the new state isn't
    derived from the old state. Otherwise, use
    [`dispatch`](https://codemirror.net/6/docs/ref/#view.EditorView.dispatch) instead.)
    */
    setState(newState) {
      if (this.updateState != 0)
        throw new Error("Calls to EditorView.setState are not allowed while an update is in progress");
      if (this.destroyed) {
        this.viewState.state = newState;
        return;
      }
      this.updateState = 2;
      let hadFocus = this.hasFocus;
      try {
        for (let plugin of this.plugins)
          plugin.destroy(this);
        this.viewState = new ViewState(this, newState);
        this.plugins = newState.facet(viewPlugin).map((spec) => new PluginInstance(spec));
        this.pluginMap.clear();
        for (let plugin of this.plugins)
          plugin.update(this);
        this.docView.destroy();
        this.docView = new DocView(this);
        this.inputState.ensureHandlers(this.plugins);
        this.mountStyles();
        this.updateAttrs();
        this.bidiCache = [];
      } finally {
        this.updateState = 0;
      }
      if (hadFocus)
        this.focus();
      this.requestMeasure();
    }
    updatePlugins(update) {
      let prevSpecs = update.startState.facet(viewPlugin), specs = update.state.facet(viewPlugin);
      if (prevSpecs != specs) {
        let newPlugins = [];
        for (let spec of specs) {
          let found = prevSpecs.indexOf(spec);
          if (found < 0) {
            newPlugins.push(new PluginInstance(spec));
          } else {
            let plugin = this.plugins[found];
            plugin.mustUpdate = update;
            newPlugins.push(plugin);
          }
        }
        for (let plugin of this.plugins)
          if (plugin.mustUpdate != update)
            plugin.destroy(this);
        this.plugins = newPlugins;
        this.pluginMap.clear();
      } else {
        for (let p of this.plugins)
          p.mustUpdate = update;
      }
      for (let i = 0; i < this.plugins.length; i++)
        this.plugins[i].update(this);
      if (prevSpecs != specs)
        this.inputState.ensureHandlers(this.plugins);
    }
    docViewUpdate() {
      for (let plugin of this.plugins) {
        let val = plugin.value;
        if (val && val.docViewUpdate) {
          try {
            val.docViewUpdate(this);
          } catch (e) {
            logException(this.state, e, "doc view update listener");
          }
        }
      }
    }
    /**
    @internal
    */
    measure(flush = true) {
      if (this.destroyed)
        return;
      if (this.measureScheduled > -1)
        this.win.cancelAnimationFrame(this.measureScheduled);
      if (this.observer.delayedAndroidKey) {
        this.measureScheduled = -1;
        this.requestMeasure();
        return;
      }
      this.measureScheduled = 0;
      if (flush)
        this.observer.forceFlush();
      let updated = null;
      let scroll = this.viewState.scrollParent, scrollOffset = this.viewState.getScrollOffset();
      let { scrollAnchorPos, scrollAnchorHeight } = this.viewState;
      if (Math.abs(scrollOffset - this.viewState.scrollOffset) > 1)
        scrollAnchorHeight = -1;
      this.viewState.scrollAnchorHeight = -1;
      try {
        for (let i = 0; ; i++) {
          if (scrollAnchorHeight < 0) {
            if (isScrolledToBottom(scroll || this.win)) {
              scrollAnchorPos = -1;
              scrollAnchorHeight = this.viewState.heightMap.height;
            } else {
              let block = this.viewState.scrollAnchorAt(scrollOffset);
              scrollAnchorPos = block.from;
              scrollAnchorHeight = block.top;
            }
          }
          this.updateState = 1;
          let changed = this.viewState.measure();
          if (!changed && !this.measureRequests.length && this.viewState.scrollTarget == null)
            break;
          if (i > 5) {
            console.warn(this.measureRequests.length ? "Measure loop restarted more than 5 times" : "Viewport failed to stabilize");
            break;
          }
          let measuring = [];
          if (!(changed & 4))
            [this.measureRequests, measuring] = [measuring, this.measureRequests];
          let measured = measuring.map((m) => {
            try {
              return m.read(this);
            } catch (e) {
              logException(this.state, e);
              return BadMeasure;
            }
          });
          let update = ViewUpdate.create(this, this.state, []), redrawn = false;
          update.flags |= changed;
          if (!updated)
            updated = update;
          else
            updated.flags |= changed;
          this.updateState = 2;
          if (!update.empty) {
            this.updatePlugins(update);
            this.inputState.update(update);
            this.updateAttrs();
            redrawn = this.docView.update(update);
            if (redrawn)
              this.docViewUpdate();
          }
          for (let i2 = 0; i2 < measuring.length; i2++)
            if (measured[i2] != BadMeasure) {
              try {
                let m = measuring[i2];
                if (m.write)
                  m.write(measured[i2], this);
              } catch (e) {
                logException(this.state, e);
              }
            }
          if (redrawn)
            this.docView.updateSelection(true);
          if (!update.viewportChanged && this.measureRequests.length == 0) {
            if (this.viewState.editorHeight) {
              if (this.viewState.scrollTarget) {
                this.docView.scrollIntoView(this.viewState.scrollTarget);
                this.viewState.scrollTarget = null;
                scrollAnchorHeight = -1;
                continue;
              } else {
                let newAnchorHeight = scrollAnchorPos < 0 ? this.viewState.heightMap.height : this.viewState.lineBlockAt(scrollAnchorPos).top;
                let diff = (newAnchorHeight - scrollAnchorHeight) / this.scaleY;
                if ((diff > 1 || diff < -1) && (scroll == this.scrollDOM || this.hasFocus || Math.max(this.inputState.lastWheelEvent, this.inputState.lastTouchTime) > Date.now() - 100)) {
                  scrollOffset = scrollOffset + diff;
                  if (scroll)
                    scroll.scrollTop += diff;
                  else
                    this.win.scrollBy(0, diff);
                  scrollAnchorHeight = -1;
                  continue;
                }
              }
            }
            break;
          }
        }
      } finally {
        this.updateState = 0;
        this.measureScheduled = -1;
      }
      if (updated && !updated.empty)
        for (let listener of this.state.facet(updateListener))
          listener(updated);
    }
    /**
    Get the CSS classes for the currently active editor themes.
    */
    get themeClasses() {
      return baseThemeID + " " + (this.state.facet(darkTheme) ? baseDarkID : baseLightID) + " " + this.state.facet(theme);
    }
    updateAttrs() {
      let editorAttrs = attrsFromFacet(this, editorAttributes, {
        class: "cm-editor" + (this.hasFocus ? " cm-focused " : " ") + this.themeClasses
      });
      let contentAttrs = {
        spellcheck: "false",
        autocorrect: "off",
        autocapitalize: "off",
        writingsuggestions: "false",
        translate: "no",
        contenteditable: !this.state.facet(editable) ? "false" : "true",
        class: "cm-content",
        style: `${browser.tabSize}: ${this.state.tabSize}`,
        role: "textbox",
        "aria-multiline": "true"
      };
      if (this.state.readOnly)
        contentAttrs["aria-readonly"] = "true";
      attrsFromFacet(this, contentAttributes, contentAttrs);
      let changed = this.observer.ignore(() => {
        let changedContent = updateAttrs(this.contentDOM, this.contentAttrs, contentAttrs);
        let changedEditor = updateAttrs(this.dom, this.editorAttrs, editorAttrs);
        return changedContent || changedEditor;
      });
      this.editorAttrs = editorAttrs;
      this.contentAttrs = contentAttrs;
      return changed;
    }
    showAnnouncements(trs) {
      let first = true;
      for (let tr of trs)
        for (let effect of tr.effects)
          if (effect.is(_EditorView.announce)) {
            if (first)
              this.announceDOM.textContent = "";
            first = false;
            let div = this.announceDOM.appendChild(document.createElement("div"));
            div.textContent = effect.value;
          }
    }
    mountStyles() {
      this.styleModules = this.state.facet(styleModule);
      let nonce = this.state.facet(_EditorView.cspNonce);
      StyleModule.mount(this.root, this.styleModules.concat(baseTheme$1).reverse(), nonce ? { nonce } : void 0);
    }
    readMeasured() {
      if (this.updateState == 2)
        throw new Error("Reading the editor layout isn't allowed during an update");
      if (this.updateState == 0 && this.measureScheduled > -1)
        this.measure(false);
    }
    /**
    Schedule a layout measurement, optionally providing callbacks to
    do custom DOM measuring followed by a DOM write phase. Using
    this is preferable reading DOM layout directly from, for
    example, an event handler, because it'll make sure measuring and
    drawing done by other components is synchronized, avoiding
    unnecessary DOM layout computations.
    */
    requestMeasure(request) {
      if (this.measureScheduled < 0)
        this.measureScheduled = this.win.requestAnimationFrame(() => this.measure());
      if (request) {
        if (this.measureRequests.indexOf(request) > -1)
          return;
        if (request.key != null)
          for (let i = 0; i < this.measureRequests.length; i++) {
            if (this.measureRequests[i].key === request.key) {
              this.measureRequests[i] = request;
              return;
            }
          }
        this.measureRequests.push(request);
      }
    }
    /**
    Get the value of a specific plugin, if present. Note that
    plugins that crash can be dropped from a view, so even when you
    know you registered a given plugin, it is recommended to check
    the return value of this method.
    */
    plugin(plugin) {
      let known = this.pluginMap.get(plugin);
      if (known === void 0 || known && known.plugin != plugin)
        this.pluginMap.set(plugin, known = this.plugins.find((p) => p.plugin == plugin) || null);
      return known && known.update(this).value;
    }
    /**
    The top position of the document, in screen coordinates. This
    may be negative when the editor is scrolled down. Points
    directly to the top of the first line, not above the padding.
    */
    get documentTop() {
      return this.contentDOM.getBoundingClientRect().top + this.viewState.paddingTop;
    }
    /**
    Reports the padding above and below the document.
    */
    get documentPadding() {
      return { top: this.viewState.paddingTop, bottom: this.viewState.paddingBottom };
    }
    /**
    If the editor is transformed with CSS, this provides the scale
    along the X axis. Otherwise, it will just be 1. Note that
    transforms other than translation and scaling are not supported.
    */
    get scaleX() {
      return this.viewState.scaleX;
    }
    /**
    Provide the CSS transformed scale along the Y axis.
    */
    get scaleY() {
      return this.viewState.scaleY;
    }
    /**
    Find the text line or block widget at the given vertical
    position (which is interpreted as relative to the [top of the
    document](https://codemirror.net/6/docs/ref/#view.EditorView.documentTop)).
    */
    elementAtHeight(height) {
      this.readMeasured();
      return this.viewState.elementAtHeight(height);
    }
    /**
    Find the line block (see
    [`lineBlockAt`](https://codemirror.net/6/docs/ref/#view.EditorView.lineBlockAt)) at the given
    height, again interpreted relative to the [top of the
    document](https://codemirror.net/6/docs/ref/#view.EditorView.documentTop).
    */
    lineBlockAtHeight(height) {
      this.readMeasured();
      return this.viewState.lineBlockAtHeight(height);
    }
    /**
    Get the extent and vertical position of all [line
    blocks](https://codemirror.net/6/docs/ref/#view.EditorView.lineBlockAt) in the viewport. Positions
    are relative to the [top of the
    document](https://codemirror.net/6/docs/ref/#view.EditorView.documentTop);
    */
    get viewportLineBlocks() {
      return this.viewState.viewportLines;
    }
    /**
    Find the line block around the given document position. A line
    block is a range delimited on both sides by either a
    non-[hidden](https://codemirror.net/6/docs/ref/#view.Decoration^replace) line break, or the
    start/end of the document. It will usually just hold a line of
    text, but may be broken into multiple textblocks by block
    widgets.
    */
    lineBlockAt(pos) {
      return this.viewState.lineBlockAt(pos);
    }
    /**
    The editor's total content height.
    */
    get contentHeight() {
      return this.viewState.contentHeight;
    }
    /**
    Move a cursor position by [grapheme
    cluster](https://codemirror.net/6/docs/ref/#state.findClusterBreak). `forward` determines whether
    the motion is away from the line start, or towards it. In
    bidirectional text, the line is traversed in visual order, using
    the editor's [text direction](https://codemirror.net/6/docs/ref/#view.EditorView.textDirection).
    When the start position was the last one on the line, the
    returned position will be across the line break. If there is no
    further line, the original position is returned.
    
    By default, this method moves over a single cluster. The
    optional `by` argument can be used to move across more. It will
    be called with the first cluster as argument, and should return
    a predicate that determines, for each subsequent cluster,
    whether it should also be moved over.
    */
    moveByChar(start, forward, by) {
      return skipAtoms(this, start, moveByChar(this, start, forward, by));
    }
    /**
    Move a cursor position across the next group of either
    [letters](https://codemirror.net/6/docs/ref/#state.EditorState.charCategorizer) or non-letter
    non-whitespace characters.
    */
    moveByGroup(start, forward) {
      return skipAtoms(this, start, moveByChar(this, start, forward, (initial) => byGroup(this, start.head, initial)));
    }
    /**
    Get the cursor position visually at the start or end of a line.
    Note that this may differ from the _logical_ position at its
    start or end (which is simply at `line.from`/`line.to`) if text
    at the start or end goes against the line's base text direction.
    */
    visualLineSide(line, end) {
      let order = this.bidiSpans(line), dir = this.textDirectionAt(line.from);
      let span = order[end ? order.length - 1 : 0];
      return EditorSelection.cursor(span.side(end, dir) + line.from, span.forward(!end, dir) ? 1 : -1);
    }
    /**
    Move to the next line boundary in the given direction. If
    `includeWrap` is true, line wrapping is on, and there is a
    further wrap point on the current line, the wrap point will be
    returned. Otherwise this function will return the start or end
    of the line.
    */
    moveToLineBoundary(start, forward, includeWrap = true) {
      return moveToLineBoundary(this, start, forward, includeWrap);
    }
    /**
    Move a cursor position vertically. When `distance` isn't given,
    it defaults to moving to the next line (including wrapped
    lines). Otherwise, `distance` should provide a positive distance
    in pixels.
    
    When `start` has a
    [`goalColumn`](https://codemirror.net/6/docs/ref/#state.SelectionRange.goalColumn), the vertical
    motion will use that as a target horizontal position. Otherwise,
    the cursor's own horizontal position is used. The returned
    cursor will have its goal column set to whichever column was
    used.
    */
    moveVertically(start, forward, distance) {
      return skipAtoms(this, start, moveVertically(this, start, forward, distance));
    }
    /**
    Find the DOM parent node and offset (child offset if `node` is
    an element, character offset when it is a text node) at the
    given document position.
    
    Note that for positions that aren't currently in
    `visibleRanges`, the resulting DOM position isn't necessarily
    meaningful (it may just point before or after a placeholder
    element).
    */
    domAtPos(pos, side = 1) {
      return this.docView.domAtPos(pos, side);
    }
    /**
    Find the document position at the given DOM node. Can be useful
    for associating positions with DOM events. Will raise an error
    when `node` isn't part of the editor content.
    */
    posAtDOM(node, offset = 0) {
      return this.docView.posFromDOM(node, offset);
    }
    posAtCoords(coords, precise = true) {
      this.readMeasured();
      let found = posAtCoords(this, coords, precise);
      return found && found.pos;
    }
    posAndSideAtCoords(coords, precise = true) {
      this.readMeasured();
      return posAtCoords(this, coords, precise);
    }
    /**
    Get the screen coordinates at the given document position.
    `side` determines whether the coordinates are based on the
    element before (-1) or after (1) the position (if no element is
    available on the given side, the method will transparently use
    another strategy to get reasonable coordinates).
    */
    coordsAtPos(pos, side = 1) {
      this.readMeasured();
      let rect = this.docView.coordsAt(pos, side);
      if (!rect || rect.left == rect.right)
        return rect;
      let line = this.state.doc.lineAt(pos), order = this.bidiSpans(line);
      let span = order[BidiSpan.find(order, pos - line.from, -1, side)];
      return flattenRect(rect, span.dir == Direction.LTR == side > 0);
    }
    /**
    Return the rectangle around a given character. If `pos` does not
    point in front of a character that is in the viewport and
    rendered (i.e. not replaced, not a line break), this will return
    null. For space characters that are a line wrap point, this will
    return the position before the line break.
    */
    coordsForChar(pos) {
      this.readMeasured();
      return this.docView.coordsForChar(pos);
    }
    /**
    The default width of a character in the editor. May not
    accurately reflect the width of all characters (given variable
    width fonts or styling of invididual ranges).
    */
    get defaultCharacterWidth() {
      return this.viewState.heightOracle.charWidth;
    }
    /**
    The default height of a line in the editor. May not be accurate
    for all lines.
    */
    get defaultLineHeight() {
      return this.viewState.heightOracle.lineHeight;
    }
    /**
    The text direction
    ([`direction`](https://developer.mozilla.org/en-US/docs/Web/CSS/direction)
    CSS property) of the editor's content element.
    */
    get textDirection() {
      return this.viewState.defaultTextDirection;
    }
    /**
    Find the text direction of the block at the given position, as
    assigned by CSS. If
    [`perLineTextDirection`](https://codemirror.net/6/docs/ref/#view.EditorView^perLineTextDirection)
    isn't enabled, or the given position is outside of the viewport,
    this will always return the same as
    [`textDirection`](https://codemirror.net/6/docs/ref/#view.EditorView.textDirection). Note that
    this may trigger a DOM layout.
    */
    textDirectionAt(pos) {
      let perLine = this.state.facet(perLineTextDirection);
      if (!perLine || pos < this.viewport.from || pos > this.viewport.to)
        return this.textDirection;
      this.readMeasured();
      return this.docView.textDirectionAt(pos);
    }
    /**
    Whether this editor [wraps lines](https://codemirror.net/6/docs/ref/#view.EditorView.lineWrapping)
    (as determined by the
    [`white-space`](https://developer.mozilla.org/en-US/docs/Web/CSS/white-space)
    CSS property of its content element).
    */
    get lineWrapping() {
      return this.viewState.heightOracle.lineWrapping;
    }
    /**
    Returns the bidirectional text structure of the given line
    (which should be in the current document) as an array of span
    objects. The order of these spans matches the [text
    direction](https://codemirror.net/6/docs/ref/#view.EditorView.textDirection)—if that is
    left-to-right, the leftmost spans come first, otherwise the
    rightmost spans come first.
    */
    bidiSpans(line) {
      if (line.length > MaxBidiLine)
        return trivialOrder(line.length);
      let dir = this.textDirectionAt(line.from), isolates;
      for (let entry of this.bidiCache) {
        if (entry.from == line.from && entry.dir == dir && (entry.fresh || isolatesEq(entry.isolates, isolates = getIsolatedRanges(this, line))))
          return entry.order;
      }
      if (!isolates)
        isolates = getIsolatedRanges(this, line);
      let order = computeOrder(line.text, dir, isolates);
      this.bidiCache.push(new CachedOrder(line.from, line.to, dir, isolates, true, order));
      return order;
    }
    /**
    Check whether the editor has focus.
    */
    get hasFocus() {
      var _a2;
      return (this.dom.ownerDocument.hasFocus() || browser.safari && ((_a2 = this.inputState) === null || _a2 === void 0 ? void 0 : _a2.lastContextMenu) > Date.now() - 3e4) && this.root.activeElement == this.contentDOM;
    }
    /**
    Put focus on the editor.
    */
    focus() {
      this.observer.ignore(() => {
        focusPreventScroll(this.contentDOM);
        this.docView.updateSelection();
      });
    }
    /**
    Update the [root](https://codemirror.net/6/docs/ref/##view.EditorViewConfig.root) in which the editor lives. This is only
    necessary when moving the editor's existing DOM to a new window or shadow root.
    */
    setRoot(root) {
      if (this._root != root) {
        this._root = root;
        this.observer.setWindow((root.nodeType == 9 ? root : root.ownerDocument).defaultView || window);
        this.mountStyles();
      }
    }
    /**
    Clean up this editor view, removing its element from the
    document, unregistering event handlers, and notifying
    plugins. The view instance can no longer be used after
    calling this.
    */
    destroy() {
      if (this.root.activeElement == this.contentDOM)
        this.contentDOM.blur();
      for (let plugin of this.plugins)
        plugin.destroy(this);
      this.plugins = [];
      this.inputState.destroy();
      this.docView.destroy();
      this.dom.remove();
      this.observer.destroy();
      if (this.measureScheduled > -1)
        this.win.cancelAnimationFrame(this.measureScheduled);
      this.destroyed = true;
    }
    /**
    Returns an effect that can be
    [added](https://codemirror.net/6/docs/ref/#state.TransactionSpec.effects) to a transaction to
    cause it to scroll the given position or range into view.
    */
    static scrollIntoView(pos, options = {}) {
      return scrollIntoView.of(new ScrollTarget(typeof pos == "number" ? EditorSelection.cursor(pos) : pos, options.y, options.x, options.yMargin, options.xMargin));
    }
    /**
    Return an effect that resets the editor to its current (at the
    time this method was called) scroll position. Note that this
    only affects the editor's own scrollable element, not parents.
    See also
    [`EditorViewConfig.scrollTo`](https://codemirror.net/6/docs/ref/#view.EditorViewConfig.scrollTo).
    
    The effect should be used with a document identical to the one
    it was created for. Failing to do so is not an error, but may
    not scroll to the expected position. You can
    [map](https://codemirror.net/6/docs/ref/#state.StateEffect.map) the effect to account for changes.
    */
    scrollSnapshot() {
      let { scrollTop, scrollLeft } = this.scrollDOM;
      let ref = this.viewState.scrollAnchorAt(scrollTop);
      return scrollIntoView.of(new ScrollTarget(EditorSelection.cursor(ref.from), "start", "start", ref.top - scrollTop, scrollLeft, true));
    }
    /**
    Enable or disable tab-focus mode, which disables key bindings
    for Tab and Shift-Tab, letting the browser's default
    focus-changing behavior go through instead. This is useful to
    prevent trapping keyboard users in your editor.
    
    Without argument, this toggles the mode. With a boolean, it
    enables (true) or disables it (false). Given a number, it
    temporarily enables the mode until that number of milliseconds
    have passed or another non-Tab key is pressed.
    */
    setTabFocusMode(to) {
      if (to == null)
        this.inputState.tabFocusMode = this.inputState.tabFocusMode < 0 ? 0 : -1;
      else if (typeof to == "boolean")
        this.inputState.tabFocusMode = to ? 0 : -1;
      else if (this.inputState.tabFocusMode != 0)
        this.inputState.tabFocusMode = Date.now() + to;
    }
    /**
    Returns an extension that can be used to add DOM event handlers.
    The value should be an object mapping event names to handler
    functions. For any given event, such functions are ordered by
    extension precedence, and the first handler to return true will
    be assumed to have handled that event, and no other handlers or
    built-in behavior will be activated for it. These are registered
    on the [content element](https://codemirror.net/6/docs/ref/#view.EditorView.contentDOM), except
    for `scroll` handlers, which will be called any time the
    editor's [scroll element](https://codemirror.net/6/docs/ref/#view.EditorView.scrollDOM) or one of
    its parent nodes is scrolled.
    */
    static domEventHandlers(handlers2) {
      return ViewPlugin.define(() => ({}), { eventHandlers: handlers2 });
    }
    /**
    Create an extension that registers DOM event observers. Contrary
    to event [handlers](https://codemirror.net/6/docs/ref/#view.EditorView^domEventHandlers),
    observers can't be prevented from running by a higher-precedence
    handler returning true. They also don't prevent other handlers
    and observers from running when they return true, and should not
    call `preventDefault`.
    */
    static domEventObservers(observers2) {
      return ViewPlugin.define(() => ({}), { eventObservers: observers2 });
    }
    /**
    Create a theme extension. The first argument can be a
    [`style-mod`](https://github.com/marijnh/style-mod#documentation)
    style spec providing the styles for the theme. These will be
    prefixed with a generated class for the style.
    
    Because the selectors will be prefixed with a scope class, rule
    that directly match the editor's [wrapper
    element](https://codemirror.net/6/docs/ref/#view.EditorView.dom)—to which the scope class will be
    added—need to be explicitly differentiated by adding an `&` to
    the selector for that element—for example
    `&.cm-focused`.
    
    When `dark` is set to true, the theme will be marked as dark,
    which will cause the `&dark` rules from [base
    themes](https://codemirror.net/6/docs/ref/#view.EditorView^baseTheme) to be used (as opposed to
    `&light` when a light theme is active).
    */
    static theme(spec, options) {
      let prefix = StyleModule.newName();
      let result = [theme.of(prefix), styleModule.of(buildTheme(`.${prefix}`, spec))];
      if (options && options.dark)
        result.push(darkTheme.of(true));
      return result;
    }
    /**
    Create an extension that adds styles to the base theme. Like
    with [`theme`](https://codemirror.net/6/docs/ref/#view.EditorView^theme), use `&` to indicate the
    place of the editor wrapper element when directly targeting
    that. You can also use `&dark` or `&light` instead to only
    target editors with a dark or light theme.
    */
    static baseTheme(spec) {
      return Prec.lowest(styleModule.of(buildTheme("." + baseThemeID, spec, lightDarkIDs)));
    }
    /**
    Retrieve an editor view instance from the view's DOM
    representation.
    */
    static findFromDOM(dom) {
      var _a2;
      let content2 = dom.querySelector(".cm-content");
      let tile = content2 && Tile.get(content2) || Tile.get(dom);
      return ((_a2 = tile === null || tile === void 0 ? void 0 : tile.root) === null || _a2 === void 0 ? void 0 : _a2.view) || null;
    }
  };
  EditorView.styleModule = styleModule;
  EditorView.inputHandler = inputHandler;
  EditorView.clipboardInputFilter = clipboardInputFilter;
  EditorView.clipboardOutputFilter = clipboardOutputFilter;
  EditorView.scrollHandler = scrollHandler;
  EditorView.focusChangeEffect = focusChangeEffect;
  EditorView.perLineTextDirection = perLineTextDirection;
  EditorView.exceptionSink = exceptionSink;
  EditorView.updateListener = updateListener;
  EditorView.editable = editable;
  EditorView.mouseSelectionStyle = mouseSelectionStyle;
  EditorView.dragMovesSelection = dragMovesSelection$1;
  EditorView.clickAddsSelectionRange = clickAddsSelectionRange;
  EditorView.decorations = decorations;
  EditorView.blockWrappers = blockWrappers;
  EditorView.outerDecorations = outerDecorations;
  EditorView.atomicRanges = atomicRanges;
  EditorView.bidiIsolatedRanges = bidiIsolatedRanges;
  EditorView.scrollMargins = scrollMargins;
  EditorView.darkTheme = darkTheme;
  EditorView.cspNonce = /* @__PURE__ */ Facet.define({ combine: (values) => values.length ? values[0] : "" });
  EditorView.contentAttributes = contentAttributes;
  EditorView.editorAttributes = editorAttributes;
  EditorView.lineWrapping = /* @__PURE__ */ EditorView.contentAttributes.of({ "class": "cm-lineWrapping" });
  EditorView.announce = /* @__PURE__ */ StateEffect.define();
  var MaxBidiLine = 4096;
  var BadMeasure = {};
  var CachedOrder = class _CachedOrder {
    constructor(from, to, dir, isolates, fresh, order) {
      this.from = from;
      this.to = to;
      this.dir = dir;
      this.isolates = isolates;
      this.fresh = fresh;
      this.order = order;
    }
    static update(cache, changes) {
      if (changes.empty && !cache.some((c) => c.fresh))
        return cache;
      let result = [], lastDir = cache.length ? cache[cache.length - 1].dir : Direction.LTR;
      for (let i = Math.max(0, cache.length - 10); i < cache.length; i++) {
        let entry = cache[i];
        if (entry.dir == lastDir && !changes.touchesRange(entry.from, entry.to))
          result.push(new _CachedOrder(changes.mapPos(entry.from, 1), changes.mapPos(entry.to, -1), entry.dir, entry.isolates, false, entry.order));
      }
      return result;
    }
  };
  function attrsFromFacet(view, facet, base2) {
    for (let sources = view.state.facet(facet), i = sources.length - 1; i >= 0; i--) {
      let source = sources[i], value = typeof source == "function" ? source(view) : source;
      if (value)
        combineAttrs(value, base2);
    }
    return base2;
  }
  var currentPlatform = browser.mac ? "mac" : browser.windows ? "win" : browser.linux ? "linux" : "key";
  function normalizeKeyName(name2, platform) {
    const parts = name2.split(/-(?!$)/);
    let result = parts[parts.length - 1];
    if (result == "Space")
      result = " ";
    let alt, ctrl, shift2, meta2;
    for (let i = 0; i < parts.length - 1; ++i) {
      const mod = parts[i];
      if (/^(cmd|meta|m)$/i.test(mod))
        meta2 = true;
      else if (/^a(lt)?$/i.test(mod))
        alt = true;
      else if (/^(c|ctrl|control)$/i.test(mod))
        ctrl = true;
      else if (/^s(hift)?$/i.test(mod))
        shift2 = true;
      else if (/^mod$/i.test(mod)) {
        if (platform == "mac")
          meta2 = true;
        else
          ctrl = true;
      } else
        throw new Error("Unrecognized modifier name: " + mod);
    }
    if (alt)
      result = "Alt-" + result;
    if (ctrl)
      result = "Ctrl-" + result;
    if (meta2)
      result = "Meta-" + result;
    if (shift2)
      result = "Shift-" + result;
    return result;
  }
  function modifiers(name2, event, shift2) {
    if (event.altKey)
      name2 = "Alt-" + name2;
    if (event.ctrlKey)
      name2 = "Ctrl-" + name2;
    if (event.metaKey)
      name2 = "Meta-" + name2;
    if (shift2 !== false && event.shiftKey)
      name2 = "Shift-" + name2;
    return name2;
  }
  var handleKeyEvents = /* @__PURE__ */ Prec.default(/* @__PURE__ */ EditorView.domEventHandlers({
    keydown(event, view) {
      return runHandlers(getKeymap(view.state), event, view, "editor");
    }
  }));
  var keymap = /* @__PURE__ */ Facet.define({ enables: handleKeyEvents });
  var Keymaps = /* @__PURE__ */ new WeakMap();
  function getKeymap(state) {
    let bindings = state.facet(keymap);
    let map = Keymaps.get(bindings);
    if (!map)
      Keymaps.set(bindings, map = buildKeymap(bindings.reduce((a, b) => a.concat(b), [])));
    return map;
  }
  function runScopeHandlers(view, event, scope) {
    return runHandlers(getKeymap(view.state), event, view, scope);
  }
  var storedPrefix = null;
  var PrefixTimeout = 4e3;
  function buildKeymap(bindings, platform = currentPlatform) {
    let bound = /* @__PURE__ */ Object.create(null);
    let isPrefix = /* @__PURE__ */ Object.create(null);
    let checkPrefix = (name2, is) => {
      let current = isPrefix[name2];
      if (current == null)
        isPrefix[name2] = is;
      else if (current != is)
        throw new Error("Key binding " + name2 + " is used both as a regular binding and as a multi-stroke prefix");
    };
    let add2 = (scope, key, command2, preventDefault, stopPropagation) => {
      var _a2, _b;
      let scopeObj = bound[scope] || (bound[scope] = /* @__PURE__ */ Object.create(null));
      let parts = key.split(/ (?!$)/).map((k) => normalizeKeyName(k, platform));
      for (let i = 1; i < parts.length; i++) {
        let prefix = parts.slice(0, i).join(" ");
        checkPrefix(prefix, true);
        if (!scopeObj[prefix])
          scopeObj[prefix] = {
            preventDefault: true,
            stopPropagation: false,
            run: [(view) => {
              let ourObj = storedPrefix = { view, prefix, scope };
              setTimeout(() => {
                if (storedPrefix == ourObj)
                  storedPrefix = null;
              }, PrefixTimeout);
              return true;
            }]
          };
      }
      let full = parts.join(" ");
      checkPrefix(full, false);
      let binding = scopeObj[full] || (scopeObj[full] = {
        preventDefault: false,
        stopPropagation: false,
        run: ((_b = (_a2 = scopeObj._any) === null || _a2 === void 0 ? void 0 : _a2.run) === null || _b === void 0 ? void 0 : _b.slice()) || []
      });
      if (command2)
        binding.run.push(command2);
      if (preventDefault)
        binding.preventDefault = true;
      if (stopPropagation)
        binding.stopPropagation = true;
    };
    for (let b of bindings) {
      let scopes = b.scope ? b.scope.split(" ") : ["editor"];
      if (b.any)
        for (let scope of scopes) {
          let scopeObj = bound[scope] || (bound[scope] = /* @__PURE__ */ Object.create(null));
          if (!scopeObj._any)
            scopeObj._any = { preventDefault: false, stopPropagation: false, run: [] };
          let { any } = b;
          for (let key in scopeObj)
            scopeObj[key].run.push((view) => any(view, currentKeyEvent));
        }
      let name2 = b[platform] || b.key;
      if (!name2)
        continue;
      for (let scope of scopes) {
        add2(scope, name2, b.run, b.preventDefault, b.stopPropagation);
        if (b.shift)
          add2(scope, "Shift-" + name2, b.shift, b.preventDefault, b.stopPropagation);
      }
    }
    return bound;
  }
  var currentKeyEvent = null;
  function runHandlers(map, event, view, scope) {
    currentKeyEvent = event;
    let name2 = keyName(event);
    let charCode = codePointAt2(name2, 0), isChar = codePointSize2(charCode) == name2.length && name2 != " ";
    let prefix = "", handled = false, prevented = false, stopPropagation = false;
    if (storedPrefix && storedPrefix.view == view && storedPrefix.scope == scope) {
      prefix = storedPrefix.prefix + " ";
      if (modifierCodes.indexOf(event.keyCode) < 0) {
        prevented = true;
        storedPrefix = null;
      }
    }
    let ran = /* @__PURE__ */ new Set();
    let runFor = (binding) => {
      if (binding) {
        for (let cmd2 of binding.run)
          if (!ran.has(cmd2)) {
            ran.add(cmd2);
            if (cmd2(view)) {
              if (binding.stopPropagation)
                stopPropagation = true;
              return true;
            }
          }
        if (binding.preventDefault) {
          if (binding.stopPropagation)
            stopPropagation = true;
          prevented = true;
        }
      }
      return false;
    };
    let scopeObj = map[scope], baseName, shiftName;
    if (scopeObj) {
      if (runFor(scopeObj[prefix + modifiers(name2, event, !isChar)])) {
        handled = true;
      } else if (isChar && (event.altKey || event.metaKey || event.ctrlKey) && // Ctrl-Alt may be used for AltGr on Windows
      !(browser.windows && event.ctrlKey && event.altKey) && // Alt-combinations on macOS tend to be typed characters
      !(browser.mac && event.altKey && !(event.ctrlKey || event.metaKey)) && (baseName = base[event.keyCode]) && baseName != name2) {
        if (runFor(scopeObj[prefix + modifiers(baseName, event, true)])) {
          handled = true;
        } else if (event.shiftKey && (shiftName = shift[event.keyCode]) != name2 && shiftName != baseName && runFor(scopeObj[prefix + modifiers(shiftName, event, false)])) {
          handled = true;
        }
      } else if (isChar && event.shiftKey && runFor(scopeObj[prefix + modifiers(name2, event, true)])) {
        handled = true;
      }
      if (!handled && runFor(scopeObj._any))
        handled = true;
    }
    if (prevented)
      handled = true;
    if (handled && stopPropagation)
      event.stopPropagation();
    currentKeyEvent = null;
    return handled;
  }
  var RectangleMarker = class _RectangleMarker {
    /**
    Create a marker with the given class and dimensions. If `width`
    is null, the DOM element will get no width style.
    */
    constructor(className, left, top2, width, height) {
      this.className = className;
      this.left = left;
      this.top = top2;
      this.width = width;
      this.height = height;
    }
    draw() {
      let elt = document.createElement("div");
      elt.className = this.className;
      this.adjust(elt);
      return elt;
    }
    update(elt, prev) {
      if (prev.className != this.className)
        return false;
      this.adjust(elt);
      return true;
    }
    adjust(elt) {
      elt.style.left = this.left + "px";
      elt.style.top = this.top + "px";
      if (this.width != null)
        elt.style.width = this.width + "px";
      elt.style.height = this.height + "px";
    }
    eq(p) {
      return this.left == p.left && this.top == p.top && this.width == p.width && this.height == p.height && this.className == p.className;
    }
    /**
    Create a set of rectangles for the given selection range,
    assigning them theclass`className`. Will create a single
    rectangle for empty ranges, and a set of selection-style
    rectangles covering the range's content (in a bidi-aware
    way) for non-empty ones.
    */
    static forRange(view, className, range) {
      if (range.empty) {
        let pos = view.coordsAtPos(range.head, range.assoc || 1);
        if (!pos)
          return [];
        let base2 = getBase(view);
        return [new _RectangleMarker(className, pos.left - base2.left, pos.top - base2.top, null, pos.bottom - pos.top)];
      } else {
        return rectanglesForRange(view, className, range);
      }
    }
  };
  function getBase(view) {
    let rect = view.scrollDOM.getBoundingClientRect();
    let left = view.textDirection == Direction.LTR ? rect.left : rect.right - view.scrollDOM.clientWidth * view.scaleX;
    return { left: left - view.scrollDOM.scrollLeft * view.scaleX, top: rect.top - view.scrollDOM.scrollTop * view.scaleY };
  }
  function wrappedLine(view, pos, side, inside) {
    let coords = view.coordsAtPos(pos, side * 2);
    if (!coords)
      return inside;
    let editorRect = view.dom.getBoundingClientRect();
    let y = (coords.top + coords.bottom) / 2;
    let left = view.posAtCoords({ x: editorRect.left + 1, y });
    let right = view.posAtCoords({ x: editorRect.right - 1, y });
    if (left == null || right == null)
      return inside;
    return { from: Math.max(inside.from, Math.min(left, right)), to: Math.min(inside.to, Math.max(left, right)) };
  }
  function rectanglesForRange(view, className, range) {
    if (range.to <= view.viewport.from || range.from >= view.viewport.to)
      return [];
    let from = Math.max(range.from, view.viewport.from), to = Math.min(range.to, view.viewport.to);
    let ltr = view.textDirection == Direction.LTR;
    let content2 = view.contentDOM, contentRect = content2.getBoundingClientRect(), base2 = getBase(view);
    let lineElt = content2.querySelector(".cm-line"), lineStyle = lineElt && window.getComputedStyle(lineElt);
    let leftSide = contentRect.left + (lineStyle ? parseInt(lineStyle.paddingLeft) + Math.min(0, parseInt(lineStyle.textIndent)) : 0);
    let rightSide = contentRect.right - (lineStyle ? parseInt(lineStyle.paddingRight) : 0);
    let startBlock = blockAt(view, from, 1), endBlock = blockAt(view, to, -1);
    let visualStart = startBlock.type == BlockType.Text ? startBlock : null;
    let visualEnd = endBlock.type == BlockType.Text ? endBlock : null;
    if (visualStart && (view.lineWrapping || startBlock.widgetLineBreaks))
      visualStart = wrappedLine(view, from, 1, visualStart);
    if (visualEnd && (view.lineWrapping || endBlock.widgetLineBreaks))
      visualEnd = wrappedLine(view, to, -1, visualEnd);
    if (visualStart && visualEnd && visualStart.from == visualEnd.from && visualStart.to == visualEnd.to) {
      return pieces(drawForLine(range.from, range.to, visualStart));
    } else {
      let top2 = visualStart ? drawForLine(range.from, null, visualStart) : drawForWidget(startBlock, false);
      let bottom = visualEnd ? drawForLine(null, range.to, visualEnd) : drawForWidget(endBlock, true);
      let between = [];
      if ((visualStart || startBlock).to < (visualEnd || endBlock).from - (visualStart && visualEnd ? 1 : 0) || startBlock.widgetLineBreaks > 1 && top2.bottom + view.defaultLineHeight / 2 < bottom.top)
        between.push(piece(leftSide, top2.bottom, rightSide, bottom.top));
      else if (top2.bottom < bottom.top && view.elementAtHeight((top2.bottom + bottom.top) / 2).type == BlockType.Text)
        top2.bottom = bottom.top = (top2.bottom + bottom.top) / 2;
      return pieces(top2).concat(between).concat(pieces(bottom));
    }
    function piece(left, top2, right, bottom) {
      return new RectangleMarker(className, left - base2.left, top2 - base2.top, Math.max(0, right - left), bottom - top2);
    }
    function pieces({ top: top2, bottom, horizontal }) {
      let pieces2 = [];
      for (let i = 0; i < horizontal.length; i += 2)
        pieces2.push(piece(horizontal[i], top2, horizontal[i + 1], bottom));
      return pieces2;
    }
    function drawForLine(from2, to2, line) {
      let top2 = 1e9, bottom = -1e9, horizontal = [];
      function addSpan(from3, fromOpen, to3, toOpen, dir) {
        let fromCoords = view.coordsAtPos(from3, from3 == line.to ? -2 : 2);
        let toCoords = view.coordsAtPos(to3, to3 == line.from ? 2 : -2);
        if (!fromCoords || !toCoords)
          return;
        top2 = Math.min(fromCoords.top, toCoords.top, top2);
        bottom = Math.max(fromCoords.bottom, toCoords.bottom, bottom);
        if (dir == Direction.LTR)
          horizontal.push(ltr && fromOpen ? leftSide : fromCoords.left, ltr && toOpen ? rightSide : toCoords.right);
        else
          horizontal.push(!ltr && toOpen ? leftSide : toCoords.left, !ltr && fromOpen ? rightSide : fromCoords.right);
      }
      let start = from2 !== null && from2 !== void 0 ? from2 : line.from, end = to2 !== null && to2 !== void 0 ? to2 : line.to;
      for (let r of view.visibleRanges)
        if (r.to > start && r.from < end) {
          for (let pos = Math.max(r.from, start), endPos = Math.min(r.to, end); ; ) {
            let docLine = view.state.doc.lineAt(pos);
            for (let span of view.bidiSpans(docLine)) {
              let spanFrom = span.from + docLine.from, spanTo = span.to + docLine.from;
              if (spanFrom >= endPos)
                break;
              if (spanTo > pos)
                addSpan(Math.max(spanFrom, pos), from2 == null && spanFrom <= start, Math.min(spanTo, endPos), to2 == null && spanTo >= end, span.dir);
            }
            pos = docLine.to + 1;
            if (pos >= endPos)
              break;
          }
        }
      if (horizontal.length == 0)
        addSpan(start, from2 == null, end, to2 == null, view.textDirection);
      return { top: top2, bottom, horizontal };
    }
    function drawForWidget(block, top2) {
      let y = contentRect.top + (top2 ? block.top : block.bottom);
      return { top: y, bottom: y, horizontal: [] };
    }
  }
  function sameMarker(a, b) {
    return a.constructor == b.constructor && a.eq(b);
  }
  var LayerView = class {
    constructor(view, layer2) {
      this.view = view;
      this.layer = layer2;
      this.drawn = [];
      this.scaleX = 1;
      this.scaleY = 1;
      this.measureReq = { read: this.measure.bind(this), write: this.draw.bind(this) };
      this.dom = view.scrollDOM.appendChild(document.createElement("div"));
      this.dom.classList.add("cm-layer");
      if (layer2.above)
        this.dom.classList.add("cm-layer-above");
      if (layer2.class)
        this.dom.classList.add(layer2.class);
      this.scale();
      this.dom.setAttribute("aria-hidden", "true");
      this.setOrder(view.state);
      view.requestMeasure(this.measureReq);
      if (layer2.mount)
        layer2.mount(this.dom, view);
    }
    update(update) {
      if (update.startState.facet(layerOrder) != update.state.facet(layerOrder))
        this.setOrder(update.state);
      if (this.layer.update(update, this.dom) || update.geometryChanged) {
        this.scale();
        update.view.requestMeasure(this.measureReq);
      }
    }
    docViewUpdate(view) {
      if (this.layer.updateOnDocViewUpdate !== false)
        view.requestMeasure(this.measureReq);
    }
    setOrder(state) {
      let pos = 0, order = state.facet(layerOrder);
      while (pos < order.length && order[pos] != this.layer)
        pos++;
      this.dom.style.zIndex = String((this.layer.above ? 150 : -1) - pos);
    }
    measure() {
      return this.layer.markers(this.view);
    }
    scale() {
      let { scaleX, scaleY } = this.view;
      if (scaleX != this.scaleX || scaleY != this.scaleY) {
        this.scaleX = scaleX;
        this.scaleY = scaleY;
        this.dom.style.transform = `scale(${1 / scaleX}, ${1 / scaleY})`;
      }
    }
    draw(markers) {
      if (markers.length != this.drawn.length || markers.some((p, i) => !sameMarker(p, this.drawn[i]))) {
        let old = this.dom.firstChild, oldI = 0;
        for (let marker of markers) {
          if (marker.update && old && marker.constructor && this.drawn[oldI].constructor && marker.update(old, this.drawn[oldI])) {
            old = old.nextSibling;
            oldI++;
          } else {
            this.dom.insertBefore(marker.draw(), old);
          }
        }
        while (old) {
          let next = old.nextSibling;
          old.remove();
          old = next;
        }
        this.drawn = markers;
        if (browser.safari && browser.safari_version >= 26)
          this.dom.style.display = this.dom.firstChild ? "" : "none";
      }
    }
    destroy() {
      if (this.layer.destroy)
        this.layer.destroy(this.dom, this.view);
      this.dom.remove();
    }
  };
  var layerOrder = /* @__PURE__ */ Facet.define();
  function layer(config) {
    return [
      ViewPlugin.define((v) => new LayerView(v, config)),
      layerOrder.of(config)
    ];
  }
  var selectionConfig = /* @__PURE__ */ Facet.define({
    combine(configs) {
      return combineConfig(configs, {
        cursorBlinkRate: 1200,
        drawRangeCursor: true
      }, {
        cursorBlinkRate: (a, b) => Math.min(a, b),
        drawRangeCursor: (a, b) => a || b
      });
    }
  });
  function drawSelection(config = {}) {
    return [
      selectionConfig.of(config),
      cursorLayer,
      selectionLayer,
      hideNativeSelection,
      nativeSelectionHidden.of(true)
    ];
  }
  function configChanged(update) {
    return update.startState.facet(selectionConfig) != update.state.facet(selectionConfig);
  }
  var cursorLayer = /* @__PURE__ */ layer({
    above: true,
    markers(view) {
      let { state } = view, conf = state.facet(selectionConfig);
      let cursors = [];
      for (let r of state.selection.ranges) {
        let prim = r == state.selection.main;
        if (r.empty || conf.drawRangeCursor) {
          let className = prim ? "cm-cursor cm-cursor-primary" : "cm-cursor cm-cursor-secondary";
          let cursor = r.empty ? r : EditorSelection.cursor(r.head, r.head > r.anchor ? -1 : 1);
          for (let piece of RectangleMarker.forRange(view, className, cursor))
            cursors.push(piece);
        }
      }
      return cursors;
    },
    update(update, dom) {
      if (update.transactions.some((tr) => tr.selection))
        dom.style.animationName = dom.style.animationName == "cm-blink" ? "cm-blink2" : "cm-blink";
      let confChange = configChanged(update);
      if (confChange)
        setBlinkRate(update.state, dom);
      return update.docChanged || update.selectionSet || confChange;
    },
    mount(dom, view) {
      setBlinkRate(view.state, dom);
    },
    class: "cm-cursorLayer"
  });
  function setBlinkRate(state, dom) {
    dom.style.animationDuration = state.facet(selectionConfig).cursorBlinkRate + "ms";
  }
  var selectionLayer = /* @__PURE__ */ layer({
    above: false,
    markers(view) {
      return view.state.selection.ranges.map((r) => r.empty ? [] : RectangleMarker.forRange(view, "cm-selectionBackground", r)).reduce((a, b) => a.concat(b));
    },
    update(update, dom) {
      return update.docChanged || update.selectionSet || update.viewportChanged || configChanged(update);
    },
    class: "cm-selectionLayer"
  });
  var hideNativeSelection = /* @__PURE__ */ Prec.highest(/* @__PURE__ */ EditorView.theme({
    ".cm-line": {
      "& ::selection, &::selection": { backgroundColor: "transparent !important" },
      caretColor: "transparent !important"
    },
    ".cm-content": {
      caretColor: "transparent !important",
      "& :focus": {
        caretColor: "initial !important",
        "&::selection, & ::selection": {
          backgroundColor: "Highlight !important"
        }
      }
    }
  }));
  var UnicodeRegexpSupport = /x/.unicode != null ? "gu" : "g";
  var Outside = "-10000px";
  var TooltipViewManager = class {
    constructor(view, facet, createTooltipView, removeTooltipView) {
      this.facet = facet;
      this.createTooltipView = createTooltipView;
      this.removeTooltipView = removeTooltipView;
      this.input = view.state.facet(facet);
      this.tooltips = this.input.filter((t2) => t2);
      let prev = null;
      this.tooltipViews = this.tooltips.map((t2) => prev = createTooltipView(t2, prev));
    }
    update(update, above) {
      var _a2;
      let input = update.state.facet(this.facet);
      let tooltips = input.filter((x) => x);
      if (input === this.input) {
        for (let t2 of this.tooltipViews)
          if (t2.update)
            t2.update(update);
        return false;
      }
      let tooltipViews = [], newAbove = above ? [] : null;
      for (let i = 0; i < tooltips.length; i++) {
        let tip = tooltips[i], known = -1;
        if (!tip)
          continue;
        for (let i2 = 0; i2 < this.tooltips.length; i2++) {
          let other = this.tooltips[i2];
          if (other && other.create == tip.create)
            known = i2;
        }
        if (known < 0) {
          tooltipViews[i] = this.createTooltipView(tip, i ? tooltipViews[i - 1] : null);
          if (newAbove)
            newAbove[i] = !!tip.above;
        } else {
          let tooltipView = tooltipViews[i] = this.tooltipViews[known];
          if (newAbove)
            newAbove[i] = above[known];
          if (tooltipView.update)
            tooltipView.update(update);
        }
      }
      for (let t2 of this.tooltipViews)
        if (tooltipViews.indexOf(t2) < 0) {
          this.removeTooltipView(t2);
          (_a2 = t2.destroy) === null || _a2 === void 0 ? void 0 : _a2.call(t2);
        }
      if (above) {
        newAbove.forEach((val, i) => above[i] = val);
        above.length = newAbove.length;
      }
      this.input = input;
      this.tooltips = tooltips;
      this.tooltipViews = tooltipViews;
      return true;
    }
  };
  function windowSpace(view) {
    let docElt = view.dom.ownerDocument.documentElement;
    return { top: 0, left: 0, bottom: docElt.clientHeight, right: docElt.clientWidth };
  }
  var tooltipConfig = /* @__PURE__ */ Facet.define({
    combine: (values) => {
      var _a2, _b, _c;
      return {
        position: browser.ios ? "absolute" : ((_a2 = values.find((conf) => conf.position)) === null || _a2 === void 0 ? void 0 : _a2.position) || "fixed",
        parent: ((_b = values.find((conf) => conf.parent)) === null || _b === void 0 ? void 0 : _b.parent) || null,
        tooltipSpace: ((_c = values.find((conf) => conf.tooltipSpace)) === null || _c === void 0 ? void 0 : _c.tooltipSpace) || windowSpace
      };
    }
  });
  var knownHeight = /* @__PURE__ */ new WeakMap();
  var tooltipPlugin = /* @__PURE__ */ ViewPlugin.fromClass(class {
    constructor(view) {
      this.view = view;
      this.above = [];
      this.inView = true;
      this.madeAbsolute = false;
      this.lastTransaction = 0;
      this.measureTimeout = -1;
      let config = view.state.facet(tooltipConfig);
      this.position = config.position;
      this.parent = config.parent;
      this.classes = view.themeClasses;
      this.createContainer();
      this.measureReq = { read: this.readMeasure.bind(this), write: this.writeMeasure.bind(this), key: this };
      this.resizeObserver = typeof ResizeObserver == "function" ? new ResizeObserver(() => this.measureSoon()) : null;
      this.manager = new TooltipViewManager(view, showTooltip, (t2, p) => this.createTooltip(t2, p), (t2) => {
        if (this.resizeObserver)
          this.resizeObserver.unobserve(t2.dom);
        t2.dom.remove();
      });
      this.above = this.manager.tooltips.map((t2) => !!t2.above);
      this.intersectionObserver = typeof IntersectionObserver == "function" ? new IntersectionObserver((entries) => {
        if (Date.now() > this.lastTransaction - 50 && entries.length > 0 && entries[entries.length - 1].intersectionRatio < 1)
          this.measureSoon();
      }, { threshold: [1] }) : null;
      this.observeIntersection();
      view.win.addEventListener("resize", this.measureSoon = this.measureSoon.bind(this));
      this.maybeMeasure();
    }
    createContainer() {
      if (this.parent) {
        this.container = document.createElement("div");
        this.container.style.position = "relative";
        this.container.className = this.view.themeClasses;
        this.parent.appendChild(this.container);
      } else {
        this.container = this.view.dom;
      }
    }
    observeIntersection() {
      if (this.intersectionObserver) {
        this.intersectionObserver.disconnect();
        for (let tooltip of this.manager.tooltipViews)
          this.intersectionObserver.observe(tooltip.dom);
      }
    }
    measureSoon() {
      if (this.measureTimeout < 0)
        this.measureTimeout = setTimeout(() => {
          this.measureTimeout = -1;
          this.maybeMeasure();
        }, 50);
    }
    update(update) {
      if (update.transactions.length)
        this.lastTransaction = Date.now();
      let updated = this.manager.update(update, this.above);
      if (updated)
        this.observeIntersection();
      let shouldMeasure = updated || update.geometryChanged;
      let newConfig = update.state.facet(tooltipConfig);
      if (newConfig.position != this.position && !this.madeAbsolute) {
        this.position = newConfig.position;
        for (let t2 of this.manager.tooltipViews)
          t2.dom.style.position = this.position;
        shouldMeasure = true;
      }
      if (newConfig.parent != this.parent) {
        if (this.parent)
          this.container.remove();
        this.parent = newConfig.parent;
        this.createContainer();
        for (let t2 of this.manager.tooltipViews)
          this.container.appendChild(t2.dom);
        shouldMeasure = true;
      } else if (this.parent && this.view.themeClasses != this.classes) {
        this.classes = this.container.className = this.view.themeClasses;
      }
      if (shouldMeasure)
        this.maybeMeasure();
    }
    createTooltip(tooltip, prev) {
      let tooltipView = tooltip.create(this.view);
      let before = prev ? prev.dom : null;
      tooltipView.dom.classList.add("cm-tooltip");
      if (tooltip.arrow && !tooltipView.dom.querySelector(".cm-tooltip > .cm-tooltip-arrow")) {
        let arrow = document.createElement("div");
        arrow.className = "cm-tooltip-arrow";
        tooltipView.dom.appendChild(arrow);
      }
      tooltipView.dom.style.position = this.position;
      tooltipView.dom.style.top = Outside;
      tooltipView.dom.style.left = "0px";
      this.container.insertBefore(tooltipView.dom, before);
      if (tooltipView.mount)
        tooltipView.mount(this.view);
      if (this.resizeObserver)
        this.resizeObserver.observe(tooltipView.dom);
      return tooltipView;
    }
    destroy() {
      var _a2, _b, _c;
      this.view.win.removeEventListener("resize", this.measureSoon);
      for (let tooltipView of this.manager.tooltipViews) {
        tooltipView.dom.remove();
        (_a2 = tooltipView.destroy) === null || _a2 === void 0 ? void 0 : _a2.call(tooltipView);
      }
      if (this.parent)
        this.container.remove();
      (_b = this.resizeObserver) === null || _b === void 0 ? void 0 : _b.disconnect();
      (_c = this.intersectionObserver) === null || _c === void 0 ? void 0 : _c.disconnect();
      clearTimeout(this.measureTimeout);
    }
    readMeasure() {
      let scaleX = 1, scaleY = 1, makeAbsolute = false;
      if (this.position == "fixed" && this.manager.tooltipViews.length) {
        let { dom } = this.manager.tooltipViews[0];
        if (browser.safari) {
          let rect = dom.getBoundingClientRect();
          makeAbsolute = Math.abs(rect.top + 1e4) > 1 || Math.abs(rect.left) > 1;
        } else {
          makeAbsolute = !!dom.offsetParent && dom.offsetParent != this.container.ownerDocument.body;
        }
      }
      if (makeAbsolute || this.position == "absolute") {
        if (this.parent) {
          let rect = this.parent.getBoundingClientRect();
          if (rect.width && rect.height) {
            scaleX = rect.width / this.parent.offsetWidth;
            scaleY = rect.height / this.parent.offsetHeight;
          }
        } else {
          ({ scaleX, scaleY } = this.view.viewState);
        }
      }
      let visible = this.view.scrollDOM.getBoundingClientRect(), margins = getScrollMargins(this.view);
      return {
        visible: {
          left: visible.left + margins.left,
          top: visible.top + margins.top,
          right: visible.right - margins.right,
          bottom: visible.bottom - margins.bottom
        },
        parent: this.parent ? this.container.getBoundingClientRect() : this.view.dom.getBoundingClientRect(),
        pos: this.manager.tooltips.map((t2, i) => {
          let tv = this.manager.tooltipViews[i];
          return tv.getCoords ? tv.getCoords(t2.pos) : this.view.coordsAtPos(t2.pos);
        }),
        size: this.manager.tooltipViews.map(({ dom }) => dom.getBoundingClientRect()),
        space: this.view.state.facet(tooltipConfig).tooltipSpace(this.view),
        scaleX,
        scaleY,
        makeAbsolute
      };
    }
    writeMeasure(measured) {
      var _a2;
      if (measured.makeAbsolute) {
        this.madeAbsolute = true;
        this.position = "absolute";
        for (let t2 of this.manager.tooltipViews)
          t2.dom.style.position = "absolute";
      }
      let { visible, space, scaleX, scaleY } = measured;
      let others = [];
      for (let i = 0; i < this.manager.tooltips.length; i++) {
        let tooltip = this.manager.tooltips[i], tView = this.manager.tooltipViews[i], { dom } = tView;
        let pos = measured.pos[i], size = measured.size[i];
        if (!pos || tooltip.clip !== false && (pos.bottom <= Math.max(visible.top, space.top) || pos.top >= Math.min(visible.bottom, space.bottom) || pos.right < Math.max(visible.left, space.left) - 0.1 || pos.left > Math.min(visible.right, space.right) + 0.1)) {
          dom.style.top = Outside;
          continue;
        }
        let arrow = tooltip.arrow ? tView.dom.querySelector(".cm-tooltip-arrow") : null;
        let arrowHeight = arrow ? 7 : 0;
        let width = size.right - size.left, height = (_a2 = knownHeight.get(tView)) !== null && _a2 !== void 0 ? _a2 : size.bottom - size.top;
        let offset = tView.offset || noOffset, ltr = this.view.textDirection == Direction.LTR;
        let left = size.width > space.right - space.left ? ltr ? space.left : space.right - size.width : ltr ? Math.max(space.left, Math.min(pos.left - (arrow ? 14 : 0) + offset.x, space.right - width)) : Math.min(Math.max(space.left, pos.left - width + (arrow ? 14 : 0) - offset.x), space.right - width);
        let above = this.above[i];
        if (!tooltip.strictSide && (above ? pos.top - height - arrowHeight - offset.y < space.top : pos.bottom + height + arrowHeight + offset.y > space.bottom) && above == space.bottom - pos.bottom > pos.top - space.top)
          above = this.above[i] = !above;
        let spaceVert = (above ? pos.top - space.top : space.bottom - pos.bottom) - arrowHeight;
        if (spaceVert < height && tView.resize !== false) {
          if (spaceVert < this.view.defaultLineHeight) {
            dom.style.top = Outside;
            continue;
          }
          knownHeight.set(tView, height);
          dom.style.height = (height = spaceVert) / scaleY + "px";
        } else if (dom.style.height) {
          dom.style.height = "";
        }
        let top2 = above ? pos.top - height - arrowHeight - offset.y : pos.bottom + arrowHeight + offset.y;
        let right = left + width;
        if (tView.overlap !== true) {
          for (let r of others)
            if (r.left < right && r.right > left && r.top < top2 + height && r.bottom > top2)
              top2 = above ? r.top - height - 2 - arrowHeight : r.bottom + arrowHeight + 2;
        }
        if (this.position == "absolute") {
          dom.style.top = (top2 - measured.parent.top) / scaleY + "px";
          setLeftStyle(dom, (left - measured.parent.left) / scaleX);
        } else {
          dom.style.top = top2 / scaleY + "px";
          setLeftStyle(dom, left / scaleX);
        }
        if (arrow) {
          let arrowLeft = pos.left + (ltr ? offset.x : -offset.x) - (left + 14 - 7);
          arrow.style.left = arrowLeft / scaleX + "px";
        }
        if (tView.overlap !== true)
          others.push({ left, top: top2, right, bottom: top2 + height });
        dom.classList.toggle("cm-tooltip-above", above);
        dom.classList.toggle("cm-tooltip-below", !above);
        if (tView.positioned)
          tView.positioned(measured.space);
      }
    }
    maybeMeasure() {
      if (this.manager.tooltips.length) {
        if (this.view.inView)
          this.view.requestMeasure(this.measureReq);
        if (this.inView != this.view.inView) {
          this.inView = this.view.inView;
          if (!this.inView)
            for (let tv of this.manager.tooltipViews)
              tv.dom.style.top = Outside;
        }
      }
    }
  }, {
    eventObservers: {
      scroll() {
        this.maybeMeasure();
      }
    }
  });
  function setLeftStyle(elt, value) {
    let current = parseInt(elt.style.left, 10);
    if (isNaN(current) || Math.abs(value - current) > 1)
      elt.style.left = value + "px";
  }
  var baseTheme = /* @__PURE__ */ EditorView.baseTheme({
    ".cm-tooltip": {
      zIndex: 500,
      boxSizing: "border-box"
    },
    "&light .cm-tooltip": {
      border: "1px solid #bbb",
      backgroundColor: "#f5f5f5"
    },
    "&light .cm-tooltip-section:not(:first-child)": {
      borderTop: "1px solid #bbb"
    },
    "&dark .cm-tooltip": {
      backgroundColor: "#333338",
      color: "white"
    },
    ".cm-tooltip-arrow": {
      height: `${7}px`,
      width: `${7 * 2}px`,
      position: "absolute",
      zIndex: -1,
      overflow: "hidden",
      "&:before, &:after": {
        content: "''",
        position: "absolute",
        width: 0,
        height: 0,
        borderLeft: `${7}px solid transparent`,
        borderRight: `${7}px solid transparent`
      },
      ".cm-tooltip-above &": {
        bottom: `-${7}px`,
        "&:before": {
          borderTop: `${7}px solid #bbb`
        },
        "&:after": {
          borderTop: `${7}px solid #f5f5f5`,
          bottom: "1px"
        }
      },
      ".cm-tooltip-below &": {
        top: `-${7}px`,
        "&:before": {
          borderBottom: `${7}px solid #bbb`
        },
        "&:after": {
          borderBottom: `${7}px solid #f5f5f5`,
          top: "1px"
        }
      }
    },
    "&dark .cm-tooltip .cm-tooltip-arrow": {
      "&:before": {
        borderTopColor: "#333338",
        borderBottomColor: "#333338"
      },
      "&:after": {
        borderTopColor: "transparent",
        borderBottomColor: "transparent"
      }
    }
  });
  var noOffset = { x: 0, y: 0 };
  var showTooltip = /* @__PURE__ */ Facet.define({
    enables: [tooltipPlugin, baseTheme]
  });
  function getTooltip(view, tooltip) {
    let plugin = view.plugin(tooltipPlugin);
    if (!plugin)
      return null;
    let found = plugin.manager.tooltips.indexOf(tooltip);
    return found < 0 ? null : plugin.manager.tooltipViews[found];
  }
  var panelConfig = /* @__PURE__ */ Facet.define({
    combine(configs) {
      let topContainer, bottomContainer;
      for (let c of configs) {
        topContainer = topContainer || c.topContainer;
        bottomContainer = bottomContainer || c.bottomContainer;
      }
      return { topContainer, bottomContainer };
    }
  });
  function getPanel(view, panel) {
    let plugin = view.plugin(panelPlugin);
    let index = plugin ? plugin.specs.indexOf(panel) : -1;
    return index > -1 ? plugin.panels[index] : null;
  }
  var panelPlugin = /* @__PURE__ */ ViewPlugin.fromClass(class {
    constructor(view) {
      this.input = view.state.facet(showPanel);
      this.specs = this.input.filter((s) => s);
      this.panels = this.specs.map((spec) => spec(view));
      let conf = view.state.facet(panelConfig);
      this.top = new PanelGroup(view, true, conf.topContainer);
      this.bottom = new PanelGroup(view, false, conf.bottomContainer);
      this.top.sync(this.panels.filter((p) => p.top));
      this.bottom.sync(this.panels.filter((p) => !p.top));
      for (let p of this.panels) {
        p.dom.classList.add("cm-panel");
        if (p.mount)
          p.mount();
      }
    }
    update(update) {
      let conf = update.state.facet(panelConfig);
      if (this.top.container != conf.topContainer) {
        this.top.sync([]);
        this.top = new PanelGroup(update.view, true, conf.topContainer);
      }
      if (this.bottom.container != conf.bottomContainer) {
        this.bottom.sync([]);
        this.bottom = new PanelGroup(update.view, false, conf.bottomContainer);
      }
      this.top.syncClasses();
      this.bottom.syncClasses();
      let input = update.state.facet(showPanel);
      if (input != this.input) {
        let specs = input.filter((x) => x);
        let panels = [], top2 = [], bottom = [], mount = [];
        for (let spec of specs) {
          let known = this.specs.indexOf(spec), panel;
          if (known < 0) {
            panel = spec(update.view);
            mount.push(panel);
          } else {
            panel = this.panels[known];
            if (panel.update)
              panel.update(update);
          }
          panels.push(panel);
          (panel.top ? top2 : bottom).push(panel);
        }
        this.specs = specs;
        this.panels = panels;
        this.top.sync(top2);
        this.bottom.sync(bottom);
        for (let p of mount) {
          p.dom.classList.add("cm-panel");
          if (p.mount)
            p.mount();
        }
      } else {
        for (let p of this.panels)
          if (p.update)
            p.update(update);
      }
    }
    destroy() {
      this.top.sync([]);
      this.bottom.sync([]);
    }
  }, {
    provide: (plugin) => EditorView.scrollMargins.of((view) => {
      let value = view.plugin(plugin);
      return value && { top: value.top.scrollMargin(), bottom: value.bottom.scrollMargin() };
    })
  });
  var PanelGroup = class {
    constructor(view, top2, container) {
      this.view = view;
      this.top = top2;
      this.container = container;
      this.dom = void 0;
      this.classes = "";
      this.panels = [];
      this.syncClasses();
    }
    sync(panels) {
      for (let p of this.panels)
        if (p.destroy && panels.indexOf(p) < 0)
          p.destroy();
      this.panels = panels;
      this.syncDOM();
    }
    syncDOM() {
      if (this.panels.length == 0) {
        if (this.dom) {
          this.dom.remove();
          this.dom = void 0;
        }
        return;
      }
      if (!this.dom) {
        this.dom = document.createElement("div");
        this.dom.className = this.top ? "cm-panels cm-panels-top" : "cm-panels cm-panels-bottom";
        this.dom.style[this.top ? "top" : "bottom"] = "0";
        let parent = this.container || this.view.dom;
        parent.insertBefore(this.dom, this.top ? parent.firstChild : null);
      }
      let curDOM = this.dom.firstChild;
      for (let panel of this.panels) {
        if (panel.dom.parentNode == this.dom) {
          while (curDOM != panel.dom)
            curDOM = rm(curDOM);
          curDOM = curDOM.nextSibling;
        } else {
          this.dom.insertBefore(panel.dom, curDOM);
        }
      }
      while (curDOM)
        curDOM = rm(curDOM);
    }
    scrollMargin() {
      return !this.dom || this.container ? 0 : Math.max(0, this.top ? this.dom.getBoundingClientRect().bottom - Math.max(0, this.view.scrollDOM.getBoundingClientRect().top) : Math.min(innerHeight, this.view.scrollDOM.getBoundingClientRect().bottom) - this.dom.getBoundingClientRect().top);
    }
    syncClasses() {
      if (!this.container || this.classes == this.view.themeClasses)
        return;
      for (let cls of this.classes.split(" "))
        if (cls)
          this.container.classList.remove(cls);
      for (let cls of (this.classes = this.view.themeClasses).split(" "))
        if (cls)
          this.container.classList.add(cls);
    }
  };
  function rm(node) {
    let next = node.nextSibling;
    node.remove();
    return next;
  }
  var showPanel = /* @__PURE__ */ Facet.define({
    enables: panelPlugin
  });
  function showDialog(view, config) {
    let resolve;
    let promise = new Promise((r) => resolve = r);
    let panelCtor = (view2) => createDialog(view2, config, resolve);
    if (view.state.field(dialogField, false)) {
      view.dispatch({ effects: openDialogEffect.of(panelCtor) });
    } else {
      view.dispatch({ effects: StateEffect.appendConfig.of(dialogField.init(() => [panelCtor])) });
    }
    let close = closeDialogEffect.of(panelCtor);
    return { close, result: promise.then((form) => {
      let queue = view.win.queueMicrotask || ((f) => view.win.setTimeout(f, 10));
      queue(() => {
        if (view.state.field(dialogField).indexOf(panelCtor) > -1)
          view.dispatch({ effects: close });
      });
      return form;
    }) };
  }
  var dialogField = /* @__PURE__ */ StateField.define({
    create() {
      return [];
    },
    update(dialogs, tr) {
      for (let e of tr.effects) {
        if (e.is(openDialogEffect))
          dialogs = [e.value].concat(dialogs);
        else if (e.is(closeDialogEffect))
          dialogs = dialogs.filter((d) => d != e.value);
      }
      return dialogs;
    },
    provide: (f) => showPanel.computeN([f], (state) => state.field(f))
  });
  var openDialogEffect = /* @__PURE__ */ StateEffect.define();
  var closeDialogEffect = /* @__PURE__ */ StateEffect.define();
  function createDialog(view, config, result) {
    let content2 = config.content ? config.content(view, () => done(null)) : null;
    if (!content2) {
      content2 = crelt("form");
      if (config.input) {
        let input = crelt("input", config.input);
        if (/^(text|password|number|email|tel|url)$/.test(input.type))
          input.classList.add("cm-textfield");
        if (!input.name)
          input.name = "input";
        content2.appendChild(crelt("label", (config.label || "") + ": ", input));
      } else {
        content2.appendChild(document.createTextNode(config.label || ""));
      }
      content2.appendChild(document.createTextNode(" "));
      content2.appendChild(crelt("button", { class: "cm-button", type: "submit" }, config.submitLabel || "OK"));
    }
    let forms = content2.nodeName == "FORM" ? [content2] : content2.querySelectorAll("form");
    for (let i = 0; i < forms.length; i++) {
      let form = forms[i];
      form.addEventListener("keydown", (event) => {
        if (event.keyCode == 27) {
          event.preventDefault();
          done(null);
        } else if (event.keyCode == 13) {
          event.preventDefault();
          done(form);
        }
      });
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        done(form);
      });
    }
    let panel = crelt("div", content2, crelt("button", {
      onclick: () => done(null),
      "aria-label": view.state.phrase("close"),
      class: "cm-dialog-close",
      type: "button"
    }, ["\xD7"]));
    if (config.class)
      panel.className = config.class;
    panel.classList.add("cm-dialog");
    function done(form) {
      if (panel.contains(panel.ownerDocument.activeElement))
        view.focus();
      result(form);
    }
    return {
      dom: panel,
      top: config.top,
      mount: () => {
        if (config.focus) {
          let focus;
          if (typeof config.focus == "string")
            focus = content2.querySelector(config.focus);
          else
            focus = content2.querySelector("input") || content2.querySelector("button");
          if (focus && "select" in focus)
            focus.select();
          else if (focus && "focus" in focus)
            focus.focus();
        }
      }
    };
  }
  var GutterMarker = class extends RangeValue {
    /**
    @internal
    */
    compare(other) {
      return this == other || this.constructor == other.constructor && this.eq(other);
    }
    /**
    Compare this marker to another marker of the same type.
    */
    eq(other) {
      return false;
    }
    /**
    Called if the marker has a `toDOM` method and its representation
    was removed from a gutter.
    */
    destroy(dom) {
    }
  };
  GutterMarker.prototype.elementClass = "";
  GutterMarker.prototype.toDOM = void 0;
  GutterMarker.prototype.mapMode = MapMode.TrackBefore;
  GutterMarker.prototype.startSide = GutterMarker.prototype.endSide = -1;
  GutterMarker.prototype.point = true;
  var gutterLineClass = /* @__PURE__ */ Facet.define();
  var gutterWidgetClass = /* @__PURE__ */ Facet.define();
  var activeGutters = /* @__PURE__ */ Facet.define();
  var unfixGutters = /* @__PURE__ */ Facet.define({
    combine: (values) => values.some((x) => x)
  });
  function gutters(config) {
    let result = [
      gutterView
    ];
    if (config && config.fixed === false)
      result.push(unfixGutters.of(true));
    return result;
  }
  var gutterView = /* @__PURE__ */ ViewPlugin.fromClass(class {
    constructor(view) {
      this.view = view;
      this.domAfter = null;
      this.prevViewport = view.viewport;
      this.dom = document.createElement("div");
      this.dom.className = "cm-gutters cm-gutters-before";
      this.dom.setAttribute("aria-hidden", "true");
      this.dom.style.minHeight = this.view.contentHeight / this.view.scaleY + "px";
      this.gutters = view.state.facet(activeGutters).map((conf) => new SingleGutterView(view, conf));
      this.fixed = !view.state.facet(unfixGutters);
      for (let gutter2 of this.gutters) {
        if (gutter2.config.side == "after")
          this.getDOMAfter().appendChild(gutter2.dom);
        else
          this.dom.appendChild(gutter2.dom);
      }
      if (this.fixed) {
        this.dom.style.position = "sticky";
      }
      this.syncGutters(false);
      view.scrollDOM.insertBefore(this.dom, view.contentDOM);
    }
    getDOMAfter() {
      if (!this.domAfter) {
        this.domAfter = document.createElement("div");
        this.domAfter.className = "cm-gutters cm-gutters-after";
        this.domAfter.setAttribute("aria-hidden", "true");
        this.domAfter.style.minHeight = this.view.contentHeight / this.view.scaleY + "px";
        this.domAfter.style.position = this.fixed ? "sticky" : "";
        this.view.scrollDOM.appendChild(this.domAfter);
      }
      return this.domAfter;
    }
    update(update) {
      if (this.updateGutters(update)) {
        let vpA = this.prevViewport, vpB = update.view.viewport;
        let vpOverlap = Math.min(vpA.to, vpB.to) - Math.max(vpA.from, vpB.from);
        this.syncGutters(vpOverlap < (vpB.to - vpB.from) * 0.8);
      }
      if (update.geometryChanged) {
        let min = this.view.contentHeight / this.view.scaleY + "px";
        this.dom.style.minHeight = min;
        if (this.domAfter)
          this.domAfter.style.minHeight = min;
      }
      if (this.view.state.facet(unfixGutters) != !this.fixed) {
        this.fixed = !this.fixed;
        this.dom.style.position = this.fixed ? "sticky" : "";
        if (this.domAfter)
          this.domAfter.style.position = this.fixed ? "sticky" : "";
      }
      this.prevViewport = update.view.viewport;
    }
    syncGutters(detach) {
      let after = this.dom.nextSibling;
      if (detach) {
        this.dom.remove();
        if (this.domAfter)
          this.domAfter.remove();
      }
      let lineClasses = RangeSet.iter(this.view.state.facet(gutterLineClass), this.view.viewport.from);
      let classSet = [];
      let contexts = this.gutters.map((gutter2) => new UpdateContext(gutter2, this.view.viewport, -this.view.documentPadding.top));
      for (let line of this.view.viewportLineBlocks) {
        if (classSet.length)
          classSet = [];
        if (Array.isArray(line.type)) {
          let first = true;
          for (let b of line.type) {
            if (b.type == BlockType.Text && first) {
              advanceCursor(lineClasses, classSet, b.from);
              for (let cx of contexts)
                cx.line(this.view, b, classSet);
              first = false;
            } else if (b.widget) {
              for (let cx of contexts)
                cx.widget(this.view, b);
            }
          }
        } else if (line.type == BlockType.Text) {
          advanceCursor(lineClasses, classSet, line.from);
          for (let cx of contexts)
            cx.line(this.view, line, classSet);
        } else if (line.widget) {
          for (let cx of contexts)
            cx.widget(this.view, line);
        }
      }
      for (let cx of contexts)
        cx.finish();
      if (detach) {
        this.view.scrollDOM.insertBefore(this.dom, after);
        if (this.domAfter)
          this.view.scrollDOM.appendChild(this.domAfter);
      }
    }
    updateGutters(update) {
      let prev = update.startState.facet(activeGutters), cur2 = update.state.facet(activeGutters);
      let change = update.docChanged || update.heightChanged || update.viewportChanged || !RangeSet.eq(update.startState.facet(gutterLineClass), update.state.facet(gutterLineClass), update.view.viewport.from, update.view.viewport.to);
      if (prev == cur2) {
        for (let gutter2 of this.gutters)
          if (gutter2.update(update))
            change = true;
      } else {
        change = true;
        let gutters2 = [];
        for (let conf of cur2) {
          let known = prev.indexOf(conf);
          if (known < 0) {
            gutters2.push(new SingleGutterView(this.view, conf));
          } else {
            this.gutters[known].update(update);
            gutters2.push(this.gutters[known]);
          }
        }
        for (let g of this.gutters) {
          g.dom.remove();
          if (gutters2.indexOf(g) < 0)
            g.destroy();
        }
        for (let g of gutters2) {
          if (g.config.side == "after")
            this.getDOMAfter().appendChild(g.dom);
          else
            this.dom.appendChild(g.dom);
        }
        this.gutters = gutters2;
      }
      return change;
    }
    destroy() {
      for (let view of this.gutters)
        view.destroy();
      this.dom.remove();
      if (this.domAfter)
        this.domAfter.remove();
    }
  }, {
    provide: (plugin) => EditorView.scrollMargins.of((view) => {
      let value = view.plugin(plugin);
      if (!value || value.gutters.length == 0 || !value.fixed)
        return null;
      let before = value.dom.offsetWidth * view.scaleX, after = value.domAfter ? value.domAfter.offsetWidth * view.scaleX : 0;
      return view.textDirection == Direction.LTR ? { left: before, right: after } : { right: before, left: after };
    })
  });
  function asArray2(val) {
    return Array.isArray(val) ? val : [val];
  }
  function advanceCursor(cursor, collect, pos) {
    while (cursor.value && cursor.from <= pos) {
      if (cursor.from == pos)
        collect.push(cursor.value);
      cursor.next();
    }
  }
  var UpdateContext = class {
    constructor(gutter2, viewport, height) {
      this.gutter = gutter2;
      this.height = height;
      this.i = 0;
      this.cursor = RangeSet.iter(gutter2.markers, viewport.from);
    }
    addElement(view, block, markers) {
      let { gutter: gutter2 } = this, above = (block.top - this.height) / view.scaleY, height = block.height / view.scaleY;
      if (this.i == gutter2.elements.length) {
        let newElt = new GutterElement(view, height, above, markers);
        gutter2.elements.push(newElt);
        gutter2.dom.appendChild(newElt.dom);
      } else {
        gutter2.elements[this.i].update(view, height, above, markers);
      }
      this.height = block.bottom;
      this.i++;
    }
    line(view, line, extraMarkers) {
      let localMarkers = [];
      advanceCursor(this.cursor, localMarkers, line.from);
      if (extraMarkers.length)
        localMarkers = localMarkers.concat(extraMarkers);
      let forLine = this.gutter.config.lineMarker(view, line, localMarkers);
      if (forLine)
        localMarkers.unshift(forLine);
      let gutter2 = this.gutter;
      if (localMarkers.length == 0 && !gutter2.config.renderEmptyElements)
        return;
      this.addElement(view, line, localMarkers);
    }
    widget(view, block) {
      let marker = this.gutter.config.widgetMarker(view, block.widget, block), markers = marker ? [marker] : null;
      for (let cls of view.state.facet(gutterWidgetClass)) {
        let marker2 = cls(view, block.widget, block);
        if (marker2)
          (markers || (markers = [])).push(marker2);
      }
      if (markers)
        this.addElement(view, block, markers);
    }
    finish() {
      let gutter2 = this.gutter;
      while (gutter2.elements.length > this.i) {
        let last = gutter2.elements.pop();
        gutter2.dom.removeChild(last.dom);
        last.destroy();
      }
    }
  };
  var SingleGutterView = class {
    constructor(view, config) {
      this.view = view;
      this.config = config;
      this.elements = [];
      this.spacer = null;
      this.dom = document.createElement("div");
      this.dom.className = "cm-gutter" + (this.config.class ? " " + this.config.class : "");
      for (let prop in config.domEventHandlers) {
        this.dom.addEventListener(prop, (event) => {
          let target = event.target, y;
          if (target != this.dom && this.dom.contains(target)) {
            while (target.parentNode != this.dom)
              target = target.parentNode;
            let rect = target.getBoundingClientRect();
            y = (rect.top + rect.bottom) / 2;
          } else {
            y = event.clientY;
          }
          let line = view.lineBlockAtHeight(y - view.documentTop);
          if (config.domEventHandlers[prop](view, line, event))
            event.preventDefault();
        });
      }
      this.markers = asArray2(config.markers(view));
      if (config.initialSpacer) {
        this.spacer = new GutterElement(view, 0, 0, [config.initialSpacer(view)]);
        this.dom.appendChild(this.spacer.dom);
        this.spacer.dom.style.cssText += "visibility: hidden; pointer-events: none";
      }
    }
    update(update) {
      let prevMarkers = this.markers;
      this.markers = asArray2(this.config.markers(update.view));
      if (this.spacer && this.config.updateSpacer) {
        let updated = this.config.updateSpacer(this.spacer.markers[0], update);
        if (updated != this.spacer.markers[0])
          this.spacer.update(update.view, 0, 0, [updated]);
      }
      let vp = update.view.viewport;
      return !RangeSet.eq(this.markers, prevMarkers, vp.from, vp.to) || (this.config.lineMarkerChange ? this.config.lineMarkerChange(update) : false);
    }
    destroy() {
      for (let elt of this.elements)
        elt.destroy();
    }
  };
  var GutterElement = class {
    constructor(view, height, above, markers) {
      this.height = -1;
      this.above = 0;
      this.markers = [];
      this.dom = document.createElement("div");
      this.dom.className = "cm-gutterElement";
      this.update(view, height, above, markers);
    }
    update(view, height, above, markers) {
      if (this.height != height) {
        this.height = height;
        this.dom.style.height = height + "px";
      }
      if (this.above != above)
        this.dom.style.marginTop = (this.above = above) ? above + "px" : "";
      if (!sameMarkers(this.markers, markers))
        this.setMarkers(view, markers);
    }
    setMarkers(view, markers) {
      let cls = "cm-gutterElement", domPos = this.dom.firstChild;
      for (let iNew = 0, iOld = 0; ; ) {
        let skipTo = iOld, marker = iNew < markers.length ? markers[iNew++] : null, matched = false;
        if (marker) {
          let c = marker.elementClass;
          if (c)
            cls += " " + c;
          for (let i = iOld; i < this.markers.length; i++)
            if (this.markers[i].compare(marker)) {
              skipTo = i;
              matched = true;
              break;
            }
        } else {
          skipTo = this.markers.length;
        }
        while (iOld < skipTo) {
          let next = this.markers[iOld++];
          if (next.toDOM) {
            next.destroy(domPos);
            let after = domPos.nextSibling;
            domPos.remove();
            domPos = after;
          }
        }
        if (!marker)
          break;
        if (marker.toDOM) {
          if (matched)
            domPos = domPos.nextSibling;
          else
            this.dom.insertBefore(marker.toDOM(view), domPos);
        }
        if (matched)
          iOld++;
      }
      this.dom.className = cls;
      this.markers = markers;
    }
    destroy() {
      this.setMarkers(null, []);
    }
  };
  function sameMarkers(a, b) {
    if (a.length != b.length)
      return false;
    for (let i = 0; i < a.length; i++)
      if (!a[i].compare(b[i]))
        return false;
    return true;
  }
  var lineNumberMarkers = /* @__PURE__ */ Facet.define();
  var lineNumberWidgetMarker = /* @__PURE__ */ Facet.define();
  var lineNumberConfig = /* @__PURE__ */ Facet.define({
    combine(values) {
      return combineConfig(values, { formatNumber: String, domEventHandlers: {} }, {
        domEventHandlers(a, b) {
          let result = Object.assign({}, a);
          for (let event in b) {
            let exists = result[event], add2 = b[event];
            result[event] = exists ? (view, line, event2) => exists(view, line, event2) || add2(view, line, event2) : add2;
          }
          return result;
        }
      });
    }
  });
  var NumberMarker = class extends GutterMarker {
    constructor(number2) {
      super();
      this.number = number2;
    }
    eq(other) {
      return this.number == other.number;
    }
    toDOM() {
      return document.createTextNode(this.number);
    }
  };
  function formatNumber(view, number2) {
    return view.state.facet(lineNumberConfig).formatNumber(number2, view.state);
  }
  var lineNumberGutter = /* @__PURE__ */ activeGutters.compute([lineNumberConfig], (state) => ({
    class: "cm-lineNumbers",
    renderEmptyElements: false,
    markers(view) {
      return view.state.facet(lineNumberMarkers);
    },
    lineMarker(view, line, others) {
      if (others.some((m) => m.toDOM))
        return null;
      return new NumberMarker(formatNumber(view, view.state.doc.lineAt(line.from).number));
    },
    widgetMarker: (view, widget, block) => {
      for (let m of view.state.facet(lineNumberWidgetMarker)) {
        let result = m(view, widget, block);
        if (result)
          return result;
      }
      return null;
    },
    lineMarkerChange: (update) => update.startState.facet(lineNumberConfig) != update.state.facet(lineNumberConfig),
    initialSpacer(view) {
      return new NumberMarker(formatNumber(view, maxLineNumber(view.state.doc.lines)));
    },
    updateSpacer(spacer, update) {
      let max = formatNumber(update.view, maxLineNumber(update.view.state.doc.lines));
      return max == spacer.number ? spacer : new NumberMarker(max);
    },
    domEventHandlers: state.facet(lineNumberConfig).domEventHandlers,
    side: "before"
  }));
  function lineNumbers(config = {}) {
    return [
      lineNumberConfig.of(config),
      gutters(),
      lineNumberGutter
    ];
  }
  function maxLineNumber(lines) {
    let last = 9;
    while (last < lines)
      last = last * 10 + 9;
    return last;
  }
  var activeLineGutterMarker = /* @__PURE__ */ new class extends GutterMarker {
    constructor() {
      super(...arguments);
      this.elementClass = "cm-activeLineGutter";
    }
  }();
  var activeLineGutterHighlighter = /* @__PURE__ */ gutterLineClass.compute(["selection"], (state) => {
    let marks2 = [], last = -1;
    for (let range of state.selection.ranges) {
      let linePos = state.doc.lineAt(range.head).from;
      if (linePos > last) {
        last = linePos;
        marks2.push(activeLineGutterMarker.range(linePos));
      }
    }
    return RangeSet.of(marks2);
  });
  function highlightActiveLineGutter() {
    return activeLineGutterHighlighter;
  }

  // node_modules/@lezer/common/dist/index.js
  var DefaultBufferLength = 1024;
  var nextPropID = 0;
  var Range2 = class {
    constructor(from, to) {
      this.from = from;
      this.to = to;
    }
  };
  var NodeProp = class {
    /**
    Create a new node prop type.
    */
    constructor(config = {}) {
      this.id = nextPropID++;
      this.perNode = !!config.perNode;
      this.deserialize = config.deserialize || (() => {
        throw new Error("This node type doesn't define a deserialize function");
      });
      this.combine = config.combine || null;
    }
    /**
    This is meant to be used with
    [`NodeSet.extend`](#common.NodeSet.extend) or
    [`LRParser.configure`](#lr.ParserConfig.props) to compute
    prop values for each node type in the set. Takes a [match
    object](#common.NodeType^match) or function that returns undefined
    if the node type doesn't get this prop, and the prop's value if
    it does.
    */
    add(match) {
      if (this.perNode)
        throw new RangeError("Can't add per-node props to node types");
      if (typeof match != "function")
        match = NodeType.match(match);
      return (type) => {
        let result = match(type);
        return result === void 0 ? null : [this, result];
      };
    }
  };
  NodeProp.closedBy = new NodeProp({ deserialize: (str) => str.split(" ") });
  NodeProp.openedBy = new NodeProp({ deserialize: (str) => str.split(" ") });
  NodeProp.group = new NodeProp({ deserialize: (str) => str.split(" ") });
  NodeProp.isolate = new NodeProp({ deserialize: (value) => {
    if (value && value != "rtl" && value != "ltr" && value != "auto")
      throw new RangeError("Invalid value for isolate: " + value);
    return value || "auto";
  } });
  NodeProp.contextHash = new NodeProp({ perNode: true });
  NodeProp.lookAhead = new NodeProp({ perNode: true });
  NodeProp.mounted = new NodeProp({ perNode: true });
  var MountedTree = class {
    constructor(tree, overlay, parser, bracketed = false) {
      this.tree = tree;
      this.overlay = overlay;
      this.parser = parser;
      this.bracketed = bracketed;
    }
    /**
    @internal
    */
    static get(tree) {
      return tree && tree.props && tree.props[NodeProp.mounted.id];
    }
  };
  var noProps = /* @__PURE__ */ Object.create(null);
  var NodeType = class _NodeType {
    /**
    @internal
    */
    constructor(name2, props, id, flags = 0) {
      this.name = name2;
      this.props = props;
      this.id = id;
      this.flags = flags;
    }
    /**
    Define a node type.
    */
    static define(spec) {
      let props = spec.props && spec.props.length ? /* @__PURE__ */ Object.create(null) : noProps;
      let flags = (spec.top ? 1 : 0) | (spec.skipped ? 2 : 0) | (spec.error ? 4 : 0) | (spec.name == null ? 8 : 0);
      let type = new _NodeType(spec.name || "", props, spec.id, flags);
      if (spec.props)
        for (let src of spec.props) {
          if (!Array.isArray(src))
            src = src(type);
          if (src) {
            if (src[0].perNode)
              throw new RangeError("Can't store a per-node prop on a node type");
            props[src[0].id] = src[1];
          }
        }
      return type;
    }
    /**
    Retrieves a node prop for this type. Will return `undefined` if
    the prop isn't present on this node.
    */
    prop(prop) {
      return this.props[prop.id];
    }
    /**
    True when this is the top node of a grammar.
    */
    get isTop() {
      return (this.flags & 1) > 0;
    }
    /**
    True when this node is produced by a skip rule.
    */
    get isSkipped() {
      return (this.flags & 2) > 0;
    }
    /**
    Indicates whether this is an error node.
    */
    get isError() {
      return (this.flags & 4) > 0;
    }
    /**
    When true, this node type doesn't correspond to a user-declared
    named node, for example because it is used to cache repetition.
    */
    get isAnonymous() {
      return (this.flags & 8) > 0;
    }
    /**
    Returns true when this node's name or one of its
    [groups](#common.NodeProp^group) matches the given string.
    */
    is(name2) {
      if (typeof name2 == "string") {
        if (this.name == name2)
          return true;
        let group = this.prop(NodeProp.group);
        return group ? group.indexOf(name2) > -1 : false;
      }
      return this.id == name2;
    }
    /**
    Create a function from node types to arbitrary values by
    specifying an object whose property names are node or
    [group](#common.NodeProp^group) names. Often useful with
    [`NodeProp.add`](#common.NodeProp.add). You can put multiple
    names, separated by spaces, in a single property name to map
    multiple node names to a single value.
    */
    static match(map) {
      let direct = /* @__PURE__ */ Object.create(null);
      for (let prop in map)
        for (let name2 of prop.split(" "))
          direct[name2] = map[prop];
      return (node) => {
        for (let groups = node.prop(NodeProp.group), i = -1; i < (groups ? groups.length : 0); i++) {
          let found = direct[i < 0 ? node.name : groups[i]];
          if (found)
            return found;
        }
      };
    }
  };
  NodeType.none = new NodeType(
    "",
    /* @__PURE__ */ Object.create(null),
    0,
    8
    /* NodeFlag.Anonymous */
  );
  var CachedNode = /* @__PURE__ */ new WeakMap();
  var CachedInnerNode = /* @__PURE__ */ new WeakMap();
  var IterMode;
  (function(IterMode2) {
    IterMode2[IterMode2["ExcludeBuffers"] = 1] = "ExcludeBuffers";
    IterMode2[IterMode2["IncludeAnonymous"] = 2] = "IncludeAnonymous";
    IterMode2[IterMode2["IgnoreMounts"] = 4] = "IgnoreMounts";
    IterMode2[IterMode2["IgnoreOverlays"] = 8] = "IgnoreOverlays";
    IterMode2[IterMode2["EnterBracketed"] = 16] = "EnterBracketed";
  })(IterMode || (IterMode = {}));
  var Tree = class _Tree {
    /**
    Construct a new tree. See also [`Tree.build`](#common.Tree^build).
    */
    constructor(type, children, positions, length, props) {
      this.type = type;
      this.children = children;
      this.positions = positions;
      this.length = length;
      this.props = null;
      if (props && props.length) {
        this.props = /* @__PURE__ */ Object.create(null);
        for (let [prop, value] of props)
          this.props[typeof prop == "number" ? prop : prop.id] = value;
      }
    }
    /**
    @internal
    */
    toString() {
      let mounted = MountedTree.get(this);
      if (mounted && !mounted.overlay)
        return mounted.tree.toString();
      let children = "";
      for (let ch of this.children) {
        let str = ch.toString();
        if (str) {
          if (children)
            children += ",";
          children += str;
        }
      }
      return !this.type.name ? children : (/\W/.test(this.type.name) && !this.type.isError ? JSON.stringify(this.type.name) : this.type.name) + (children.length ? "(" + children + ")" : "");
    }
    /**
    Get a [tree cursor](#common.TreeCursor) positioned at the top of
    the tree. Mode can be used to [control](#common.IterMode) which
    nodes the cursor visits.
    */
    cursor(mode = 0) {
      return new TreeCursor(this.topNode, mode);
    }
    /**
    Get a [tree cursor](#common.TreeCursor) pointing into this tree
    at the given position and side (see
    [`moveTo`](#common.TreeCursor.moveTo).
    */
    cursorAt(pos, side = 0, mode = 0) {
      let scope = CachedNode.get(this) || this.topNode;
      let cursor = new TreeCursor(scope);
      cursor.moveTo(pos, side);
      CachedNode.set(this, cursor._tree);
      return cursor;
    }
    /**
    Get a [syntax node](#common.SyntaxNode) object for the top of the
    tree.
    */
    get topNode() {
      return new TreeNode(this, 0, 0, null);
    }
    /**
    Get the [syntax node](#common.SyntaxNode) at the given position.
    If `side` is -1, this will move into nodes that end at the
    position. If 1, it'll move into nodes that start at the
    position. With 0, it'll only enter nodes that cover the position
    from both sides.
    
    Note that this will not enter
    [overlays](#common.MountedTree.overlay), and you often want
    [`resolveInner`](#common.Tree.resolveInner) instead.
    */
    resolve(pos, side = 0) {
      let node = resolveNode(CachedNode.get(this) || this.topNode, pos, side, false);
      CachedNode.set(this, node);
      return node;
    }
    /**
    Like [`resolve`](#common.Tree.resolve), but will enter
    [overlaid](#common.MountedTree.overlay) nodes, producing a syntax node
    pointing into the innermost overlaid tree at the given position
    (with parent links going through all parent structure, including
    the host trees).
    */
    resolveInner(pos, side = 0) {
      let node = resolveNode(CachedInnerNode.get(this) || this.topNode, pos, side, true);
      CachedInnerNode.set(this, node);
      return node;
    }
    /**
    In some situations, it can be useful to iterate through all
    nodes around a position, including those in overlays that don't
    directly cover the position. This method gives you an iterator
    that will produce all nodes, from small to big, around the given
    position.
    */
    resolveStack(pos, side = 0) {
      return stackIterator(this, pos, side);
    }
    /**
    Iterate over the tree and its children, calling `enter` for any
    node that touches the `from`/`to` region (if given) before
    running over such a node's children, and `leave` (if given) when
    leaving the node. When `enter` returns `false`, that node will
    not have its children iterated over (or `leave` called).
    */
    iterate(spec) {
      let { enter, leave, from = 0, to = this.length } = spec;
      let mode = spec.mode || 0, anon = (mode & IterMode.IncludeAnonymous) > 0;
      for (let c = this.cursor(mode | IterMode.IncludeAnonymous); ; ) {
        let entered = false;
        if (c.from <= to && c.to >= from && (!anon && c.type.isAnonymous || enter(c) !== false)) {
          if (c.firstChild())
            continue;
          entered = true;
        }
        for (; ; ) {
          if (entered && leave && (anon || !c.type.isAnonymous))
            leave(c);
          if (c.nextSibling())
            break;
          if (!c.parent())
            return;
          entered = true;
        }
      }
    }
    /**
    Get the value of the given [node prop](#common.NodeProp) for this
    node. Works with both per-node and per-type props.
    */
    prop(prop) {
      return !prop.perNode ? this.type.prop(prop) : this.props ? this.props[prop.id] : void 0;
    }
    /**
    Returns the node's [per-node props](#common.NodeProp.perNode) in a
    format that can be passed to the [`Tree`](#common.Tree)
    constructor.
    */
    get propValues() {
      let result = [];
      if (this.props)
        for (let id in this.props)
          result.push([+id, this.props[id]]);
      return result;
    }
    /**
    Balance the direct children of this tree, producing a copy of
    which may have children grouped into subtrees with type
    [`NodeType.none`](#common.NodeType^none).
    */
    balance(config = {}) {
      return this.children.length <= 8 ? this : balanceRange(NodeType.none, this.children, this.positions, 0, this.children.length, 0, this.length, (children, positions, length) => new _Tree(this.type, children, positions, length, this.propValues), config.makeTree || ((children, positions, length) => new _Tree(NodeType.none, children, positions, length)));
    }
    /**
    Build a tree from a postfix-ordered buffer of node information,
    or a cursor over such a buffer.
    */
    static build(data) {
      return buildTree(data);
    }
  };
  Tree.empty = new Tree(NodeType.none, [], [], 0);
  var FlatBufferCursor = class _FlatBufferCursor {
    constructor(buffer, index) {
      this.buffer = buffer;
      this.index = index;
    }
    get id() {
      return this.buffer[this.index - 4];
    }
    get start() {
      return this.buffer[this.index - 3];
    }
    get end() {
      return this.buffer[this.index - 2];
    }
    get size() {
      return this.buffer[this.index - 1];
    }
    get pos() {
      return this.index;
    }
    next() {
      this.index -= 4;
    }
    fork() {
      return new _FlatBufferCursor(this.buffer, this.index);
    }
  };
  var TreeBuffer = class _TreeBuffer {
    /**
    Create a tree buffer.
    */
    constructor(buffer, length, set) {
      this.buffer = buffer;
      this.length = length;
      this.set = set;
    }
    /**
    @internal
    */
    get type() {
      return NodeType.none;
    }
    /**
    @internal
    */
    toString() {
      let result = [];
      for (let index = 0; index < this.buffer.length; ) {
        result.push(this.childString(index));
        index = this.buffer[index + 3];
      }
      return result.join(",");
    }
    /**
    @internal
    */
    childString(index) {
      let id = this.buffer[index], endIndex = this.buffer[index + 3];
      let type = this.set.types[id], result = type.name;
      if (/\W/.test(result) && !type.isError)
        result = JSON.stringify(result);
      index += 4;
      if (endIndex == index)
        return result;
      let children = [];
      while (index < endIndex) {
        children.push(this.childString(index));
        index = this.buffer[index + 3];
      }
      return result + "(" + children.join(",") + ")";
    }
    /**
    @internal
    */
    findChild(startIndex, endIndex, dir, pos, side) {
      let { buffer } = this, pick = -1;
      for (let i = startIndex; i != endIndex; i = buffer[i + 3]) {
        if (checkSide(side, pos, buffer[i + 1], buffer[i + 2])) {
          pick = i;
          if (dir > 0)
            break;
        }
      }
      return pick;
    }
    /**
    @internal
    */
    slice(startI, endI, from) {
      let b = this.buffer;
      let copy = new Uint16Array(endI - startI), len = 0;
      for (let i = startI, j = 0; i < endI; ) {
        copy[j++] = b[i++];
        copy[j++] = b[i++] - from;
        let to = copy[j++] = b[i++] - from;
        copy[j++] = b[i++] - startI;
        len = Math.max(len, to);
      }
      return new _TreeBuffer(copy, len, this.set);
    }
  };
  function checkSide(side, pos, from, to) {
    switch (side) {
      case -2:
        return from < pos;
      case -1:
        return to >= pos && from < pos;
      case 0:
        return from < pos && to > pos;
      case 1:
        return from <= pos && to > pos;
      case 2:
        return to > pos;
      case 4:
        return true;
    }
  }
  function resolveNode(node, pos, side, overlays) {
    var _a2;
    while (node.from == node.to || (side < 1 ? node.from >= pos : node.from > pos) || (side > -1 ? node.to <= pos : node.to < pos)) {
      let parent = !overlays && node instanceof TreeNode && node.index < 0 ? null : node.parent;
      if (!parent)
        return node;
      node = parent;
    }
    let mode = overlays ? 0 : IterMode.IgnoreOverlays;
    if (overlays)
      for (let scan = node, parent = scan.parent; parent; scan = parent, parent = scan.parent) {
        if (scan instanceof TreeNode && scan.index < 0 && ((_a2 = parent.enter(pos, side, mode)) === null || _a2 === void 0 ? void 0 : _a2.from) != scan.from)
          node = parent;
      }
    for (; ; ) {
      let inner = node.enter(pos, side, mode);
      if (!inner)
        return node;
      node = inner;
    }
  }
  var BaseNode = class {
    cursor(mode = 0) {
      return new TreeCursor(this, mode);
    }
    getChild(type, before = null, after = null) {
      let r = getChildren(this, type, before, after);
      return r.length ? r[0] : null;
    }
    getChildren(type, before = null, after = null) {
      return getChildren(this, type, before, after);
    }
    resolve(pos, side = 0) {
      return resolveNode(this, pos, side, false);
    }
    resolveInner(pos, side = 0) {
      return resolveNode(this, pos, side, true);
    }
    matchContext(context) {
      return matchNodeContext(this.parent, context);
    }
    enterUnfinishedNodesBefore(pos) {
      let scan = this.childBefore(pos), node = this;
      while (scan) {
        let last = scan.lastChild;
        if (!last || last.to != scan.to)
          break;
        if (last.type.isError && last.from == last.to) {
          node = scan;
          scan = last.prevSibling;
        } else {
          scan = last;
        }
      }
      return node;
    }
    get node() {
      return this;
    }
    get next() {
      return this.parent;
    }
  };
  var TreeNode = class _TreeNode extends BaseNode {
    constructor(_tree, from, index, _parent) {
      super();
      this._tree = _tree;
      this.from = from;
      this.index = index;
      this._parent = _parent;
    }
    get type() {
      return this._tree.type;
    }
    get name() {
      return this._tree.type.name;
    }
    get to() {
      return this.from + this._tree.length;
    }
    nextChild(i, dir, pos, side, mode = 0) {
      for (let parent = this; ; ) {
        for (let { children, positions } = parent._tree, e = dir > 0 ? children.length : -1; i != e; i += dir) {
          let next = children[i], start = positions[i] + parent.from, mounted;
          if (!(mode & IterMode.EnterBracketed && next instanceof Tree && (mounted = MountedTree.get(next)) && !mounted.overlay && mounted.bracketed && pos >= start && pos <= start + next.length) && !checkSide(side, pos, start, start + next.length))
            continue;
          if (next instanceof TreeBuffer) {
            if (mode & IterMode.ExcludeBuffers)
              continue;
            let index = next.findChild(0, next.buffer.length, dir, pos - start, side);
            if (index > -1)
              return new BufferNode(new BufferContext(parent, next, i, start), null, index);
          } else if (mode & IterMode.IncludeAnonymous || (!next.type.isAnonymous || hasChild(next))) {
            let mounted2;
            if (!(mode & IterMode.IgnoreMounts) && (mounted2 = MountedTree.get(next)) && !mounted2.overlay)
              return new _TreeNode(mounted2.tree, start, i, parent);
            let inner = new _TreeNode(next, start, i, parent);
            return mode & IterMode.IncludeAnonymous || !inner.type.isAnonymous ? inner : inner.nextChild(dir < 0 ? next.children.length - 1 : 0, dir, pos, side, mode);
          }
        }
        if (mode & IterMode.IncludeAnonymous || !parent.type.isAnonymous)
          return null;
        if (parent.index >= 0)
          i = parent.index + dir;
        else
          i = dir < 0 ? -1 : parent._parent._tree.children.length;
        parent = parent._parent;
        if (!parent)
          return null;
      }
    }
    get firstChild() {
      return this.nextChild(
        0,
        1,
        0,
        4
        /* Side.DontCare */
      );
    }
    get lastChild() {
      return this.nextChild(
        this._tree.children.length - 1,
        -1,
        0,
        4
        /* Side.DontCare */
      );
    }
    childAfter(pos) {
      return this.nextChild(
        0,
        1,
        pos,
        2
        /* Side.After */
      );
    }
    childBefore(pos) {
      return this.nextChild(
        this._tree.children.length - 1,
        -1,
        pos,
        -2
        /* Side.Before */
      );
    }
    prop(prop) {
      return this._tree.prop(prop);
    }
    enter(pos, side, mode = 0) {
      let mounted;
      if (!(mode & IterMode.IgnoreOverlays) && (mounted = MountedTree.get(this._tree)) && mounted.overlay) {
        let rPos = pos - this.from, enterBracketed = mode & IterMode.EnterBracketed && mounted.bracketed;
        for (let { from, to } of mounted.overlay) {
          if ((side > 0 || enterBracketed ? from <= rPos : from < rPos) && (side < 0 || enterBracketed ? to >= rPos : to > rPos))
            return new _TreeNode(mounted.tree, mounted.overlay[0].from + this.from, -1, this);
        }
      }
      return this.nextChild(0, 1, pos, side, mode);
    }
    nextSignificantParent() {
      let val = this;
      while (val.type.isAnonymous && val._parent)
        val = val._parent;
      return val;
    }
    get parent() {
      return this._parent ? this._parent.nextSignificantParent() : null;
    }
    get nextSibling() {
      return this._parent && this.index >= 0 ? this._parent.nextChild(
        this.index + 1,
        1,
        0,
        4
        /* Side.DontCare */
      ) : null;
    }
    get prevSibling() {
      return this._parent && this.index >= 0 ? this._parent.nextChild(
        this.index - 1,
        -1,
        0,
        4
        /* Side.DontCare */
      ) : null;
    }
    get tree() {
      return this._tree;
    }
    toTree() {
      return this._tree;
    }
    /**
    @internal
    */
    toString() {
      return this._tree.toString();
    }
  };
  function getChildren(node, type, before, after) {
    let cur2 = node.cursor(), result = [];
    if (!cur2.firstChild())
      return result;
    if (before != null)
      for (let found = false; !found; ) {
        found = cur2.type.is(before);
        if (!cur2.nextSibling())
          return result;
      }
    for (; ; ) {
      if (after != null && cur2.type.is(after))
        return result;
      if (cur2.type.is(type))
        result.push(cur2.node);
      if (!cur2.nextSibling())
        return after == null ? result : [];
    }
  }
  function matchNodeContext(node, context, i = context.length - 1) {
    for (let p = node; i >= 0; p = p.parent) {
      if (!p)
        return false;
      if (!p.type.isAnonymous) {
        if (context[i] && context[i] != p.name)
          return false;
        i--;
      }
    }
    return true;
  }
  var BufferContext = class {
    constructor(parent, buffer, index, start) {
      this.parent = parent;
      this.buffer = buffer;
      this.index = index;
      this.start = start;
    }
  };
  var BufferNode = class _BufferNode extends BaseNode {
    get name() {
      return this.type.name;
    }
    get from() {
      return this.context.start + this.context.buffer.buffer[this.index + 1];
    }
    get to() {
      return this.context.start + this.context.buffer.buffer[this.index + 2];
    }
    constructor(context, _parent, index) {
      super();
      this.context = context;
      this._parent = _parent;
      this.index = index;
      this.type = context.buffer.set.types[context.buffer.buffer[index]];
    }
    child(dir, pos, side) {
      let { buffer } = this.context;
      let index = buffer.findChild(this.index + 4, buffer.buffer[this.index + 3], dir, pos - this.context.start, side);
      return index < 0 ? null : new _BufferNode(this.context, this, index);
    }
    get firstChild() {
      return this.child(
        1,
        0,
        4
        /* Side.DontCare */
      );
    }
    get lastChild() {
      return this.child(
        -1,
        0,
        4
        /* Side.DontCare */
      );
    }
    childAfter(pos) {
      return this.child(
        1,
        pos,
        2
        /* Side.After */
      );
    }
    childBefore(pos) {
      return this.child(
        -1,
        pos,
        -2
        /* Side.Before */
      );
    }
    prop(prop) {
      return this.type.prop(prop);
    }
    enter(pos, side, mode = 0) {
      if (mode & IterMode.ExcludeBuffers)
        return null;
      let { buffer } = this.context;
      let index = buffer.findChild(this.index + 4, buffer.buffer[this.index + 3], side > 0 ? 1 : -1, pos - this.context.start, side);
      return index < 0 ? null : new _BufferNode(this.context, this, index);
    }
    get parent() {
      return this._parent || this.context.parent.nextSignificantParent();
    }
    externalSibling(dir) {
      return this._parent ? null : this.context.parent.nextChild(
        this.context.index + dir,
        dir,
        0,
        4
        /* Side.DontCare */
      );
    }
    get nextSibling() {
      let { buffer } = this.context;
      let after = buffer.buffer[this.index + 3];
      if (after < (this._parent ? buffer.buffer[this._parent.index + 3] : buffer.buffer.length))
        return new _BufferNode(this.context, this._parent, after);
      return this.externalSibling(1);
    }
    get prevSibling() {
      let { buffer } = this.context;
      let parentStart = this._parent ? this._parent.index + 4 : 0;
      if (this.index == parentStart)
        return this.externalSibling(-1);
      return new _BufferNode(this.context, this._parent, buffer.findChild(
        parentStart,
        this.index,
        -1,
        0,
        4
        /* Side.DontCare */
      ));
    }
    get tree() {
      return null;
    }
    toTree() {
      let children = [], positions = [];
      let { buffer } = this.context;
      let startI = this.index + 4, endI = buffer.buffer[this.index + 3];
      if (endI > startI) {
        let from = buffer.buffer[this.index + 1];
        children.push(buffer.slice(startI, endI, from));
        positions.push(0);
      }
      return new Tree(this.type, children, positions, this.to - this.from);
    }
    /**
    @internal
    */
    toString() {
      return this.context.buffer.childString(this.index);
    }
  };
  function iterStack(heads) {
    if (!heads.length)
      return null;
    let pick = 0, picked = heads[0];
    for (let i = 1; i < heads.length; i++) {
      let node = heads[i];
      if (node.from > picked.from || node.to < picked.to) {
        picked = node;
        pick = i;
      }
    }
    let next = picked instanceof TreeNode && picked.index < 0 ? null : picked.parent;
    let newHeads = heads.slice();
    if (next)
      newHeads[pick] = next;
    else
      newHeads.splice(pick, 1);
    return new StackIterator(newHeads, picked);
  }
  var StackIterator = class {
    constructor(heads, node) {
      this.heads = heads;
      this.node = node;
    }
    get next() {
      return iterStack(this.heads);
    }
  };
  function stackIterator(tree, pos, side) {
    let inner = tree.resolveInner(pos, side), layers = null;
    for (let scan = inner instanceof TreeNode ? inner : inner.context.parent; scan; scan = scan.parent) {
      if (scan.index < 0) {
        let parent = scan.parent;
        (layers || (layers = [inner])).push(parent.resolve(pos, side));
        scan = parent;
      } else {
        let mount = MountedTree.get(scan.tree);
        if (mount && mount.overlay && mount.overlay[0].from <= pos && mount.overlay[mount.overlay.length - 1].to >= pos) {
          let root = new TreeNode(mount.tree, mount.overlay[0].from + scan.from, -1, scan);
          (layers || (layers = [inner])).push(resolveNode(root, pos, side, false));
        }
      }
    }
    return layers ? iterStack(layers) : inner;
  }
  var TreeCursor = class {
    /**
    Shorthand for `.type.name`.
    */
    get name() {
      return this.type.name;
    }
    /**
    @internal
    */
    constructor(node, mode = 0) {
      this.buffer = null;
      this.stack = [];
      this.index = 0;
      this.bufferNode = null;
      this.mode = mode & ~IterMode.EnterBracketed;
      if (node instanceof TreeNode) {
        this.yieldNode(node);
      } else {
        this._tree = node.context.parent;
        this.buffer = node.context;
        for (let n = node._parent; n; n = n._parent)
          this.stack.unshift(n.index);
        this.bufferNode = node;
        this.yieldBuf(node.index);
      }
    }
    yieldNode(node) {
      if (!node)
        return false;
      this._tree = node;
      this.type = node.type;
      this.from = node.from;
      this.to = node.to;
      return true;
    }
    yieldBuf(index, type) {
      this.index = index;
      let { start, buffer } = this.buffer;
      this.type = type || buffer.set.types[buffer.buffer[index]];
      this.from = start + buffer.buffer[index + 1];
      this.to = start + buffer.buffer[index + 2];
      return true;
    }
    /**
    @internal
    */
    yield(node) {
      if (!node)
        return false;
      if (node instanceof TreeNode) {
        this.buffer = null;
        return this.yieldNode(node);
      }
      this.buffer = node.context;
      return this.yieldBuf(node.index, node.type);
    }
    /**
    @internal
    */
    toString() {
      return this.buffer ? this.buffer.buffer.childString(this.index) : this._tree.toString();
    }
    /**
    @internal
    */
    enterChild(dir, pos, side) {
      if (!this.buffer)
        return this.yield(this._tree.nextChild(dir < 0 ? this._tree._tree.children.length - 1 : 0, dir, pos, side, this.mode));
      let { buffer } = this.buffer;
      let index = buffer.findChild(this.index + 4, buffer.buffer[this.index + 3], dir, pos - this.buffer.start, side);
      if (index < 0)
        return false;
      this.stack.push(this.index);
      return this.yieldBuf(index);
    }
    /**
    Move the cursor to this node's first child. When this returns
    false, the node has no child, and the cursor has not been moved.
    */
    firstChild() {
      return this.enterChild(
        1,
        0,
        4
        /* Side.DontCare */
      );
    }
    /**
    Move the cursor to this node's last child.
    */
    lastChild() {
      return this.enterChild(
        -1,
        0,
        4
        /* Side.DontCare */
      );
    }
    /**
    Move the cursor to the first child that ends after `pos`.
    */
    childAfter(pos) {
      return this.enterChild(
        1,
        pos,
        2
        /* Side.After */
      );
    }
    /**
    Move to the last child that starts before `pos`.
    */
    childBefore(pos) {
      return this.enterChild(
        -1,
        pos,
        -2
        /* Side.Before */
      );
    }
    /**
    Move the cursor to the child around `pos`. If side is -1 the
    child may end at that position, when 1 it may start there. This
    will also enter [overlaid](#common.MountedTree.overlay)
    [mounted](#common.NodeProp^mounted) trees unless `overlays` is
    set to false.
    */
    enter(pos, side, mode = this.mode) {
      if (!this.buffer)
        return this.yield(this._tree.enter(pos, side, mode));
      return mode & IterMode.ExcludeBuffers ? false : this.enterChild(1, pos, side);
    }
    /**
    Move to the node's parent node, if this isn't the top node.
    */
    parent() {
      if (!this.buffer)
        return this.yieldNode(this.mode & IterMode.IncludeAnonymous ? this._tree._parent : this._tree.parent);
      if (this.stack.length)
        return this.yieldBuf(this.stack.pop());
      let parent = this.mode & IterMode.IncludeAnonymous ? this.buffer.parent : this.buffer.parent.nextSignificantParent();
      this.buffer = null;
      return this.yieldNode(parent);
    }
    /**
    @internal
    */
    sibling(dir) {
      if (!this.buffer)
        return !this._tree._parent ? false : this.yield(this._tree.index < 0 ? null : this._tree._parent.nextChild(this._tree.index + dir, dir, 0, 4, this.mode));
      let { buffer } = this.buffer, d = this.stack.length - 1;
      if (dir < 0) {
        let parentStart = d < 0 ? 0 : this.stack[d] + 4;
        if (this.index != parentStart)
          return this.yieldBuf(buffer.findChild(
            parentStart,
            this.index,
            -1,
            0,
            4
            /* Side.DontCare */
          ));
      } else {
        let after = buffer.buffer[this.index + 3];
        if (after < (d < 0 ? buffer.buffer.length : buffer.buffer[this.stack[d] + 3]))
          return this.yieldBuf(after);
      }
      return d < 0 ? this.yield(this.buffer.parent.nextChild(this.buffer.index + dir, dir, 0, 4, this.mode)) : false;
    }
    /**
    Move to this node's next sibling, if any.
    */
    nextSibling() {
      return this.sibling(1);
    }
    /**
    Move to this node's previous sibling, if any.
    */
    prevSibling() {
      return this.sibling(-1);
    }
    atLastNode(dir) {
      let index, parent, { buffer } = this;
      if (buffer) {
        if (dir > 0) {
          if (this.index < buffer.buffer.buffer.length)
            return false;
        } else {
          for (let i = 0; i < this.index; i++)
            if (buffer.buffer.buffer[i + 3] < this.index)
              return false;
        }
        ({ index, parent } = buffer);
      } else {
        ({ index, _parent: parent } = this._tree);
      }
      for (; parent; { index, _parent: parent } = parent) {
        if (index > -1)
          for (let i = index + dir, e = dir < 0 ? -1 : parent._tree.children.length; i != e; i += dir) {
            let child = parent._tree.children[i];
            if (this.mode & IterMode.IncludeAnonymous || child instanceof TreeBuffer || !child.type.isAnonymous || hasChild(child))
              return false;
          }
      }
      return true;
    }
    move(dir, enter) {
      if (enter && this.enterChild(
        dir,
        0,
        4
        /* Side.DontCare */
      ))
        return true;
      for (; ; ) {
        if (this.sibling(dir))
          return true;
        if (this.atLastNode(dir) || !this.parent())
          return false;
      }
    }
    /**
    Move to the next node in a
    [pre-order](https://en.wikipedia.org/wiki/Tree_traversal#Pre-order,_NLR)
    traversal, going from a node to its first child or, if the
    current node is empty or `enter` is false, its next sibling or
    the next sibling of the first parent node that has one.
    */
    next(enter = true) {
      return this.move(1, enter);
    }
    /**
    Move to the next node in a last-to-first pre-order traversal. A
    node is followed by its last child or, if it has none, its
    previous sibling or the previous sibling of the first parent
    node that has one.
    */
    prev(enter = true) {
      return this.move(-1, enter);
    }
    /**
    Move the cursor to the innermost node that covers `pos`. If
    `side` is -1, it will enter nodes that end at `pos`. If it is 1,
    it will enter nodes that start at `pos`.
    */
    moveTo(pos, side = 0) {
      while (this.from == this.to || (side < 1 ? this.from >= pos : this.from > pos) || (side > -1 ? this.to <= pos : this.to < pos))
        if (!this.parent())
          break;
      while (this.enterChild(1, pos, side)) {
      }
      return this;
    }
    /**
    Get a [syntax node](#common.SyntaxNode) at the cursor's current
    position.
    */
    get node() {
      if (!this.buffer)
        return this._tree;
      let cache = this.bufferNode, result = null, depth = 0;
      if (cache && cache.context == this.buffer) {
        scan: for (let index = this.index, d = this.stack.length; d >= 0; ) {
          for (let c = cache; c; c = c._parent)
            if (c.index == index) {
              if (index == this.index)
                return c;
              result = c;
              depth = d + 1;
              break scan;
            }
          index = this.stack[--d];
        }
      }
      for (let i = depth; i < this.stack.length; i++)
        result = new BufferNode(this.buffer, result, this.stack[i]);
      return this.bufferNode = new BufferNode(this.buffer, result, this.index);
    }
    /**
    Get the [tree](#common.Tree) that represents the current node, if
    any. Will return null when the node is in a [tree
    buffer](#common.TreeBuffer).
    */
    get tree() {
      return this.buffer ? null : this._tree._tree;
    }
    /**
    Iterate over the current node and all its descendants, calling
    `enter` when entering a node and `leave`, if given, when leaving
    one. When `enter` returns `false`, any children of that node are
    skipped, and `leave` isn't called for it.
    */
    iterate(enter, leave) {
      for (let depth = 0; ; ) {
        let mustLeave = false;
        if (this.type.isAnonymous || enter(this) !== false) {
          if (this.firstChild()) {
            depth++;
            continue;
          }
          if (!this.type.isAnonymous)
            mustLeave = true;
        }
        for (; ; ) {
          if (mustLeave && leave)
            leave(this);
          mustLeave = this.type.isAnonymous;
          if (!depth)
            return;
          if (this.nextSibling())
            break;
          this.parent();
          depth--;
          mustLeave = true;
        }
      }
    }
    /**
    Test whether the current node matches a given context—a sequence
    of direct parent node names. Empty strings in the context array
    are treated as wildcards.
    */
    matchContext(context) {
      if (!this.buffer)
        return matchNodeContext(this.node.parent, context);
      let { buffer } = this.buffer, { types: types2 } = buffer.set;
      for (let i = context.length - 1, d = this.stack.length - 1; i >= 0; d--) {
        if (d < 0)
          return matchNodeContext(this._tree, context, i);
        let type = types2[buffer.buffer[this.stack[d]]];
        if (!type.isAnonymous) {
          if (context[i] && context[i] != type.name)
            return false;
          i--;
        }
      }
      return true;
    }
  };
  function hasChild(tree) {
    return tree.children.some((ch) => ch instanceof TreeBuffer || !ch.type.isAnonymous || hasChild(ch));
  }
  function buildTree(data) {
    var _a2;
    let { buffer, nodeSet, maxBufferLength = DefaultBufferLength, reused = [], minRepeatType = nodeSet.types.length } = data;
    let cursor = Array.isArray(buffer) ? new FlatBufferCursor(buffer, buffer.length) : buffer;
    let types2 = nodeSet.types;
    let contextHash = 0, lookAhead = 0;
    function takeNode(parentStart, minPos, children2, positions2, inRepeat, depth) {
      let { id, start, end, size } = cursor;
      let lookAheadAtStart = lookAhead, contextAtStart = contextHash;
      if (size < 0) {
        cursor.next();
        if (size == -1) {
          let node2 = reused[id];
          children2.push(node2);
          positions2.push(start - parentStart);
          return;
        } else if (size == -3) {
          contextHash = id;
          return;
        } else if (size == -4) {
          lookAhead = id;
          return;
        } else {
          throw new RangeError(`Unrecognized record size: ${size}`);
        }
      }
      let type = types2[id], node, buffer2;
      let startPos = start - parentStart;
      if (end - start <= maxBufferLength && (buffer2 = findBufferSize(cursor.pos - minPos, inRepeat))) {
        let data2 = new Uint16Array(buffer2.size - buffer2.skip);
        let endPos = cursor.pos - buffer2.size, index = data2.length;
        while (cursor.pos > endPos)
          index = copyToBuffer(buffer2.start, data2, index);
        node = new TreeBuffer(data2, end - buffer2.start, nodeSet);
        startPos = buffer2.start - parentStart;
      } else {
        let endPos = cursor.pos - size;
        cursor.next();
        let localChildren = [], localPositions = [];
        let localInRepeat = id >= minRepeatType ? id : -1;
        let lastGroup = 0, lastEnd = end;
        while (cursor.pos > endPos) {
          if (localInRepeat >= 0 && cursor.id == localInRepeat && cursor.size >= 0) {
            if (cursor.end <= lastEnd - maxBufferLength) {
              makeRepeatLeaf(localChildren, localPositions, start, lastGroup, cursor.end, lastEnd, localInRepeat, lookAheadAtStart, contextAtStart);
              lastGroup = localChildren.length;
              lastEnd = cursor.end;
            }
            cursor.next();
          } else if (depth > 2500) {
            takeFlatNode(start, endPos, localChildren, localPositions);
          } else {
            takeNode(start, endPos, localChildren, localPositions, localInRepeat, depth + 1);
          }
        }
        if (localInRepeat >= 0 && lastGroup > 0 && lastGroup < localChildren.length)
          makeRepeatLeaf(localChildren, localPositions, start, lastGroup, start, lastEnd, localInRepeat, lookAheadAtStart, contextAtStart);
        localChildren.reverse();
        localPositions.reverse();
        if (localInRepeat > -1 && lastGroup > 0) {
          let make = makeBalanced(type, contextAtStart);
          node = balanceRange(type, localChildren, localPositions, 0, localChildren.length, 0, end - start, make, make);
        } else {
          node = makeTree(type, localChildren, localPositions, end - start, lookAheadAtStart - end, contextAtStart);
        }
      }
      children2.push(node);
      positions2.push(startPos);
    }
    function takeFlatNode(parentStart, minPos, children2, positions2) {
      let nodes = [];
      let nodeCount = 0, stopAt = -1;
      while (cursor.pos > minPos) {
        let { id, start, end, size } = cursor;
        if (size > 4) {
          cursor.next();
        } else if (stopAt > -1 && start < stopAt) {
          break;
        } else {
          if (stopAt < 0)
            stopAt = end - maxBufferLength;
          nodes.push(id, start, end);
          nodeCount++;
          cursor.next();
        }
      }
      if (nodeCount) {
        let buffer2 = new Uint16Array(nodeCount * 4);
        let start = nodes[nodes.length - 2];
        for (let i = nodes.length - 3, j = 0; i >= 0; i -= 3) {
          buffer2[j++] = nodes[i];
          buffer2[j++] = nodes[i + 1] - start;
          buffer2[j++] = nodes[i + 2] - start;
          buffer2[j++] = j;
        }
        children2.push(new TreeBuffer(buffer2, nodes[2] - start, nodeSet));
        positions2.push(start - parentStart);
      }
    }
    function makeBalanced(type, contextHash2) {
      return (children2, positions2, length2) => {
        let lookAhead2 = 0, lastI = children2.length - 1, last, lookAheadProp;
        if (lastI >= 0 && (last = children2[lastI]) instanceof Tree) {
          if (!lastI && last.type == type && last.length == length2)
            return last;
          if (lookAheadProp = last.prop(NodeProp.lookAhead))
            lookAhead2 = positions2[lastI] + last.length + lookAheadProp;
        }
        return makeTree(type, children2, positions2, length2, lookAhead2, contextHash2);
      };
    }
    function makeRepeatLeaf(children2, positions2, base2, i, from, to, type, lookAhead2, contextHash2) {
      let localChildren = [], localPositions = [];
      while (children2.length > i) {
        localChildren.push(children2.pop());
        localPositions.push(positions2.pop() + base2 - from);
      }
      children2.push(makeTree(nodeSet.types[type], localChildren, localPositions, to - from, lookAhead2 - to, contextHash2));
      positions2.push(from - base2);
    }
    function makeTree(type, children2, positions2, length2, lookAhead2, contextHash2, props) {
      if (contextHash2) {
        let pair = [NodeProp.contextHash, contextHash2];
        props = props ? [pair].concat(props) : [pair];
      }
      if (lookAhead2 > 25) {
        let pair = [NodeProp.lookAhead, lookAhead2];
        props = props ? [pair].concat(props) : [pair];
      }
      return new Tree(type, children2, positions2, length2, props);
    }
    function findBufferSize(maxSize, inRepeat) {
      let fork = cursor.fork();
      let size = 0, start = 0, skip = 0, minStart = fork.end - maxBufferLength;
      let result = { size: 0, start: 0, skip: 0 };
      scan: for (let minPos = fork.pos - maxSize; fork.pos > minPos; ) {
        let nodeSize2 = fork.size;
        if (fork.id == inRepeat && nodeSize2 >= 0) {
          result.size = size;
          result.start = start;
          result.skip = skip;
          skip += 4;
          size += 4;
          fork.next();
          continue;
        }
        let startPos = fork.pos - nodeSize2;
        if (nodeSize2 < 0 || startPos < minPos || fork.start < minStart)
          break;
        let localSkipped = fork.id >= minRepeatType ? 4 : 0;
        let nodeStart = fork.start;
        fork.next();
        while (fork.pos > startPos) {
          if (fork.size < 0) {
            if (fork.size == -3 || fork.size == -4)
              localSkipped += 4;
            else
              break scan;
          } else if (fork.id >= minRepeatType) {
            localSkipped += 4;
          }
          fork.next();
        }
        start = nodeStart;
        size += nodeSize2;
        skip += localSkipped;
      }
      if (inRepeat < 0 || size == maxSize) {
        result.size = size;
        result.start = start;
        result.skip = skip;
      }
      return result.size > 4 ? result : void 0;
    }
    function copyToBuffer(bufferStart, buffer2, index) {
      let { id, start, end, size } = cursor;
      cursor.next();
      if (size >= 0 && id < minRepeatType) {
        let startIndex = index;
        if (size > 4) {
          let endPos = cursor.pos - (size - 4);
          while (cursor.pos > endPos)
            index = copyToBuffer(bufferStart, buffer2, index);
        }
        buffer2[--index] = startIndex;
        buffer2[--index] = end - bufferStart;
        buffer2[--index] = start - bufferStart;
        buffer2[--index] = id;
      } else if (size == -3) {
        contextHash = id;
      } else if (size == -4) {
        lookAhead = id;
      }
      return index;
    }
    let children = [], positions = [];
    while (cursor.pos > 0)
      takeNode(data.start || 0, data.bufferStart || 0, children, positions, -1, 0);
    let length = (_a2 = data.length) !== null && _a2 !== void 0 ? _a2 : children.length ? positions[0] + children[0].length : 0;
    return new Tree(types2[data.topID], children.reverse(), positions.reverse(), length);
  }
  var nodeSizeCache = /* @__PURE__ */ new WeakMap();
  function nodeSize(balanceType, node) {
    if (!balanceType.isAnonymous || node instanceof TreeBuffer || node.type != balanceType)
      return 1;
    let size = nodeSizeCache.get(node);
    if (size == null) {
      size = 1;
      for (let child of node.children) {
        if (child.type != balanceType || !(child instanceof Tree)) {
          size = 1;
          break;
        }
        size += nodeSize(balanceType, child);
      }
      nodeSizeCache.set(node, size);
    }
    return size;
  }
  function balanceRange(balanceType, children, positions, from, to, start, length, mkTop, mkTree) {
    let total = 0;
    for (let i = from; i < to; i++)
      total += nodeSize(balanceType, children[i]);
    let maxChild = Math.ceil(
      total * 1.5 / 8
      /* Balance.BranchFactor */
    );
    let localChildren = [], localPositions = [];
    function divide(children2, positions2, from2, to2, offset) {
      for (let i = from2; i < to2; ) {
        let groupFrom = i, groupStart = positions2[i], groupSize = nodeSize(balanceType, children2[i]);
        i++;
        for (; i < to2; i++) {
          let nextSize = nodeSize(balanceType, children2[i]);
          if (groupSize + nextSize >= maxChild)
            break;
          groupSize += nextSize;
        }
        if (i == groupFrom + 1) {
          if (groupSize > maxChild) {
            let only = children2[groupFrom];
            divide(only.children, only.positions, 0, only.children.length, positions2[groupFrom] + offset);
            continue;
          }
          localChildren.push(children2[groupFrom]);
        } else {
          let length2 = positions2[i - 1] + children2[i - 1].length - groupStart;
          localChildren.push(balanceRange(balanceType, children2, positions2, groupFrom, i, groupStart, length2, null, mkTree));
        }
        localPositions.push(groupStart + offset - start);
      }
    }
    divide(children, positions, from, to, 0);
    return (mkTop || mkTree)(localChildren, localPositions, length);
  }
  var TreeFragment = class _TreeFragment {
    /**
    Construct a tree fragment. You'll usually want to use
    [`addTree`](#common.TreeFragment^addTree) and
    [`applyChanges`](#common.TreeFragment^applyChanges) instead of
    calling this directly.
    */
    constructor(from, to, tree, offset, openStart = false, openEnd = false) {
      this.from = from;
      this.to = to;
      this.tree = tree;
      this.offset = offset;
      this.open = (openStart ? 1 : 0) | (openEnd ? 2 : 0);
    }
    /**
    Whether the start of the fragment represents the start of a
    parse, or the end of a change. (In the second case, it may not
    be safe to reuse some nodes at the start, depending on the
    parsing algorithm.)
    */
    get openStart() {
      return (this.open & 1) > 0;
    }
    /**
    Whether the end of the fragment represents the end of a
    full-document parse, or the start of a change.
    */
    get openEnd() {
      return (this.open & 2) > 0;
    }
    /**
    Create a set of fragments from a freshly parsed tree, or update
    an existing set of fragments by replacing the ones that overlap
    with a tree with content from the new tree. When `partial` is
    true, the parse is treated as incomplete, and the resulting
    fragment has [`openEnd`](#common.TreeFragment.openEnd) set to
    true.
    */
    static addTree(tree, fragments = [], partial = false) {
      let result = [new _TreeFragment(0, tree.length, tree, 0, false, partial)];
      for (let f of fragments)
        if (f.to > tree.length)
          result.push(f);
      return result;
    }
    /**
    Apply a set of edits to an array of fragments, removing or
    splitting fragments as necessary to remove edited ranges, and
    adjusting offsets for fragments that moved.
    */
    static applyChanges(fragments, changes, minGap = 128) {
      if (!changes.length)
        return fragments;
      let result = [];
      let fI = 1, nextF = fragments.length ? fragments[0] : null;
      for (let cI = 0, pos = 0, off = 0; ; cI++) {
        let nextC = cI < changes.length ? changes[cI] : null;
        let nextPos = nextC ? nextC.fromA : 1e9;
        if (nextPos - pos >= minGap)
          while (nextF && nextF.from < nextPos) {
            let cut = nextF;
            if (pos >= cut.from || nextPos <= cut.to || off) {
              let fFrom = Math.max(cut.from, pos) - off, fTo = Math.min(cut.to, nextPos) - off;
              cut = fFrom >= fTo ? null : new _TreeFragment(fFrom, fTo, cut.tree, cut.offset + off, cI > 0, !!nextC);
            }
            if (cut)
              result.push(cut);
            if (nextF.to > nextPos)
              break;
            nextF = fI < fragments.length ? fragments[fI++] : null;
          }
        if (!nextC)
          break;
        pos = nextC.toA;
        off = nextC.toA - nextC.toB;
      }
      return result;
    }
  };
  var Parser = class {
    /**
    Start a parse, returning a [partial parse](#common.PartialParse)
    object. [`fragments`](#common.TreeFragment) can be passed in to
    make the parse incremental.
    
    By default, the entire input is parsed. You can pass `ranges`,
    which should be a sorted array of non-empty, non-overlapping
    ranges, to parse only those ranges. The tree returned in that
    case will start at `ranges[0].from`.
    */
    startParse(input, fragments, ranges) {
      if (typeof input == "string")
        input = new StringInput(input);
      ranges = !ranges ? [new Range2(0, input.length)] : ranges.length ? ranges.map((r) => new Range2(r.from, r.to)) : [new Range2(0, 0)];
      return this.createParse(input, fragments || [], ranges);
    }
    /**
    Run a full parse, returning the resulting tree.
    */
    parse(input, fragments, ranges) {
      let parse = this.startParse(input, fragments, ranges);
      for (; ; ) {
        let done = parse.advance();
        if (done)
          return done;
      }
    }
  };
  var StringInput = class {
    constructor(string2) {
      this.string = string2;
    }
    get length() {
      return this.string.length;
    }
    chunk(from) {
      return this.string.slice(from);
    }
    get lineChunks() {
      return false;
    }
    read(from, to) {
      return this.string.slice(from, to);
    }
  };
  var stoppedInner = new NodeProp({ perNode: true });

  // node_modules/@lezer/highlight/dist/index.js
  var nextTagID = 0;
  var Tag = class _Tag {
    /**
    @internal
    */
    constructor(name2, set, base2, modified) {
      this.name = name2;
      this.set = set;
      this.base = base2;
      this.modified = modified;
      this.id = nextTagID++;
    }
    toString() {
      let { name: name2 } = this;
      for (let mod of this.modified)
        if (mod.name)
          name2 = `${mod.name}(${name2})`;
      return name2;
    }
    static define(nameOrParent, parent) {
      let name2 = typeof nameOrParent == "string" ? nameOrParent : "?";
      if (nameOrParent instanceof _Tag)
        parent = nameOrParent;
      if (parent === null || parent === void 0 ? void 0 : parent.base)
        throw new Error("Can not derive from a modified tag");
      let tag = new _Tag(name2, [], null, []);
      tag.set.push(tag);
      if (parent)
        for (let t2 of parent.set)
          tag.set.push(t2);
      return tag;
    }
    /**
    Define a tag _modifier_, which is a function that, given a tag,
    will return a tag that is a subtag of the original. Applying the
    same modifier to a twice tag will return the same value (`m1(t1)
    == m1(t1)`) and applying multiple modifiers will, regardless or
    order, produce the same tag (`m1(m2(t1)) == m2(m1(t1))`).
    
    When multiple modifiers are applied to a given base tag, each
    smaller set of modifiers is registered as a parent, so that for
    example `m1(m2(m3(t1)))` is a subtype of `m1(m2(t1))`,
    `m1(m3(t1)`, and so on.
    */
    static defineModifier(name2) {
      let mod = new Modifier(name2);
      return (tag) => {
        if (tag.modified.indexOf(mod) > -1)
          return tag;
        return Modifier.get(tag.base || tag, tag.modified.concat(mod).sort((a, b) => a.id - b.id));
      };
    }
  };
  var nextModifierID = 0;
  var Modifier = class _Modifier {
    constructor(name2) {
      this.name = name2;
      this.instances = [];
      this.id = nextModifierID++;
    }
    static get(base2, mods) {
      if (!mods.length)
        return base2;
      let exists = mods[0].instances.find((t2) => t2.base == base2 && sameArray2(mods, t2.modified));
      if (exists)
        return exists;
      let set = [], tag = new Tag(base2.name, set, base2, mods);
      for (let m of mods)
        m.instances.push(tag);
      let configs = powerSet(mods);
      for (let parent of base2.set)
        if (!parent.modified.length)
          for (let config of configs)
            set.push(_Modifier.get(parent, config));
      return tag;
    }
  };
  function sameArray2(a, b) {
    return a.length == b.length && a.every((x, i) => x == b[i]);
  }
  function powerSet(array) {
    let sets = [[]];
    for (let i = 0; i < array.length; i++) {
      for (let j = 0, e = sets.length; j < e; j++) {
        sets.push(sets[j].concat(array[i]));
      }
    }
    return sets.sort((a, b) => b.length - a.length);
  }
  function styleTags(spec) {
    let byName = /* @__PURE__ */ Object.create(null);
    for (let prop in spec) {
      let tags2 = spec[prop];
      if (!Array.isArray(tags2))
        tags2 = [tags2];
      for (let part of prop.split(" "))
        if (part) {
          let pieces = [], mode = 2, rest = part;
          for (let pos = 0; ; ) {
            if (rest == "..." && pos > 0 && pos + 3 == part.length) {
              mode = 1;
              break;
            }
            let m = /^"(?:[^"\\]|\\.)*?"|[^\/!]+/.exec(rest);
            if (!m)
              throw new RangeError("Invalid path: " + part);
            pieces.push(m[0] == "*" ? "" : m[0][0] == '"' ? JSON.parse(m[0]) : m[0]);
            pos += m[0].length;
            if (pos == part.length)
              break;
            let next = part[pos++];
            if (pos == part.length && next == "!") {
              mode = 0;
              break;
            }
            if (next != "/")
              throw new RangeError("Invalid path: " + part);
            rest = part.slice(pos);
          }
          let last = pieces.length - 1, inner = pieces[last];
          if (!inner)
            throw new RangeError("Invalid path: " + part);
          let rule = new Rule(tags2, mode, last > 0 ? pieces.slice(0, last) : null);
          byName[inner] = rule.sort(byName[inner]);
        }
    }
    return ruleNodeProp.add(byName);
  }
  var ruleNodeProp = new NodeProp({
    combine(a, b) {
      let cur2, root, take;
      while (a || b) {
        if (!a || b && a.depth >= b.depth) {
          take = b;
          b = b.next;
        } else {
          take = a;
          a = a.next;
        }
        if (cur2 && cur2.mode == take.mode && !take.context && !cur2.context)
          continue;
        let copy = new Rule(take.tags, take.mode, take.context);
        if (cur2)
          cur2.next = copy;
        else
          root = copy;
        cur2 = copy;
      }
      return root;
    }
  });
  var Rule = class {
    constructor(tags2, mode, context, next) {
      this.tags = tags2;
      this.mode = mode;
      this.context = context;
      this.next = next;
    }
    get opaque() {
      return this.mode == 0;
    }
    get inherit() {
      return this.mode == 1;
    }
    sort(other) {
      if (!other || other.depth < this.depth) {
        this.next = other;
        return this;
      }
      other.next = this.sort(other.next);
      return other;
    }
    get depth() {
      return this.context ? this.context.length : 0;
    }
  };
  Rule.empty = new Rule([], 2, null);
  function tagHighlighter(tags2, options) {
    let map = /* @__PURE__ */ Object.create(null);
    for (let style of tags2) {
      if (!Array.isArray(style.tag))
        map[style.tag.id] = style.class;
      else
        for (let tag of style.tag)
          map[tag.id] = style.class;
    }
    let { scope, all = null } = options || {};
    return {
      style: (tags3) => {
        let cls = all;
        for (let tag of tags3) {
          for (let sub of tag.set) {
            let tagClass = map[sub.id];
            if (tagClass) {
              cls = cls ? cls + " " + tagClass : tagClass;
              break;
            }
          }
        }
        return cls;
      },
      scope
    };
  }
  var t = Tag.define;
  var comment = t();
  var name = t();
  var typeName = t(name);
  var propertyName = t(name);
  var literal = t();
  var string = t(literal);
  var number = t(literal);
  var content = t();
  var heading = t(content);
  var keyword = t();
  var operator = t();
  var punctuation = t();
  var bracket = t(punctuation);
  var meta = t();
  var tags = {
    /**
    A comment.
    */
    comment,
    /**
    A line [comment](#highlight.tags.comment).
    */
    lineComment: t(comment),
    /**
    A block [comment](#highlight.tags.comment).
    */
    blockComment: t(comment),
    /**
    A documentation [comment](#highlight.tags.comment).
    */
    docComment: t(comment),
    /**
    Any kind of identifier.
    */
    name,
    /**
    The [name](#highlight.tags.name) of a variable.
    */
    variableName: t(name),
    /**
    A type [name](#highlight.tags.name).
    */
    typeName,
    /**
    A tag name (subtag of [`typeName`](#highlight.tags.typeName)).
    */
    tagName: t(typeName),
    /**
    A property or field [name](#highlight.tags.name).
    */
    propertyName,
    /**
    An attribute name (subtag of [`propertyName`](#highlight.tags.propertyName)).
    */
    attributeName: t(propertyName),
    /**
    The [name](#highlight.tags.name) of a class.
    */
    className: t(name),
    /**
    A label [name](#highlight.tags.name).
    */
    labelName: t(name),
    /**
    A namespace [name](#highlight.tags.name).
    */
    namespace: t(name),
    /**
    The [name](#highlight.tags.name) of a macro.
    */
    macroName: t(name),
    /**
    A literal value.
    */
    literal,
    /**
    A string [literal](#highlight.tags.literal).
    */
    string,
    /**
    A documentation [string](#highlight.tags.string).
    */
    docString: t(string),
    /**
    A character literal (subtag of [string](#highlight.tags.string)).
    */
    character: t(string),
    /**
    An attribute value (subtag of [string](#highlight.tags.string)).
    */
    attributeValue: t(string),
    /**
    A number [literal](#highlight.tags.literal).
    */
    number,
    /**
    An integer [number](#highlight.tags.number) literal.
    */
    integer: t(number),
    /**
    A floating-point [number](#highlight.tags.number) literal.
    */
    float: t(number),
    /**
    A boolean [literal](#highlight.tags.literal).
    */
    bool: t(literal),
    /**
    Regular expression [literal](#highlight.tags.literal).
    */
    regexp: t(literal),
    /**
    An escape [literal](#highlight.tags.literal), for example a
    backslash escape in a string.
    */
    escape: t(literal),
    /**
    A color [literal](#highlight.tags.literal).
    */
    color: t(literal),
    /**
    A URL [literal](#highlight.tags.literal).
    */
    url: t(literal),
    /**
    A language keyword.
    */
    keyword,
    /**
    The [keyword](#highlight.tags.keyword) for the self or this
    object.
    */
    self: t(keyword),
    /**
    The [keyword](#highlight.tags.keyword) for null.
    */
    null: t(keyword),
    /**
    A [keyword](#highlight.tags.keyword) denoting some atomic value.
    */
    atom: t(keyword),
    /**
    A [keyword](#highlight.tags.keyword) that represents a unit.
    */
    unit: t(keyword),
    /**
    A modifier [keyword](#highlight.tags.keyword).
    */
    modifier: t(keyword),
    /**
    A [keyword](#highlight.tags.keyword) that acts as an operator.
    */
    operatorKeyword: t(keyword),
    /**
    A control-flow related [keyword](#highlight.tags.keyword).
    */
    controlKeyword: t(keyword),
    /**
    A [keyword](#highlight.tags.keyword) that defines something.
    */
    definitionKeyword: t(keyword),
    /**
    A [keyword](#highlight.tags.keyword) related to defining or
    interfacing with modules.
    */
    moduleKeyword: t(keyword),
    /**
    An operator.
    */
    operator,
    /**
    An [operator](#highlight.tags.operator) that dereferences something.
    */
    derefOperator: t(operator),
    /**
    Arithmetic-related [operator](#highlight.tags.operator).
    */
    arithmeticOperator: t(operator),
    /**
    Logical [operator](#highlight.tags.operator).
    */
    logicOperator: t(operator),
    /**
    Bit [operator](#highlight.tags.operator).
    */
    bitwiseOperator: t(operator),
    /**
    Comparison [operator](#highlight.tags.operator).
    */
    compareOperator: t(operator),
    /**
    [Operator](#highlight.tags.operator) that updates its operand.
    */
    updateOperator: t(operator),
    /**
    [Operator](#highlight.tags.operator) that defines something.
    */
    definitionOperator: t(operator),
    /**
    Type-related [operator](#highlight.tags.operator).
    */
    typeOperator: t(operator),
    /**
    Control-flow [operator](#highlight.tags.operator).
    */
    controlOperator: t(operator),
    /**
    Program or markup punctuation.
    */
    punctuation,
    /**
    [Punctuation](#highlight.tags.punctuation) that separates
    things.
    */
    separator: t(punctuation),
    /**
    Bracket-style [punctuation](#highlight.tags.punctuation).
    */
    bracket,
    /**
    Angle [brackets](#highlight.tags.bracket) (usually `<` and `>`
    tokens).
    */
    angleBracket: t(bracket),
    /**
    Square [brackets](#highlight.tags.bracket) (usually `[` and `]`
    tokens).
    */
    squareBracket: t(bracket),
    /**
    Parentheses (usually `(` and `)` tokens). Subtag of
    [bracket](#highlight.tags.bracket).
    */
    paren: t(bracket),
    /**
    Braces (usually `{` and `}` tokens). Subtag of
    [bracket](#highlight.tags.bracket).
    */
    brace: t(bracket),
    /**
    Content, for example plain text in XML or markup documents.
    */
    content,
    /**
    [Content](#highlight.tags.content) that represents a heading.
    */
    heading,
    /**
    A level 1 [heading](#highlight.tags.heading).
    */
    heading1: t(heading),
    /**
    A level 2 [heading](#highlight.tags.heading).
    */
    heading2: t(heading),
    /**
    A level 3 [heading](#highlight.tags.heading).
    */
    heading3: t(heading),
    /**
    A level 4 [heading](#highlight.tags.heading).
    */
    heading4: t(heading),
    /**
    A level 5 [heading](#highlight.tags.heading).
    */
    heading5: t(heading),
    /**
    A level 6 [heading](#highlight.tags.heading).
    */
    heading6: t(heading),
    /**
    A prose [content](#highlight.tags.content) separator (such as a horizontal rule).
    */
    contentSeparator: t(content),
    /**
    [Content](#highlight.tags.content) that represents a list.
    */
    list: t(content),
    /**
    [Content](#highlight.tags.content) that represents a quote.
    */
    quote: t(content),
    /**
    [Content](#highlight.tags.content) that is emphasized.
    */
    emphasis: t(content),
    /**
    [Content](#highlight.tags.content) that is styled strong.
    */
    strong: t(content),
    /**
    [Content](#highlight.tags.content) that is part of a link.
    */
    link: t(content),
    /**
    [Content](#highlight.tags.content) that is styled as code or
    monospace.
    */
    monospace: t(content),
    /**
    [Content](#highlight.tags.content) that has a strike-through
    style.
    */
    strikethrough: t(content),
    /**
    Inserted text in a change-tracking format.
    */
    inserted: t(),
    /**
    Deleted text.
    */
    deleted: t(),
    /**
    Changed text.
    */
    changed: t(),
    /**
    An invalid or unsyntactic element.
    */
    invalid: t(),
    /**
    Metadata or meta-instruction.
    */
    meta,
    /**
    [Metadata](#highlight.tags.meta) that applies to the entire
    document.
    */
    documentMeta: t(meta),
    /**
    [Metadata](#highlight.tags.meta) that annotates or adds
    attributes to a given syntactic element.
    */
    annotation: t(meta),
    /**
    Processing instruction or preprocessor directive. Subtag of
    [meta](#highlight.tags.meta).
    */
    processingInstruction: t(meta),
    /**
    [Modifier](#highlight.Tag^defineModifier) that indicates that a
    given element is being defined. Expected to be used with the
    various [name](#highlight.tags.name) tags.
    */
    definition: Tag.defineModifier("definition"),
    /**
    [Modifier](#highlight.Tag^defineModifier) that indicates that
    something is constant. Mostly expected to be used with
    [variable names](#highlight.tags.variableName).
    */
    constant: Tag.defineModifier("constant"),
    /**
    [Modifier](#highlight.Tag^defineModifier) used to indicate that
    a [variable](#highlight.tags.variableName) or [property
    name](#highlight.tags.propertyName) is being called or defined
    as a function.
    */
    function: Tag.defineModifier("function"),
    /**
    [Modifier](#highlight.Tag^defineModifier) that can be applied to
    [names](#highlight.tags.name) to indicate that they belong to
    the language's standard environment.
    */
    standard: Tag.defineModifier("standard"),
    /**
    [Modifier](#highlight.Tag^defineModifier) that indicates a given
    [names](#highlight.tags.name) is local to some scope.
    */
    local: Tag.defineModifier("local"),
    /**
    A generic variant [modifier](#highlight.Tag^defineModifier) that
    can be used to tag language-specific alternative variants of
    some common tag. It is recommended for themes to define special
    forms of at least the [string](#highlight.tags.string) and
    [variable name](#highlight.tags.variableName) tags, since those
    come up a lot.
    */
    special: Tag.defineModifier("special")
  };
  for (let name2 in tags) {
    let val = tags[name2];
    if (val instanceof Tag)
      val.name = name2;
  }
  var classHighlighter = tagHighlighter([
    { tag: tags.link, class: "tok-link" },
    { tag: tags.heading, class: "tok-heading" },
    { tag: tags.emphasis, class: "tok-emphasis" },
    { tag: tags.strong, class: "tok-strong" },
    { tag: tags.keyword, class: "tok-keyword" },
    { tag: tags.atom, class: "tok-atom" },
    { tag: tags.bool, class: "tok-bool" },
    { tag: tags.url, class: "tok-url" },
    { tag: tags.labelName, class: "tok-labelName" },
    { tag: tags.inserted, class: "tok-inserted" },
    { tag: tags.deleted, class: "tok-deleted" },
    { tag: tags.literal, class: "tok-literal" },
    { tag: tags.string, class: "tok-string" },
    { tag: tags.number, class: "tok-number" },
    { tag: [tags.regexp, tags.escape, tags.special(tags.string)], class: "tok-string2" },
    { tag: tags.variableName, class: "tok-variableName" },
    { tag: tags.local(tags.variableName), class: "tok-variableName tok-local" },
    { tag: tags.definition(tags.variableName), class: "tok-variableName tok-definition" },
    { tag: tags.special(tags.variableName), class: "tok-variableName2" },
    { tag: tags.definition(tags.propertyName), class: "tok-propertyName tok-definition" },
    { tag: tags.typeName, class: "tok-typeName" },
    { tag: tags.namespace, class: "tok-namespace" },
    { tag: tags.className, class: "tok-className" },
    { tag: tags.macroName, class: "tok-macroName" },
    { tag: tags.propertyName, class: "tok-propertyName" },
    { tag: tags.operator, class: "tok-operator" },
    { tag: tags.comment, class: "tok-comment" },
    { tag: tags.meta, class: "tok-meta" },
    { tag: tags.invalid, class: "tok-invalid" },
    { tag: tags.punctuation, class: "tok-punctuation" }
  ]);

  // node_modules/@codemirror/language/dist/index.js
  var _a;
  var languageDataProp = /* @__PURE__ */ new NodeProp();
  var sublanguageProp = /* @__PURE__ */ new NodeProp();
  var Language = class {
    /**
    Construct a language object. If you need to invoke this
    directly, first define a data facet with
    [`defineLanguageFacet`](https://codemirror.net/6/docs/ref/#language.defineLanguageFacet), and then
    configure your parser to [attach](https://codemirror.net/6/docs/ref/#language.languageDataProp) it
    to the language's outer syntax node.
    */
    constructor(data, parser, extraExtensions = [], name2 = "") {
      this.data = data;
      this.name = name2;
      if (!EditorState.prototype.hasOwnProperty("tree"))
        Object.defineProperty(EditorState.prototype, "tree", { get() {
          return syntaxTree(this);
        } });
      this.parser = parser;
      this.extension = [
        language.of(this),
        EditorState.languageData.of((state, pos, side) => {
          let top2 = topNodeAt(state, pos, side), data2 = top2.type.prop(languageDataProp);
          if (!data2)
            return [];
          let base2 = state.facet(data2), sub = top2.type.prop(sublanguageProp);
          if (sub) {
            let innerNode = top2.resolve(pos - top2.from, side);
            for (let sublang of sub)
              if (sublang.test(innerNode, state)) {
                let data3 = state.facet(sublang.facet);
                return sublang.type == "replace" ? data3 : data3.concat(base2);
              }
          }
          return base2;
        })
      ].concat(extraExtensions);
    }
    /**
    Query whether this language is active at the given position.
    */
    isActiveAt(state, pos, side = -1) {
      return topNodeAt(state, pos, side).type.prop(languageDataProp) == this.data;
    }
    /**
    Find the document regions that were parsed using this language.
    The returned regions will _include_ any nested languages rooted
    in this language, when those exist.
    */
    findRegions(state) {
      let lang = state.facet(language);
      if ((lang === null || lang === void 0 ? void 0 : lang.data) == this.data)
        return [{ from: 0, to: state.doc.length }];
      if (!lang || !lang.allowsNesting)
        return [];
      let result = [];
      let explore = (tree, from) => {
        if (tree.prop(languageDataProp) == this.data) {
          result.push({ from, to: from + tree.length });
          return;
        }
        let mount = tree.prop(NodeProp.mounted);
        if (mount) {
          if (mount.tree.prop(languageDataProp) == this.data) {
            if (mount.overlay)
              for (let r of mount.overlay)
                result.push({ from: r.from + from, to: r.to + from });
            else
              result.push({ from, to: from + tree.length });
            return;
          } else if (mount.overlay) {
            let size = result.length;
            explore(mount.tree, mount.overlay[0].from + from);
            if (result.length > size)
              return;
          }
        }
        for (let i = 0; i < tree.children.length; i++) {
          let ch = tree.children[i];
          if (ch instanceof Tree)
            explore(ch, tree.positions[i] + from);
        }
      };
      explore(syntaxTree(state), 0);
      return result;
    }
    /**
    Indicates whether this language allows nested languages. The
    default implementation returns true.
    */
    get allowsNesting() {
      return true;
    }
  };
  Language.setState = /* @__PURE__ */ StateEffect.define();
  function topNodeAt(state, pos, side) {
    let topLang = state.facet(language), tree = syntaxTree(state).topNode;
    if (!topLang || topLang.allowsNesting) {
      for (let node = tree; node; node = node.enter(pos, side, IterMode.ExcludeBuffers | IterMode.EnterBracketed))
        if (node.type.isTop)
          tree = node;
    }
    return tree;
  }
  function syntaxTree(state) {
    let field = state.field(Language.state, false);
    return field ? field.tree : Tree.empty;
  }
  var DocInput = class {
    /**
    Create an input object for the given document.
    */
    constructor(doc2) {
      this.doc = doc2;
      this.cursorPos = 0;
      this.string = "";
      this.cursor = doc2.iter();
    }
    get length() {
      return this.doc.length;
    }
    syncTo(pos) {
      this.string = this.cursor.next(pos - this.cursorPos).value;
      this.cursorPos = pos + this.string.length;
      return this.cursorPos - this.string.length;
    }
    chunk(pos) {
      this.syncTo(pos);
      return this.string;
    }
    get lineChunks() {
      return true;
    }
    read(from, to) {
      let stringStart = this.cursorPos - this.string.length;
      if (from < stringStart || to >= this.cursorPos)
        return this.doc.sliceString(from, to);
      else
        return this.string.slice(from - stringStart, to - stringStart);
    }
  };
  var currentContext = null;
  var ParseContext = class _ParseContext {
    constructor(parser, state, fragments = [], tree, treeLen, viewport, skipped, scheduleOn) {
      this.parser = parser;
      this.state = state;
      this.fragments = fragments;
      this.tree = tree;
      this.treeLen = treeLen;
      this.viewport = viewport;
      this.skipped = skipped;
      this.scheduleOn = scheduleOn;
      this.parse = null;
      this.tempSkipped = [];
    }
    /**
    @internal
    */
    static create(parser, state, viewport) {
      return new _ParseContext(parser, state, [], Tree.empty, 0, viewport, [], null);
    }
    startParse() {
      return this.parser.startParse(new DocInput(this.state.doc), this.fragments);
    }
    /**
    @internal
    */
    work(until, upto) {
      if (upto != null && upto >= this.state.doc.length)
        upto = void 0;
      if (this.tree != Tree.empty && this.isDone(upto !== null && upto !== void 0 ? upto : this.state.doc.length)) {
        this.takeTree();
        return true;
      }
      return this.withContext(() => {
        var _a2;
        if (typeof until == "number") {
          let endTime = Date.now() + until;
          until = () => Date.now() > endTime;
        }
        if (!this.parse)
          this.parse = this.startParse();
        if (upto != null && (this.parse.stoppedAt == null || this.parse.stoppedAt > upto) && upto < this.state.doc.length)
          this.parse.stopAt(upto);
        for (; ; ) {
          let done = this.parse.advance();
          if (done) {
            this.fragments = this.withoutTempSkipped(TreeFragment.addTree(done, this.fragments, this.parse.stoppedAt != null));
            this.treeLen = (_a2 = this.parse.stoppedAt) !== null && _a2 !== void 0 ? _a2 : this.state.doc.length;
            this.tree = done;
            this.parse = null;
            if (this.treeLen < (upto !== null && upto !== void 0 ? upto : this.state.doc.length))
              this.parse = this.startParse();
            else
              return true;
          }
          if (until())
            return false;
        }
      });
    }
    /**
    @internal
    */
    takeTree() {
      let pos, tree;
      if (this.parse && (pos = this.parse.parsedPos) >= this.treeLen) {
        if (this.parse.stoppedAt == null || this.parse.stoppedAt > pos)
          this.parse.stopAt(pos);
        this.withContext(() => {
          while (!(tree = this.parse.advance())) {
          }
        });
        this.treeLen = pos;
        this.tree = tree;
        this.fragments = this.withoutTempSkipped(TreeFragment.addTree(this.tree, this.fragments, true));
        this.parse = null;
      }
    }
    withContext(f) {
      let prev = currentContext;
      currentContext = this;
      try {
        return f();
      } finally {
        currentContext = prev;
      }
    }
    withoutTempSkipped(fragments) {
      for (let r; r = this.tempSkipped.pop(); )
        fragments = cutFragments(fragments, r.from, r.to);
      return fragments;
    }
    /**
    @internal
    */
    changes(changes, newState) {
      let { fragments, tree, treeLen, viewport, skipped } = this;
      this.takeTree();
      if (!changes.empty) {
        let ranges = [];
        changes.iterChangedRanges((fromA, toA, fromB, toB) => ranges.push({ fromA, toA, fromB, toB }));
        fragments = TreeFragment.applyChanges(fragments, ranges);
        tree = Tree.empty;
        treeLen = 0;
        viewport = { from: changes.mapPos(viewport.from, -1), to: changes.mapPos(viewport.to, 1) };
        if (this.skipped.length) {
          skipped = [];
          for (let r of this.skipped) {
            let from = changes.mapPos(r.from, 1), to = changes.mapPos(r.to, -1);
            if (from < to)
              skipped.push({ from, to });
          }
        }
      }
      return new _ParseContext(this.parser, newState, fragments, tree, treeLen, viewport, skipped, this.scheduleOn);
    }
    /**
    @internal
    */
    updateViewport(viewport) {
      if (this.viewport.from == viewport.from && this.viewport.to == viewport.to)
        return false;
      this.viewport = viewport;
      let startLen = this.skipped.length;
      for (let i = 0; i < this.skipped.length; i++) {
        let { from, to } = this.skipped[i];
        if (from < viewport.to && to > viewport.from) {
          this.fragments = cutFragments(this.fragments, from, to);
          this.skipped.splice(i--, 1);
        }
      }
      if (this.skipped.length >= startLen)
        return false;
      this.reset();
      return true;
    }
    /**
    @internal
    */
    reset() {
      if (this.parse) {
        this.takeTree();
        this.parse = null;
      }
    }
    /**
    Notify the parse scheduler that the given region was skipped
    because it wasn't in view, and the parse should be restarted
    when it comes into view.
    */
    skipUntilInView(from, to) {
      this.skipped.push({ from, to });
    }
    /**
    Returns a parser intended to be used as placeholder when
    asynchronously loading a nested parser. It'll skip its input and
    mark it as not-really-parsed, so that the next update will parse
    it again.
    
    When `until` is given, a reparse will be scheduled when that
    promise resolves.
    */
    static getSkippingParser(until) {
      return new class extends Parser {
        createParse(input, fragments, ranges) {
          let from = ranges[0].from, to = ranges[ranges.length - 1].to;
          let parser = {
            parsedPos: from,
            advance() {
              let cx = currentContext;
              if (cx) {
                for (let r of ranges)
                  cx.tempSkipped.push(r);
                if (until)
                  cx.scheduleOn = cx.scheduleOn ? Promise.all([cx.scheduleOn, until]) : until;
              }
              this.parsedPos = to;
              return new Tree(NodeType.none, [], [], to - from);
            },
            stoppedAt: null,
            stopAt() {
            }
          };
          return parser;
        }
      }();
    }
    /**
    @internal
    */
    isDone(upto) {
      upto = Math.min(upto, this.state.doc.length);
      let frags = this.fragments;
      return this.treeLen >= upto && frags.length && frags[0].from == 0 && frags[0].to >= upto;
    }
    /**
    Get the context for the current parse, or `null` if no editor
    parse is in progress.
    */
    static get() {
      return currentContext;
    }
  };
  function cutFragments(fragments, from, to) {
    return TreeFragment.applyChanges(fragments, [{ fromA: from, toA: to, fromB: from, toB: to }]);
  }
  var LanguageState = class _LanguageState {
    constructor(context) {
      this.context = context;
      this.tree = context.tree;
    }
    apply(tr) {
      if (!tr.docChanged && this.tree == this.context.tree)
        return this;
      let newCx = this.context.changes(tr.changes, tr.state);
      let upto = this.context.treeLen == tr.startState.doc.length ? void 0 : Math.max(tr.changes.mapPos(this.context.treeLen), newCx.viewport.to);
      if (!newCx.work(20, upto))
        newCx.takeTree();
      return new _LanguageState(newCx);
    }
    static init(state) {
      let vpTo = Math.min(3e3, state.doc.length);
      let parseState = ParseContext.create(state.facet(language).parser, state, { from: 0, to: vpTo });
      if (!parseState.work(20, vpTo))
        parseState.takeTree();
      return new _LanguageState(parseState);
    }
  };
  Language.state = /* @__PURE__ */ StateField.define({
    create: LanguageState.init,
    update(value, tr) {
      for (let e of tr.effects)
        if (e.is(Language.setState))
          return e.value;
      if (tr.startState.facet(language) != tr.state.facet(language))
        return LanguageState.init(tr.state);
      return value.apply(tr);
    }
  });
  var requestIdle = (callback) => {
    let timeout = setTimeout(
      () => callback(),
      500
      /* Work.MaxPause */
    );
    return () => clearTimeout(timeout);
  };
  if (typeof requestIdleCallback != "undefined")
    requestIdle = (callback) => {
      let idle = -1, timeout = setTimeout(
        () => {
          idle = requestIdleCallback(callback, {
            timeout: 500 - 100
            /* Work.MinPause */
          });
        },
        100
        /* Work.MinPause */
      );
      return () => idle < 0 ? clearTimeout(timeout) : cancelIdleCallback(idle);
    };
  var isInputPending = typeof navigator != "undefined" && ((_a = navigator.scheduling) === null || _a === void 0 ? void 0 : _a.isInputPending) ? () => navigator.scheduling.isInputPending() : null;
  var parseWorker = /* @__PURE__ */ ViewPlugin.fromClass(class ParseWorker {
    constructor(view) {
      this.view = view;
      this.working = null;
      this.workScheduled = 0;
      this.chunkEnd = -1;
      this.chunkBudget = -1;
      this.work = this.work.bind(this);
      this.scheduleWork();
    }
    update(update) {
      let cx = this.view.state.field(Language.state).context;
      if (cx.updateViewport(update.view.viewport) || this.view.viewport.to > cx.treeLen)
        this.scheduleWork();
      if (update.docChanged || update.selectionSet) {
        if (this.view.hasFocus)
          this.chunkBudget += 50;
        this.scheduleWork();
      }
      this.checkAsyncSchedule(cx);
    }
    scheduleWork() {
      if (this.working)
        return;
      let { state } = this.view, field = state.field(Language.state);
      if (field.tree != field.context.tree || !field.context.isDone(state.doc.length))
        this.working = requestIdle(this.work);
    }
    work(deadline) {
      this.working = null;
      let now = Date.now();
      if (this.chunkEnd < now && (this.chunkEnd < 0 || this.view.hasFocus)) {
        this.chunkEnd = now + 3e4;
        this.chunkBudget = 3e3;
      }
      if (this.chunkBudget <= 0)
        return;
      let { state, viewport: { to: vpTo } } = this.view, field = state.field(Language.state);
      if (field.tree == field.context.tree && field.context.isDone(
        vpTo + 1e5
        /* Work.MaxParseAhead */
      ))
        return;
      let endTime = Date.now() + Math.min(this.chunkBudget, 100, deadline && !isInputPending ? Math.max(25, deadline.timeRemaining() - 5) : 1e9);
      let viewportFirst = field.context.treeLen < vpTo && state.doc.length > vpTo + 1e3;
      let done = field.context.work(() => {
        return isInputPending && isInputPending() || Date.now() > endTime;
      }, vpTo + (viewportFirst ? 0 : 1e5));
      this.chunkBudget -= Date.now() - now;
      if (done || this.chunkBudget <= 0) {
        field.context.takeTree();
        this.view.dispatch({ effects: Language.setState.of(new LanguageState(field.context)) });
      }
      if (this.chunkBudget > 0 && !(done && !viewportFirst))
        this.scheduleWork();
      this.checkAsyncSchedule(field.context);
    }
    checkAsyncSchedule(cx) {
      if (cx.scheduleOn) {
        this.workScheduled++;
        cx.scheduleOn.then(() => this.scheduleWork()).catch((err) => logException(this.view.state, err)).then(() => this.workScheduled--);
        cx.scheduleOn = null;
      }
    }
    destroy() {
      if (this.working)
        this.working();
    }
    isWorking() {
      return !!(this.working || this.workScheduled > 0);
    }
  }, {
    eventHandlers: { focus() {
      this.scheduleWork();
    } }
  });
  var language = /* @__PURE__ */ Facet.define({
    combine(languages) {
      return languages.length ? languages[0] : null;
    },
    enables: (language2) => [
      Language.state,
      parseWorker,
      EditorView.contentAttributes.compute([language2], (state) => {
        let lang = state.facet(language2);
        return lang && lang.name ? { "data-language": lang.name } : {};
      })
    ]
  });
  var indentService = /* @__PURE__ */ Facet.define();
  var indentUnit = /* @__PURE__ */ Facet.define({
    combine: (values) => {
      if (!values.length)
        return "  ";
      let unit = values[0];
      if (!unit || /\S/.test(unit) || Array.from(unit).some((e) => e != unit[0]))
        throw new Error("Invalid indent unit: " + JSON.stringify(values[0]));
      return unit;
    }
  });
  function getIndentUnit(state) {
    let unit = state.facet(indentUnit);
    return unit.charCodeAt(0) == 9 ? state.tabSize * unit.length : unit.length;
  }
  function indentString(state, cols) {
    let result = "", ts = state.tabSize, ch = state.facet(indentUnit)[0];
    if (ch == "	") {
      while (cols >= ts) {
        result += "	";
        cols -= ts;
      }
      ch = " ";
    }
    for (let i = 0; i < cols; i++)
      result += ch;
    return result;
  }
  function getIndentation(context, pos) {
    if (context instanceof EditorState)
      context = new IndentContext(context);
    for (let service of context.state.facet(indentService)) {
      let result = service(context, pos);
      if (result !== void 0)
        return result;
    }
    let tree = syntaxTree(context.state);
    return tree.length >= pos ? syntaxIndentation(context, tree, pos) : null;
  }
  var IndentContext = class {
    /**
    Create an indent context.
    */
    constructor(state, options = {}) {
      this.state = state;
      this.options = options;
      this.unit = getIndentUnit(state);
    }
    /**
    Get a description of the line at the given position, taking
    [simulated line
    breaks](https://codemirror.net/6/docs/ref/#language.IndentContext.constructor^options.simulateBreak)
    into account. If there is such a break at `pos`, the `bias`
    argument determines whether the part of the line line before or
    after the break is used.
    */
    lineAt(pos, bias = 1) {
      let line = this.state.doc.lineAt(pos);
      let { simulateBreak, simulateDoubleBreak } = this.options;
      if (simulateBreak != null && simulateBreak >= line.from && simulateBreak <= line.to) {
        if (simulateDoubleBreak && simulateBreak == pos)
          return { text: "", from: pos };
        else if (bias < 0 ? simulateBreak < pos : simulateBreak <= pos)
          return { text: line.text.slice(simulateBreak - line.from), from: simulateBreak };
        else
          return { text: line.text.slice(0, simulateBreak - line.from), from: line.from };
      }
      return line;
    }
    /**
    Get the text directly after `pos`, either the entire line
    or the next 100 characters, whichever is shorter.
    */
    textAfterPos(pos, bias = 1) {
      if (this.options.simulateDoubleBreak && pos == this.options.simulateBreak)
        return "";
      let { text, from } = this.lineAt(pos, bias);
      return text.slice(pos - from, Math.min(text.length, pos + 100 - from));
    }
    /**
    Find the column for the given position.
    */
    column(pos, bias = 1) {
      let { text, from } = this.lineAt(pos, bias);
      let result = this.countColumn(text, pos - from);
      let override = this.options.overrideIndentation ? this.options.overrideIndentation(from) : -1;
      if (override > -1)
        result += override - this.countColumn(text, text.search(/\S|$/));
      return result;
    }
    /**
    Find the column position (taking tabs into account) of the given
    position in the given string.
    */
    countColumn(line, pos = line.length) {
      return countColumn(line, this.state.tabSize, pos);
    }
    /**
    Find the indentation column of the line at the given point.
    */
    lineIndent(pos, bias = 1) {
      let { text, from } = this.lineAt(pos, bias);
      let override = this.options.overrideIndentation;
      if (override) {
        let overriden = override(from);
        if (overriden > -1)
          return overriden;
      }
      return this.countColumn(text, text.search(/\S|$/));
    }
    /**
    Returns the [simulated line
    break](https://codemirror.net/6/docs/ref/#language.IndentContext.constructor^options.simulateBreak)
    for this context, if any.
    */
    get simulatedBreak() {
      return this.options.simulateBreak || null;
    }
  };
  var indentNodeProp = /* @__PURE__ */ new NodeProp();
  function syntaxIndentation(cx, ast, pos) {
    let stack = ast.resolveStack(pos);
    let inner = ast.resolveInner(pos, -1).resolve(pos, 0).enterUnfinishedNodesBefore(pos);
    if (inner != stack.node) {
      let add2 = [];
      for (let cur2 = inner; cur2 && !(cur2.from < stack.node.from || cur2.to > stack.node.to || cur2.from == stack.node.from && cur2.type == stack.node.type); cur2 = cur2.parent)
        add2.push(cur2);
      for (let i = add2.length - 1; i >= 0; i--)
        stack = { node: add2[i], next: stack };
    }
    return indentFor(stack, cx, pos);
  }
  function indentFor(stack, cx, pos) {
    for (let cur2 = stack; cur2; cur2 = cur2.next) {
      let strategy = indentStrategy(cur2.node);
      if (strategy)
        return strategy(TreeIndentContext.create(cx, pos, cur2));
    }
    return 0;
  }
  function ignoreClosed(cx) {
    return cx.pos == cx.options.simulateBreak && cx.options.simulateDoubleBreak;
  }
  function indentStrategy(tree) {
    let strategy = tree.type.prop(indentNodeProp);
    if (strategy)
      return strategy;
    let first = tree.firstChild, close;
    if (first && (close = first.type.prop(NodeProp.closedBy))) {
      let last = tree.lastChild, closed = last && close.indexOf(last.name) > -1;
      return (cx) => delimitedStrategy(cx, true, 1, void 0, closed && !ignoreClosed(cx) ? last.from : void 0);
    }
    return tree.parent == null ? topIndent : null;
  }
  function topIndent() {
    return 0;
  }
  var TreeIndentContext = class _TreeIndentContext extends IndentContext {
    constructor(base2, pos, context) {
      super(base2.state, base2.options);
      this.base = base2;
      this.pos = pos;
      this.context = context;
    }
    /**
    The syntax tree node to which the indentation strategy
    applies.
    */
    get node() {
      return this.context.node;
    }
    /**
    @internal
    */
    static create(base2, pos, context) {
      return new _TreeIndentContext(base2, pos, context);
    }
    /**
    Get the text directly after `this.pos`, either the entire line
    or the next 100 characters, whichever is shorter.
    */
    get textAfter() {
      return this.textAfterPos(this.pos);
    }
    /**
    Get the indentation at the reference line for `this.node`, which
    is the line on which it starts, unless there is a node that is
    _not_ a parent of this node covering the start of that line. If
    so, the line at the start of that node is tried, again skipping
    on if it is covered by another such node.
    */
    get baseIndent() {
      return this.baseIndentFor(this.node);
    }
    /**
    Get the indentation for the reference line of the given node
    (see [`baseIndent`](https://codemirror.net/6/docs/ref/#language.TreeIndentContext.baseIndent)).
    */
    baseIndentFor(node) {
      let line = this.state.doc.lineAt(node.from);
      for (; ; ) {
        let atBreak = node.resolve(line.from);
        while (atBreak.parent && atBreak.parent.from == atBreak.from)
          atBreak = atBreak.parent;
        if (isParent(atBreak, node))
          break;
        line = this.state.doc.lineAt(atBreak.from);
      }
      return this.lineIndent(line.from);
    }
    /**
    Continue looking for indentations in the node's parent nodes,
    and return the result of that.
    */
    continue() {
      return indentFor(this.context.next, this.base, this.pos);
    }
  };
  function isParent(parent, of) {
    for (let cur2 = of; cur2; cur2 = cur2.parent)
      if (parent == cur2)
        return true;
    return false;
  }
  function bracketedAligned(context) {
    let tree = context.node;
    let openToken = tree.childAfter(tree.from), last = tree.lastChild;
    if (!openToken)
      return null;
    let sim = context.options.simulateBreak;
    let openLine = context.state.doc.lineAt(openToken.from);
    let lineEnd = sim == null || sim <= openLine.from ? openLine.to : Math.min(openLine.to, sim);
    for (let pos = openToken.to; ; ) {
      let next = tree.childAfter(pos);
      if (!next || next == last)
        return null;
      if (!next.type.isSkipped) {
        if (next.from >= lineEnd)
          return null;
        let space = /^ */.exec(openLine.text.slice(openToken.to - openLine.from))[0].length;
        return { from: openToken.from, to: openToken.to + space };
      }
      pos = next.to;
    }
  }
  function delimitedStrategy(context, align, units, closing, closedAt) {
    let after = context.textAfter, space = after.match(/^\s*/)[0].length;
    let closed = closing && after.slice(space, space + closing.length) == closing || closedAt == context.pos + space;
    let aligned = align ? bracketedAligned(context) : null;
    if (aligned)
      return closed ? context.column(aligned.from) : context.column(aligned.to);
    return context.baseIndent + (closed ? 0 : context.unit * units);
  }
  var HighlightStyle = class _HighlightStyle {
    constructor(specs, options) {
      this.specs = specs;
      let modSpec;
      function def(spec) {
        let cls = StyleModule.newName();
        (modSpec || (modSpec = /* @__PURE__ */ Object.create(null)))["." + cls] = spec;
        return cls;
      }
      const all = typeof options.all == "string" ? options.all : options.all ? def(options.all) : void 0;
      const scopeOpt = options.scope;
      this.scope = scopeOpt instanceof Language ? (type) => type.prop(languageDataProp) == scopeOpt.data : scopeOpt ? (type) => type == scopeOpt : void 0;
      this.style = tagHighlighter(specs.map((style) => ({
        tag: style.tag,
        class: style.class || def(Object.assign({}, style, { tag: null }))
      })), {
        all
      }).style;
      this.module = modSpec ? new StyleModule(modSpec) : null;
      this.themeType = options.themeType;
    }
    /**
    Create a highlighter style that associates the given styles to
    the given tags. The specs must be objects that hold a style tag
    or array of tags in their `tag` property, and either a single
    `class` property providing a static CSS class (for highlighter
    that rely on external styling), or a
    [`style-mod`](https://github.com/marijnh/style-mod#documentation)-style
    set of CSS properties (which define the styling for those tags).
    
    The CSS rules created for a highlighter will be emitted in the
    order of the spec's properties. That means that for elements that
    have multiple tags associated with them, styles defined further
    down in the list will have a higher CSS precedence than styles
    defined earlier.
    */
    static define(specs, options) {
      return new _HighlightStyle(specs, options || {});
    }
  };
  var defaultHighlightStyle = /* @__PURE__ */ HighlightStyle.define([
    {
      tag: tags.meta,
      color: "#404740"
    },
    {
      tag: tags.link,
      textDecoration: "underline"
    },
    {
      tag: tags.heading,
      textDecoration: "underline",
      fontWeight: "bold"
    },
    {
      tag: tags.emphasis,
      fontStyle: "italic"
    },
    {
      tag: tags.strong,
      fontWeight: "bold"
    },
    {
      tag: tags.strikethrough,
      textDecoration: "line-through"
    },
    {
      tag: tags.keyword,
      color: "#708"
    },
    {
      tag: [tags.atom, tags.bool, tags.url, tags.contentSeparator, tags.labelName],
      color: "#219"
    },
    {
      tag: [tags.literal, tags.inserted],
      color: "#164"
    },
    {
      tag: [tags.string, tags.deleted],
      color: "#a11"
    },
    {
      tag: [tags.regexp, tags.escape, /* @__PURE__ */ tags.special(tags.string)],
      color: "#e40"
    },
    {
      tag: /* @__PURE__ */ tags.definition(tags.variableName),
      color: "#00f"
    },
    {
      tag: /* @__PURE__ */ tags.local(tags.variableName),
      color: "#30a"
    },
    {
      tag: [tags.typeName, tags.namespace],
      color: "#085"
    },
    {
      tag: tags.className,
      color: "#167"
    },
    {
      tag: [/* @__PURE__ */ tags.special(tags.variableName), tags.macroName],
      color: "#256"
    },
    {
      tag: /* @__PURE__ */ tags.definition(tags.propertyName),
      color: "#00c"
    },
    {
      tag: tags.comment,
      color: "#940"
    },
    {
      tag: tags.invalid,
      color: "#f00"
    }
  ]);
  var DefaultScanDist = 1e4;
  var DefaultBrackets = "()[]{}";
  var bracketMatchingHandle = /* @__PURE__ */ new NodeProp();
  function matchingNodes(node, dir, brackets) {
    let byProp = node.prop(dir < 0 ? NodeProp.openedBy : NodeProp.closedBy);
    if (byProp)
      return byProp;
    if (node.name.length == 1) {
      let index = brackets.indexOf(node.name);
      if (index > -1 && index % 2 == (dir < 0 ? 1 : 0))
        return [brackets[index + dir]];
    }
    return null;
  }
  function findHandle(node) {
    let hasHandle = node.type.prop(bracketMatchingHandle);
    return hasHandle ? hasHandle(node.node) : node;
  }
  function matchBrackets(state, pos, dir, config = {}) {
    let maxScanDistance = config.maxScanDistance || DefaultScanDist, brackets = config.brackets || DefaultBrackets;
    let tree = syntaxTree(state), node = tree.resolveInner(pos, dir);
    for (let cur2 = node; cur2; cur2 = cur2.parent) {
      let matches = matchingNodes(cur2.type, dir, brackets);
      if (matches && cur2.from < cur2.to) {
        let handle = findHandle(cur2);
        if (handle && (dir > 0 ? pos >= handle.from && pos < handle.to : pos > handle.from && pos <= handle.to))
          return matchMarkedBrackets(state, pos, dir, cur2, handle, matches, brackets);
      }
    }
    return matchPlainBrackets(state, pos, dir, tree, node.type, maxScanDistance, brackets);
  }
  function matchMarkedBrackets(_state, _pos, dir, token, handle, matching, brackets) {
    let parent = token.parent, firstToken = { from: handle.from, to: handle.to };
    let depth = 0, cursor = parent === null || parent === void 0 ? void 0 : parent.cursor();
    if (cursor && (dir < 0 ? cursor.childBefore(token.from) : cursor.childAfter(token.to)))
      do {
        if (dir < 0 ? cursor.to <= token.from : cursor.from >= token.to) {
          if (depth == 0 && matching.indexOf(cursor.type.name) > -1 && cursor.from < cursor.to) {
            let endHandle = findHandle(cursor);
            return { start: firstToken, end: endHandle ? { from: endHandle.from, to: endHandle.to } : void 0, matched: true };
          } else if (matchingNodes(cursor.type, dir, brackets)) {
            depth++;
          } else if (matchingNodes(cursor.type, -dir, brackets)) {
            if (depth == 0) {
              let endHandle = findHandle(cursor);
              return {
                start: firstToken,
                end: endHandle && endHandle.from < endHandle.to ? { from: endHandle.from, to: endHandle.to } : void 0,
                matched: false
              };
            }
            depth--;
          }
        }
      } while (dir < 0 ? cursor.prevSibling() : cursor.nextSibling());
    return { start: firstToken, matched: false };
  }
  function matchPlainBrackets(state, pos, dir, tree, tokenType, maxScanDistance, brackets) {
    let startCh = dir < 0 ? state.sliceDoc(pos - 1, pos) : state.sliceDoc(pos, pos + 1);
    let bracket2 = brackets.indexOf(startCh);
    if (bracket2 < 0 || bracket2 % 2 == 0 != dir > 0)
      return null;
    let startToken = { from: dir < 0 ? pos - 1 : pos, to: dir > 0 ? pos + 1 : pos };
    let iter = state.doc.iterRange(pos, dir > 0 ? state.doc.length : 0), depth = 0;
    for (let distance = 0; !iter.next().done && distance <= maxScanDistance; ) {
      let text = iter.value;
      if (dir < 0)
        distance += text.length;
      let basePos = pos + distance * dir;
      for (let pos2 = dir > 0 ? 0 : text.length - 1, end = dir > 0 ? text.length : -1; pos2 != end; pos2 += dir) {
        let found = brackets.indexOf(text[pos2]);
        if (found < 0 || tree.resolveInner(basePos + pos2, 1).type != tokenType)
          continue;
        if (found % 2 == 0 == dir > 0) {
          depth++;
        } else if (depth == 1) {
          return { start: startToken, end: { from: basePos + pos2, to: basePos + pos2 + 1 }, matched: found >> 1 == bracket2 >> 1 };
        } else {
          depth--;
        }
      }
      if (dir > 0)
        distance += text.length;
    }
    return iter.done ? { start: startToken, matched: false } : null;
  }
  var noTokens = /* @__PURE__ */ Object.create(null);
  var typeArray = [NodeType.none];
  var warned = [];
  var byTag = /* @__PURE__ */ Object.create(null);
  var defaultTable = /* @__PURE__ */ Object.create(null);
  for (let [legacyName, name2] of [
    ["variable", "variableName"],
    ["variable-2", "variableName.special"],
    ["string-2", "string.special"],
    ["def", "variableName.definition"],
    ["tag", "tagName"],
    ["attribute", "attributeName"],
    ["type", "typeName"],
    ["builtin", "variableName.standard"],
    ["qualifier", "modifier"],
    ["error", "invalid"],
    ["header", "heading"],
    ["property", "propertyName"]
  ])
    defaultTable[legacyName] = /* @__PURE__ */ createTokenType(noTokens, name2);
  function warnForPart(part, msg) {
    if (warned.indexOf(part) > -1)
      return;
    warned.push(part);
    console.warn(msg);
  }
  function createTokenType(extra, tagStr) {
    let tags$1 = [];
    for (let name3 of tagStr.split(" ")) {
      let found = [];
      for (let part of name3.split(".")) {
        let value = extra[part] || tags[part];
        if (!value) {
          warnForPart(part, `Unknown highlighting tag ${part}`);
        } else if (typeof value == "function") {
          if (!found.length)
            warnForPart(part, `Modifier ${part} used at start of tag`);
          else
            found = found.map(value);
        } else {
          if (found.length)
            warnForPart(part, `Tag ${part} used as modifier`);
          else
            found = Array.isArray(value) ? value : [value];
        }
      }
      for (let tag of found)
        tags$1.push(tag);
    }
    if (!tags$1.length)
      return 0;
    let name2 = tagStr.replace(/ /g, "_"), key = name2 + " " + tags$1.map((t2) => t2.id);
    let known = byTag[key];
    if (known)
      return known.id;
    let type = byTag[key] = NodeType.define({
      id: typeArray.length,
      name: name2,
      props: [styleTags({ [name2]: tags$1 })]
    });
    typeArray.push(type);
    return type.id;
  }
  var marks = {
    rtl: /* @__PURE__ */ Decoration.mark({ class: "cm-iso", inclusive: true, attributes: { dir: "rtl" }, bidiIsolate: Direction.RTL }),
    ltr: /* @__PURE__ */ Decoration.mark({ class: "cm-iso", inclusive: true, attributes: { dir: "ltr" }, bidiIsolate: Direction.LTR }),
    auto: /* @__PURE__ */ Decoration.mark({ class: "cm-iso", inclusive: true, attributes: { dir: "auto" }, bidiIsolate: null })
  };

  // node_modules/@codemirror/commands/dist/index.js
  var toggleComment = (target) => {
    let { state } = target, line = state.doc.lineAt(state.selection.main.from), config = getConfig(target.state, line.from);
    return config.line ? toggleLineComment(target) : config.block ? toggleBlockCommentByLine(target) : false;
  };
  function command(f, option) {
    return ({ state, dispatch }) => {
      if (state.readOnly)
        return false;
      let tr = f(option, state);
      if (!tr)
        return false;
      dispatch(state.update(tr));
      return true;
    };
  }
  var toggleLineComment = /* @__PURE__ */ command(
    changeLineComment,
    0
    /* CommentOption.Toggle */
  );
  var toggleBlockComment = /* @__PURE__ */ command(
    changeBlockComment,
    0
    /* CommentOption.Toggle */
  );
  var toggleBlockCommentByLine = /* @__PURE__ */ command(
    (o, s) => changeBlockComment(o, s, selectedLineRanges(s)),
    0
    /* CommentOption.Toggle */
  );
  function getConfig(state, pos) {
    let data = state.languageDataAt("commentTokens", pos, 1);
    return data.length ? data[0] : {};
  }
  var SearchMargin = 50;
  function findBlockComment(state, { open, close }, from, to) {
    let textBefore = state.sliceDoc(from - SearchMargin, from);
    let textAfter = state.sliceDoc(to, to + SearchMargin);
    let spaceBefore = /\s*$/.exec(textBefore)[0].length, spaceAfter = /^\s*/.exec(textAfter)[0].length;
    let beforeOff = textBefore.length - spaceBefore;
    if (textBefore.slice(beforeOff - open.length, beforeOff) == open && textAfter.slice(spaceAfter, spaceAfter + close.length) == close) {
      return {
        open: { pos: from - spaceBefore, margin: spaceBefore && 1 },
        close: { pos: to + spaceAfter, margin: spaceAfter && 1 }
      };
    }
    let startText, endText;
    if (to - from <= 2 * SearchMargin) {
      startText = endText = state.sliceDoc(from, to);
    } else {
      startText = state.sliceDoc(from, from + SearchMargin);
      endText = state.sliceDoc(to - SearchMargin, to);
    }
    let startSpace = /^\s*/.exec(startText)[0].length, endSpace = /\s*$/.exec(endText)[0].length;
    let endOff = endText.length - endSpace - close.length;
    if (startText.slice(startSpace, startSpace + open.length) == open && endText.slice(endOff, endOff + close.length) == close) {
      return {
        open: {
          pos: from + startSpace + open.length,
          margin: /\s/.test(startText.charAt(startSpace + open.length)) ? 1 : 0
        },
        close: {
          pos: to - endSpace - close.length,
          margin: /\s/.test(endText.charAt(endOff - 1)) ? 1 : 0
        }
      };
    }
    return null;
  }
  function selectedLineRanges(state) {
    let ranges = [];
    for (let r of state.selection.ranges) {
      let fromLine = state.doc.lineAt(r.from);
      let toLine = r.to <= fromLine.to ? fromLine : state.doc.lineAt(r.to);
      if (toLine.from > fromLine.from && toLine.from == r.to)
        toLine = r.to == fromLine.to + 1 ? fromLine : state.doc.lineAt(r.to - 1);
      let last = ranges.length - 1;
      if (last >= 0 && ranges[last].to > fromLine.from)
        ranges[last].to = toLine.to;
      else
        ranges.push({ from: fromLine.from + /^\s*/.exec(fromLine.text)[0].length, to: toLine.to });
    }
    return ranges;
  }
  function changeBlockComment(option, state, ranges = state.selection.ranges) {
    let tokens = ranges.map((r) => getConfig(state, r.from).block);
    if (!tokens.every((c) => c))
      return null;
    let comments = ranges.map((r, i) => findBlockComment(state, tokens[i], r.from, r.to));
    if (option != 2 && !comments.every((c) => c)) {
      return { changes: state.changes(ranges.map((range, i) => {
        if (comments[i])
          return [];
        return [{ from: range.from, insert: tokens[i].open + " " }, { from: range.to, insert: " " + tokens[i].close }];
      })) };
    } else if (option != 1 && comments.some((c) => c)) {
      let changes = [];
      for (let i = 0, comment2; i < comments.length; i++)
        if (comment2 = comments[i]) {
          let token = tokens[i], { open, close } = comment2;
          changes.push({ from: open.pos - token.open.length, to: open.pos + open.margin }, { from: close.pos - close.margin, to: close.pos + token.close.length });
        }
      return { changes };
    }
    return null;
  }
  function changeLineComment(option, state, ranges = state.selection.ranges) {
    let lines = [];
    let prevLine = -1;
    ranges: for (let { from, to } of ranges) {
      let startI = lines.length, minIndent = 1e9, token;
      for (let pos = from; pos <= to; ) {
        let line = state.doc.lineAt(pos);
        if (token == void 0) {
          token = getConfig(state, line.from).line;
          if (!token)
            continue ranges;
        }
        if (line.from > prevLine && (from == to || to > line.from)) {
          prevLine = line.from;
          let indent = /^\s*/.exec(line.text)[0].length;
          let empty2 = indent == line.length;
          let comment2 = line.text.slice(indent, indent + token.length) == token ? indent : -1;
          if (indent < line.text.length && indent < minIndent)
            minIndent = indent;
          lines.push({ line, comment: comment2, token, indent, empty: empty2, single: false });
        }
        pos = line.to + 1;
      }
      if (minIndent < 1e9) {
        for (let i = startI; i < lines.length; i++)
          if (lines[i].indent < lines[i].line.text.length)
            lines[i].indent = minIndent;
      }
      if (lines.length == startI + 1)
        lines[startI].single = true;
    }
    if (option != 2 && lines.some((l) => l.comment < 0 && (!l.empty || l.single))) {
      let changes = [];
      for (let { line, token, indent, empty: empty2, single } of lines)
        if (single || !empty2)
          changes.push({ from: line.from + indent, insert: token + " " });
      let changeSet = state.changes(changes);
      return { changes: changeSet, selection: state.selection.map(changeSet, 1) };
    } else if (option != 1 && lines.some((l) => l.comment >= 0)) {
      let changes = [];
      for (let { line, comment: comment2, token } of lines)
        if (comment2 >= 0) {
          let from = line.from + comment2, to = from + token.length;
          if (line.text[to - line.from] == " ")
            to++;
          changes.push({ from, to });
        }
      return { changes };
    }
    return null;
  }
  var fromHistory = /* @__PURE__ */ Annotation.define();
  var isolateHistory = /* @__PURE__ */ Annotation.define();
  var invertedEffects = /* @__PURE__ */ Facet.define();
  var historyConfig = /* @__PURE__ */ Facet.define({
    combine(configs) {
      return combineConfig(configs, {
        minDepth: 100,
        newGroupDelay: 500,
        joinToEvent: (_t, isAdjacent2) => isAdjacent2
      }, {
        minDepth: Math.max,
        newGroupDelay: Math.min,
        joinToEvent: (a, b) => (tr, adj) => a(tr, adj) || b(tr, adj)
      });
    }
  });
  var historyField_ = /* @__PURE__ */ StateField.define({
    create() {
      return HistoryState.empty;
    },
    update(state, tr) {
      let config = tr.state.facet(historyConfig);
      let fromHist = tr.annotation(fromHistory);
      if (fromHist) {
        let item = HistEvent.fromTransaction(tr, fromHist.selection), from = fromHist.side;
        let other = from == 0 ? state.undone : state.done;
        if (item)
          other = updateBranch(other, other.length, config.minDepth, item);
        else
          other = addSelection(other, tr.startState.selection);
        return new HistoryState(from == 0 ? fromHist.rest : other, from == 0 ? other : fromHist.rest);
      }
      let isolate = tr.annotation(isolateHistory);
      if (isolate == "full" || isolate == "before")
        state = state.isolate();
      if (tr.annotation(Transaction.addToHistory) === false)
        return !tr.changes.empty ? state.addMapping(tr.changes.desc) : state;
      let event = HistEvent.fromTransaction(tr);
      let time = tr.annotation(Transaction.time), userEvent = tr.annotation(Transaction.userEvent);
      if (event)
        state = state.addChanges(event, time, userEvent, config, tr);
      else if (tr.selection)
        state = state.addSelection(tr.startState.selection, time, userEvent, config.newGroupDelay);
      if (isolate == "full" || isolate == "after")
        state = state.isolate();
      return state;
    },
    toJSON(value) {
      return { done: value.done.map((e) => e.toJSON()), undone: value.undone.map((e) => e.toJSON()) };
    },
    fromJSON(json) {
      return new HistoryState(json.done.map(HistEvent.fromJSON), json.undone.map(HistEvent.fromJSON));
    }
  });
  function history(config = {}) {
    return [
      historyField_,
      historyConfig.of(config),
      EditorView.domEventHandlers({
        beforeinput(e, view) {
          let command2 = e.inputType == "historyUndo" ? undo : e.inputType == "historyRedo" ? redo : null;
          if (!command2)
            return false;
          e.preventDefault();
          return command2(view);
        }
      })
    ];
  }
  function cmd(side, selection) {
    return function({ state, dispatch }) {
      if (!selection && state.readOnly)
        return false;
      let historyState = state.field(historyField_, false);
      if (!historyState)
        return false;
      let tr = historyState.pop(side, state, selection);
      if (!tr)
        return false;
      dispatch(tr);
      return true;
    };
  }
  var undo = /* @__PURE__ */ cmd(0, false);
  var redo = /* @__PURE__ */ cmd(1, false);
  var undoSelection = /* @__PURE__ */ cmd(0, true);
  var redoSelection = /* @__PURE__ */ cmd(1, true);
  var HistEvent = class _HistEvent {
    constructor(changes, effects, mapped, startSelection, selectionsAfter) {
      this.changes = changes;
      this.effects = effects;
      this.mapped = mapped;
      this.startSelection = startSelection;
      this.selectionsAfter = selectionsAfter;
    }
    setSelAfter(after) {
      return new _HistEvent(this.changes, this.effects, this.mapped, this.startSelection, after);
    }
    toJSON() {
      var _a2, _b, _c;
      return {
        changes: (_a2 = this.changes) === null || _a2 === void 0 ? void 0 : _a2.toJSON(),
        mapped: (_b = this.mapped) === null || _b === void 0 ? void 0 : _b.toJSON(),
        startSelection: (_c = this.startSelection) === null || _c === void 0 ? void 0 : _c.toJSON(),
        selectionsAfter: this.selectionsAfter.map((s) => s.toJSON())
      };
    }
    static fromJSON(json) {
      return new _HistEvent(json.changes && ChangeSet.fromJSON(json.changes), [], json.mapped && ChangeDesc.fromJSON(json.mapped), json.startSelection && EditorSelection.fromJSON(json.startSelection), json.selectionsAfter.map(EditorSelection.fromJSON));
    }
    // This does not check `addToHistory` and such, it assumes the
    // transaction needs to be converted to an item. Returns null when
    // there are no changes or effects in the transaction.
    static fromTransaction(tr, selection) {
      let effects = none2;
      for (let invert of tr.startState.facet(invertedEffects)) {
        let result = invert(tr);
        if (result.length)
          effects = effects.concat(result);
      }
      if (!effects.length && tr.changes.empty)
        return null;
      return new _HistEvent(tr.changes.invert(tr.startState.doc), effects, void 0, selection || tr.startState.selection, none2);
    }
    static selection(selections) {
      return new _HistEvent(void 0, none2, void 0, void 0, selections);
    }
  };
  function updateBranch(branch, to, maxLen, newEvent) {
    let start = to + 1 > maxLen + 20 ? to - maxLen - 1 : 0;
    let newBranch = branch.slice(start, to);
    newBranch.push(newEvent);
    return newBranch;
  }
  function isAdjacent(a, b) {
    let ranges = [], isAdjacent2 = false;
    a.iterChangedRanges((f, t2) => ranges.push(f, t2));
    b.iterChangedRanges((_f, _t, f, t2) => {
      for (let i = 0; i < ranges.length; ) {
        let from = ranges[i++], to = ranges[i++];
        if (t2 >= from && f <= to)
          isAdjacent2 = true;
      }
    });
    return isAdjacent2;
  }
  function eqSelectionShape(a, b) {
    return a.ranges.length == b.ranges.length && a.ranges.filter((r, i) => r.empty != b.ranges[i].empty).length === 0;
  }
  function conc(a, b) {
    return !a.length ? b : !b.length ? a : a.concat(b);
  }
  var none2 = [];
  var MaxSelectionsPerEvent = 200;
  function addSelection(branch, selection) {
    if (!branch.length) {
      return [HistEvent.selection([selection])];
    } else {
      let lastEvent = branch[branch.length - 1];
      let sels = lastEvent.selectionsAfter.slice(Math.max(0, lastEvent.selectionsAfter.length - MaxSelectionsPerEvent));
      if (sels.length && sels[sels.length - 1].eq(selection))
        return branch;
      sels.push(selection);
      return updateBranch(branch, branch.length - 1, 1e9, lastEvent.setSelAfter(sels));
    }
  }
  function popSelection(branch) {
    let last = branch[branch.length - 1];
    let newBranch = branch.slice();
    newBranch[branch.length - 1] = last.setSelAfter(last.selectionsAfter.slice(0, last.selectionsAfter.length - 1));
    return newBranch;
  }
  function addMappingToBranch(branch, mapping) {
    if (!branch.length)
      return branch;
    let length = branch.length, selections = none2;
    while (length) {
      let event = mapEvent(branch[length - 1], mapping, selections);
      if (event.changes && !event.changes.empty || event.effects.length) {
        let result = branch.slice(0, length);
        result[length - 1] = event;
        return result;
      } else {
        mapping = event.mapped;
        length--;
        selections = event.selectionsAfter;
      }
    }
    return selections.length ? [HistEvent.selection(selections)] : none2;
  }
  function mapEvent(event, mapping, extraSelections) {
    let selections = conc(event.selectionsAfter.length ? event.selectionsAfter.map((s) => s.map(mapping)) : none2, extraSelections);
    if (!event.changes)
      return HistEvent.selection(selections);
    let mappedChanges = event.changes.map(mapping), before = mapping.mapDesc(event.changes, true);
    let fullMapping = event.mapped ? event.mapped.composeDesc(before) : before;
    return new HistEvent(mappedChanges, StateEffect.mapEffects(event.effects, mapping), fullMapping, event.startSelection.map(before), selections);
  }
  var joinableUserEvent = /^(input\.type|delete)($|\.)/;
  var HistoryState = class _HistoryState {
    constructor(done, undone, prevTime = 0, prevUserEvent = void 0) {
      this.done = done;
      this.undone = undone;
      this.prevTime = prevTime;
      this.prevUserEvent = prevUserEvent;
    }
    isolate() {
      return this.prevTime ? new _HistoryState(this.done, this.undone) : this;
    }
    addChanges(event, time, userEvent, config, tr) {
      let done = this.done, lastEvent = done[done.length - 1];
      if (lastEvent && lastEvent.changes && !lastEvent.changes.empty && event.changes && (!userEvent || joinableUserEvent.test(userEvent)) && (!lastEvent.selectionsAfter.length && time - this.prevTime < config.newGroupDelay && config.joinToEvent(tr, isAdjacent(lastEvent.changes, event.changes)) || // For compose (but not compose.start) events, always join with previous event
      userEvent == "input.type.compose")) {
        done = updateBranch(done, done.length - 1, config.minDepth, new HistEvent(event.changes.compose(lastEvent.changes), conc(StateEffect.mapEffects(event.effects, lastEvent.changes), lastEvent.effects), lastEvent.mapped, lastEvent.startSelection, none2));
      } else {
        done = updateBranch(done, done.length, config.minDepth, event);
      }
      return new _HistoryState(done, none2, time, userEvent);
    }
    addSelection(selection, time, userEvent, newGroupDelay) {
      let last = this.done.length ? this.done[this.done.length - 1].selectionsAfter : none2;
      if (last.length > 0 && time - this.prevTime < newGroupDelay && userEvent == this.prevUserEvent && userEvent && /^select($|\.)/.test(userEvent) && eqSelectionShape(last[last.length - 1], selection))
        return this;
      return new _HistoryState(addSelection(this.done, selection), this.undone, time, userEvent);
    }
    addMapping(mapping) {
      return new _HistoryState(addMappingToBranch(this.done, mapping), addMappingToBranch(this.undone, mapping), this.prevTime, this.prevUserEvent);
    }
    pop(side, state, onlySelection) {
      let branch = side == 0 ? this.done : this.undone;
      if (branch.length == 0)
        return null;
      let event = branch[branch.length - 1], selection = event.selectionsAfter[0] || (event.startSelection ? event.startSelection.map(event.changes.invertedDesc, 1) : state.selection);
      if (onlySelection && event.selectionsAfter.length) {
        return state.update({
          selection: event.selectionsAfter[event.selectionsAfter.length - 1],
          annotations: fromHistory.of({ side, rest: popSelection(branch), selection }),
          userEvent: side == 0 ? "select.undo" : "select.redo",
          scrollIntoView: true
        });
      } else if (!event.changes) {
        return null;
      } else {
        let rest = branch.length == 1 ? none2 : branch.slice(0, branch.length - 1);
        if (event.mapped)
          rest = addMappingToBranch(rest, event.mapped);
        return state.update({
          changes: event.changes,
          selection: event.startSelection,
          effects: event.effects,
          annotations: fromHistory.of({ side, rest, selection }),
          filter: false,
          userEvent: side == 0 ? "undo" : "redo",
          scrollIntoView: true
        });
      }
    }
  };
  HistoryState.empty = /* @__PURE__ */ new HistoryState(none2, none2);
  var historyKeymap = [
    { key: "Mod-z", run: undo, preventDefault: true },
    { key: "Mod-y", mac: "Mod-Shift-z", run: redo, preventDefault: true },
    { linux: "Ctrl-Shift-z", run: redo, preventDefault: true },
    { key: "Mod-u", run: undoSelection, preventDefault: true },
    { key: "Alt-u", mac: "Mod-Shift-u", run: redoSelection, preventDefault: true }
  ];
  function updateSel(sel, by) {
    return EditorSelection.create(sel.ranges.map(by), sel.mainIndex);
  }
  function setSel(state, selection) {
    return state.update({ selection, scrollIntoView: true, userEvent: "select" });
  }
  function moveSel({ state, dispatch }, how) {
    let selection = updateSel(state.selection, how);
    if (selection.eq(state.selection, true))
      return false;
    dispatch(setSel(state, selection));
    return true;
  }
  function rangeEnd(range, forward) {
    return EditorSelection.cursor(forward ? range.to : range.from);
  }
  function cursorByChar(view, forward) {
    return moveSel(view, (range) => range.empty ? view.moveByChar(range, forward) : rangeEnd(range, forward));
  }
  function ltrAtCursor(view) {
    return view.textDirectionAt(view.state.selection.main.head) == Direction.LTR;
  }
  var cursorCharLeft = (view) => cursorByChar(view, !ltrAtCursor(view));
  var cursorCharRight = (view) => cursorByChar(view, ltrAtCursor(view));
  function cursorByGroup(view, forward) {
    return moveSel(view, (range) => range.empty ? view.moveByGroup(range, forward) : rangeEnd(range, forward));
  }
  var cursorGroupLeft = (view) => cursorByGroup(view, !ltrAtCursor(view));
  var cursorGroupRight = (view) => cursorByGroup(view, ltrAtCursor(view));
  var segmenter = typeof Intl != "undefined" && Intl.Segmenter ? /* @__PURE__ */ new Intl.Segmenter(void 0, { granularity: "word" }) : null;
  function interestingNode(state, node, bracketProp) {
    if (node.type.prop(bracketProp))
      return true;
    let len = node.to - node.from;
    return len && (len > 2 || /[^\s,.;:]/.test(state.sliceDoc(node.from, node.to))) || node.firstChild;
  }
  function moveBySyntax(state, start, forward) {
    let pos = syntaxTree(state).resolveInner(start.head);
    let bracketProp = forward ? NodeProp.closedBy : NodeProp.openedBy;
    for (let at = start.head; ; ) {
      let next = forward ? pos.childAfter(at) : pos.childBefore(at);
      if (!next)
        break;
      if (interestingNode(state, next, bracketProp))
        pos = next;
      else
        at = forward ? next.to : next.from;
    }
    let bracket2 = pos.type.prop(bracketProp), match, newPos;
    if (bracket2 && (match = forward ? matchBrackets(state, pos.from, 1) : matchBrackets(state, pos.to, -1)) && match.matched)
      newPos = forward ? match.end.to : match.end.from;
    else
      newPos = forward ? pos.to : pos.from;
    return EditorSelection.cursor(newPos, forward ? -1 : 1);
  }
  var cursorSyntaxLeft = (view) => moveSel(view, (range) => moveBySyntax(view.state, range, !ltrAtCursor(view)));
  var cursorSyntaxRight = (view) => moveSel(view, (range) => moveBySyntax(view.state, range, ltrAtCursor(view)));
  function cursorByLine(view, forward) {
    return moveSel(view, (range) => {
      if (!range.empty)
        return rangeEnd(range, forward);
      let moved = view.moveVertically(range, forward);
      return moved.head != range.head ? moved : view.moveToLineBoundary(range, forward);
    });
  }
  var cursorLineUp = (view) => cursorByLine(view, false);
  var cursorLineDown = (view) => cursorByLine(view, true);
  function pageInfo(view) {
    let selfScroll = view.scrollDOM.clientHeight < view.scrollDOM.scrollHeight - 2;
    let marginTop = 0, marginBottom = 0, height;
    if (selfScroll) {
      for (let source of view.state.facet(EditorView.scrollMargins)) {
        let margins = source(view);
        if (margins === null || margins === void 0 ? void 0 : margins.top)
          marginTop = Math.max(margins === null || margins === void 0 ? void 0 : margins.top, marginTop);
        if (margins === null || margins === void 0 ? void 0 : margins.bottom)
          marginBottom = Math.max(margins === null || margins === void 0 ? void 0 : margins.bottom, marginBottom);
      }
      height = view.scrollDOM.clientHeight - marginTop - marginBottom;
    } else {
      height = (view.dom.ownerDocument.defaultView || window).innerHeight;
    }
    return {
      marginTop,
      marginBottom,
      selfScroll,
      height: Math.max(view.defaultLineHeight, height - 5)
    };
  }
  function cursorByPage(view, forward) {
    let page = pageInfo(view);
    let { state } = view, selection = updateSel(state.selection, (range) => {
      return range.empty ? view.moveVertically(range, forward, page.height) : rangeEnd(range, forward);
    });
    if (selection.eq(state.selection))
      return false;
    let effect;
    if (page.selfScroll) {
      let startPos = view.coordsAtPos(state.selection.main.head);
      let scrollRect = view.scrollDOM.getBoundingClientRect();
      let scrollTop = scrollRect.top + page.marginTop, scrollBottom = scrollRect.bottom - page.marginBottom;
      if (startPos && startPos.top > scrollTop && startPos.bottom < scrollBottom)
        effect = EditorView.scrollIntoView(selection.main.head, { y: "start", yMargin: startPos.top - scrollTop });
    }
    view.dispatch(setSel(state, selection), { effects: effect });
    return true;
  }
  var cursorPageUp = (view) => cursorByPage(view, false);
  var cursorPageDown = (view) => cursorByPage(view, true);
  function moveByLineBoundary(view, start, forward) {
    let line = view.lineBlockAt(start.head), moved = view.moveToLineBoundary(start, forward);
    if (moved.head == start.head && moved.head != (forward ? line.to : line.from))
      moved = view.moveToLineBoundary(start, forward, false);
    if (!forward && moved.head == line.from && line.length) {
      let space = /^\s*/.exec(view.state.sliceDoc(line.from, Math.min(line.from + 100, line.to)))[0].length;
      if (space && start.head != line.from + space)
        moved = EditorSelection.cursor(line.from + space);
    }
    return moved;
  }
  var cursorLineBoundaryForward = (view) => moveSel(view, (range) => moveByLineBoundary(view, range, true));
  var cursorLineBoundaryBackward = (view) => moveSel(view, (range) => moveByLineBoundary(view, range, false));
  var cursorLineBoundaryLeft = (view) => moveSel(view, (range) => moveByLineBoundary(view, range, !ltrAtCursor(view)));
  var cursorLineBoundaryRight = (view) => moveSel(view, (range) => moveByLineBoundary(view, range, ltrAtCursor(view)));
  var cursorLineStart = (view) => moveSel(view, (range) => EditorSelection.cursor(view.lineBlockAt(range.head).from, 1));
  var cursorLineEnd = (view) => moveSel(view, (range) => EditorSelection.cursor(view.lineBlockAt(range.head).to, -1));
  function toMatchingBracket(state, dispatch, extend) {
    let found = false, selection = updateSel(state.selection, (range) => {
      let matching = matchBrackets(state, range.head, -1) || matchBrackets(state, range.head, 1) || range.head > 0 && matchBrackets(state, range.head - 1, 1) || range.head < state.doc.length && matchBrackets(state, range.head + 1, -1);
      if (!matching || !matching.end)
        return range;
      found = true;
      let head = matching.start.from == range.head ? matching.end.to : matching.end.from;
      return extend ? EditorSelection.range(range.anchor, head) : EditorSelection.cursor(head);
    });
    if (!found)
      return false;
    dispatch(setSel(state, selection));
    return true;
  }
  var cursorMatchingBracket = ({ state, dispatch }) => toMatchingBracket(state, dispatch, false);
  function extendSel(target, how) {
    let selection = updateSel(target.state.selection, (range) => {
      let head = how(range);
      return EditorSelection.range(range.anchor, head.head, head.goalColumn, head.bidiLevel || void 0);
    });
    if (selection.eq(target.state.selection))
      return false;
    target.dispatch(setSel(target.state, selection));
    return true;
  }
  function selectByChar(view, forward) {
    return extendSel(view, (range) => view.moveByChar(range, forward));
  }
  var selectCharLeft = (view) => selectByChar(view, !ltrAtCursor(view));
  var selectCharRight = (view) => selectByChar(view, ltrAtCursor(view));
  function selectByGroup(view, forward) {
    return extendSel(view, (range) => view.moveByGroup(range, forward));
  }
  var selectGroupLeft = (view) => selectByGroup(view, !ltrAtCursor(view));
  var selectGroupRight = (view) => selectByGroup(view, ltrAtCursor(view));
  var selectSyntaxLeft = (view) => extendSel(view, (range) => moveBySyntax(view.state, range, !ltrAtCursor(view)));
  var selectSyntaxRight = (view) => extendSel(view, (range) => moveBySyntax(view.state, range, ltrAtCursor(view)));
  function selectByLine(view, forward) {
    return extendSel(view, (range) => view.moveVertically(range, forward));
  }
  var selectLineUp = (view) => selectByLine(view, false);
  var selectLineDown = (view) => selectByLine(view, true);
  function selectByPage(view, forward) {
    return extendSel(view, (range) => view.moveVertically(range, forward, pageInfo(view).height));
  }
  var selectPageUp = (view) => selectByPage(view, false);
  var selectPageDown = (view) => selectByPage(view, true);
  var selectLineBoundaryForward = (view) => extendSel(view, (range) => moveByLineBoundary(view, range, true));
  var selectLineBoundaryBackward = (view) => extendSel(view, (range) => moveByLineBoundary(view, range, false));
  var selectLineBoundaryLeft = (view) => extendSel(view, (range) => moveByLineBoundary(view, range, !ltrAtCursor(view)));
  var selectLineBoundaryRight = (view) => extendSel(view, (range) => moveByLineBoundary(view, range, ltrAtCursor(view)));
  var selectLineStart = (view) => extendSel(view, (range) => EditorSelection.cursor(view.lineBlockAt(range.head).from));
  var selectLineEnd = (view) => extendSel(view, (range) => EditorSelection.cursor(view.lineBlockAt(range.head).to));
  var cursorDocStart = ({ state, dispatch }) => {
    dispatch(setSel(state, { anchor: 0 }));
    return true;
  };
  var cursorDocEnd = ({ state, dispatch }) => {
    dispatch(setSel(state, { anchor: state.doc.length }));
    return true;
  };
  var selectDocStart = ({ state, dispatch }) => {
    dispatch(setSel(state, { anchor: state.selection.main.anchor, head: 0 }));
    return true;
  };
  var selectDocEnd = ({ state, dispatch }) => {
    dispatch(setSel(state, { anchor: state.selection.main.anchor, head: state.doc.length }));
    return true;
  };
  var selectAll = ({ state, dispatch }) => {
    dispatch(state.update({ selection: { anchor: 0, head: state.doc.length }, userEvent: "select" }));
    return true;
  };
  var selectLine = ({ state, dispatch }) => {
    let ranges = selectedLineBlocks(state).map(({ from, to }) => EditorSelection.range(from, Math.min(to + 1, state.doc.length)));
    dispatch(state.update({ selection: EditorSelection.create(ranges), userEvent: "select" }));
    return true;
  };
  var selectParentSyntax = ({ state, dispatch }) => {
    let selection = updateSel(state.selection, (range) => {
      let tree = syntaxTree(state), stack = tree.resolveStack(range.from, 1);
      if (range.empty) {
        let stackBefore = tree.resolveStack(range.from, -1);
        if (stackBefore.node.from >= stack.node.from && stackBefore.node.to <= stack.node.to)
          stack = stackBefore;
      }
      for (let cur2 = stack; cur2; cur2 = cur2.next) {
        let { node } = cur2;
        if ((node.from < range.from && node.to >= range.to || node.to > range.to && node.from <= range.from) && cur2.next)
          return EditorSelection.range(node.to, node.from);
      }
      return range;
    });
    if (selection.eq(state.selection))
      return false;
    dispatch(setSel(state, selection));
    return true;
  };
  function addCursorVertically(view, forward) {
    let { state } = view, sel = state.selection, ranges = state.selection.ranges.slice();
    for (let range of state.selection.ranges) {
      let line = state.doc.lineAt(range.head);
      if (forward ? line.to < view.state.doc.length : line.from > 0)
        for (let cur2 = range; ; ) {
          let next = view.moveVertically(cur2, forward);
          if (next.head < line.from || next.head > line.to) {
            if (!ranges.some((r) => r.head == next.head))
              ranges.push(next);
            break;
          } else if (next.head == cur2.head) {
            break;
          } else {
            cur2 = next;
          }
        }
    }
    if (ranges.length == sel.ranges.length)
      return false;
    view.dispatch(setSel(state, EditorSelection.create(ranges, ranges.length - 1)));
    return true;
  }
  var addCursorAbove = (view) => addCursorVertically(view, false);
  var addCursorBelow = (view) => addCursorVertically(view, true);
  var simplifySelection = ({ state, dispatch }) => {
    let cur2 = state.selection, selection = null;
    if (cur2.ranges.length > 1)
      selection = EditorSelection.create([cur2.main]);
    else if (!cur2.main.empty)
      selection = EditorSelection.create([EditorSelection.cursor(cur2.main.head)]);
    if (!selection)
      return false;
    dispatch(setSel(state, selection));
    return true;
  };
  function deleteBy(target, by) {
    if (target.state.readOnly)
      return false;
    let event = "delete.selection", { state } = target;
    let changes = state.changeByRange((range) => {
      let { from, to } = range;
      if (from == to) {
        let towards = by(range);
        if (towards < from) {
          event = "delete.backward";
          towards = skipAtomic(target, towards, false);
        } else if (towards > from) {
          event = "delete.forward";
          towards = skipAtomic(target, towards, true);
        }
        from = Math.min(from, towards);
        to = Math.max(to, towards);
      } else {
        from = skipAtomic(target, from, false);
        to = skipAtomic(target, to, true);
      }
      return from == to ? { range } : { changes: { from, to }, range: EditorSelection.cursor(from, from < range.head ? -1 : 1) };
    });
    if (changes.changes.empty)
      return false;
    target.dispatch(state.update(changes, {
      scrollIntoView: true,
      userEvent: event,
      effects: event == "delete.selection" ? EditorView.announce.of(state.phrase("Selection deleted")) : void 0
    }));
    return true;
  }
  function skipAtomic(target, pos, forward) {
    if (target instanceof EditorView)
      for (let ranges of target.state.facet(EditorView.atomicRanges).map((f) => f(target)))
        ranges.between(pos, pos, (from, to) => {
          if (from < pos && to > pos)
            pos = forward ? to : from;
        });
    return pos;
  }
  var deleteByChar = (target, forward, byIndentUnit) => deleteBy(target, (range) => {
    let pos = range.from, { state } = target, line = state.doc.lineAt(pos), before, targetPos;
    if (byIndentUnit && !forward && pos > line.from && pos < line.from + 200 && !/[^ \t]/.test(before = line.text.slice(0, pos - line.from))) {
      if (before[before.length - 1] == "	")
        return pos - 1;
      let col = countColumn(before, state.tabSize), drop = col % getIndentUnit(state) || getIndentUnit(state);
      for (let i = 0; i < drop && before[before.length - 1 - i] == " "; i++)
        pos--;
      targetPos = pos;
    } else {
      targetPos = findClusterBreak2(line.text, pos - line.from, forward, forward) + line.from;
      if (targetPos == pos && line.number != (forward ? state.doc.lines : 1))
        targetPos += forward ? 1 : -1;
      else if (!forward && /[\ufe00-\ufe0f]/.test(line.text.slice(targetPos - line.from, pos - line.from)))
        targetPos = findClusterBreak2(line.text, targetPos - line.from, false, false) + line.from;
    }
    return targetPos;
  });
  var deleteCharBackward = (view) => deleteByChar(view, false, true);
  var deleteCharForward = (view) => deleteByChar(view, true, false);
  var deleteByGroup = (target, forward) => deleteBy(target, (range) => {
    let pos = range.head, { state } = target, line = state.doc.lineAt(pos);
    let categorize = state.charCategorizer(pos);
    for (let cat = null; ; ) {
      if (pos == (forward ? line.to : line.from)) {
        if (pos == range.head && line.number != (forward ? state.doc.lines : 1))
          pos += forward ? 1 : -1;
        break;
      }
      let next = findClusterBreak2(line.text, pos - line.from, forward) + line.from;
      let nextChar = line.text.slice(Math.min(pos, next) - line.from, Math.max(pos, next) - line.from);
      let nextCat = categorize(nextChar);
      if (cat != null && nextCat != cat)
        break;
      if (nextChar != " " || pos != range.head)
        cat = nextCat;
      pos = next;
    }
    return pos;
  });
  var deleteGroupBackward = (target) => deleteByGroup(target, false);
  var deleteGroupForward = (target) => deleteByGroup(target, true);
  var deleteToLineEnd = (view) => deleteBy(view, (range) => {
    let lineEnd = view.lineBlockAt(range.head).to;
    return range.head < lineEnd ? lineEnd : Math.min(view.state.doc.length, range.head + 1);
  });
  var deleteLineBoundaryBackward = (view) => deleteBy(view, (range) => {
    let lineStart = view.moveToLineBoundary(range, false).head;
    return range.head > lineStart ? lineStart : Math.max(0, range.head - 1);
  });
  var deleteLineBoundaryForward = (view) => deleteBy(view, (range) => {
    let lineStart = view.moveToLineBoundary(range, true).head;
    return range.head < lineStart ? lineStart : Math.min(view.state.doc.length, range.head + 1);
  });
  var splitLine = ({ state, dispatch }) => {
    if (state.readOnly)
      return false;
    let changes = state.changeByRange((range) => {
      return {
        changes: { from: range.from, to: range.to, insert: Text.of(["", ""]) },
        range: EditorSelection.cursor(range.from)
      };
    });
    dispatch(state.update(changes, { scrollIntoView: true, userEvent: "input" }));
    return true;
  };
  var transposeChars = ({ state, dispatch }) => {
    if (state.readOnly)
      return false;
    let changes = state.changeByRange((range) => {
      if (!range.empty || range.from == 0 || range.from == state.doc.length)
        return { range };
      let pos = range.from, line = state.doc.lineAt(pos);
      let from = pos == line.from ? pos - 1 : findClusterBreak2(line.text, pos - line.from, false) + line.from;
      let to = pos == line.to ? pos + 1 : findClusterBreak2(line.text, pos - line.from, true) + line.from;
      return {
        changes: { from, to, insert: state.doc.slice(pos, to).append(state.doc.slice(from, pos)) },
        range: EditorSelection.cursor(to)
      };
    });
    if (changes.changes.empty)
      return false;
    dispatch(state.update(changes, { scrollIntoView: true, userEvent: "move.character" }));
    return true;
  };
  function selectedLineBlocks(state) {
    let blocks = [], upto = -1;
    for (let range of state.selection.ranges) {
      let startLine = state.doc.lineAt(range.from), endLine = state.doc.lineAt(range.to);
      if (!range.empty && range.to == endLine.from)
        endLine = state.doc.lineAt(range.to - 1);
      if (upto >= startLine.number) {
        let prev = blocks[blocks.length - 1];
        prev.to = endLine.to;
        prev.ranges.push(range);
      } else {
        blocks.push({ from: startLine.from, to: endLine.to, ranges: [range] });
      }
      upto = endLine.number + 1;
    }
    return blocks;
  }
  function moveLine(state, dispatch, forward) {
    if (state.readOnly)
      return false;
    let changes = [], ranges = [];
    for (let block of selectedLineBlocks(state)) {
      if (forward ? block.to == state.doc.length : block.from == 0)
        continue;
      let nextLine = state.doc.lineAt(forward ? block.to + 1 : block.from - 1);
      let size = nextLine.length + 1;
      if (forward) {
        changes.push({ from: block.to, to: nextLine.to }, { from: block.from, insert: nextLine.text + state.lineBreak });
        for (let r of block.ranges)
          ranges.push(EditorSelection.range(Math.min(state.doc.length, r.anchor + size), Math.min(state.doc.length, r.head + size)));
      } else {
        changes.push({ from: nextLine.from, to: block.from }, { from: block.to, insert: state.lineBreak + nextLine.text });
        for (let r of block.ranges)
          ranges.push(EditorSelection.range(r.anchor - size, r.head - size));
      }
    }
    if (!changes.length)
      return false;
    dispatch(state.update({
      changes,
      scrollIntoView: true,
      selection: EditorSelection.create(ranges, state.selection.mainIndex),
      userEvent: "move.line"
    }));
    return true;
  }
  var moveLineUp = ({ state, dispatch }) => moveLine(state, dispatch, false);
  var moveLineDown = ({ state, dispatch }) => moveLine(state, dispatch, true);
  function copyLine(state, dispatch, forward) {
    if (state.readOnly)
      return false;
    let changes = [];
    for (let block of selectedLineBlocks(state)) {
      if (forward)
        changes.push({ from: block.from, insert: state.doc.slice(block.from, block.to) + state.lineBreak });
      else
        changes.push({ from: block.to, insert: state.lineBreak + state.doc.slice(block.from, block.to) });
    }
    let changeSet = state.changes(changes);
    dispatch(state.update({
      changes: changeSet,
      selection: state.selection.map(changeSet, forward ? 1 : -1),
      scrollIntoView: true,
      userEvent: "input.copyline"
    }));
    return true;
  }
  var copyLineUp = ({ state, dispatch }) => copyLine(state, dispatch, false);
  var copyLineDown = ({ state, dispatch }) => copyLine(state, dispatch, true);
  var deleteLine = (view) => {
    if (view.state.readOnly)
      return false;
    let { state } = view, changes = state.changes(selectedLineBlocks(state).map(({ from, to }) => {
      if (from > 0)
        from--;
      else if (to < state.doc.length)
        to++;
      return { from, to };
    }));
    let selection = updateSel(state.selection, (range) => {
      let dist2 = void 0;
      if (view.lineWrapping) {
        let block = view.lineBlockAt(range.head), pos = view.coordsAtPos(range.head, range.assoc || 1);
        if (pos)
          dist2 = block.bottom + view.documentTop - pos.bottom + view.defaultLineHeight / 2;
      }
      return view.moveVertically(range, true, dist2);
    }).map(changes);
    view.dispatch({ changes, selection, scrollIntoView: true, userEvent: "delete.line" });
    return true;
  };
  function isBetweenBrackets(state, pos) {
    if (/\(\)|\[\]|\{\}/.test(state.sliceDoc(pos - 1, pos + 1)))
      return { from: pos, to: pos };
    let context = syntaxTree(state).resolveInner(pos);
    let before = context.childBefore(pos), after = context.childAfter(pos), closedBy;
    if (before && after && before.to <= pos && after.from >= pos && (closedBy = before.type.prop(NodeProp.closedBy)) && closedBy.indexOf(after.name) > -1 && state.doc.lineAt(before.to).from == state.doc.lineAt(after.from).from && !/\S/.test(state.sliceDoc(before.to, after.from)))
      return { from: before.to, to: after.from };
    return null;
  }
  var insertNewlineAndIndent = /* @__PURE__ */ newlineAndIndent(false);
  var insertBlankLine = /* @__PURE__ */ newlineAndIndent(true);
  function newlineAndIndent(atEof) {
    return ({ state, dispatch }) => {
      if (state.readOnly)
        return false;
      let changes = state.changeByRange((range) => {
        let { from, to } = range, line = state.doc.lineAt(from);
        let explode = !atEof && from == to && isBetweenBrackets(state, from);
        if (atEof)
          from = to = (to <= line.to ? line : state.doc.lineAt(to)).to;
        let cx = new IndentContext(state, { simulateBreak: from, simulateDoubleBreak: !!explode });
        let indent = getIndentation(cx, from);
        if (indent == null)
          indent = countColumn(/^\s*/.exec(state.doc.lineAt(from).text)[0], state.tabSize);
        while (to < line.to && /\s/.test(line.text[to - line.from]))
          to++;
        if (explode)
          ({ from, to } = explode);
        else if (from > line.from && from < line.from + 100 && !/\S/.test(line.text.slice(0, from)))
          from = line.from;
        let insert2 = ["", indentString(state, indent)];
        if (explode)
          insert2.push(indentString(state, cx.lineIndent(line.from, -1)));
        return {
          changes: { from, to, insert: Text.of(insert2) },
          range: EditorSelection.cursor(from + 1 + insert2[1].length)
        };
      });
      dispatch(state.update(changes, { scrollIntoView: true, userEvent: "input" }));
      return true;
    };
  }
  function changeBySelectedLine(state, f) {
    let atLine = -1;
    return state.changeByRange((range) => {
      let changes = [];
      for (let pos = range.from; pos <= range.to; ) {
        let line = state.doc.lineAt(pos);
        if (line.number > atLine && (range.empty || range.to > line.from)) {
          f(line, changes, range);
          atLine = line.number;
        }
        pos = line.to + 1;
      }
      let changeSet = state.changes(changes);
      return {
        changes,
        range: EditorSelection.range(changeSet.mapPos(range.anchor, 1), changeSet.mapPos(range.head, 1))
      };
    });
  }
  var indentSelection = ({ state, dispatch }) => {
    if (state.readOnly)
      return false;
    let updated = /* @__PURE__ */ Object.create(null);
    let context = new IndentContext(state, { overrideIndentation: (start) => {
      let found = updated[start];
      return found == null ? -1 : found;
    } });
    let changes = changeBySelectedLine(state, (line, changes2, range) => {
      let indent = getIndentation(context, line.from);
      if (indent == null)
        return;
      if (!/\S/.test(line.text))
        indent = 0;
      let cur2 = /^\s*/.exec(line.text)[0];
      let norm = indentString(state, indent);
      if (cur2 != norm || range.from < line.from + cur2.length) {
        updated[line.from] = indent;
        changes2.push({ from: line.from, to: line.from + cur2.length, insert: norm });
      }
    });
    if (!changes.changes.empty)
      dispatch(state.update(changes, { userEvent: "indent" }));
    return true;
  };
  var indentMore = ({ state, dispatch }) => {
    if (state.readOnly)
      return false;
    dispatch(state.update(changeBySelectedLine(state, (line, changes) => {
      changes.push({ from: line.from, insert: state.facet(indentUnit) });
    }), { userEvent: "input.indent" }));
    return true;
  };
  var indentLess = ({ state, dispatch }) => {
    if (state.readOnly)
      return false;
    dispatch(state.update(changeBySelectedLine(state, (line, changes) => {
      let space = /^\s*/.exec(line.text)[0];
      if (!space)
        return;
      let col = countColumn(space, state.tabSize), keep = 0;
      let insert2 = indentString(state, Math.max(0, col - getIndentUnit(state)));
      while (keep < space.length && keep < insert2.length && space.charCodeAt(keep) == insert2.charCodeAt(keep))
        keep++;
      changes.push({ from: line.from + keep, to: line.from + space.length, insert: insert2.slice(keep) });
    }), { userEvent: "delete.dedent" }));
    return true;
  };
  var toggleTabFocusMode = (view) => {
    view.setTabFocusMode();
    return true;
  };
  var emacsStyleKeymap = [
    { key: "Ctrl-b", run: cursorCharLeft, shift: selectCharLeft, preventDefault: true },
    { key: "Ctrl-f", run: cursorCharRight, shift: selectCharRight },
    { key: "Ctrl-p", run: cursorLineUp, shift: selectLineUp },
    { key: "Ctrl-n", run: cursorLineDown, shift: selectLineDown },
    { key: "Ctrl-a", run: cursorLineStart, shift: selectLineStart },
    { key: "Ctrl-e", run: cursorLineEnd, shift: selectLineEnd },
    { key: "Ctrl-d", run: deleteCharForward },
    { key: "Ctrl-h", run: deleteCharBackward },
    { key: "Ctrl-k", run: deleteToLineEnd },
    { key: "Ctrl-Alt-h", run: deleteGroupBackward },
    { key: "Ctrl-o", run: splitLine },
    { key: "Ctrl-t", run: transposeChars },
    { key: "Ctrl-v", run: cursorPageDown }
  ];
  var standardKeymap = /* @__PURE__ */ [
    { key: "ArrowLeft", run: cursorCharLeft, shift: selectCharLeft, preventDefault: true },
    { key: "Mod-ArrowLeft", mac: "Alt-ArrowLeft", run: cursorGroupLeft, shift: selectGroupLeft, preventDefault: true },
    { mac: "Cmd-ArrowLeft", run: cursorLineBoundaryLeft, shift: selectLineBoundaryLeft, preventDefault: true },
    { key: "ArrowRight", run: cursorCharRight, shift: selectCharRight, preventDefault: true },
    { key: "Mod-ArrowRight", mac: "Alt-ArrowRight", run: cursorGroupRight, shift: selectGroupRight, preventDefault: true },
    { mac: "Cmd-ArrowRight", run: cursorLineBoundaryRight, shift: selectLineBoundaryRight, preventDefault: true },
    { key: "ArrowUp", run: cursorLineUp, shift: selectLineUp, preventDefault: true },
    { mac: "Cmd-ArrowUp", run: cursorDocStart, shift: selectDocStart },
    { mac: "Ctrl-ArrowUp", run: cursorPageUp, shift: selectPageUp },
    { key: "ArrowDown", run: cursorLineDown, shift: selectLineDown, preventDefault: true },
    { mac: "Cmd-ArrowDown", run: cursorDocEnd, shift: selectDocEnd },
    { mac: "Ctrl-ArrowDown", run: cursorPageDown, shift: selectPageDown },
    { key: "PageUp", run: cursorPageUp, shift: selectPageUp },
    { key: "PageDown", run: cursorPageDown, shift: selectPageDown },
    { key: "Home", run: cursorLineBoundaryBackward, shift: selectLineBoundaryBackward, preventDefault: true },
    { key: "Mod-Home", run: cursorDocStart, shift: selectDocStart },
    { key: "End", run: cursorLineBoundaryForward, shift: selectLineBoundaryForward, preventDefault: true },
    { key: "Mod-End", run: cursorDocEnd, shift: selectDocEnd },
    { key: "Enter", run: insertNewlineAndIndent, shift: insertNewlineAndIndent },
    { key: "Mod-a", run: selectAll },
    { key: "Backspace", run: deleteCharBackward, shift: deleteCharBackward, preventDefault: true },
    { key: "Delete", run: deleteCharForward, preventDefault: true },
    { key: "Mod-Backspace", mac: "Alt-Backspace", run: deleteGroupBackward, preventDefault: true },
    { key: "Mod-Delete", mac: "Alt-Delete", run: deleteGroupForward, preventDefault: true },
    { mac: "Mod-Backspace", run: deleteLineBoundaryBackward, preventDefault: true },
    { mac: "Mod-Delete", run: deleteLineBoundaryForward, preventDefault: true }
  ].concat(/* @__PURE__ */ emacsStyleKeymap.map((b) => ({ mac: b.key, run: b.run, shift: b.shift })));
  var defaultKeymap = /* @__PURE__ */ [
    { key: "Alt-ArrowLeft", mac: "Ctrl-ArrowLeft", run: cursorSyntaxLeft, shift: selectSyntaxLeft },
    { key: "Alt-ArrowRight", mac: "Ctrl-ArrowRight", run: cursorSyntaxRight, shift: selectSyntaxRight },
    { key: "Alt-ArrowUp", run: moveLineUp },
    { key: "Shift-Alt-ArrowUp", run: copyLineUp },
    { key: "Alt-ArrowDown", run: moveLineDown },
    { key: "Shift-Alt-ArrowDown", run: copyLineDown },
    { key: "Mod-Alt-ArrowUp", run: addCursorAbove },
    { key: "Mod-Alt-ArrowDown", run: addCursorBelow },
    { key: "Escape", run: simplifySelection },
    { key: "Mod-Enter", run: insertBlankLine },
    { key: "Alt-l", mac: "Ctrl-l", run: selectLine },
    { key: "Mod-i", run: selectParentSyntax, preventDefault: true },
    { key: "Mod-[", run: indentLess },
    { key: "Mod-]", run: indentMore },
    { key: "Mod-Alt-\\", run: indentSelection },
    { key: "Shift-Mod-k", run: deleteLine },
    { key: "Shift-Mod-\\", run: cursorMatchingBracket },
    { key: "Mod-/", run: toggleComment },
    { key: "Alt-A", run: toggleBlockComment },
    { key: "Ctrl-m", mac: "Shift-Alt-m", run: toggleTabFocusMode }
  ].concat(standardKeymap);
  var indentWithTab = { key: "Tab", run: indentMore, shift: indentLess };

  // node_modules/@codemirror/autocomplete/dist/index.js
  var CompletionContext = class {
    /**
    Create a new completion context. (Mostly useful for testing
    completion sources—in the editor, the extension will create
    these for you.)
    */
    constructor(state, pos, explicit, view) {
      this.state = state;
      this.pos = pos;
      this.explicit = explicit;
      this.view = view;
      this.abortListeners = [];
      this.abortOnDocChange = false;
    }
    /**
    Get the extent, content, and (if there is a token) type of the
    token before `this.pos`.
    */
    tokenBefore(types2) {
      let token = syntaxTree(this.state).resolveInner(this.pos, -1);
      while (token && types2.indexOf(token.name) < 0)
        token = token.parent;
      return token ? {
        from: token.from,
        to: this.pos,
        text: this.state.sliceDoc(token.from, this.pos),
        type: token.type
      } : null;
    }
    /**
    Get the match of the given expression directly before the
    cursor.
    */
    matchBefore(expr) {
      let line = this.state.doc.lineAt(this.pos);
      let start = Math.max(line.from, this.pos - 250);
      let str = line.text.slice(start - line.from, this.pos - line.from);
      let found = str.search(ensureAnchor(expr, false));
      return found < 0 ? null : { from: start + found, to: this.pos, text: str.slice(found) };
    }
    /**
    Yields true when the query has been aborted. Can be useful in
    asynchronous queries to avoid doing work that will be ignored.
    */
    get aborted() {
      return this.abortListeners == null;
    }
    /**
    Allows you to register abort handlers, which will be called when
    the query is
    [aborted](https://codemirror.net/6/docs/ref/#autocomplete.CompletionContext.aborted).
    
    By default, running queries will not be aborted for regular
    typing or backspacing, on the assumption that they are likely to
    return a result with a
    [`validFor`](https://codemirror.net/6/docs/ref/#autocomplete.CompletionResult.validFor) field that
    allows the result to be used after all. Passing `onDocChange:
    true` will cause this query to be aborted for any document
    change.
    */
    addEventListener(type, listener, options) {
      if (type == "abort" && this.abortListeners) {
        this.abortListeners.push(listener);
        if (options && options.onDocChange)
          this.abortOnDocChange = true;
      }
    }
  };
  function toSet(chars) {
    let flat = Object.keys(chars).join("");
    let words = /\w/.test(flat);
    if (words)
      flat = flat.replace(/\w/g, "");
    return `[${words ? "\\w" : ""}${flat.replace(/[^\w\s]/g, "\\$&")}]`;
  }
  function prefixMatch(options) {
    let first = /* @__PURE__ */ Object.create(null), rest = /* @__PURE__ */ Object.create(null);
    for (let { label } of options) {
      first[label[0]] = true;
      for (let i = 1; i < label.length; i++)
        rest[label[i]] = true;
    }
    let source = toSet(first) + toSet(rest) + "*$";
    return [new RegExp("^" + source), new RegExp(source)];
  }
  function completeFromList(list) {
    let options = list.map((o) => typeof o == "string" ? { label: o } : o);
    let [validFor, match] = options.every((o) => /^\w+$/.test(o.label)) ? [/\w*$/, /\w+$/] : prefixMatch(options);
    return (context) => {
      let token = context.matchBefore(match);
      return token || context.explicit ? { from: token ? token.from : context.pos, options, validFor } : null;
    };
  }
  var Option = class {
    constructor(completion, source, match, score2) {
      this.completion = completion;
      this.source = source;
      this.match = match;
      this.score = score2;
    }
  };
  function cur(state) {
    return state.selection.main.from;
  }
  function ensureAnchor(expr, start) {
    var _a2;
    let { source } = expr;
    let addStart = start && source[0] != "^", addEnd = source[source.length - 1] != "$";
    if (!addStart && !addEnd)
      return expr;
    return new RegExp(`${addStart ? "^" : ""}(?:${source})${addEnd ? "$" : ""}`, (_a2 = expr.flags) !== null && _a2 !== void 0 ? _a2 : expr.ignoreCase ? "i" : "");
  }
  var pickedCompletion = /* @__PURE__ */ Annotation.define();
  function insertCompletionText(state, text, from, to) {
    let { main } = state.selection, fromOff = from - main.from, toOff = to - main.from;
    return {
      ...state.changeByRange((range) => {
        if (range != main && from != to && state.sliceDoc(range.from + fromOff, range.from + toOff) != state.sliceDoc(from, to))
          return { range };
        let lines = state.toText(text);
        return {
          changes: { from: range.from + fromOff, to: to == main.from ? range.to : range.from + toOff, insert: lines },
          range: EditorSelection.cursor(range.from + fromOff + lines.length)
        };
      }),
      scrollIntoView: true,
      userEvent: "input.complete"
    };
  }
  var SourceCache = /* @__PURE__ */ new WeakMap();
  function asSource(source) {
    if (!Array.isArray(source))
      return source;
    let known = SourceCache.get(source);
    if (!known)
      SourceCache.set(source, known = completeFromList(source));
    return known;
  }
  var startCompletionEffect = /* @__PURE__ */ StateEffect.define();
  var closeCompletionEffect = /* @__PURE__ */ StateEffect.define();
  var FuzzyMatcher = class {
    constructor(pattern) {
      this.pattern = pattern;
      this.chars = [];
      this.folded = [];
      this.any = [];
      this.precise = [];
      this.byWord = [];
      this.score = 0;
      this.matched = [];
      for (let p = 0; p < pattern.length; ) {
        let char = codePointAt2(pattern, p), size = codePointSize2(char);
        this.chars.push(char);
        let part = pattern.slice(p, p + size), upper = part.toUpperCase();
        this.folded.push(codePointAt2(upper == part ? part.toLowerCase() : upper, 0));
        p += size;
      }
      this.astral = pattern.length != this.chars.length;
    }
    ret(score2, matched) {
      this.score = score2;
      this.matched = matched;
      return this;
    }
    // Matches a given word (completion) against the pattern (input).
    // Will return a boolean indicating whether there was a match and,
    // on success, set `this.score` to the score, `this.matched` to an
    // array of `from, to` pairs indicating the matched parts of `word`.
    //
    // The score is a number that is more negative the worse the match
    // is. See `Penalty` above.
    match(word) {
      if (this.pattern.length == 0)
        return this.ret(-100, []);
      if (word.length < this.pattern.length)
        return null;
      let { chars, folded, any, precise, byWord } = this;
      if (chars.length == 1) {
        let first = codePointAt2(word, 0), firstSize = codePointSize2(first);
        let score2 = firstSize == word.length ? 0 : -100;
        if (first == chars[0]) ;
        else if (first == folded[0])
          score2 += -200;
        else
          return null;
        return this.ret(score2, [0, firstSize]);
      }
      let direct = word.indexOf(this.pattern);
      if (direct == 0)
        return this.ret(word.length == this.pattern.length ? 0 : -100, [0, this.pattern.length]);
      let len = chars.length, anyTo = 0;
      if (direct < 0) {
        for (let i = 0, e = Math.min(word.length, 200); i < e && anyTo < len; ) {
          let next = codePointAt2(word, i);
          if (next == chars[anyTo] || next == folded[anyTo])
            any[anyTo++] = i;
          i += codePointSize2(next);
        }
        if (anyTo < len)
          return null;
      }
      let preciseTo = 0;
      let byWordTo = 0, byWordFolded = false;
      let adjacentTo = 0, adjacentStart = -1, adjacentEnd = -1;
      let hasLower = /[a-z]/.test(word), wordAdjacent = true;
      for (let i = 0, e = Math.min(word.length, 200), prevType = 0; i < e && byWordTo < len; ) {
        let next = codePointAt2(word, i);
        if (direct < 0) {
          if (preciseTo < len && next == chars[preciseTo])
            precise[preciseTo++] = i;
          if (adjacentTo < len) {
            if (next == chars[adjacentTo] || next == folded[adjacentTo]) {
              if (adjacentTo == 0)
                adjacentStart = i;
              adjacentEnd = i + 1;
              adjacentTo++;
            } else {
              adjacentTo = 0;
            }
          }
        }
        let ch, type = next < 255 ? next >= 48 && next <= 57 || next >= 97 && next <= 122 ? 2 : next >= 65 && next <= 90 ? 1 : 0 : (ch = fromCodePoint(next)) != ch.toLowerCase() ? 1 : ch != ch.toUpperCase() ? 2 : 0;
        if (!i || type == 1 && hasLower || prevType == 0 && type != 0) {
          if (chars[byWordTo] == next || folded[byWordTo] == next && (byWordFolded = true))
            byWord[byWordTo++] = i;
          else if (byWord.length)
            wordAdjacent = false;
        }
        prevType = type;
        i += codePointSize2(next);
      }
      if (byWordTo == len && byWord[0] == 0 && wordAdjacent)
        return this.result(-100 + (byWordFolded ? -200 : 0), byWord, word);
      if (adjacentTo == len && adjacentStart == 0)
        return this.ret(-200 - word.length + (adjacentEnd == word.length ? 0 : -100), [0, adjacentEnd]);
      if (direct > -1)
        return this.ret(-700 - word.length, [direct, direct + this.pattern.length]);
      if (adjacentTo == len)
        return this.ret(-200 + -700 - word.length, [adjacentStart, adjacentEnd]);
      if (byWordTo == len)
        return this.result(-100 + (byWordFolded ? -200 : 0) + -700 + (wordAdjacent ? 0 : -1100), byWord, word);
      return chars.length == 2 ? null : this.result((any[0] ? -700 : 0) + -200 + -1100, any, word);
    }
    result(score2, positions, word) {
      let result = [], i = 0;
      for (let pos of positions) {
        let to = pos + (this.astral ? codePointSize2(codePointAt2(word, pos)) : 1);
        if (i && result[i - 1] == pos)
          result[i - 1] = to;
        else {
          result[i++] = pos;
          result[i++] = to;
        }
      }
      return this.ret(score2 - word.length, result);
    }
  };
  var StrictMatcher = class {
    constructor(pattern) {
      this.pattern = pattern;
      this.matched = [];
      this.score = 0;
      this.folded = pattern.toLowerCase();
    }
    match(word) {
      if (word.length < this.pattern.length)
        return null;
      let start = word.slice(0, this.pattern.length);
      let match = start == this.pattern ? 0 : start.toLowerCase() == this.folded ? -200 : null;
      if (match == null)
        return null;
      this.matched = [0, start.length];
      this.score = match + (word.length == this.pattern.length ? 0 : -100);
      return this;
    }
  };
  var completionConfig = /* @__PURE__ */ Facet.define({
    combine(configs) {
      return combineConfig(configs, {
        activateOnTyping: true,
        activateOnCompletion: () => false,
        activateOnTypingDelay: 100,
        selectOnOpen: true,
        override: null,
        closeOnBlur: true,
        maxRenderedOptions: 100,
        defaultKeymap: true,
        tooltipClass: () => "",
        optionClass: () => "",
        aboveCursor: false,
        icons: true,
        addToOptions: [],
        positionInfo: defaultPositionInfo,
        filterStrict: false,
        compareCompletions: (a, b) => (a.sortText || a.label).localeCompare(b.sortText || b.label),
        interactionDelay: 75,
        updateSyncTime: 100
      }, {
        defaultKeymap: (a, b) => a && b,
        closeOnBlur: (a, b) => a && b,
        icons: (a, b) => a && b,
        tooltipClass: (a, b) => (c) => joinClass(a(c), b(c)),
        optionClass: (a, b) => (c) => joinClass(a(c), b(c)),
        addToOptions: (a, b) => a.concat(b),
        filterStrict: (a, b) => a || b
      });
    }
  });
  function joinClass(a, b) {
    return a ? b ? a + " " + b : a : b;
  }
  function defaultPositionInfo(view, list, option, info, space, tooltip) {
    let rtl = view.textDirection == Direction.RTL, left = rtl, narrow = false;
    let side = "top", offset, maxWidth;
    let spaceLeft = list.left - space.left, spaceRight = space.right - list.right;
    let infoWidth = info.right - info.left, infoHeight = info.bottom - info.top;
    if (left && spaceLeft < Math.min(infoWidth, spaceRight))
      left = false;
    else if (!left && spaceRight < Math.min(infoWidth, spaceLeft))
      left = true;
    if (infoWidth <= (left ? spaceLeft : spaceRight)) {
      offset = Math.max(space.top, Math.min(option.top, space.bottom - infoHeight)) - list.top;
      maxWidth = Math.min(400, left ? spaceLeft : spaceRight);
    } else {
      narrow = true;
      maxWidth = Math.min(
        400,
        (rtl ? list.right : space.right - list.left) - 30
        /* Info.Margin */
      );
      let spaceBelow = space.bottom - list.bottom;
      if (spaceBelow >= infoHeight || spaceBelow > list.top) {
        offset = option.bottom - list.top;
      } else {
        side = "bottom";
        offset = list.bottom - option.top;
      }
    }
    let scaleY = (list.bottom - list.top) / tooltip.offsetHeight;
    let scaleX = (list.right - list.left) / tooltip.offsetWidth;
    return {
      style: `${side}: ${offset / scaleY}px; max-width: ${maxWidth / scaleX}px`,
      class: "cm-completionInfo-" + (narrow ? rtl ? "left-narrow" : "right-narrow" : left ? "left" : "right")
    };
  }
  var setSelectedEffect = /* @__PURE__ */ StateEffect.define();
  function optionContent(config) {
    let content2 = config.addToOptions.slice();
    if (config.icons)
      content2.push({
        render(completion) {
          let icon = document.createElement("div");
          icon.classList.add("cm-completionIcon");
          if (completion.type)
            icon.classList.add(...completion.type.split(/\s+/g).map((cls) => "cm-completionIcon-" + cls));
          icon.setAttribute("aria-hidden", "true");
          return icon;
        },
        position: 20
      });
    content2.push({
      render(completion, _s, _v, match) {
        let labelElt = document.createElement("span");
        labelElt.className = "cm-completionLabel";
        let label = completion.displayLabel || completion.label, off = 0;
        for (let j = 0; j < match.length; ) {
          let from = match[j++], to = match[j++];
          if (from > off)
            labelElt.appendChild(document.createTextNode(label.slice(off, from)));
          let span = labelElt.appendChild(document.createElement("span"));
          span.appendChild(document.createTextNode(label.slice(from, to)));
          span.className = "cm-completionMatchedText";
          off = to;
        }
        if (off < label.length)
          labelElt.appendChild(document.createTextNode(label.slice(off)));
        return labelElt;
      },
      position: 50
    }, {
      render(completion) {
        if (!completion.detail)
          return null;
        let detailElt = document.createElement("span");
        detailElt.className = "cm-completionDetail";
        detailElt.textContent = completion.detail;
        return detailElt;
      },
      position: 80
    });
    return content2.sort((a, b) => a.position - b.position).map((a) => a.render);
  }
  function rangeAroundSelected(total, selected, max) {
    if (total <= max)
      return { from: 0, to: total };
    if (selected < 0)
      selected = 0;
    if (selected <= total >> 1) {
      let off2 = Math.floor(selected / max);
      return { from: off2 * max, to: (off2 + 1) * max };
    }
    let off = Math.floor((total - selected) / max);
    return { from: total - (off + 1) * max, to: total - off * max };
  }
  var CompletionTooltip = class {
    constructor(view, stateField, applyCompletion2) {
      this.view = view;
      this.stateField = stateField;
      this.applyCompletion = applyCompletion2;
      this.info = null;
      this.infoDestroy = null;
      this.placeInfoReq = {
        read: () => this.measureInfo(),
        write: (pos) => this.placeInfo(pos),
        key: this
      };
      this.space = null;
      this.currentClass = "";
      let cState = view.state.field(stateField);
      let { options, selected } = cState.open;
      let config = view.state.facet(completionConfig);
      this.optionContent = optionContent(config);
      this.optionClass = config.optionClass;
      this.tooltipClass = config.tooltipClass;
      this.range = rangeAroundSelected(options.length, selected, config.maxRenderedOptions);
      this.dom = document.createElement("div");
      this.dom.className = "cm-tooltip-autocomplete";
      this.updateTooltipClass(view.state);
      this.dom.addEventListener("mousedown", (e) => {
        let { options: options2 } = view.state.field(stateField).open;
        for (let dom = e.target, match; dom && dom != this.dom; dom = dom.parentNode) {
          if (dom.nodeName == "LI" && (match = /-(\d+)$/.exec(dom.id)) && +match[1] < options2.length) {
            this.applyCompletion(view, options2[+match[1]]);
            e.preventDefault();
            return;
          }
        }
        if (e.target == this.list) {
          let move = this.list.classList.contains("cm-completionListIncompleteTop") && e.clientY < this.list.firstChild.getBoundingClientRect().top ? this.range.from - 1 : this.list.classList.contains("cm-completionListIncompleteBottom") && e.clientY > this.list.lastChild.getBoundingClientRect().bottom ? this.range.to : null;
          if (move != null) {
            view.dispatch({ effects: setSelectedEffect.of(move) });
            e.preventDefault();
          }
        }
      });
      this.dom.addEventListener("focusout", (e) => {
        let state = view.state.field(this.stateField, false);
        if (state && state.tooltip && view.state.facet(completionConfig).closeOnBlur && e.relatedTarget != view.contentDOM)
          view.dispatch({ effects: closeCompletionEffect.of(null) });
      });
      this.showOptions(options, cState.id);
    }
    mount() {
      this.updateSel();
    }
    showOptions(options, id) {
      if (this.list)
        this.list.remove();
      this.list = this.dom.appendChild(this.createListBox(options, id, this.range));
      this.list.addEventListener("scroll", () => {
        if (this.info)
          this.view.requestMeasure(this.placeInfoReq);
      });
    }
    update(update) {
      var _a2;
      let cState = update.state.field(this.stateField);
      let prevState = update.startState.field(this.stateField);
      this.updateTooltipClass(update.state);
      if (cState != prevState) {
        let { options, selected, disabled } = cState.open;
        if (!prevState.open || prevState.open.options != options) {
          this.range = rangeAroundSelected(options.length, selected, update.state.facet(completionConfig).maxRenderedOptions);
          this.showOptions(options, cState.id);
        }
        this.updateSel();
        if (disabled != ((_a2 = prevState.open) === null || _a2 === void 0 ? void 0 : _a2.disabled))
          this.dom.classList.toggle("cm-tooltip-autocomplete-disabled", !!disabled);
      }
    }
    updateTooltipClass(state) {
      let cls = this.tooltipClass(state);
      if (cls != this.currentClass) {
        for (let c of this.currentClass.split(" "))
          if (c)
            this.dom.classList.remove(c);
        for (let c of cls.split(" "))
          if (c)
            this.dom.classList.add(c);
        this.currentClass = cls;
      }
    }
    positioned(space) {
      this.space = space;
      if (this.info)
        this.view.requestMeasure(this.placeInfoReq);
    }
    updateSel() {
      let cState = this.view.state.field(this.stateField), open = cState.open;
      if (open.selected > -1 && open.selected < this.range.from || open.selected >= this.range.to) {
        this.range = rangeAroundSelected(open.options.length, open.selected, this.view.state.facet(completionConfig).maxRenderedOptions);
        this.showOptions(open.options, cState.id);
      }
      let newSel = this.updateSelectedOption(open.selected);
      if (newSel) {
        this.destroyInfo();
        let { completion } = open.options[open.selected];
        let { info } = completion;
        if (!info)
          return;
        let infoResult = typeof info === "string" ? document.createTextNode(info) : info(completion);
        if (!infoResult)
          return;
        if ("then" in infoResult) {
          infoResult.then((obj) => {
            if (obj && this.view.state.field(this.stateField, false) == cState)
              this.addInfoPane(obj, completion);
          }).catch((e) => logException(this.view.state, e, "completion info"));
        } else {
          this.addInfoPane(infoResult, completion);
          newSel.setAttribute("aria-describedby", this.info.id);
        }
      }
    }
    addInfoPane(content2, completion) {
      this.destroyInfo();
      let wrap = this.info = document.createElement("div");
      wrap.className = "cm-tooltip cm-completionInfo";
      wrap.id = "cm-completionInfo-" + Math.floor(Math.random() * 65535).toString(16);
      if (content2.nodeType != null) {
        wrap.appendChild(content2);
        this.infoDestroy = null;
      } else {
        let { dom, destroy } = content2;
        wrap.appendChild(dom);
        this.infoDestroy = destroy || null;
      }
      this.dom.appendChild(wrap);
      this.view.requestMeasure(this.placeInfoReq);
    }
    updateSelectedOption(selected) {
      let set = null;
      for (let opt = this.list.firstChild, i = this.range.from; opt; opt = opt.nextSibling, i++) {
        if (opt.nodeName != "LI" || !opt.id) {
          i--;
        } else if (i == selected) {
          if (!opt.hasAttribute("aria-selected")) {
            opt.setAttribute("aria-selected", "true");
            set = opt;
          }
        } else {
          if (opt.hasAttribute("aria-selected")) {
            opt.removeAttribute("aria-selected");
            opt.removeAttribute("aria-describedby");
          }
        }
      }
      if (set)
        scrollIntoView2(this.list, set);
      return set;
    }
    measureInfo() {
      let sel = this.dom.querySelector("[aria-selected]");
      if (!sel || !this.info)
        return null;
      let listRect = this.dom.getBoundingClientRect();
      let infoRect = this.info.getBoundingClientRect();
      let selRect = sel.getBoundingClientRect();
      let space = this.space;
      if (!space) {
        let docElt = this.dom.ownerDocument.documentElement;
        space = { left: 0, top: 0, right: docElt.clientWidth, bottom: docElt.clientHeight };
      }
      if (selRect.top > Math.min(space.bottom, listRect.bottom) - 10 || selRect.bottom < Math.max(space.top, listRect.top) + 10)
        return null;
      return this.view.state.facet(completionConfig).positionInfo(this.view, listRect, selRect, infoRect, space, this.dom);
    }
    placeInfo(pos) {
      if (this.info) {
        if (pos) {
          if (pos.style)
            this.info.style.cssText = pos.style;
          this.info.className = "cm-tooltip cm-completionInfo " + (pos.class || "");
        } else {
          this.info.style.cssText = "top: -1e6px";
        }
      }
    }
    createListBox(options, id, range) {
      const ul = document.createElement("ul");
      ul.id = id;
      ul.setAttribute("role", "listbox");
      ul.setAttribute("aria-expanded", "true");
      ul.setAttribute("aria-label", this.view.state.phrase("Completions"));
      ul.addEventListener("mousedown", (e) => {
        if (e.target == ul)
          e.preventDefault();
      });
      let curSection = null;
      for (let i = range.from; i < range.to; i++) {
        let { completion, match } = options[i], { section } = completion;
        if (section) {
          let name2 = typeof section == "string" ? section : section.name;
          if (name2 != curSection && (i > range.from || range.from == 0)) {
            curSection = name2;
            if (typeof section != "string" && section.header) {
              ul.appendChild(section.header(section));
            } else {
              let header = ul.appendChild(document.createElement("completion-section"));
              header.textContent = name2;
            }
          }
        }
        const li = ul.appendChild(document.createElement("li"));
        li.id = id + "-" + i;
        li.setAttribute("role", "option");
        let cls = this.optionClass(completion);
        if (cls)
          li.className = cls;
        for (let source of this.optionContent) {
          let node = source(completion, this.view.state, this.view, match);
          if (node)
            li.appendChild(node);
        }
      }
      if (range.from)
        ul.classList.add("cm-completionListIncompleteTop");
      if (range.to < options.length)
        ul.classList.add("cm-completionListIncompleteBottom");
      return ul;
    }
    destroyInfo() {
      if (this.info) {
        if (this.infoDestroy)
          this.infoDestroy();
        this.info.remove();
        this.info = null;
      }
    }
    destroy() {
      this.destroyInfo();
    }
  };
  function completionTooltip(stateField, applyCompletion2) {
    return (view) => new CompletionTooltip(view, stateField, applyCompletion2);
  }
  function scrollIntoView2(container, element) {
    let parent = container.getBoundingClientRect();
    let self = element.getBoundingClientRect();
    let scaleY = parent.height / container.offsetHeight;
    if (self.top < parent.top)
      container.scrollTop -= (parent.top - self.top) / scaleY;
    else if (self.bottom > parent.bottom)
      container.scrollTop += (self.bottom - parent.bottom) / scaleY;
  }
  function score(option) {
    return (option.boost || 0) * 100 + (option.apply ? 10 : 0) + (option.info ? 5 : 0) + (option.type ? 1 : 0);
  }
  function sortOptions(active, state) {
    let options = [];
    let sections = null, dynamicSectionScore = null;
    let addOption = (option) => {
      options.push(option);
      let { section } = option.completion;
      if (section) {
        if (!sections)
          sections = [];
        let name2 = typeof section == "string" ? section : section.name;
        if (!sections.some((s) => s.name == name2))
          sections.push(typeof section == "string" ? { name: name2 } : section);
      }
    };
    let conf = state.facet(completionConfig);
    for (let a of active)
      if (a.hasResult()) {
        let getMatch = a.result.getMatch;
        if (a.result.filter === false) {
          for (let option of a.result.options) {
            addOption(new Option(option, a.source, getMatch ? getMatch(option) : [], 1e9 - options.length));
          }
        } else {
          let pattern = state.sliceDoc(a.from, a.to), match;
          let matcher = conf.filterStrict ? new StrictMatcher(pattern) : new FuzzyMatcher(pattern);
          for (let option of a.result.options)
            if (match = matcher.match(option.label)) {
              let matched = !option.displayLabel ? match.matched : getMatch ? getMatch(option, match.matched) : [];
              let score2 = match.score + (option.boost || 0);
              addOption(new Option(option, a.source, matched, score2));
              if (typeof option.section == "object" && option.section.rank === "dynamic") {
                let { name: name2 } = option.section;
                if (!dynamicSectionScore)
                  dynamicSectionScore = /* @__PURE__ */ Object.create(null);
                dynamicSectionScore[name2] = Math.max(score2, dynamicSectionScore[name2] || -1e9);
              }
            }
        }
      }
    if (sections) {
      let sectionOrder = /* @__PURE__ */ Object.create(null), pos = 0;
      let cmp = (a, b) => {
        return (a.rank === "dynamic" && b.rank === "dynamic" ? dynamicSectionScore[b.name] - dynamicSectionScore[a.name] : 0) || (typeof a.rank == "number" ? a.rank : 1e9) - (typeof b.rank == "number" ? b.rank : 1e9) || (a.name < b.name ? -1 : 1);
      };
      for (let s of sections.sort(cmp)) {
        pos -= 1e5;
        sectionOrder[s.name] = pos;
      }
      for (let option of options) {
        let { section } = option.completion;
        if (section)
          option.score += sectionOrder[typeof section == "string" ? section : section.name];
      }
    }
    let result = [], prev = null;
    let compare2 = conf.compareCompletions;
    for (let opt of options.sort((a, b) => b.score - a.score || compare2(a.completion, b.completion))) {
      let cur2 = opt.completion;
      if (!prev || prev.label != cur2.label || prev.detail != cur2.detail || prev.type != null && cur2.type != null && prev.type != cur2.type || prev.apply != cur2.apply || prev.boost != cur2.boost)
        result.push(opt);
      else if (score(opt.completion) > score(prev))
        result[result.length - 1] = opt;
      prev = opt.completion;
    }
    return result;
  }
  var CompletionDialog = class _CompletionDialog {
    constructor(options, attrs, tooltip, timestamp, selected, disabled) {
      this.options = options;
      this.attrs = attrs;
      this.tooltip = tooltip;
      this.timestamp = timestamp;
      this.selected = selected;
      this.disabled = disabled;
    }
    setSelected(selected, id) {
      return selected == this.selected || selected >= this.options.length ? this : new _CompletionDialog(this.options, makeAttrs(id, selected), this.tooltip, this.timestamp, selected, this.disabled);
    }
    static build(active, state, id, prev, conf, didSetActive) {
      if (prev && !didSetActive && active.some((s) => s.isPending))
        return prev.setDisabled();
      let options = sortOptions(active, state);
      if (!options.length)
        return prev && active.some((a) => a.isPending) ? prev.setDisabled() : null;
      let selected = state.facet(completionConfig).selectOnOpen ? 0 : -1;
      if (prev && prev.selected != selected && prev.selected != -1) {
        let selectedValue = prev.options[prev.selected].completion;
        for (let i = 0; i < options.length; i++)
          if (options[i].completion == selectedValue) {
            selected = i;
            break;
          }
      }
      return new _CompletionDialog(options, makeAttrs(id, selected), {
        pos: active.reduce((a, b) => b.hasResult() ? Math.min(a, b.from) : a, 1e8),
        create: createTooltip,
        above: conf.aboveCursor
      }, prev ? prev.timestamp : Date.now(), selected, false);
    }
    map(changes) {
      return new _CompletionDialog(this.options, this.attrs, { ...this.tooltip, pos: changes.mapPos(this.tooltip.pos) }, this.timestamp, this.selected, this.disabled);
    }
    setDisabled() {
      return new _CompletionDialog(this.options, this.attrs, this.tooltip, this.timestamp, this.selected, true);
    }
  };
  var CompletionState = class _CompletionState {
    constructor(active, id, open) {
      this.active = active;
      this.id = id;
      this.open = open;
    }
    static start() {
      return new _CompletionState(none3, "cm-ac-" + Math.floor(Math.random() * 2e6).toString(36), null);
    }
    update(tr) {
      let { state } = tr, conf = state.facet(completionConfig);
      let sources = conf.override || state.languageDataAt("autocomplete", cur(state)).map(asSource);
      let active = sources.map((source) => {
        let value = this.active.find((s) => s.source == source) || new ActiveSource(
          source,
          this.active.some(
            (a) => a.state != 0
            /* State.Inactive */
          ) ? 1 : 0
          /* State.Inactive */
        );
        return value.update(tr, conf);
      });
      if (active.length == this.active.length && active.every((a, i) => a == this.active[i]))
        active = this.active;
      let open = this.open, didSet = tr.effects.some((e) => e.is(setActiveEffect));
      if (open && tr.docChanged)
        open = open.map(tr.changes);
      if (tr.selection || active.some((a) => a.hasResult() && tr.changes.touchesRange(a.from, a.to)) || !sameResults(active, this.active) || didSet)
        open = CompletionDialog.build(active, state, this.id, open, conf, didSet);
      else if (open && open.disabled && !active.some((a) => a.isPending))
        open = null;
      if (!open && active.every((a) => !a.isPending) && active.some((a) => a.hasResult()))
        active = active.map((a) => a.hasResult() ? new ActiveSource(
          a.source,
          0
          /* State.Inactive */
        ) : a);
      for (let effect of tr.effects)
        if (effect.is(setSelectedEffect))
          open = open && open.setSelected(effect.value, this.id);
      return active == this.active && open == this.open ? this : new _CompletionState(active, this.id, open);
    }
    get tooltip() {
      return this.open ? this.open.tooltip : null;
    }
    get attrs() {
      return this.open ? this.open.attrs : this.active.length ? baseAttrs : noAttrs2;
    }
  };
  function sameResults(a, b) {
    if (a == b)
      return true;
    for (let iA = 0, iB = 0; ; ) {
      while (iA < a.length && !a[iA].hasResult())
        iA++;
      while (iB < b.length && !b[iB].hasResult())
        iB++;
      let endA = iA == a.length, endB = iB == b.length;
      if (endA || endB)
        return endA == endB;
      if (a[iA++].result != b[iB++].result)
        return false;
    }
  }
  var baseAttrs = {
    "aria-autocomplete": "list"
  };
  var noAttrs2 = {};
  function makeAttrs(id, selected) {
    let result = {
      "aria-autocomplete": "list",
      "aria-haspopup": "listbox",
      "aria-controls": id
    };
    if (selected > -1)
      result["aria-activedescendant"] = id + "-" + selected;
    return result;
  }
  var none3 = [];
  function getUpdateType(tr, conf) {
    if (tr.isUserEvent("input.complete")) {
      let completion = tr.annotation(pickedCompletion);
      if (completion && conf.activateOnCompletion(completion))
        return 4 | 8;
    }
    let typing = tr.isUserEvent("input.type");
    return typing && conf.activateOnTyping ? 4 | 1 : typing ? 1 : tr.isUserEvent("delete.backward") ? 2 : tr.selection ? 8 : tr.docChanged ? 16 : 0;
  }
  var ActiveSource = class _ActiveSource {
    constructor(source, state, explicit = false) {
      this.source = source;
      this.state = state;
      this.explicit = explicit;
    }
    hasResult() {
      return false;
    }
    get isPending() {
      return this.state == 1;
    }
    update(tr, conf) {
      let type = getUpdateType(tr, conf), value = this;
      if (type & 8 || type & 16 && this.touches(tr))
        value = new _ActiveSource(
          value.source,
          0
          /* State.Inactive */
        );
      if (type & 4 && value.state == 0)
        value = new _ActiveSource(
          this.source,
          1
          /* State.Pending */
        );
      value = value.updateFor(tr, type);
      for (let effect of tr.effects) {
        if (effect.is(startCompletionEffect))
          value = new _ActiveSource(value.source, 1, effect.value);
        else if (effect.is(closeCompletionEffect))
          value = new _ActiveSource(
            value.source,
            0
            /* State.Inactive */
          );
        else if (effect.is(setActiveEffect)) {
          for (let active of effect.value)
            if (active.source == value.source)
              value = active;
        }
      }
      return value;
    }
    updateFor(tr, type) {
      return this.map(tr.changes);
    }
    map(changes) {
      return this;
    }
    touches(tr) {
      return tr.changes.touchesRange(cur(tr.state));
    }
  };
  var ActiveResult = class _ActiveResult extends ActiveSource {
    constructor(source, explicit, limit, result, from, to) {
      super(source, 3, explicit);
      this.limit = limit;
      this.result = result;
      this.from = from;
      this.to = to;
    }
    hasResult() {
      return true;
    }
    updateFor(tr, type) {
      var _a2;
      if (!(type & 3))
        return this.map(tr.changes);
      let result = this.result;
      if (result.map && !tr.changes.empty)
        result = result.map(result, tr.changes);
      let from = tr.changes.mapPos(this.from), to = tr.changes.mapPos(this.to, 1);
      let pos = cur(tr.state);
      if (pos > to || !result || type & 2 && (cur(tr.startState) == this.from || pos < this.limit))
        return new ActiveSource(
          this.source,
          type & 4 ? 1 : 0
          /* State.Inactive */
        );
      let limit = tr.changes.mapPos(this.limit);
      if (checkValid(result.validFor, tr.state, from, to))
        return new _ActiveResult(this.source, this.explicit, limit, result, from, to);
      if (result.update && (result = result.update(result, from, to, new CompletionContext(tr.state, pos, false))))
        return new _ActiveResult(this.source, this.explicit, limit, result, result.from, (_a2 = result.to) !== null && _a2 !== void 0 ? _a2 : cur(tr.state));
      return new ActiveSource(this.source, 1, this.explicit);
    }
    map(mapping) {
      if (mapping.empty)
        return this;
      let result = this.result.map ? this.result.map(this.result, mapping) : this.result;
      if (!result)
        return new ActiveSource(
          this.source,
          0
          /* State.Inactive */
        );
      return new _ActiveResult(this.source, this.explicit, mapping.mapPos(this.limit), this.result, mapping.mapPos(this.from), mapping.mapPos(this.to, 1));
    }
    touches(tr) {
      return tr.changes.touchesRange(this.from, this.to);
    }
  };
  function checkValid(validFor, state, from, to) {
    if (!validFor)
      return false;
    let text = state.sliceDoc(from, to);
    return typeof validFor == "function" ? validFor(text, from, to, state) : ensureAnchor(validFor, true).test(text);
  }
  var setActiveEffect = /* @__PURE__ */ StateEffect.define({
    map(sources, mapping) {
      return sources.map((s) => s.map(mapping));
    }
  });
  var completionState = /* @__PURE__ */ StateField.define({
    create() {
      return CompletionState.start();
    },
    update(value, tr) {
      return value.update(tr);
    },
    provide: (f) => [
      showTooltip.from(f, (val) => val.tooltip),
      EditorView.contentAttributes.from(f, (state) => state.attrs)
    ]
  });
  function applyCompletion(view, option) {
    const apply = option.completion.apply || option.completion.label;
    let result = view.state.field(completionState).active.find((a) => a.source == option.source);
    if (!(result instanceof ActiveResult))
      return false;
    if (typeof apply == "string")
      view.dispatch({
        ...insertCompletionText(view.state, apply, result.from, result.to),
        annotations: pickedCompletion.of(option.completion)
      });
    else
      apply(view, option.completion, result.from, result.to);
    return true;
  }
  var createTooltip = /* @__PURE__ */ completionTooltip(completionState, applyCompletion);
  function moveCompletionSelection(forward, by = "option") {
    return (view) => {
      let cState = view.state.field(completionState, false);
      if (!cState || !cState.open || cState.open.disabled || Date.now() - cState.open.timestamp < view.state.facet(completionConfig).interactionDelay)
        return false;
      let step = 1, tooltip;
      if (by == "page" && (tooltip = getTooltip(view, cState.open.tooltip)))
        step = Math.max(2, Math.floor(tooltip.dom.offsetHeight / tooltip.dom.querySelector("li").offsetHeight) - 1);
      let { length } = cState.open.options;
      let selected = cState.open.selected > -1 ? cState.open.selected + step * (forward ? 1 : -1) : forward ? 0 : length - 1;
      if (selected < 0)
        selected = by == "page" ? 0 : length - 1;
      else if (selected >= length)
        selected = by == "page" ? length - 1 : 0;
      view.dispatch({ effects: setSelectedEffect.of(selected) });
      return true;
    };
  }
  var acceptCompletion = (view) => {
    let cState = view.state.field(completionState, false);
    if (view.state.readOnly || !cState || !cState.open || cState.open.selected < 0 || cState.open.disabled || Date.now() - cState.open.timestamp < view.state.facet(completionConfig).interactionDelay)
      return false;
    return applyCompletion(view, cState.open.options[cState.open.selected]);
  };
  var startCompletion = (view) => {
    let cState = view.state.field(completionState, false);
    if (!cState)
      return false;
    view.dispatch({ effects: startCompletionEffect.of(true) });
    return true;
  };
  var closeCompletion = (view) => {
    let cState = view.state.field(completionState, false);
    if (!cState || !cState.active.some(
      (a) => a.state != 0
      /* State.Inactive */
    ))
      return false;
    view.dispatch({ effects: closeCompletionEffect.of(null) });
    return true;
  };
  var RunningQuery = class {
    constructor(active, context) {
      this.active = active;
      this.context = context;
      this.time = Date.now();
      this.updates = [];
      this.done = void 0;
    }
  };
  var MaxUpdateCount = 50;
  var MinAbortTime = 1e3;
  var completionPlugin = /* @__PURE__ */ ViewPlugin.fromClass(class {
    constructor(view) {
      this.view = view;
      this.debounceUpdate = -1;
      this.running = [];
      this.debounceAccept = -1;
      this.pendingStart = false;
      this.composing = 0;
      for (let active of view.state.field(completionState).active)
        if (active.isPending)
          this.startQuery(active);
    }
    update(update) {
      let cState = update.state.field(completionState);
      let conf = update.state.facet(completionConfig);
      if (!update.selectionSet && !update.docChanged && update.startState.field(completionState) == cState)
        return;
      let doesReset = update.transactions.some((tr) => {
        let type = getUpdateType(tr, conf);
        return type & 8 || (tr.selection || tr.docChanged) && !(type & 3);
      });
      for (let i = 0; i < this.running.length; i++) {
        let query = this.running[i];
        if (doesReset || query.context.abortOnDocChange && update.docChanged || query.updates.length + update.transactions.length > MaxUpdateCount && Date.now() - query.time > MinAbortTime) {
          for (let handler of query.context.abortListeners) {
            try {
              handler();
            } catch (e) {
              logException(this.view.state, e);
            }
          }
          query.context.abortListeners = null;
          this.running.splice(i--, 1);
        } else {
          query.updates.push(...update.transactions);
        }
      }
      if (this.debounceUpdate > -1)
        clearTimeout(this.debounceUpdate);
      if (update.transactions.some((tr) => tr.effects.some((e) => e.is(startCompletionEffect))))
        this.pendingStart = true;
      let delay = this.pendingStart ? 50 : conf.activateOnTypingDelay;
      this.debounceUpdate = cState.active.some((a) => a.isPending && !this.running.some((q) => q.active.source == a.source)) ? setTimeout(() => this.startUpdate(), delay) : -1;
      if (this.composing != 0)
        for (let tr of update.transactions) {
          if (tr.isUserEvent("input.type"))
            this.composing = 2;
          else if (this.composing == 2 && tr.selection)
            this.composing = 3;
        }
    }
    startUpdate() {
      this.debounceUpdate = -1;
      this.pendingStart = false;
      let { state } = this.view, cState = state.field(completionState);
      for (let active of cState.active) {
        if (active.isPending && !this.running.some((r) => r.active.source == active.source))
          this.startQuery(active);
      }
      if (this.running.length && cState.open && cState.open.disabled)
        this.debounceAccept = setTimeout(() => this.accept(), this.view.state.facet(completionConfig).updateSyncTime);
    }
    startQuery(active) {
      let { state } = this.view, pos = cur(state);
      let context = new CompletionContext(state, pos, active.explicit, this.view);
      let pending = new RunningQuery(active, context);
      this.running.push(pending);
      Promise.resolve(active.source(context)).then((result) => {
        if (!pending.context.aborted) {
          pending.done = result || null;
          this.scheduleAccept();
        }
      }, (err) => {
        this.view.dispatch({ effects: closeCompletionEffect.of(null) });
        logException(this.view.state, err);
      });
    }
    scheduleAccept() {
      if (this.running.every((q) => q.done !== void 0))
        this.accept();
      else if (this.debounceAccept < 0)
        this.debounceAccept = setTimeout(() => this.accept(), this.view.state.facet(completionConfig).updateSyncTime);
    }
    // For each finished query in this.running, try to create a result
    // or, if appropriate, restart the query.
    accept() {
      var _a2;
      if (this.debounceAccept > -1)
        clearTimeout(this.debounceAccept);
      this.debounceAccept = -1;
      let updated = [];
      let conf = this.view.state.facet(completionConfig), cState = this.view.state.field(completionState);
      for (let i = 0; i < this.running.length; i++) {
        let query = this.running[i];
        if (query.done === void 0)
          continue;
        this.running.splice(i--, 1);
        if (query.done) {
          let pos = cur(query.updates.length ? query.updates[0].startState : this.view.state);
          let limit = Math.min(pos, query.done.from + (query.active.explicit ? 0 : 1));
          let active = new ActiveResult(query.active.source, query.active.explicit, limit, query.done, query.done.from, (_a2 = query.done.to) !== null && _a2 !== void 0 ? _a2 : pos);
          for (let tr of query.updates)
            active = active.update(tr, conf);
          if (active.hasResult()) {
            updated.push(active);
            continue;
          }
        }
        let current = cState.active.find((a) => a.source == query.active.source);
        if (current && current.isPending) {
          if (query.done == null) {
            let active = new ActiveSource(
              query.active.source,
              0
              /* State.Inactive */
            );
            for (let tr of query.updates)
              active = active.update(tr, conf);
            if (!active.isPending)
              updated.push(active);
          } else {
            this.startQuery(current);
          }
        }
      }
      if (updated.length || cState.open && cState.open.disabled)
        this.view.dispatch({ effects: setActiveEffect.of(updated) });
    }
  }, {
    eventHandlers: {
      blur(event) {
        let state = this.view.state.field(completionState, false);
        if (state && state.tooltip && this.view.state.facet(completionConfig).closeOnBlur) {
          let dialog = state.open && getTooltip(this.view, state.open.tooltip);
          if (!dialog || !dialog.dom.contains(event.relatedTarget))
            setTimeout(() => this.view.dispatch({ effects: closeCompletionEffect.of(null) }), 10);
        }
      },
      compositionstart() {
        this.composing = 1;
      },
      compositionend() {
        if (this.composing == 3) {
          setTimeout(() => this.view.dispatch({ effects: startCompletionEffect.of(false) }), 20);
        }
        this.composing = 0;
      }
    }
  });
  var windows = typeof navigator == "object" && /* @__PURE__ */ /Win/.test(navigator.platform);
  var commitCharacters = /* @__PURE__ */ Prec.highest(/* @__PURE__ */ EditorView.domEventHandlers({
    keydown(event, view) {
      let field = view.state.field(completionState, false);
      if (!field || !field.open || field.open.disabled || field.open.selected < 0 || event.key.length > 1 || event.ctrlKey && !(windows && event.altKey) || event.metaKey)
        return false;
      let option = field.open.options[field.open.selected];
      let result = field.active.find((a) => a.source == option.source);
      let commitChars = option.completion.commitCharacters || result.result.commitCharacters;
      if (commitChars && commitChars.indexOf(event.key) > -1)
        applyCompletion(view, option);
      return false;
    }
  }));
  var baseTheme2 = /* @__PURE__ */ EditorView.baseTheme({
    ".cm-tooltip.cm-tooltip-autocomplete": {
      "& > ul": {
        fontFamily: "monospace",
        whiteSpace: "nowrap",
        overflow: "hidden auto",
        maxWidth_fallback: "700px",
        maxWidth: "min(700px, 95vw)",
        minWidth: "250px",
        maxHeight: "10em",
        height: "100%",
        listStyle: "none",
        margin: 0,
        padding: 0,
        "& > li, & > completion-section": {
          padding: "1px 3px",
          lineHeight: 1.2
        },
        "& > li": {
          overflowX: "hidden",
          textOverflow: "ellipsis",
          cursor: "pointer"
        },
        "& > completion-section": {
          display: "list-item",
          borderBottom: "1px solid silver",
          paddingLeft: "0.5em",
          opacity: 0.7
        }
      }
    },
    "&light .cm-tooltip-autocomplete ul li[aria-selected]": {
      background: "#17c",
      color: "white"
    },
    "&light .cm-tooltip-autocomplete-disabled ul li[aria-selected]": {
      background: "#777"
    },
    "&dark .cm-tooltip-autocomplete ul li[aria-selected]": {
      background: "#347",
      color: "white"
    },
    "&dark .cm-tooltip-autocomplete-disabled ul li[aria-selected]": {
      background: "#444"
    },
    ".cm-completionListIncompleteTop:before, .cm-completionListIncompleteBottom:after": {
      content: '"\xB7\xB7\xB7"',
      opacity: 0.5,
      display: "block",
      textAlign: "center"
    },
    ".cm-tooltip.cm-completionInfo": {
      position: "absolute",
      padding: "3px 9px",
      width: "max-content",
      maxWidth: `${400}px`,
      boxSizing: "border-box",
      whiteSpace: "pre-line"
    },
    ".cm-completionInfo.cm-completionInfo-left": { right: "100%" },
    ".cm-completionInfo.cm-completionInfo-right": { left: "100%" },
    ".cm-completionInfo.cm-completionInfo-left-narrow": { right: `${30}px` },
    ".cm-completionInfo.cm-completionInfo-right-narrow": { left: `${30}px` },
    "&light .cm-snippetField": { backgroundColor: "#00000022" },
    "&dark .cm-snippetField": { backgroundColor: "#ffffff22" },
    ".cm-snippetFieldPosition": {
      verticalAlign: "text-top",
      width: 0,
      height: "1.15em",
      display: "inline-block",
      margin: "0 -0.7px -.7em",
      borderLeft: "1.4px dotted #888"
    },
    ".cm-completionMatchedText": {
      textDecoration: "underline"
    },
    ".cm-completionDetail": {
      marginLeft: "0.5em",
      fontStyle: "italic"
    },
    ".cm-completionIcon": {
      fontSize: "90%",
      width: ".8em",
      display: "inline-block",
      textAlign: "center",
      paddingRight: ".6em",
      opacity: "0.6",
      boxSizing: "content-box"
    },
    ".cm-completionIcon-function, .cm-completionIcon-method": {
      "&:after": { content: "'\u0192'" }
    },
    ".cm-completionIcon-class": {
      "&:after": { content: "'\u25CB'" }
    },
    ".cm-completionIcon-interface": {
      "&:after": { content: "'\u25CC'" }
    },
    ".cm-completionIcon-variable": {
      "&:after": { content: "'\u{1D465}'" }
    },
    ".cm-completionIcon-constant": {
      "&:after": { content: "'\u{1D436}'" }
    },
    ".cm-completionIcon-type": {
      "&:after": { content: "'\u{1D461}'" }
    },
    ".cm-completionIcon-enum": {
      "&:after": { content: "'\u222A'" }
    },
    ".cm-completionIcon-property": {
      "&:after": { content: "'\u25A1'" }
    },
    ".cm-completionIcon-keyword": {
      "&:after": { content: "'\u{1F511}\uFE0E'" }
      // Disable emoji rendering
    },
    ".cm-completionIcon-namespace": {
      "&:after": { content: "'\u25A2'" }
    },
    ".cm-completionIcon-text": {
      "&:after": { content: "'abc'", fontSize: "50%", verticalAlign: "middle" }
    }
  });
  var closedBracket = /* @__PURE__ */ new class extends RangeValue {
  }();
  closedBracket.startSide = 1;
  closedBracket.endSide = -1;
  var android = typeof navigator == "object" && /* @__PURE__ */ /Android\b/.test(navigator.userAgent);
  function autocompletion(config = {}) {
    return [
      commitCharacters,
      completionState,
      completionConfig.of(config),
      completionPlugin,
      completionKeymapExt,
      baseTheme2
    ];
  }
  var completionKeymap = [
    { key: "Ctrl-Space", run: startCompletion },
    { mac: "Alt-`", run: startCompletion },
    { mac: "Alt-i", run: startCompletion },
    { key: "Escape", run: closeCompletion },
    { key: "ArrowDown", run: /* @__PURE__ */ moveCompletionSelection(true) },
    { key: "ArrowUp", run: /* @__PURE__ */ moveCompletionSelection(false) },
    { key: "PageDown", run: /* @__PURE__ */ moveCompletionSelection(true, "page") },
    { key: "PageUp", run: /* @__PURE__ */ moveCompletionSelection(false, "page") },
    { key: "Enter", run: acceptCompletion }
  ];
  var completionKeymapExt = /* @__PURE__ */ Prec.highest(/* @__PURE__ */ keymap.computeN([completionConfig], (state) => state.facet(completionConfig).defaultKeymap ? [completionKeymap] : []));

  // node_modules/@codemirror/search/dist/index.js
  var basicNormalize = typeof String.prototype.normalize == "function" ? (x) => x.normalize("NFKD") : (x) => x;
  var SearchCursor = class {
    /**
    Create a text cursor. The query is the search string, `from` to
    `to` provides the region to search.
    
    When `normalize` is given, it will be called, on both the query
    string and the content it is matched against, before comparing.
    You can, for example, create a case-insensitive search by
    passing `s => s.toLowerCase()`.
    
    Text is always normalized with
    [`.normalize("NFKD")`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/normalize)
    (when supported).
    */
    constructor(text, query, from = 0, to = text.length, normalize, test) {
      this.test = test;
      this.value = { from: 0, to: 0 };
      this.done = false;
      this.matches = [];
      this.buffer = "";
      this.bufferPos = 0;
      this.iter = text.iterRange(from, to);
      this.bufferStart = from;
      this.normalize = normalize ? (x) => normalize(basicNormalize(x)) : basicNormalize;
      this.query = this.normalize(query);
    }
    peek() {
      if (this.bufferPos == this.buffer.length) {
        this.bufferStart += this.buffer.length;
        this.iter.next();
        if (this.iter.done)
          return -1;
        this.bufferPos = 0;
        this.buffer = this.iter.value;
      }
      return codePointAt2(this.buffer, this.bufferPos);
    }
    /**
    Look for the next match. Updates the iterator's
    [`value`](https://codemirror.net/6/docs/ref/#search.SearchCursor.value) and
    [`done`](https://codemirror.net/6/docs/ref/#search.SearchCursor.done) properties. Should be called
    at least once before using the cursor.
    */
    next() {
      while (this.matches.length)
        this.matches.pop();
      return this.nextOverlapping();
    }
    /**
    The `next` method will ignore matches that partially overlap a
    previous match. This method behaves like `next`, but includes
    such matches.
    */
    nextOverlapping() {
      for (; ; ) {
        let next = this.peek();
        if (next < 0) {
          this.done = true;
          return this;
        }
        let str = fromCodePoint(next), start = this.bufferStart + this.bufferPos;
        this.bufferPos += codePointSize2(next);
        let norm = this.normalize(str);
        if (norm.length)
          for (let i = 0, pos = start; ; i++) {
            let code = norm.charCodeAt(i);
            let match = this.match(code, pos, this.bufferPos + this.bufferStart);
            if (i == norm.length - 1) {
              if (match) {
                this.value = match;
                return this;
              }
              break;
            }
            if (pos == start && i < str.length && str.charCodeAt(i) == code)
              pos++;
          }
      }
    }
    match(code, pos, end) {
      let match = null;
      for (let i = 0; i < this.matches.length; i += 2) {
        let index = this.matches[i], keep = false;
        if (this.query.charCodeAt(index) == code) {
          if (index == this.query.length - 1) {
            match = { from: this.matches[i + 1], to: end };
          } else {
            this.matches[i]++;
            keep = true;
          }
        }
        if (!keep) {
          this.matches.splice(i, 2);
          i -= 2;
        }
      }
      if (this.query.charCodeAt(0) == code) {
        if (this.query.length == 1)
          match = { from: pos, to: end };
        else
          this.matches.push(1, pos);
      }
      if (match && this.test && !this.test(match.from, match.to, this.buffer, this.bufferStart))
        match = null;
      return match;
    }
  };
  if (typeof Symbol != "undefined")
    SearchCursor.prototype[Symbol.iterator] = function() {
      return this;
    };
  var empty = { from: -1, to: -1, match: /* @__PURE__ */ /.*/.exec("") };
  var baseFlags = "gm" + (/x/.unicode == null ? "" : "u");
  var RegExpCursor = class {
    /**
    Create a cursor that will search the given range in the given
    document. `query` should be the raw pattern (as you'd pass it to
    `new RegExp`).
    */
    constructor(text, query, options, from = 0, to = text.length) {
      this.text = text;
      this.to = to;
      this.curLine = "";
      this.done = false;
      this.value = empty;
      if (/\\[sWDnr]|\n|\r|\[\^/.test(query))
        return new MultilineRegExpCursor(text, query, options, from, to);
      this.re = new RegExp(query, baseFlags + ((options === null || options === void 0 ? void 0 : options.ignoreCase) ? "i" : ""));
      this.test = options === null || options === void 0 ? void 0 : options.test;
      this.iter = text.iter();
      let startLine = text.lineAt(from);
      this.curLineStart = startLine.from;
      this.matchPos = toCharEnd(text, from);
      this.getLine(this.curLineStart);
    }
    getLine(skip) {
      this.iter.next(skip);
      if (this.iter.lineBreak) {
        this.curLine = "";
      } else {
        this.curLine = this.iter.value;
        if (this.curLineStart + this.curLine.length > this.to)
          this.curLine = this.curLine.slice(0, this.to - this.curLineStart);
        this.iter.next();
      }
    }
    nextLine() {
      this.curLineStart = this.curLineStart + this.curLine.length + 1;
      if (this.curLineStart > this.to)
        this.curLine = "";
      else
        this.getLine(0);
    }
    /**
    Move to the next match, if there is one.
    */
    next() {
      for (let off = this.matchPos - this.curLineStart; ; ) {
        this.re.lastIndex = off;
        let match = this.matchPos <= this.to && this.re.exec(this.curLine);
        if (match) {
          let from = this.curLineStart + match.index, to = from + match[0].length;
          this.matchPos = toCharEnd(this.text, to + (from == to ? 1 : 0));
          if (from == this.curLineStart + this.curLine.length)
            this.nextLine();
          if ((from < to || from > this.value.to) && (!this.test || this.test(from, to, match))) {
            this.value = { from, to, match };
            return this;
          }
          off = this.matchPos - this.curLineStart;
        } else if (this.curLineStart + this.curLine.length < this.to) {
          this.nextLine();
          off = 0;
        } else {
          this.done = true;
          return this;
        }
      }
    }
  };
  var flattened = /* @__PURE__ */ new WeakMap();
  var FlattenedDoc = class _FlattenedDoc {
    constructor(from, text) {
      this.from = from;
      this.text = text;
    }
    get to() {
      return this.from + this.text.length;
    }
    static get(doc2, from, to) {
      let cached = flattened.get(doc2);
      if (!cached || cached.from >= to || cached.to <= from) {
        let flat = new _FlattenedDoc(from, doc2.sliceString(from, to));
        flattened.set(doc2, flat);
        return flat;
      }
      if (cached.from == from && cached.to == to)
        return cached;
      let { text, from: cachedFrom } = cached;
      if (cachedFrom > from) {
        text = doc2.sliceString(from, cachedFrom) + text;
        cachedFrom = from;
      }
      if (cached.to < to)
        text += doc2.sliceString(cached.to, to);
      flattened.set(doc2, new _FlattenedDoc(cachedFrom, text));
      return new _FlattenedDoc(from, text.slice(from - cachedFrom, to - cachedFrom));
    }
  };
  var MultilineRegExpCursor = class {
    constructor(text, query, options, from, to) {
      this.text = text;
      this.to = to;
      this.done = false;
      this.value = empty;
      this.matchPos = toCharEnd(text, from);
      this.re = new RegExp(query, baseFlags + ((options === null || options === void 0 ? void 0 : options.ignoreCase) ? "i" : ""));
      this.test = options === null || options === void 0 ? void 0 : options.test;
      this.flat = FlattenedDoc.get(text, from, this.chunkEnd(
        from + 5e3
        /* Chunk.Base */
      ));
    }
    chunkEnd(pos) {
      return pos >= this.to ? this.to : this.text.lineAt(pos).to;
    }
    next() {
      for (; ; ) {
        let off = this.re.lastIndex = this.matchPos - this.flat.from;
        let match = this.re.exec(this.flat.text);
        if (match && !match[0] && match.index == off) {
          this.re.lastIndex = off + 1;
          match = this.re.exec(this.flat.text);
        }
        if (match) {
          let from = this.flat.from + match.index, to = from + match[0].length;
          if ((this.flat.to >= this.to || match.index + match[0].length <= this.flat.text.length - 10) && (!this.test || this.test(from, to, match))) {
            this.value = { from, to, match };
            this.matchPos = toCharEnd(this.text, to + (from == to ? 1 : 0));
            return this;
          }
        }
        if (this.flat.to == this.to) {
          this.done = true;
          return this;
        }
        this.flat = FlattenedDoc.get(this.text, this.flat.from, this.chunkEnd(this.flat.from + this.flat.text.length * 2));
      }
    }
  };
  if (typeof Symbol != "undefined") {
    RegExpCursor.prototype[Symbol.iterator] = MultilineRegExpCursor.prototype[Symbol.iterator] = function() {
      return this;
    };
  }
  function validRegExp(source) {
    try {
      new RegExp(source, baseFlags);
      return true;
    } catch (_a2) {
      return false;
    }
  }
  function toCharEnd(text, pos) {
    if (pos >= text.length)
      return pos;
    let line = text.lineAt(pos), next;
    while (pos < line.to && (next = line.text.charCodeAt(pos - line.from)) >= 56320 && next < 57344)
      pos++;
    return pos;
  }
  var gotoLine = (view) => {
    let { state } = view;
    let line = String(state.doc.lineAt(view.state.selection.main.head).number);
    let { close, result } = showDialog(view, {
      label: state.phrase("Go to line"),
      input: { type: "text", name: "line", value: line },
      focus: true,
      submitLabel: state.phrase("go")
    });
    result.then((form) => {
      let match = form && /^([+-])?(\d+)?(:\d+)?(%)?$/.exec(form.elements["line"].value);
      if (!match) {
        view.dispatch({ effects: close });
        return;
      }
      let startLine = state.doc.lineAt(state.selection.main.head);
      let [, sign, ln, cl, percent] = match;
      let col = cl ? +cl.slice(1) : 0;
      let line2 = ln ? +ln : startLine.number;
      if (ln && percent) {
        let pc = line2 / 100;
        if (sign)
          pc = pc * (sign == "-" ? -1 : 1) + startLine.number / state.doc.lines;
        line2 = Math.round(state.doc.lines * pc);
      } else if (ln && sign) {
        line2 = line2 * (sign == "-" ? -1 : 1) + startLine.number;
      }
      let docLine = state.doc.line(Math.max(1, Math.min(state.doc.lines, line2)));
      let selection = EditorSelection.cursor(docLine.from + Math.max(0, Math.min(col, docLine.length)));
      view.dispatch({
        effects: [close, EditorView.scrollIntoView(selection.from, { y: "center" })],
        selection
      });
    });
    return true;
  };
  var defaultHighlightOptions = {
    highlightWordAroundCursor: false,
    minSelectionLength: 1,
    maxMatches: 100,
    wholeWords: false
  };
  var highlightConfig = /* @__PURE__ */ Facet.define({
    combine(options) {
      return combineConfig(options, defaultHighlightOptions, {
        highlightWordAroundCursor: (a, b) => a || b,
        minSelectionLength: Math.min,
        maxMatches: Math.min
      });
    }
  });
  function highlightSelectionMatches(options) {
    let ext = [defaultTheme, matchHighlighter];
    if (options)
      ext.push(highlightConfig.of(options));
    return ext;
  }
  var matchDeco = /* @__PURE__ */ Decoration.mark({ class: "cm-selectionMatch" });
  var mainMatchDeco = /* @__PURE__ */ Decoration.mark({ class: "cm-selectionMatch cm-selectionMatch-main" });
  function insideWordBoundaries(check, state, from, to) {
    return (from == 0 || check(state.sliceDoc(from - 1, from)) != CharCategory.Word) && (to == state.doc.length || check(state.sliceDoc(to, to + 1)) != CharCategory.Word);
  }
  function insideWord(check, state, from, to) {
    return check(state.sliceDoc(from, from + 1)) == CharCategory.Word && check(state.sliceDoc(to - 1, to)) == CharCategory.Word;
  }
  var matchHighlighter = /* @__PURE__ */ ViewPlugin.fromClass(class {
    constructor(view) {
      this.decorations = this.getDeco(view);
    }
    update(update) {
      if (update.selectionSet || update.docChanged || update.viewportChanged)
        this.decorations = this.getDeco(update.view);
    }
    getDeco(view) {
      let conf = view.state.facet(highlightConfig);
      let { state } = view, sel = state.selection;
      if (sel.ranges.length > 1)
        return Decoration.none;
      let range = sel.main, query, check = null;
      if (range.empty) {
        if (!conf.highlightWordAroundCursor)
          return Decoration.none;
        let word = state.wordAt(range.head);
        if (!word)
          return Decoration.none;
        check = state.charCategorizer(range.head);
        query = state.sliceDoc(word.from, word.to);
      } else {
        let len = range.to - range.from;
        if (len < conf.minSelectionLength || len > 200)
          return Decoration.none;
        if (conf.wholeWords) {
          query = state.sliceDoc(range.from, range.to);
          check = state.charCategorizer(range.head);
          if (!(insideWordBoundaries(check, state, range.from, range.to) && insideWord(check, state, range.from, range.to)))
            return Decoration.none;
        } else {
          query = state.sliceDoc(range.from, range.to);
          if (!query)
            return Decoration.none;
        }
      }
      let deco = [];
      for (let part of view.visibleRanges) {
        let cursor = new SearchCursor(state.doc, query, part.from, part.to);
        while (!cursor.next().done) {
          let { from, to } = cursor.value;
          if (!check || insideWordBoundaries(check, state, from, to)) {
            if (range.empty && from <= range.from && to >= range.to)
              deco.push(mainMatchDeco.range(from, to));
            else if (from >= range.to || to <= range.from)
              deco.push(matchDeco.range(from, to));
            if (deco.length > conf.maxMatches)
              return Decoration.none;
          }
        }
      }
      return Decoration.set(deco);
    }
  }, {
    decorations: (v) => v.decorations
  });
  var defaultTheme = /* @__PURE__ */ EditorView.baseTheme({
    ".cm-selectionMatch": { backgroundColor: "#99ff7780" },
    ".cm-searchMatch .cm-selectionMatch": { backgroundColor: "transparent" }
  });
  var selectWord = ({ state, dispatch }) => {
    let { selection } = state;
    let newSel = EditorSelection.create(selection.ranges.map((range) => state.wordAt(range.head) || EditorSelection.cursor(range.head)), selection.mainIndex);
    if (newSel.eq(selection))
      return false;
    dispatch(state.update({ selection: newSel }));
    return true;
  };
  function findNextOccurrence(state, query) {
    let { main, ranges } = state.selection;
    let word = state.wordAt(main.head), fullWord = word && word.from == main.from && word.to == main.to;
    for (let cycled = false, cursor = new SearchCursor(state.doc, query, ranges[ranges.length - 1].to); ; ) {
      cursor.next();
      if (cursor.done) {
        if (cycled)
          return null;
        cursor = new SearchCursor(state.doc, query, 0, Math.max(0, ranges[ranges.length - 1].from - 1));
        cycled = true;
      } else {
        if (cycled && ranges.some((r) => r.from == cursor.value.from))
          continue;
        if (fullWord) {
          let word2 = state.wordAt(cursor.value.from);
          if (!word2 || word2.from != cursor.value.from || word2.to != cursor.value.to)
            continue;
        }
        return cursor.value;
      }
    }
  }
  var selectNextOccurrence = ({ state, dispatch }) => {
    let { ranges } = state.selection;
    if (ranges.some((sel) => sel.from === sel.to))
      return selectWord({ state, dispatch });
    let searchedText = state.sliceDoc(ranges[0].from, ranges[0].to);
    if (state.selection.ranges.some((r) => state.sliceDoc(r.from, r.to) != searchedText))
      return false;
    let range = findNextOccurrence(state, searchedText);
    if (!range)
      return false;
    dispatch(state.update({
      selection: state.selection.addRange(EditorSelection.range(range.from, range.to), false),
      effects: EditorView.scrollIntoView(range.to)
    }));
    return true;
  };
  var searchConfigFacet = /* @__PURE__ */ Facet.define({
    combine(configs) {
      return combineConfig(configs, {
        top: false,
        caseSensitive: false,
        literal: false,
        regexp: false,
        wholeWord: false,
        createPanel: (view) => new SearchPanel(view),
        scrollToMatch: (range) => EditorView.scrollIntoView(range)
      });
    }
  });
  function search(config) {
    return config ? [searchConfigFacet.of(config), searchExtensions] : searchExtensions;
  }
  var SearchQuery = class {
    /**
    Create a query object.
    */
    constructor(config) {
      this.search = config.search;
      this.caseSensitive = !!config.caseSensitive;
      this.literal = !!config.literal;
      this.regexp = !!config.regexp;
      this.replace = config.replace || "";
      this.valid = !!this.search && (!this.regexp || validRegExp(this.search));
      this.unquoted = this.unquote(this.search);
      this.wholeWord = !!config.wholeWord;
      this.test = config.test;
    }
    /**
    @internal
    */
    unquote(text) {
      return this.literal ? text : text.replace(/\\([nrt\\])/g, (_, ch) => ch == "n" ? "\n" : ch == "r" ? "\r" : ch == "t" ? "	" : "\\");
    }
    /**
    Compare this query to another query.
    */
    eq(other) {
      return this.search == other.search && this.replace == other.replace && this.caseSensitive == other.caseSensitive && this.regexp == other.regexp && this.wholeWord == other.wholeWord && this.test == other.test;
    }
    /**
    @internal
    */
    create() {
      return this.regexp ? new RegExpQuery(this) : new StringQuery(this);
    }
    /**
    Get a search cursor for this query, searching through the given
    range in the given state.
    */
    getCursor(state, from = 0, to) {
      let st = state.doc ? state : EditorState.create({ doc: state });
      if (to == null)
        to = st.doc.length;
      return this.regexp ? regexpCursor(this, st, from, to) : stringCursor(this, st, from, to);
    }
  };
  var QueryType2 = class {
    constructor(spec) {
      this.spec = spec;
    }
  };
  function wrapStringTest(test, state, inner) {
    return (from, to, buffer, bufferPos) => {
      if (inner && !inner(from, to, buffer, bufferPos))
        return false;
      let match = from >= bufferPos && to <= bufferPos + buffer.length ? buffer.slice(from - bufferPos, to - bufferPos) : state.doc.sliceString(from, to);
      return test(match, state, from, to);
    };
  }
  function stringCursor(spec, state, from, to) {
    let test;
    if (spec.wholeWord)
      test = stringWordTest(state.doc, state.charCategorizer(state.selection.main.head));
    if (spec.test)
      test = wrapStringTest(spec.test, state, test);
    return new SearchCursor(state.doc, spec.unquoted, from, to, spec.caseSensitive ? void 0 : (x) => x.toLowerCase(), test);
  }
  function stringWordTest(doc2, categorizer) {
    return (from, to, buf, bufPos) => {
      if (bufPos > from || bufPos + buf.length < to) {
        bufPos = Math.max(0, from - 2);
        buf = doc2.sliceString(bufPos, Math.min(doc2.length, to + 2));
      }
      return (categorizer(charBefore(buf, from - bufPos)) != CharCategory.Word || categorizer(charAfter(buf, from - bufPos)) != CharCategory.Word) && (categorizer(charAfter(buf, to - bufPos)) != CharCategory.Word || categorizer(charBefore(buf, to - bufPos)) != CharCategory.Word);
    };
  }
  var StringQuery = class extends QueryType2 {
    constructor(spec) {
      super(spec);
    }
    nextMatch(state, curFrom, curTo) {
      let cursor = stringCursor(this.spec, state, curTo, state.doc.length).nextOverlapping();
      if (cursor.done) {
        let end = Math.min(state.doc.length, curFrom + this.spec.unquoted.length);
        cursor = stringCursor(this.spec, state, 0, end).nextOverlapping();
      }
      return cursor.done || cursor.value.from == curFrom && cursor.value.to == curTo ? null : cursor.value;
    }
    // Searching in reverse is, rather than implementing an inverted search
    // cursor, done by scanning chunk after chunk forward.
    prevMatchInRange(state, from, to) {
      for (let pos = to; ; ) {
        let start = Math.max(from, pos - 1e4 - this.spec.unquoted.length);
        let cursor = stringCursor(this.spec, state, start, pos), range = null;
        while (!cursor.nextOverlapping().done)
          range = cursor.value;
        if (range)
          return range;
        if (start == from)
          return null;
        pos -= 1e4;
      }
    }
    prevMatch(state, curFrom, curTo) {
      let found = this.prevMatchInRange(state, 0, curFrom);
      if (!found)
        found = this.prevMatchInRange(state, Math.max(0, curTo - this.spec.unquoted.length), state.doc.length);
      return found && (found.from != curFrom || found.to != curTo) ? found : null;
    }
    getReplacement(_result) {
      return this.spec.unquote(this.spec.replace);
    }
    matchAll(state, limit) {
      let cursor = stringCursor(this.spec, state, 0, state.doc.length), ranges = [];
      while (!cursor.next().done) {
        if (ranges.length >= limit)
          return null;
        ranges.push(cursor.value);
      }
      return ranges;
    }
    highlight(state, from, to, add2) {
      let cursor = stringCursor(this.spec, state, Math.max(0, from - this.spec.unquoted.length), Math.min(to + this.spec.unquoted.length, state.doc.length));
      while (!cursor.next().done)
        add2(cursor.value.from, cursor.value.to);
    }
  };
  function wrapRegexpTest(test, state, inner) {
    return (from, to, match) => {
      return (!inner || inner(from, to, match)) && test(match[0], state, from, to);
    };
  }
  function regexpCursor(spec, state, from, to) {
    let test;
    if (spec.wholeWord)
      test = regexpWordTest(state.charCategorizer(state.selection.main.head));
    if (spec.test)
      test = wrapRegexpTest(spec.test, state, test);
    return new RegExpCursor(state.doc, spec.search, { ignoreCase: !spec.caseSensitive, test }, from, to);
  }
  function charBefore(str, index) {
    return str.slice(findClusterBreak2(str, index, false), index);
  }
  function charAfter(str, index) {
    return str.slice(index, findClusterBreak2(str, index));
  }
  function regexpWordTest(categorizer) {
    return (_from, _to, match) => !match[0].length || (categorizer(charBefore(match.input, match.index)) != CharCategory.Word || categorizer(charAfter(match.input, match.index)) != CharCategory.Word) && (categorizer(charAfter(match.input, match.index + match[0].length)) != CharCategory.Word || categorizer(charBefore(match.input, match.index + match[0].length)) != CharCategory.Word);
  }
  var RegExpQuery = class extends QueryType2 {
    nextMatch(state, curFrom, curTo) {
      let cursor = regexpCursor(this.spec, state, curTo, state.doc.length).next();
      if (cursor.done)
        cursor = regexpCursor(this.spec, state, 0, curFrom).next();
      return cursor.done ? null : cursor.value;
    }
    prevMatchInRange(state, from, to) {
      for (let size = 1; ; size++) {
        let start = Math.max(
          from,
          to - size * 1e4
          /* FindPrev.ChunkSize */
        );
        let cursor = regexpCursor(this.spec, state, start, to), range = null;
        while (!cursor.next().done)
          range = cursor.value;
        if (range && (start == from || range.from > start + 10))
          return range;
        if (start == from)
          return null;
      }
    }
    prevMatch(state, curFrom, curTo) {
      return this.prevMatchInRange(state, 0, curFrom) || this.prevMatchInRange(state, curTo, state.doc.length);
    }
    getReplacement(result) {
      return this.spec.unquote(this.spec.replace).replace(/\$([$&]|\d+)/g, (m, i) => {
        if (i == "&")
          return result.match[0];
        if (i == "$")
          return "$";
        for (let l = i.length; l > 0; l--) {
          let n = +i.slice(0, l);
          if (n > 0 && n < result.match.length)
            return result.match[n] + i.slice(l);
        }
        return m;
      });
    }
    matchAll(state, limit) {
      let cursor = regexpCursor(this.spec, state, 0, state.doc.length), ranges = [];
      while (!cursor.next().done) {
        if (ranges.length >= limit)
          return null;
        ranges.push(cursor.value);
      }
      return ranges;
    }
    highlight(state, from, to, add2) {
      let cursor = regexpCursor(this.spec, state, Math.max(
        0,
        from - 250
        /* RegExp.HighlightMargin */
      ), Math.min(to + 250, state.doc.length));
      while (!cursor.next().done)
        add2(cursor.value.from, cursor.value.to);
    }
  };
  var setSearchQuery = /* @__PURE__ */ StateEffect.define();
  var togglePanel = /* @__PURE__ */ StateEffect.define();
  var searchState = /* @__PURE__ */ StateField.define({
    create(state) {
      return new SearchState(defaultQuery(state).create(), null);
    },
    update(value, tr) {
      for (let effect of tr.effects) {
        if (effect.is(setSearchQuery))
          value = new SearchState(effect.value.create(), value.panel);
        else if (effect.is(togglePanel))
          value = new SearchState(value.query, effect.value ? createSearchPanel : null);
      }
      return value;
    },
    provide: (f) => showPanel.from(f, (val) => val.panel)
  });
  var SearchState = class {
    constructor(query, panel) {
      this.query = query;
      this.panel = panel;
    }
  };
  var matchMark = /* @__PURE__ */ Decoration.mark({ class: "cm-searchMatch" });
  var selectedMatchMark = /* @__PURE__ */ Decoration.mark({ class: "cm-searchMatch cm-searchMatch-selected" });
  var searchHighlighter = /* @__PURE__ */ ViewPlugin.fromClass(class {
    constructor(view) {
      this.view = view;
      this.decorations = this.highlight(view.state.field(searchState));
    }
    update(update) {
      let state = update.state.field(searchState);
      if (state != update.startState.field(searchState) || update.docChanged || update.selectionSet || update.viewportChanged)
        this.decorations = this.highlight(state);
    }
    highlight({ query, panel }) {
      if (!panel || !query.spec.valid)
        return Decoration.none;
      let { view } = this;
      let builder = new RangeSetBuilder();
      for (let i = 0, ranges = view.visibleRanges, l = ranges.length; i < l; i++) {
        let { from, to } = ranges[i];
        while (i < l - 1 && to > ranges[i + 1].from - 2 * 250)
          to = ranges[++i].to;
        query.highlight(view.state, from, to, (from2, to2) => {
          let selected = view.state.selection.ranges.some((r) => r.from == from2 && r.to == to2);
          builder.add(from2, to2, selected ? selectedMatchMark : matchMark);
        });
      }
      return builder.finish();
    }
  }, {
    decorations: (v) => v.decorations
  });
  function searchCommand(f) {
    return (view) => {
      let state = view.state.field(searchState, false);
      return state && state.query.spec.valid ? f(view, state) : openSearchPanel(view);
    };
  }
  var findNext = /* @__PURE__ */ searchCommand((view, { query }) => {
    let { to } = view.state.selection.main;
    let next = query.nextMatch(view.state, to, to);
    if (!next)
      return false;
    let selection = EditorSelection.single(next.from, next.to);
    let config = view.state.facet(searchConfigFacet);
    view.dispatch({
      selection,
      effects: [announceMatch(view, next), config.scrollToMatch(selection.main, view)],
      userEvent: "select.search"
    });
    selectSearchInput(view);
    return true;
  });
  var findPrevious = /* @__PURE__ */ searchCommand((view, { query }) => {
    let { state } = view, { from } = state.selection.main;
    let prev = query.prevMatch(state, from, from);
    if (!prev)
      return false;
    let selection = EditorSelection.single(prev.from, prev.to);
    let config = view.state.facet(searchConfigFacet);
    view.dispatch({
      selection,
      effects: [announceMatch(view, prev), config.scrollToMatch(selection.main, view)],
      userEvent: "select.search"
    });
    selectSearchInput(view);
    return true;
  });
  var selectMatches = /* @__PURE__ */ searchCommand((view, { query }) => {
    let ranges = query.matchAll(view.state, 1e3);
    if (!ranges || !ranges.length)
      return false;
    view.dispatch({
      selection: EditorSelection.create(ranges.map((r) => EditorSelection.range(r.from, r.to))),
      userEvent: "select.search.matches"
    });
    return true;
  });
  var selectSelectionMatches = ({ state, dispatch }) => {
    let sel = state.selection;
    if (sel.ranges.length > 1 || sel.main.empty)
      return false;
    let { from, to } = sel.main;
    let ranges = [], main = 0;
    for (let cur2 = new SearchCursor(state.doc, state.sliceDoc(from, to)); !cur2.next().done; ) {
      if (ranges.length > 1e3)
        return false;
      if (cur2.value.from == from)
        main = ranges.length;
      ranges.push(EditorSelection.range(cur2.value.from, cur2.value.to));
    }
    dispatch(state.update({
      selection: EditorSelection.create(ranges, main),
      userEvent: "select.search.matches"
    }));
    return true;
  };
  var replaceNext = /* @__PURE__ */ searchCommand((view, { query }) => {
    let { state } = view, { from, to } = state.selection.main;
    if (state.readOnly)
      return false;
    let match = query.nextMatch(state, from, from);
    if (!match)
      return false;
    let next = match;
    let changes = [], selection, replacement;
    let effects = [];
    if (next.from == from && next.to == to) {
      replacement = state.toText(query.getReplacement(next));
      changes.push({ from: next.from, to: next.to, insert: replacement });
      next = query.nextMatch(state, next.from, next.to);
      effects.push(EditorView.announce.of(state.phrase("replaced match on line $", state.doc.lineAt(from).number) + "."));
    }
    let changeSet = view.state.changes(changes);
    if (next) {
      selection = EditorSelection.single(next.from, next.to).map(changeSet);
      effects.push(announceMatch(view, next));
      effects.push(state.facet(searchConfigFacet).scrollToMatch(selection.main, view));
    }
    view.dispatch({
      changes: changeSet,
      selection,
      effects,
      userEvent: "input.replace"
    });
    return true;
  });
  var replaceAll = /* @__PURE__ */ searchCommand((view, { query }) => {
    if (view.state.readOnly)
      return false;
    let changes = query.matchAll(view.state, 1e9).map((match) => {
      let { from, to } = match;
      return { from, to, insert: query.getReplacement(match) };
    });
    if (!changes.length)
      return false;
    let announceText = view.state.phrase("replaced $ matches", changes.length) + ".";
    view.dispatch({
      changes,
      effects: EditorView.announce.of(announceText),
      userEvent: "input.replace.all"
    });
    return true;
  });
  function createSearchPanel(view) {
    return view.state.facet(searchConfigFacet).createPanel(view);
  }
  function defaultQuery(state, fallback) {
    var _a2, _b, _c, _d, _e;
    let sel = state.selection.main;
    let selText = sel.empty || sel.to > sel.from + 100 ? "" : state.sliceDoc(sel.from, sel.to);
    if (fallback && !selText)
      return fallback;
    let config = state.facet(searchConfigFacet);
    return new SearchQuery({
      search: ((_a2 = fallback === null || fallback === void 0 ? void 0 : fallback.literal) !== null && _a2 !== void 0 ? _a2 : config.literal) ? selText : selText.replace(/\n/g, "\\n"),
      caseSensitive: (_b = fallback === null || fallback === void 0 ? void 0 : fallback.caseSensitive) !== null && _b !== void 0 ? _b : config.caseSensitive,
      literal: (_c = fallback === null || fallback === void 0 ? void 0 : fallback.literal) !== null && _c !== void 0 ? _c : config.literal,
      regexp: (_d = fallback === null || fallback === void 0 ? void 0 : fallback.regexp) !== null && _d !== void 0 ? _d : config.regexp,
      wholeWord: (_e = fallback === null || fallback === void 0 ? void 0 : fallback.wholeWord) !== null && _e !== void 0 ? _e : config.wholeWord
    });
  }
  function getSearchInput(view) {
    let panel = getPanel(view, createSearchPanel);
    return panel && panel.dom.querySelector("[main-field]");
  }
  function selectSearchInput(view) {
    let input = getSearchInput(view);
    if (input && input == view.root.activeElement)
      input.select();
  }
  var openSearchPanel = (view) => {
    let state = view.state.field(searchState, false);
    if (state && state.panel) {
      let searchInput = getSearchInput(view);
      if (searchInput && searchInput != view.root.activeElement) {
        let query = defaultQuery(view.state, state.query.spec);
        if (query.valid)
          view.dispatch({ effects: setSearchQuery.of(query) });
        searchInput.focus();
        searchInput.select();
      }
    } else {
      view.dispatch({ effects: [
        togglePanel.of(true),
        state ? setSearchQuery.of(defaultQuery(view.state, state.query.spec)) : StateEffect.appendConfig.of(searchExtensions)
      ] });
    }
    return true;
  };
  var closeSearchPanel = (view) => {
    let state = view.state.field(searchState, false);
    if (!state || !state.panel)
      return false;
    let panel = getPanel(view, createSearchPanel);
    if (panel && panel.dom.contains(view.root.activeElement))
      view.focus();
    view.dispatch({ effects: togglePanel.of(false) });
    return true;
  };
  var searchKeymap = [
    { key: "Mod-f", run: openSearchPanel, scope: "editor search-panel" },
    { key: "F3", run: findNext, shift: findPrevious, scope: "editor search-panel", preventDefault: true },
    { key: "Mod-g", run: findNext, shift: findPrevious, scope: "editor search-panel", preventDefault: true },
    { key: "Escape", run: closeSearchPanel, scope: "editor search-panel" },
    { key: "Mod-Shift-l", run: selectSelectionMatches },
    { key: "Mod-Alt-g", run: gotoLine },
    { key: "Mod-d", run: selectNextOccurrence, preventDefault: true }
  ];
  var SearchPanel = class {
    constructor(view) {
      this.view = view;
      let query = this.query = view.state.field(searchState).query.spec;
      this.commit = this.commit.bind(this);
      this.searchField = crelt("input", {
        value: query.search,
        placeholder: phrase(view, "Find"),
        "aria-label": phrase(view, "Find"),
        class: "cm-textfield",
        name: "search",
        form: "",
        "main-field": "true",
        onchange: this.commit,
        onkeyup: this.commit
      });
      this.replaceField = crelt("input", {
        value: query.replace,
        placeholder: phrase(view, "Replace"),
        "aria-label": phrase(view, "Replace"),
        class: "cm-textfield",
        name: "replace",
        form: "",
        onchange: this.commit,
        onkeyup: this.commit
      });
      this.caseField = crelt("input", {
        type: "checkbox",
        name: "case",
        form: "",
        checked: query.caseSensitive,
        onchange: this.commit
      });
      this.reField = crelt("input", {
        type: "checkbox",
        name: "re",
        form: "",
        checked: query.regexp,
        onchange: this.commit
      });
      this.wordField = crelt("input", {
        type: "checkbox",
        name: "word",
        form: "",
        checked: query.wholeWord,
        onchange: this.commit
      });
      function button(name2, onclick, content2) {
        return crelt("button", { class: "cm-button", name: name2, onclick, type: "button" }, content2);
      }
      this.dom = crelt("div", { onkeydown: (e) => this.keydown(e), class: "cm-search" }, [
        this.searchField,
        button("next", () => findNext(view), [phrase(view, "next")]),
        button("prev", () => findPrevious(view), [phrase(view, "previous")]),
        button("select", () => selectMatches(view), [phrase(view, "all")]),
        crelt("label", null, [this.caseField, phrase(view, "match case")]),
        crelt("label", null, [this.reField, phrase(view, "regexp")]),
        crelt("label", null, [this.wordField, phrase(view, "by word")]),
        ...view.state.readOnly ? [] : [
          crelt("br"),
          this.replaceField,
          button("replace", () => replaceNext(view), [phrase(view, "replace")]),
          button("replaceAll", () => replaceAll(view), [phrase(view, "replace all")])
        ],
        crelt("button", {
          name: "close",
          onclick: () => closeSearchPanel(view),
          "aria-label": phrase(view, "close"),
          type: "button"
        }, ["\xD7"])
      ]);
    }
    commit() {
      let query = new SearchQuery({
        search: this.searchField.value,
        caseSensitive: this.caseField.checked,
        regexp: this.reField.checked,
        wholeWord: this.wordField.checked,
        replace: this.replaceField.value
      });
      if (!query.eq(this.query)) {
        this.query = query;
        this.view.dispatch({ effects: setSearchQuery.of(query) });
      }
    }
    keydown(e) {
      if (runScopeHandlers(this.view, e, "search-panel")) {
        e.preventDefault();
      } else if (e.keyCode == 13 && e.target == this.searchField) {
        e.preventDefault();
        (e.shiftKey ? findPrevious : findNext)(this.view);
      } else if (e.keyCode == 13 && e.target == this.replaceField) {
        e.preventDefault();
        replaceNext(this.view);
      }
    }
    update(update) {
      for (let tr of update.transactions)
        for (let effect of tr.effects) {
          if (effect.is(setSearchQuery) && !effect.value.eq(this.query))
            this.setQuery(effect.value);
        }
    }
    setQuery(query) {
      this.query = query;
      this.searchField.value = query.search;
      this.replaceField.value = query.replace;
      this.caseField.checked = query.caseSensitive;
      this.reField.checked = query.regexp;
      this.wordField.checked = query.wholeWord;
    }
    mount() {
      this.searchField.select();
    }
    get pos() {
      return 80;
    }
    get top() {
      return this.view.state.facet(searchConfigFacet).top;
    }
  };
  function phrase(view, phrase2) {
    return view.state.phrase(phrase2);
  }
  var AnnounceMargin = 30;
  var Break = /[\s\.,:;?!]/;
  function announceMatch(view, { from, to }) {
    let line = view.state.doc.lineAt(from), lineEnd = view.state.doc.lineAt(to).to;
    let start = Math.max(line.from, from - AnnounceMargin), end = Math.min(lineEnd, to + AnnounceMargin);
    let text = view.state.sliceDoc(start, end);
    if (start != line.from) {
      for (let i = 0; i < AnnounceMargin; i++)
        if (!Break.test(text[i + 1]) && Break.test(text[i])) {
          text = text.slice(i);
          break;
        }
    }
    if (end != lineEnd) {
      for (let i = text.length - 1; i > text.length - AnnounceMargin; i--)
        if (!Break.test(text[i - 1]) && Break.test(text[i])) {
          text = text.slice(0, i);
          break;
        }
    }
    return EditorView.announce.of(`${view.state.phrase("current match")}. ${text} ${view.state.phrase("on line")} ${line.number}.`);
  }
  var baseTheme3 = /* @__PURE__ */ EditorView.baseTheme({
    ".cm-panel.cm-search": {
      padding: "2px 6px 4px",
      position: "relative",
      "& [name=close]": {
        position: "absolute",
        top: "0",
        right: "4px",
        backgroundColor: "inherit",
        border: "none",
        font: "inherit",
        padding: 0,
        margin: 0
      },
      "& input, & button, & label": {
        margin: ".2em .6em .2em 0"
      },
      "& input[type=checkbox]": {
        marginRight: ".2em"
      },
      "& label": {
        fontSize: "80%",
        whiteSpace: "pre"
      }
    },
    "&light .cm-searchMatch": { backgroundColor: "#ffff0054" },
    "&dark .cm-searchMatch": { backgroundColor: "#00ffff8a" },
    "&light .cm-searchMatch-selected": { backgroundColor: "#ff6a0054" },
    "&dark .cm-searchMatch-selected": { backgroundColor: "#ff00ff8a" }
  });
  var searchExtensions = [
    searchState,
    /* @__PURE__ */ Prec.low(searchHighlighter),
    baseTheme3
  ];

  // src/js/codemirror/cm6-puzzlescript.js
  function ensureEditorCodeMirrorCompat() {
    if (!window.CodeMirror) {
      window.CodeMirror = function CodeMirror() {
      };
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
        return { line, ch };
      };
    }
    if (!cm.defineMode) {
      cm.defineMode = function(name2, modeFactory) {
        cm.modes[name2] = modeFactory;
        if (cm.defaults.mode == null) {
          cm.defaults.mode = name2;
        }
      };
    }
    if (!cm.registerHelper) {
      cm.registerHelper = function(type, name2, value) {
        if (!cm[type]) {
          cm[type] = {};
        }
        cm[type][name2] = value;
      };
    }
    return cm;
  }
  ensureEditorCodeMirrorCompat();
  function countColumn2(string2, end, tabSize, startIndex, startValue) {
    if (end == null) {
      end = string2.search(/[^\s\u00a0]/);
      if (end === -1) {
        end = string2.length;
      }
    }
    let n = startValue || 0;
    for (let i = startIndex || 0; i < end; i++) {
      if (string2.charCodeAt(i) === 9) {
        n += tabSize - n % tabSize;
      } else {
        n++;
      }
    }
    return n;
  }
  if (!window.CodeMirror.StringStream) {
    window.CodeMirror.StringStream = function(string2, tabSize) {
      this.pos = this.start = 0;
      this.string = string2;
      this.tabSize = tabSize || 8;
      this.lastColumnPos = this.lastColumnValue = 0;
      this.lineStart = 0;
    };
    window.CodeMirror.StringStream.prototype = {
      eol: function() {
        return this.pos >= this.string.length;
      },
      sol: function() {
        return this.pos === this.lineStart;
      },
      peek: function() {
        return this.string.charAt(this.pos) || void 0;
      },
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
        while (this.eat(match)) {
        }
        return this.pos > start;
      },
      eatSpace: function() {
        const start = this.pos;
        while (/[\s\u00a0]/.test(this.string.charAt(this.pos))) {
          ++this.pos;
        }
        return this.pos > start;
      },
      skipToEnd: function() {
        this.pos = this.string.length;
      },
      skipTo: function(ch) {
        const found = this.string.indexOf(ch, this.pos);
        if (found > -1) {
          this.pos = found;
          return true;
        }
      },
      backUp: function(n) {
        this.pos -= n;
      },
      column: function() {
        if (this.lastColumnPos < this.start) {
          this.lastColumnValue = countColumn2(this.string, this.start, this.tabSize, this.lastColumnPos, this.lastColumnValue);
          this.lastColumnPos = this.start;
        }
        return this.lastColumnValue - (this.lineStart ? countColumn2(this.string, this.lineStart, this.tabSize) : 0);
      },
      indentation: function() {
        return countColumn2(this.string, null, this.tabSize) - (this.lineStart ? countColumn2(this.string, this.lineStart, this.tabSize) : 0);
      },
      match: function(pattern, consume, caseInsensitive) {
        if (typeof pattern === "string") {
          const cased = function(str) {
            return caseInsensitive ? str.toLowerCase() : str;
          };
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
      current: function() {
        return this.string.slice(this.start, this.pos);
      },
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
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  var colorCache = /* @__PURE__ */ new Map();
  function styleFromHexCode(hexCode) {
    function parseColor(input) {
      input = input.trim();
      if (input.length < 4) {
        return null;
      } else if (input.length < 7) {
        return [
          parseInt(input.charAt(1), 16) * 17,
          parseInt(input.charAt(2), 16) * 17,
          parseInt(input.charAt(3), 16) * 17
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
        let hue2rgb = function(p2, q2, t2) {
          if (t2 < 0) t2 += 1;
          if (t2 > 1) t2 -= 1;
          if (t2 < 1 / 6) return p2 + (q2 - p2) * 6 * t2;
          if (t2 < 1 / 2) return q2;
          if (t2 < 2 / 3) return p2 + (q2 - p2) * (2 / 3 - t2) * 6;
          return p2;
        };
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
  var PuzzleTokenizer = class {
    constructor(doc2) {
      this.mode = createMode();
      this.reset(doc2);
    }
    reset(doc2) {
      this.doc = doc2;
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
          style
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
  };
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
      inlineStyle
    };
  }
  var decorationCache = /* @__PURE__ */ new Map();
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
  function toOffset(doc2, pos) {
    if (pos == null) {
      return doc2.length;
    }
    const lineNumber = clamp(pos.line, 0, doc2.lines - 1) + 1;
    const line = doc2.line(lineNumber);
    const ch = clamp(pos.ch || 0, 0, line.length);
    return line.from + ch;
  }
  function fromOffset(doc2, offset) {
    const line = doc2.lineAt(offset);
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
    function emit(name2, payload) {
      for (const callback of listeners[name2]) {
        callback(editorFacade, payload);
      }
    }
    function buildDecorations(view2) {
      const builder = new RangeSetBuilder();
      let lastLine = -1;
      for (const range of view2.visibleRanges) {
        let startLine = view2.state.doc.lineAt(range.from).number - 1;
        const endLine = view2.state.doc.lineAt(range.to).number - 1;
        if (startLine <= lastLine) {
          startLine = lastLine + 1;
        }
        for (let lineNumber = startLine; lineNumber <= endLine; lineNumber++) {
          const line = view2.state.doc.line(lineNumber + 1);
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
      constructor(view2) {
        this.decorations = buildDecorations(view2);
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
      return document.body.classList.contains("dark-theme") || !document.body.classList.contains("light-theme") && document.body.style.colorScheme !== "light";
    }
    function getLayerBase(view2) {
      let rect = view2.scrollDOM.getBoundingClientRect();
      let left = view2.textDirection === Direction.LTR ? rect.left : rect.right - view2.scrollDOM.clientWidth * view2.scaleX;
      return {
        left: left - view2.scrollDOM.scrollLeft * view2.scaleX,
        top: rect.top - view2.scrollDOM.scrollTop * view2.scaleY
      };
    }
    const activeLineBehindLayer = layer({
      above: false,
      class: "cm-activeLineLayer",
      markers(view2) {
        let lastLineStart = -1;
        let markers = [];
        let content2 = view2.contentDOM;
        let contentRect = content2.getBoundingClientRect();
        let lineElt = content2.querySelector(".cm-line");
        let lineStyle = lineElt ? window.getComputedStyle(lineElt) : null;
        let paddingLeft = (lineStyle ? parseInt(lineStyle.paddingLeft, 10) : 0) || 0;
        let paddingRight = (lineStyle ? parseInt(lineStyle.paddingRight, 10) : 0) || 0;
        let leftSide = contentRect.left + paddingLeft;
        let rightSide = contentRect.right - paddingRight;
        let base2 = getLayerBase(view2);
        for (let r of view2.state.selection.ranges) {
          let line = view2.lineBlockAt(r.head);
          if (line.from > lastLineStart) {
            let coords = view2.coordsAtPos(line.from);
            if (coords) {
              let left = leftSide - base2.left;
              let top2 = coords.top - base2.top;
              let width = Math.max(0, rightSide - leftSide);
              let height = coords.bottom - coords.top;
              markers.push(new RectangleMarker("cm-activeLine", left, top2, width, height));
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
    const darkTheme2 = EditorView.theme({
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
    const puzzleTheme = isDarkTheme() ? darkTheme2 : lightTheme;
    function clickToken(event, view2) {
      const pos = view2.posAtCoords({ x: event.clientX, y: event.clientY });
      if (pos == null) {
        return false;
      }
      const line = view2.state.doc.lineAt(pos);
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
        view2.contentDOM.blur();
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
            keyup: function(event, view2) {
              emit("keyup", event);
              return false;
            },
            drop: function(event, view2) {
              emit("drop", event);
              return false;
            },
            mousedown: function(event, view2) {
              emit("mousedown", event);
              return clickToken(event, view2);
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
      view,
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
      on: function(name2, callback) {
        if (!listeners[name2]) {
          listeners[name2] = [];
        }
        listeners[name2].push(callback);
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
      setOption: function(name2, value) {
        if (name2 === "theme" && value !== "midnight") {
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
        const doc2 = ensureView().state.doc;
        const fromOffsetValue = toOffset(doc2, from);
        const toOffsetValue = to == null ? fromOffsetValue : toOffset(doc2, to);
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
        const offset = toOffset(ensureView().state.doc, { line, ch });
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
      refresh: function() {
      },
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
    createEditor
  };
})();
//# sourceMappingURL=cm6.bundle.js.map
