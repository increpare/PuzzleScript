// LS is an interface built on top of the browser's localStorage.
// If you work with window.localStorage directly, you need lots of
// boilerplate code to avoid unwanted errors, e.g.:
//   try {
//      if (!!window.localStorage) {
//        localStorage["foo"]=JSON.stringify([1,2,3]);
//        var bar = JSON.parse(localStorage["bar"]);
//      }
//    } catch (ex) { }
// With this interface, this becomes:
//   LS.set("foo",[1,2,3])
//   var bar = LS.get("bar")
// Think of LS as a dictionary that persists itself to
// the browser's localStorage, if available.

var LS = {
  _cache: {}, // an in-memory copy of the browser's localStorage
  _shouldSync: false, // whether the real window.localStorage is available
};

LS.set = function(key, val) {
  LS._cache[key] = val;
  if (LS._shouldSync) window.localStorage[key] = JSON.stringify(val);
}

LS.get = function(key) {
  return LS._cache[key];
}

LS.del = function(key) {
  delete LS._cache[keys];
  if (LS._shouldSync) delete window.localStorage[key];
}

LS.clear = function() {
  LS._cache = {};
  if (LS._shouldSync) window.localStorage.clear();
}

LS._populateCache = function() {
  for (var key of Object.keys(window.localStorage)) {
    var val = window.localStorage[key];
    try {
      LS._cache[key] = JSON.parse(val);
    } catch (err) {
      console.warn(`Error parsing window.localStorage: [${key}]=${val}`);
    }
  }
}

// call this once during window.onload
function setupLocalStorage() {
  if (isLocalStorageAvailable()) {
    LS._shouldSync = true
    LS._populateCache()
  }
}

function isLocalStorageAvailable() {
  try {
    // accessing localStorage at all will throw an error if it's not available
    var x = window.localStorage;
    return true;
  } catch (e) {
    return false;
  }
}
