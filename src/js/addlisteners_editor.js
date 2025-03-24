'use strict';

for (let i=0;i<10;i++) {
	let idname = "newsound"+i;
	let el = document.getElementById(idname);
    el.addEventListener("click", (function(n){return function(){return newSound(n);};})(i), false);
}

//let soundButtonPress = document.getElementById("soundButtonPress");
//soundButtonPress.addEventListener("click", buttonPress, false);

let runClickLink = document.getElementById("runClickLink");
runClickLink.addEventListener("click", runClick, false);

let saveClickLink = document.getElementById("saveClickLink");
saveClickLink.addEventListener("click", saveClick, false);

let rebuildClickLink = document.getElementById("rebuildClickLink");
rebuildClickLink.addEventListener("click", rebuildClick, false);

let shareClickLink = document.getElementById("shareClickLink");
shareClickLink.addEventListener("click", shareClick, false);

let levelEditorClickLink = document.getElementById("levelEditorClickLink");
levelEditorClickLink.addEventListener("click", levelEditorClick_Fn, false);

let toggleThemeClickLinks = document.getElementById("toggleThemeClickLinks");
toggleThemeClickLinks.addEventListener("click", toggleThemeClick, false);

let exportClickLink = document.getElementById("exportClickLink");
exportClickLink.addEventListener("click", exportClick, false);

let exampleDropdown = document.getElementById("exampleDropdown");
exampleDropdown.addEventListener("change", dropdownChange, false);

let loadDropDown = document.getElementById("loadDropDown");
loadDropDown.addEventListener("change", loadDropDownChange, false);

let horizontalDragbar = document.getElementById("horizontaldragbar");
horizontalDragbar.addEventListener("mousedown", horizontalDragbarMouseDown, false);

let verticalDragbar = document.getElementById("verticaldragbar");
verticalDragbar.addEventListener("mousedown", verticalDragbarMouseDown, false);

window.addEventListener("resize", resize_all, false);
window.addEventListener("load", reset_panels, false);

/* https://github.com/ndrake/PuzzleScript/commit/de4ac2a38865b74e66c1d711a25f0691079a290d */
window.onbeforeunload = function (e) {
  e = e || window.event;
  let msg = 'You have unsaved changes!';

  if(_editorDirty) {      

    // For IE and Firefox prior to version 4
    if (e) {
      e.preventDefault();
      e.returnValue = msg;
    }

    // For Safari
    return msg;
  }
};

let gestureHandler = Mobile.enable();
if (gestureHandler) {
    gestureHandler.setFocusElement(document.getElementById('gameCanvas'));
}

