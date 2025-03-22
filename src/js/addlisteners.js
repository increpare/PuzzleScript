

    onmousemove="mouseMove(event)" 
    onmouseout="mouseOut()"

let el = document.getElementById("gameCanvas");
if (el.addEventListener) {
    el.addEventListener("contextmenu", rightClickCanvas, false);
    el.addEventListener("mousemove", mouseMove, false);
    el.addEventListener("mouseout", mouseOut, false);
} else {
    el.attachEvent('oncontextmenu', rightClickCanvas);
    el.attachEvent('onmousemove', mouseMove);
    el.attachEvent('onmouseout', mouseOut);
}  
