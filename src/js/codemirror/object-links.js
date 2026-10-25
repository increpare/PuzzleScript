'use strict';

function installObjectLinks(cm) {
    const wrapper = cm.getWrapperElement();
    function updateModifier(event) {
        wrapper.classList.toggle('cm-object-links', !!(event.ctrlKey || event.metaKey));
    }
    document.addEventListener('keydown', updateModifier);
    document.addEventListener('keyup', updateModifier);
    wrapper.addEventListener('mousemove', updateModifier);
    window.addEventListener('blur', function() {
        wrapper.classList.remove('cm-object-links');
    });

    cm.on('mousedown', function(cm, event) {
        if (!(event.ctrlKey || event.metaKey) || event.button !== 0 ||
            !event.target.closest('.cm-NAME')) return;
        const pos = cm.posFromMouse(event);
        if (!pos) return;
        let token = cm.getTokenAt(pos, true);
        // Mouse positions round to the nearest character boundary. At the left
        // edge of a name, getTokenAt can still return the preceding token.
        if (!token.type || !token.type.split(/\s+/).includes('NAME')) {
            token = cm.getTokenAt({ line: pos.line, ch: pos.ch + 1 }, true);
        }
        if (!token.type || !token.type.split(/\s+/).includes('NAME')) return;
        // Use the current document's parser state, including a legend declaration
        // completed at the end of this line, rather than the last compiled game.
        const state = cm.getStateAfter(pos.line, true);
        const declaration = wordAlreadyDeclared(state, token.string.trim().toLowerCase());
        if (!declaration || !Number.isInteger(declaration.lineNumber)) return;
        event.preventDefault();
        jumpToLine(declaration.lineNumber);
        cm.focus();
    });
}
