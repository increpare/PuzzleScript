# Game-to-Editor Tab Focus Design

## Context

PuzzleScript currently has a one-way keyboard workflow. `Ctrl/Cmd+Enter` and
`Ctrl/Cmd+Shift+Enter` rebuild or run the game, blur CodeMirror, and route input
to the game. Returning to the script editor requires a pointer click.

The game canvas is not currently a real DOM focus target. Gameplay routing
instead depends on `lastDownTarget`, which can describe the game as active even
when another element owns DOM focus. A keyboard-only return path therefore needs
to make the game/editor focus transfer explicit as well as add the shortcut.

## Goals

- Plain `Tab` returns focus from the active game canvas to CodeMirror.
- Returning to CodeMirror does not compile, restart, or otherwise change the
  game.
- CodeMirror restores its existing cursor, selection, and scroll position.
- Game input stops immediately when focus returns to CodeMirror.
- Tab retains its normal behavior outside active game focus.
- The interaction is discoverable to keyboard and assistive-technology users.

## Non-goals

- Do not add a symmetric `Ctrl+;` focus toggle.
- Do not change CodeMirror's existing Tab/indent behavior.
- Do not change standalone player keyboard behavior.
- Do not make compilation itself imply a focus transfer; background or utility
  compilation must not steal focus.
- Do not redesign the broader gameplay input system.

## Interaction Design

The editor-page game canvas becomes programmatically focusable with
`tabindex="-1"`. It stays out of the browser's normal tab order, but explicit
game-entry actions can give it real DOM focus.

The following actions enter game focus:

- pressing `Ctrl/Cmd+Enter` or `Ctrl/Cmd+Shift+Enter`;
- clicking RUN or REBUILD;
- activating the level editor; and
- clicking or tapping the game canvas.

While the canvas owns focus, pressing an unmodified `Tab`:

1. consumes the event before it can enter the gameplay key buffer or trigger
   browser focus traversal;
2. clears held/repeating gameplay keys;
3. changes `lastDownTarget` away from the canvas;
4. calls CodeMirror's public `focus()` method.

Modified Tab combinations, including `Shift+Tab`, retain browser behavior. Tab
also remains untouched when focus is in CodeMirror, a CodeMirror dialog, the
toolbar, the console, or any other control.

The canvas receives an accessible label that identifies it as the game preview
and states that Tab returns to the script editor. A `:focus-visible` outline
shows keyboard users when game focus is active without changing layout.

## Implementation Boundaries

### Focus helpers

Add small editor-specific helpers beside the gameplay input handler in
`inputoutput.js`, where the canvas target and key-repeat state are already
owned. They have two responsibilities:

- `focusGameCanvas()` gives the canvas DOM focus, identifies it as the gameplay
  input target, and ensures CodeMirror is blurred.
- `focusCodeEditor()` clears gameplay key state, assigns `lastDownTarget` to
  CodeMirror's input field, and focuses CodeMirror through `editor.focus()`.

Extract the three existing key-reset assignments (`keybuffer`,
`keyRepeatIndex`, and `keyRepeatTimer`) into one helper used by the window
focus/blur handlers and by `focusCodeEditor()`. This keeps focus transitions
from developing different held-key behavior.

Call `focusGameCanvas()` from explicit game-entry actions rather than from the
general `compile()` function. This prevents save, load, export, GIF generation,
and other compilation callers from acquiring game focus accidentally.

### Keyboard routing

Handle the editor-only Tab transition near the start of the document-level
`onKeyDown` path, before the existing key-buffer and `checkKey` logic. The guard
requires all of the following:

- the IDE/editor build is active;
- `event.key` is `Tab`;
- Ctrl, Meta, Alt, and Shift are not held; and
- `document.activeElement` is the game canvas.

After `focusCodeEditor()` runs, prevent the default event and stop further
handling. Gameplay never receives the Tab press.

### Markup, styling, and documentation

- Add `tabindex="-1"` and an accessible label to the canvas in `editor.html`.
- Add a non-layout-shifting `:focus-visible` style in `gamecanvas.css`.
- Add `Tab — Return to the script editor` under Game Window in
  `Documentation/keyboard_shortcuts.html`.

## Edge Cases

- Key repeat cannot oscillate focus because the first Tab press transfers focus
  away from the canvas and consumes the event; repeated keydown events no longer
  satisfy the canvas-focus guard.
- Compiler errors do not affect the return path because returning to CodeMirror
  never compiles.
- A CodeMirror search or replace field is not the canvas, so Tab continues to
  behave within that UI.
- Stale `lastDownTarget` state cannot keep moving the game after the transfer,
  because `focusCodeEditor()` explicitly resets both the target and held keys.
- The standalone player does not load the editor-only markup or focus behavior.

## Verification

Verify the interaction in the browser using both keyboard and pointer entry
paths:

1. Place the cursor and a selection in CodeMirror, run with `Ctrl/Cmd+Enter`,
   press Tab, and confirm focus, cursor, selection, and scroll position return.
2. Repeat with `Ctrl/Cmd+Shift+Enter`, RUN, REBUILD, the level-editor action, and
   a direct canvas click.
3. Hold a movement key, press Tab, and confirm movement/repeat stops immediately.
4. Confirm Tab is not delivered as a game action and does not compile or restart.
5. Confirm Tab and Shift+Tab retain their normal behavior in CodeMirror dialogs,
   toolbar controls, and the console.
6. Confirm the canvas focus indicator appears for keyboard focus without
   resizing or obscuring the game.
7. Confirm standalone games retain their existing input and Tab behavior.

Because the current automated suite does not exercise editor DOM focus, this
change should be validated with a focused browser regression check in addition
to the existing test suite.
