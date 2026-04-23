'use strict';

function normalizeInputToken(input) {
    return input;
}

function captureRandomStatePreview(byteCount) {
    if (!RandomGen || !RandomGen._state) {
        return null;
    }
    const state = RandomGen._state;
    const preview = {
        valid: true,
        i: typeof state.i === 'number' ? state.i : 0,
        j: typeof state.j === 'number' ? state.j : 0,
        preview_bytes: [],
    };
    const clone = {
        i: preview.i,
        j: preview.j,
        s: Array.isArray(state.s) ? state.s.slice() : [],
    };
    for (let index = 0; index < byteCount; index++) {
        clone.i = (clone.i + 1) % 256;
        clone.j = (clone.j + clone.s[clone.i]) % 256;
        const swap = clone.s[clone.i];
        clone.s[clone.i] = clone.s[clone.j];
        clone.s[clone.j] = swap;
        preview.preview_bytes.push(clone.s[(clone.s[clone.i] + clone.s[clone.j]) % 256]);
    }
    return preview;
}

function captureTraceSnapshot(phase, inputIndex, inputToken, substepIndex, previousSoundCount) {
    const currentLevelTarget =
        typeof curlevelTarget === 'number'
            ? curlevelTarget
            : (curlevelTarget === null ? null : curlevelTarget);
    const randomState = captureRandomStatePreview(8);

    return {
        phase,
        input_index: inputIndex,
        input: normalizeInputToken(inputToken),
        substep_index: substepIndex,
        current_level_index: typeof curlevel === 'number' ? curlevel : 0,
        current_level_target: currentLevelTarget,
        title_screen: Boolean(titleScreen),
        text_mode: Boolean(textMode),
        title_mode: typeof titleMode === 'number' ? titleMode : 0,
        title_selection: typeof titleSelection === 'number' ? titleSelection : 0,
        title_selected: Boolean(titleSelected),
        message_selected: Boolean(messageselected),
        winning: Boolean(winning),
        againing: Boolean(againing),
        loaded_level_seed: typeof loadedLevelSeed === 'string' ? loadedLevelSeed : null,
        random_state_valid: randomState ? Boolean(randomState.valid) : false,
        random_state_i: randomState ? randomState.i : 0,
        random_state_j: randomState ? randomState.j : 0,
        random_state_preview_bytes: randomState ? randomState.preview_bytes : [],
        serialized_level: typeof convertLevelToString === 'function' ? convertLevelToString() : '',
        command_queue: level && Array.isArray(level.commandQueue) ? level.commandQueue.slice() : [],
        command_queue_source_rules: level && Array.isArray(level.commandQueueSourceRules) ? level.commandQueueSourceRules.slice() : [],
        sound_history_length: Array.isArray(soundHistory) ? soundHistory.length : 0,
        new_sounds: Array.isArray(soundHistory) ? soundHistory.slice(previousSoundCount) : [],
    };
}

function executeInputToken(inputToken) {
    if (inputToken === 'undo') {
        DoUndo(false, true);
    } else if (inputToken === 'restart') {
        DoRestart();
    } else if (inputToken === 'tick') {
        processInput(-1);
    } else {
        processInput(inputToken);
    }
}

function runInputTrace(inputTokens) {
    const snapshots = [];
    let previousSoundCount = 0;

    const pushSnapshot = (phase, inputIndex, inputToken, substepIndex) => {
        const snapshot = captureTraceSnapshot(phase, inputIndex, inputToken, substepIndex, previousSoundCount);
        previousSoundCount = snapshot.sound_history_length;
        snapshots.push(snapshot);
    };

    pushSnapshot('initial', null, null, 0);

    for (let inputIndex = 0; inputIndex < inputTokens.length; inputIndex++) {
        const inputToken = inputTokens[inputIndex];
        executeInputToken(inputToken);
        pushSnapshot('input', inputIndex, inputToken, 0);

        let againSubstep = 0;
        while (againing) {
            againing = false;
            processInput(-1);
            againSubstep += 1;
            pushSnapshot('again', inputIndex, inputToken, againSubstep);
        }
    }

    return {
        input_count: inputTokens.length,
        snapshot_count: snapshots.length,
        snapshots,
    };
}

module.exports = {
    captureTraceSnapshot,
    runInputTrace,
};
