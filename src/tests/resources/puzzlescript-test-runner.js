(function () {
	"use strict";

	var GROUP_SIMULATION = "simulation";
	var GROUP_COMPILER = "compiler-messages";
	var HIDE_PASSED_KEY = "puzzlescript-test-runner-hide-passed";
	var TEST_PAGE_TITLE = document.title;
	var TEST_BATCH_BUDGET_MS = 200;
	var tests = [];
	var activeTest = null;
	var started = false;
	var currentRunTests = [];
	var emptyState = null;
	var filters = {
		hidePassed: readHidePassedPreference(),
		groups: {
			"simulation": true,
			"compiler-messages": true
		}
	};

	function readHidePassedPreference() {
		try {
			return window.sessionStorage.getItem(HIDE_PASSED_KEY) === "true";
		} catch (error) {
			return false;
		}
	}

	function writeHidePassedPreference(hidden) {
		try {
			if (hidden) {
				window.sessionStorage.setItem(HIDE_PASSED_KEY, "true");
			} else {
				window.sessionStorage.removeItem(HIDE_PASSED_KEY);
			}
		} catch (error) {
			// Filtering still works for this page when storage is unavailable.
		}
	}

	function createElement(tagName, className, text) {
		var element = document.createElement(tagName);
		if (className) {
			element.className = className;
		}
		if (text !== undefined) {
			element.textContent = text;
		}
		return element;
	}

	function registerTest(name, options, callback) {
		if (started) {
			throw new Error("Cannot register a test after the run has started.");
		}
		if (!options || (options.group !== GROUP_SIMULATION && options.group !== GROUP_COMPILER)) {
			throw new Error("Test group must be simulation or compiler-messages.");
		}
		if (typeof options.source !== "string") {
			throw new Error("Test source must be a string.");
		}
		if (typeof options.output !== "string") {
			throw new Error("Test output must be a string.");
		}
		if (typeof callback !== "function") {
			throw new Error("Test callback must be a function.");
		}

		tests.push({
			number: tests.length + 1,
			name: String(name),
			group: options.group,
			source: options.source,
			setup: options.setup || [],
			output: options.output,
			callback: callback,
			status: "pending",
			duration: 0,
			comparisons: [],
			error: null,
			expanded: false,
			detailRendered: false,
			element: null
		});
	}

	function requireActiveTest() {
		if (!activeTest) {
			throw new Error("Assertion outside test context.");
		}
	}

	function toDisplayString(value) {
		var serialized;
		if (typeof value === "string") {
			return value;
		}
		if (value === undefined) {
			return "undefined";
		}
		serialized = JSON.stringify(value);
		return serialized === undefined ? String(value) : serialized;
	}

	function pushAssertion(result, actual, expected, message) {
		requireActiveTest();
		if (result) {
			return;
		}
		if (actual === false && expected === false) {
			activeTest.error = {
				message: String(message || "Test error"),
				stack: ""
			};
			return;
		}
		activeTest.comparisons.push({
			expected: toDisplayString(expected),
			actual: toDisplayString(actual)
		});
	}

	function assertEqual(actual, expected, message) {
		pushAssertion(expected == actual, actual, expected, message);
	}

	function normalizeException(error) {
		return {
			message: error && error.message ? error.message : String(error),
			stack: error && error.stack ? error.stack : ""
		};
	}

	function executeTest(testRecord) {
		var returned;
		var startedAt = performance.now();
		testRecord.status = "pending";
		testRecord.comparisons = [];
		testRecord.error = null;
		activeTest = testRecord;
		try {
			returned = testRecord.callback();
		} catch (error) {
			testRecord.error = normalizeException(error);
		} finally {
			activeTest = null;
			testRecord.duration = performance.now() - startedAt;
			document.title = TEST_PAGE_TITLE;
		}

		if (returned === false && !testRecord.error && testRecord.comparisons.length === 0) {
			testRecord.error = {
				message: "Test returned false without failure details.",
				stack: ""
			};
		}
		testRecord.status = testRecord.error ? "errored" :
			testRecord.comparisons.length ? "failed" : "passed";
	}

	function selectedTests() {
		var value = new URLSearchParams(window.location.search).get("testNumber");
		var number;
		if (value === null || value.trim() === "") {
			return tests.slice();
		}
		number = Number(value);
		if (!Number.isInteger(number) || number < 1 || number > tests.length) {
			return tests.slice();
		}
		return [tests[number - 1]];
	}

	function createMetric(className, iconText, value, accessibleLabel) {
		var metric = createElement("span", "test-runner__metric " + className);
		var icon = createElement("span", "test-runner__metric-icon", iconText);
		icon.setAttribute("aria-hidden", "true");
		metric.setAttribute("aria-label", accessibleLabel);
		metric.appendChild(icon);
		metric.appendChild(document.createTextNode(String(value)));
		return metric;
	}

	function runningTestText(testRecord) {
		return "Running: " + testRecord.name;
	}

	function updateRunningTest(root, testRecord) {
		root.querySelector(".test-runner__running").textContent =
			runningTestText(testRecord);
	}

	function renderRunningHeader(testRecord) {
		var header = createElement("header", "test-runner__header suite-running");
		var heading = createElement("h1", "test-runner__title");
		var title = createElement("a", "test-runner__title-link", TEST_PAGE_TITLE);
		var summary = createElement("div", "test-runner__summary");

		title.href = window.location.pathname;
		heading.appendChild(title);
		header.appendChild(heading);
		summary.setAttribute("role", "status");
		summary.setAttribute("aria-live", "polite");
		summary.setAttribute("aria-atomic", "true");
		summary.appendChild(createElement("span", "test-runner__running", runningTestText(testRecord)));
		header.appendChild(summary);
		return header;
	}

	function updateCompletedHeader(header, runTests, duration) {
		var summary = header.querySelector(".test-runner__summary");
		var passed = 0;
		var failed = 0;
		var failedMetric;
		var i;
		for (i = 0; i < runTests.length; i++) {
			if (runTests[i].status === "passed") {
				passed++;
			} else {
				failed++;
			}
		}
		header.classList.remove("suite-running", "suite-passed", "suite-failed");
		header.classList.add(failed ? "suite-failed" : "suite-passed");
		failedMetric = createMetric("test-runner__metric--failed", "×", failed, failed + " failed");
		if (failed === 0) {
			failedMetric.classList.add("zero");
		}
		summary.replaceChildren(
			createMetric("test-runner__metric--passed", "✓", passed, passed + " passed"),
			failedMetric,
			createElement("span", "test-runner__metric test-runner__metric--duration", (duration / 1000).toFixed(2) + "s")
		);
	}

	function renderRunning(root, runTests) {
		var shell = createElement("section", "test-runner");
		var results = createElement("div", "test-runner__results");
		var i;
		currentRunTests = runTests;
		shell.appendChild(renderRunningHeader(runTests[0]));
		shell.appendChild(renderToolbar());
		for (i = 0; i < runTests.length; i++) {
			results.appendChild(renderResult(runTests[i]));
		}
		emptyState = createElement("p", "test-runner__empty", "No results match the active controls.");
		emptyState.hidden = true;
		results.appendChild(emptyState);
		shell.appendChild(results);
		root.replaceChildren(shell);
		updateVisibility();
	}

	function groupCount(group) {
		var count = 0;
		var i;
		for (i = 0; i < currentRunTests.length; i++) {
			if (currentRunTests[i].group === group) {
				count++;
			}
		}
		return count;
	}

	function isResultVisible(testRecord) {
		return filters.groups[testRecord.group] &&
			(!filters.hidePassed || testRecord.status !== "passed");
	}

	function updateVisibility() {
		var visibleCount = 0;
		var i;
		for (i = 0; i < currentRunTests.length; i++) {
			currentRunTests[i].element.hidden = !isResultVisible(currentRunTests[i]);
			if (!currentRunTests[i].element.hidden) {
				visibleCount++;
			}
		}
		if (emptyState) {
			emptyState.hidden = visibleCount !== 0;
		}
	}

	function createGroupFilter(group, label) {
		var button = createElement("button", "test-runner__group-filter" + (filters.groups[group] ? " is-active" : ""));
		var count = createElement("span", "test-runner__group-count", groupCount(group));
		button.type = "button";
		button.setAttribute("aria-pressed", String(filters.groups[group]));
		button.appendChild(document.createTextNode(label));
		button.appendChild(count);
		button.addEventListener("click", function () {
			filters.groups[group] = !filters.groups[group];
			button.setAttribute("aria-pressed", String(filters.groups[group]));
			button.classList.toggle("is-active", filters.groups[group]);
			updateVisibility();
		});
		return button;
	}

	function renderToolbar() {
		var toolbar = createElement("div", "test-runner__toolbar");
		var hidePassed = createElement("label", "test-runner__hide-passed");
		var checkbox = createElement("input", "test-runner__hide-passed-input");
		var track = createElement("span", "test-runner__switch-track");
		checkbox.type = "checkbox";
		checkbox.checked = filters.hidePassed;
		checkbox.setAttribute("role", "switch");
		checkbox.addEventListener("change", function () {
			filters.hidePassed = checkbox.checked;
			writeHidePassedPreference(filters.hidePassed);
			updateVisibility();
		});
		hidePassed.appendChild(checkbox);
		hidePassed.appendChild(track);
		hidePassed.appendChild(document.createTextNode("Hide passed"));
		toolbar.appendChild(hidePassed);
		toolbar.appendChild(createGroupFilter(GROUP_SIMULATION, "Simulation"));
		toolbar.appendChild(createGroupFilter(GROUP_COMPILER, "Compiler messages"));
		return toolbar;
	}

	function rerunUrl(testNumber) {
		var url = new URL(window.location.href);
		url.search = "";
		url.searchParams.set("testNumber", testNumber);
		return url.pathname + url.search;
	}

	function setExpanded(testRecord, expanded) {
		var disclosure = testRecord.element.querySelector(".test-result__disclosure");
		var name = testRecord.element.querySelector(".test-result__name");
		var detail = testRecord.element.querySelector(".test-result__detail");
		if (expanded && !testRecord.detailRendered) {
			populateDetail(detail, testRecord);
			testRecord.detailRendered = true;
		}
		testRecord.expanded = expanded;
		disclosure.setAttribute("aria-expanded", String(expanded));
		disclosure.setAttribute("aria-label", (expanded ? "Collapse " : "Expand ") + testRecord.name);
		name.setAttribute("aria-expanded", String(expanded));
		detail.hidden = !expanded;
		testRecord.element.classList.toggle("is-expanded", expanded);
	}

	function diffSequence(expected, actual) {
		var rows = expected.length + 1;
		var columns = actual.length + 1;
		var table = new Array(rows);
		var operations = [];
		var i;
		var j;

		for (i = 0; i < rows; i++) {
			table[i] = new Array(columns).fill(0);
		}
		for (i = expected.length - 1; i >= 0; i--) {
			for (j = actual.length - 1; j >= 0; j--) {
				table[i][j] = expected[i] === actual[j] ?
					table[i + 1][j + 1] + 1 :
					Math.max(table[i + 1][j], table[i][j + 1]);
			}
		}

		i = 0;
		j = 0;
		while (i < expected.length || j < actual.length) {
			if (i < expected.length && j < actual.length && expected[i] === actual[j]) {
				operations.push({ type: "equal", expected: expected[i], actual: actual[j] });
				i++;
				j++;
			} else if (j < actual.length && (i === expected.length || table[i][j + 1] >= table[i + 1][j])) {
				operations.push({ type: "insert", actual: actual[j++] });
			} else {
				operations.push({ type: "delete", expected: expected[i++] });
			}
		}
		return operations;
	}

	function appendSegment(segments, text, changed) {
		var last = segments[segments.length - 1];
		if (last && last.changed === changed) {
			last.text += text;
		} else {
			segments.push({ text: text, changed: changed });
		}
	}

	function diffLine(expected, actual) {
		var operations = diffSequence(Array.from(expected), Array.from(actual));
		var result = { expected: [], actual: [] };
		var i;
		for (i = 0; i < operations.length; i++) {
			if (operations[i].type === "equal") {
				appendSegment(result.expected, operations[i].expected, false);
				appendSegment(result.actual, operations[i].actual, false);
			} else if (operations[i].type === "delete") {
				appendSegment(result.expected, operations[i].expected, true);
			} else {
				appendSegment(result.actual, operations[i].actual, true);
			}
		}
		return result;
	}

	function addChangedLine(result, expectedLine, actualLine) {
		var lineDiff;
		if (expectedLine !== undefined && actualLine !== undefined) {
			lineDiff = diffLine(expectedLine, actualLine);
			result.expected.push(lineDiff.expected);
			result.actual.push(lineDiff.actual);
		} else if (expectedLine !== undefined) {
			result.expected.push([{ text: expectedLine, changed: true }]);
			result.actual.push([]);
		} else {
			result.expected.push([]);
			result.actual.push([{ text: actualLine, changed: true }]);
		}
	}

	function diffText(expected, actual) {
		var operations = diffSequence(expected.split("\n"), actual.split("\n"));
		var result = { expected: [], actual: [] };
		var deleted = [];
		var inserted = [];
		var i;

		function flushChanges() {
			var count = Math.max(deleted.length, inserted.length);
			var index;
			for (index = 0; index < count; index++) {
				addChangedLine(result, deleted[index], inserted[index]);
			}
			deleted = [];
			inserted = [];
		}

		for (i = 0; i < operations.length; i++) {
			if (operations[i].type === "equal") {
				flushChanges();
				result.expected.push([{ text: operations[i].expected, changed: false }]);
				result.actual.push([{ text: operations[i].actual, changed: false }]);
			} else if (operations[i].type === "delete") {
				deleted.push(operations[i].expected);
			} else {
				inserted.push(operations[i].actual);
			}
		}
		flushChanges();
		return result;
	}

	function renderDiffLines(lines, side) {
		var pre = createElement("pre", "diff__value");
		var line;
		var segment;
		var lineElement;
		var segmentElement;
		for (line = 0; line < lines.length; line++) {
			lineElement = createElement("span", "diff__line");
			for (segment = 0; segment < lines[line].length; segment++) {
				if (lines[line][segment].changed) {
					segmentElement = createElement(
						"span",
						"diff__change diff__change--" + side,
						lines[line][segment].text
					);
					lineElement.appendChild(segmentElement);
				} else {
					lineElement.appendChild(document.createTextNode(lines[line][segment].text));
				}
			}
			pre.appendChild(lineElement);
		}
		return pre;
	}

	function renderValuePane(label, lines, side) {
		var pane = createElement("section", "diff__pane diff__pane--" + side);
		var heading = createElement("h3", "test-result__section-label diff__label diff__label--" + side, label);
		var scroll = createElement("div", "diff__scroll");
		scroll.appendChild(renderDiffLines(lines, side));
		pane.appendChild(heading);
		pane.appendChild(scroll);
		return pane;
	}

	function renderComparison(comparison) {
		var diff = diffText(comparison.expected, comparison.actual);
		var comparisonElement = createElement("div", "diff");
		comparisonElement.appendChild(renderValuePane("Expected", diff.expected, "expected"));
		comparisonElement.appendChild(renderValuePane("Actual", diff.actual, "actual"));
		return comparisonElement;
	}

	function renderOutput(output) {
		var section = createElement("section", "test-output");
		var scroll = createElement("div", "test-output__scroll");
		section.appendChild(createElement("h3", "test-result__section-label", "Output"));
		scroll.appendChild(createElement("pre", "test-output__value", output));
		section.appendChild(scroll);
		return section;
	}

	function renderSetup(rows) {
		var section = createElement("section", "test-setup");
		var list = createElement("dl", "test-setup__list");
		var row;
		var i;
		for (i = 0; i < rows.length; i++) {
			row = createElement("div", "test-setup__row");
			row.appendChild(createElement("dt", "test-setup__label", rows[i].label));
			row.appendChild(createElement("dd", "test-setup__value", rows[i].value));
			list.appendChild(row);
		}
		section.appendChild(createElement("h3", "test-result__section-label", "Setup"));
		section.appendChild(list);
		return section;
	}

	function renderError(error) {
		var section = createElement("section", "test-error");
		var text = error.message;
		if (error.stack && error.stack.indexOf(error.message) === -1) {
			text += "\n\n" + error.stack;
		} else if (error.stack) {
			text = error.stack;
		}
		section.appendChild(createElement("h3", "test-result__section-label test-error__label", "Error"));
		section.appendChild(createElement("pre", "test-error__content", text));
		return section;
	}

	function selectSource(sourceElement) {
		var selection = window.getSelection();
		var range = document.createRange();
		range.selectNodeContents(sourceElement);
		selection.removeAllRanges();
		selection.addRange(range);
	}

	function setCopyState(button, state) {
		var label = "Copy source";
		if (button.copyResetTimer) {
			clearTimeout(button.copyResetTimer);
		}
		button.classList.remove("is-copied", "is-failed");
		if (state === "copied") {
			label = "Source copied";
			button.classList.add("is-copied");
		} else if (state === "failed") {
			label = "Copy failed; source selected";
			button.classList.add("is-failed");
		}
		button.setAttribute("aria-label", label);
		button.title = label;
		if (state !== "ready") {
			button.copyResetTimer = setTimeout(function () {
				setCopyState(button, "ready");
			}, 2000);
		}
	}

	function legacyCopySource() {
		try {
			return document.execCommand("copy");
		} catch (error) {
			return false;
		}
	}

	function finishCopy(button, copied) {
		setCopyState(button, copied ? "copied" : "failed");
	}

	function copySource(button, sourceElement, source) {
		var request;
		selectSource(sourceElement);
		if (!navigator.clipboard || typeof navigator.clipboard.writeText !== "function") {
			finishCopy(button, legacyCopySource());
			return;
		}
		try {
			request = navigator.clipboard.writeText(source);
			if (request && typeof request.then === "function") {
				request.then(function () {
					finishCopy(button, true);
				}, function () {
					finishCopy(button, legacyCopySource());
				});
			} else {
				finishCopy(button, true);
			}
		} catch (error) {
			finishCopy(button, legacyCopySource());
		}
	}

	function renderSource(testRecord) {
		var section = createElement("section", "test-result__source-section");
		var panel = createElement("div", "source-panel");
		var scroll = createElement("div", "source-panel__scroll");
		var pre = createElement("pre", "source-panel__code");
		var source = createElement("code", "source-panel__source", testRecord.source);
		var copy = createElement("button", "source-panel__copy");
		copy.type = "button";
		setCopyState(copy, "ready");
		copy.addEventListener("click", function (event) {
			event.stopPropagation();
			copySource(copy, source, testRecord.source);
		});
		pre.appendChild(source);
		scroll.appendChild(pre);
		panel.appendChild(scroll);
		panel.appendChild(copy);
		section.appendChild(createElement("h3", "test-result__section-label", "PuzzleScript source"));
		section.appendChild(panel);
		return section;
	}

	function populateDetail(detail, testRecord) {
		var i;
		detail.appendChild(renderSource(testRecord));
		if (testRecord.setup.length) {
			detail.appendChild(renderSetup(testRecord.setup));
		}
		if (testRecord.status === "passed") {
			detail.appendChild(renderOutput(testRecord.output));
		}
		for (i = 0; i < testRecord.comparisons.length; i++) {
			detail.appendChild(renderComparison(testRecord.comparisons[i]));
		}
		if (testRecord.error) {
			detail.appendChild(renderError(testRecord.error));
		}
	}

	function renderResult(testRecord) {
		var result = createElement("article", "test-result test-result--" + testRecord.status);
		var summary = createElement("div", "test-result__summary");
		var disclosure = createElement("button", "test-result__disclosure", "▶");
		var pending = testRecord.status === "pending";
		var statusText = testRecord.status === "errored" ? "Error" :
			pending ? "Pending" : testRecord.status === "passed" ? "Passed" : "Failed";
		var statusIcon = pending ? "•" : testRecord.status === "passed" ? "✓" : "×";
		var status = createElement("span", "test-result__status", statusIcon);
		var identity = createElement("div", "test-result__identity");
		var name = createElement("button", "test-result__name", testRecord.name);
		var rerun = createElement("a", "test-result__rerun", "Rerun");
		var detail = createElement("div", "test-result__detail");
		var detailId = "test-result-detail-" + testRecord.number;

		disclosure.type = "button";
		disclosure.setAttribute("aria-controls", detailId);
		disclosure.setAttribute("aria-expanded", "false");
		disclosure.setAttribute("aria-label", "Expand " + testRecord.name);
		status.setAttribute("role", "img");
		status.setAttribute("aria-label", statusText);
		name.type = "button";
		name.setAttribute("aria-controls", detailId);
		name.setAttribute("aria-expanded", "false");
		disclosure.disabled = pending;
		name.disabled = pending;
		rerun.href = rerunUrl(testRecord.number);
		rerun.addEventListener("click", function (event) {
			event.stopPropagation();
		});
		detail.id = detailId;
		detail.hidden = true;
		disclosure.addEventListener("click", function () {
			setExpanded(testRecord, !testRecord.expanded);
		});
		name.addEventListener("click", function () {
			setExpanded(testRecord, !testRecord.expanded);
		});
		identity.appendChild(name);
		identity.appendChild(rerun);
		summary.appendChild(disclosure);
		summary.appendChild(status);
		summary.appendChild(identity);
		summary.appendChild(createElement("span", "test-result__duration", pending ? "—" : testRecord.duration.toFixed(1) + "ms"));
		result.appendChild(summary);
		result.appendChild(detail);
		testRecord.element = result;
		return result;
	}

	function renderCompletedResult(testRecord) {
		var pendingElement = testRecord.element;
		var completedElement = renderResult(testRecord);
		pendingElement.replaceWith(completedElement);
	}

	function scheduleRunStep(callback) {
		if (document.hidden) {
			setTimeout(callback, 0);
			return;
		}
		requestAnimationFrame(callback);
	}

	function runBatch(root, runTests, nextIndex, totalDuration) {
		var batchStartedAt = performance.now();
		var completedInBatch = [];
		var testRecord;
		var i;

		do {
			testRecord = runTests[nextIndex];
			executeTest(testRecord);
			totalDuration += testRecord.duration;
			completedInBatch.push(testRecord);
			nextIndex++;
		} while (nextIndex < runTests.length &&
			performance.now() - batchStartedAt < TEST_BATCH_BUDGET_MS);

		for (i = 0; i < completedInBatch.length; i++) {
			renderCompletedResult(completedInBatch[i]);
		}
		updateVisibility();

		if (nextIndex < runTests.length) {
			updateRunningTest(root, runTests[nextIndex]);
			scheduleRunStep(function () {
				runBatch(root, runTests, nextIndex, totalDuration);
			});
			return;
		}

		updateCompletedHeader(
			root.querySelector(".test-runner__header"),
			runTests,
			totalDuration
		);
	}

	function start() {
		var root;
		var runTests;
		if (started) {
			throw new Error("PuzzleScript tests have already started.");
		}
		started = true;
		document.title = TEST_PAGE_TITLE;
		root = document.getElementById("puzzlescript-test-runner");
		if (!root) {
			throw new Error("Missing #puzzlescript-test-runner root.");
		}
		runTests = selectedTests();
		renderRunning(root, runTests);
		scheduleRunStep(function () {
			scheduleRunStep(function () {
				runBatch(root, runTests, 0, 0);
			});
		});
	}

	window.test = registerTest;
	window.PuzzleScriptTests = { start: start };
	window.PuzzleScriptTestAssertions = {
		push: pushAssertion,
		equal: assertEqual
	};
}());
