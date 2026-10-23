"""The built-in web console: served from the package, no backend state needed."""

from html.parser import HTMLParser

from fastapi.testclient import TestClient

from maxmlx.server.api_server import app


class ConsoleStructure(HTMLParser):
    """Check view ownership and balanced markup, not just matching strings."""
    def __init__(self, page):
        super().__init__()
        self.stack = []
        self.ancestors = {}
        self.elements = {}
        self.feed(page)
        assert not self.stack

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        element_id = attrs.get("id")
        if element_id:
            assert element_id not in self.ancestors, f"duplicate ID: {element_id}"
            self.ancestors[element_id] = [parent_id for _, parent_id in self.stack]
            self.elements[element_id] = (tag, attrs)
        if tag not in {"meta", "input", "br", "hr", "link", "img"}:
            self.stack.append((tag, element_id))

    def handle_endtag(self, tag):
        assert self.stack and self.stack[-1][0] == tag, f"unbalanced closing tag: {tag}"
        self.stack.pop()


def test_four_views_have_separate_responsibilities():
    page = TestClient(app).get("/console").text
    structure = ConsoleStructure(page)
    views = {key for key, (_, attrs) in structure.elements.items() if "view" in attrs.get("class", "").split()}
    assert views == {"dashboardView", "settingsView", "downloadsView", "chatView"}
    for key in views:
        tag, attrs = structure.elements[key]
        assert tag == "section" and attrs["aria-labelledby"] in structure.elements
        assert "content" in structure.ancestors[key]
    for element in ("cTokens", "cReq", "mTps", "mTtft", "memTxt", "kvRow", "diskRow", "requestLog"):
        assert "dashboardView" in structure.ancestors[element]
    for element in ("moeRange", "ctxRange", "genTemp", "genThink", "thinkRange", "reloadBtn", "apiOpenai"):
        assert "settingsView" in structure.ancestors[element]
    for element in ("sysPrompt", "messages", "input", "sendBtn", "stopBtn"):
        assert "chatView" in structure.ancestors[element]
    for element in ("dlRepo", "dlStart", "dlQueue", "dlList", "dlQuery", "dlDetail", "dlSettings"):
        assert "downloadsView" in structure.ancestors[element]
    assert 'id="logsView"' not in page and 'id="sidebar"' not in page


def test_settings_have_one_save_button_outside_all_category_panels():
    page = TestClient(app).get("/console").text
    structure = ConsoleStructure(page)
    assert page.count('id="settingsSave"') == 1
    assert "settingsView" in structure.ancestors["settingsSave"]
    assert "settingsSaveBar" in structure.ancestors["settingsSave"]
    assert not {"modelSettings", "memorySettings", "serverSettings", "integrationSettings"}.intersection(structure.ancestors["settingsSave"])
    for old in ("genApply", "ctxApply", "thinkApply", "cacheApply", "performanceApply", "diskApply", "modelApply", "serverApply"):
        assert old not in structure.elements


def test_navigation_and_controls_are_accessible():
    page = TestClient(app).get("/console").text
    structure = ConsoleStructure(page)
    assert structure.elements["navigation"][1]["aria-label"] == "Main navigation"
    assert 'aria-controls="dashboardView"' in page
    assert 'aria-controls="settingsView"' in page
    assert 'aria-controls="chatView"' in page
    assert 'aria-current="page"' in page
    assert 'for="moeRange"' in page and 'for="ctxRange"' in page
    assert 'for="genTemp"' in page and 'for="genMax"' in page
    assert 'for="genThink"' in page and 'for="thinkRange"' in page
    assert 'for="sysPrompt"' in page and 'aria-label="Message"' in page
    assert 'location.hash === "#logs"' in page
    assert '"#settings"' in page and '"hashchange"' in page
    assert 'id="memBar"' not in page, "do not invent a memory capacity for the usage bar"


def test_console_served():
    client = TestClient(app)
    r = client.get("/console")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "maxMLX Console" in r.text
    # the page drives exactly these same-origin endpoints, and reaches our own
    # ones under /admin rather than in the protocols' /v1 namespace
    for path in ("/health", "/admin/stats", "/admin/cache/status", "/admin/requests"):
        assert path in r.text


def test_a_waiting_turn_shows_a_spinner_and_a_clock():
    """An empty assistant bubble reads as a hung server.

    A long prompt or a reasoning model can take seconds before its first token,
    so the bubble carries a spinner and a running clock until then, and the
    clock has to be stopped on every exit — including an abort or an HTTP error,
    or it ticks forever in a bubble nobody is filling.
    """
    page = TestClient(app).get("/console").text
    assert ".spinner" in page and "@keyframes spin" in page
    assert "Processing prompt" in page
    assert "prefers-reduced-motion" in page, "an infinite spinner needs the opt-out"
    # The status line is derived from the message's streaming flag, and `finally` always clears
    # it (abort, HTTP error and success alike), so the clock cannot outlive the turn.
    assert "msg._streaming = false; delete chat.streams[doc.id]" in page
    assert "status.hidden = !(msg._streaming && !msg.content)" in page


def test_root_redirects_to_console():
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/console"


def test_generation_defaults_live_in_settings_not_the_chat():
    """Server-wide defaults stay separate from the chat's request controls."""
    page = TestClient(app).get("/console").text
    structure = ConsoleStructure(page)
    for element in ("genTemp", "genMax", "settingsSave"):
        assert "settingsView" in structure.ancestors[element]
        assert "chatView" not in structure.ancestors[element]
    assert 'id="maxTok"' not in page and 'id="temp"' not in page
    assert "max_output_tokens" in page
    # the chat request carries neither field any more
    assert 'temperature: parseFloat($("temp")' not in page
    assert 'max_tokens: parseInt($("maxTok")' not in page


def test_the_thinking_switch_sits_in_the_generation_row_and_the_budget_is_a_cap():
    """"Reasoning budget: off" read as "thinking off" — it is a cap. The switch
    that actually disables thinking lives beside temperature and the output cap
    and rides the same /admin/cache/rebuild call."""
    page = TestClient(app).get("/console").text
    structure = ConsoleStructure(page)
    assert "settingsView" in structure.ancestors["genThink"]
    assert '<option value="off">off</option>' in page
    assert 'thinking: $("genThink").value' in page
    assert "no cap" in page and "budget off" not in page
