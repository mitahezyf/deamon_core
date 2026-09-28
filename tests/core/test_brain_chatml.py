"""
Testy jednostkowe: TagAwareBuffer, SentenceBuffer, walidacja Safe API.
Pokrywają:
- Parsowanie <tool_call> z poszatkowanych tokenów SSE
- Filtracja <think>…</think> ze strumienia
- Flush resztek z niedokończonych tagów
- Walidacja whitelist (ALLOWED_TOOLS / ALLOWED_ACTIONS)
- SentenceBuffer zachowanie regresyjne
"""

import json
import pytest

from app.core.brain import (
    TagAwareBuffer,
    SentenceBuffer,
    ALLOWED_TOOLS,
    ALLOWED_ACTIONS,
)


# ============================================================================
# TagAwareBuffer: poprawne parsowanie <tool_call>
# ============================================================================

class TestTagAwareBuffer:

    def test_normal_text_passthrough(self):
        """Zwykły tekst bez tagów przechodzi bez zmian."""
        buf = TagAwareBuffer()
        clean, tcs = buf.feed("Cześć wodzu, wszystko gotowe.")
        assert clean == "Cześć wodzu, wszystko gotowe."
        assert tcs == []

    def test_tool_call_single_token(self):
        """<tool_call>…</tool_call> w jednym tokenie jest poprawnie wyodrębniony."""
        buf = TagAwareBuffer()
        raw = '<tool_call>\n{"name": "desktop_action", "arguments": {"action": "vol_up"}}\n</tool_call>'
        clean, tcs = buf.feed(raw)
        assert clean == ""
        assert len(tcs) == 1
        parsed = json.loads(tcs[0])
        assert parsed["name"] == "desktop_action"
        assert parsed["arguments"]["action"] == "vol_up"

    def test_tool_call_fragmented_tokens(self):
        """Poszatkowane tokeny SSE (np. '<tool_' + 'call>') są poprawnie sklejane."""
        buf = TagAwareBuffer()

        # Symulacja fragmentacji: 6 kawałków
        fragments = [
            "<tool_",
            "call>",
            '\n{"name": "desktop_action",',
            ' "arguments": {"action": "vol_down"}}',
            "\n</tool_",
            "call>",
        ]

        all_clean = ""
        all_tcs = []
        for frag in fragments:
            clean, tcs = buf.feed(frag)
            all_clean += clean
            all_tcs.extend(tcs)

        assert all_clean == ""
        assert len(all_tcs) == 1
        parsed = json.loads(all_tcs[0])
        assert parsed["arguments"]["action"] == "vol_down"

    def test_tool_call_with_surrounding_text(self):
        """Tekst przed i po <tool_call> jest poprawnie oddzielony."""
        buf = TagAwareBuffer()

        c1, t1 = buf.feed("Jasne, zrobię to. ")
        assert c1 == "Jasne, zrobię to. "
        assert t1 == []

        c2, t2 = buf.feed('<tool_call>\n{"name": "desktop_action", "arguments": {"action": "open_browser"}}\n</tool_call>')
        assert c2 == ""
        assert len(t2) == 1

        c3, t3 = buf.feed(" Gotowe!")
        assert c3 == " Gotowe!"
        assert t3 == []

    def test_think_tag_filtered(self):
        """<think>…</think> jest odrzucane w ciszy."""
        buf = TagAwareBuffer()

        c1, t1 = buf.feed("Odpowiedź: ")
        assert c1 == "Odpowiedź: "

        c2, t2 = buf.feed("<think>Hmm, zastanówmy się nad tym...</think>")
        assert c2 == ""
        assert t2 == []

        c3, t3 = buf.feed("To jest bezpośrednia odpowiedź.")
        assert c3 == "To jest bezpośrednia odpowiedź."

    def test_think_tag_fragmented(self):
        """Poszatkowane <think> + </think> poprawnie filtrowane."""
        buf = TagAwareBuffer()
        fragments = ["<thi", "nk>", "wewnętrzne myślenie", "</thi", "nk>", "Tekst po myśleniu."]

        clean_all = ""
        for f in fragments:
            c, _ = buf.feed(f)
            clean_all += c

        assert clean_all == "Tekst po myśleniu."

    def test_non_matching_angle_bracket(self):
        """< nie będący początkiem znanego tagu jest przepuszczany jako tekst."""
        buf = TagAwareBuffer()
        c, t = buf.feed("Temperatura < 100 stopni.")
        remaining, _ = buf.flush()
        full = c + remaining
        assert "<" in full
        assert "100" in full
        assert t == []

    def test_flush_incomplete_tag_candidate(self):
        """Flush wypycha niedokończony tag candidate jako zwykły tekst."""
        buf = TagAwareBuffer()
        c, t = buf.feed("Dane: <to")
        assert c == "Dane: "
        assert t == []

        remaining, t2 = buf.flush()
        assert remaining == "<to"
        assert t2 == []

    def test_flush_incomplete_tool_call(self):
        """Niedokończony <tool_call> (model urwał) jest ignorowany w flush."""
        buf = TagAwareBuffer()
        buf.feed('<tool_call>\n{"name": "desktop_action"')
        remaining, tcs = buf.flush()
        assert remaining == ""
        assert tcs == []

    def test_multiple_tool_calls_in_sequence(self):
        """Dwa <tool_call> bloki w sekwencji."""
        buf = TagAwareBuffer()
        raw = (
            '<tool_call>\n{"name": "desktop_action", "arguments": {"action": "vol_up"}}\n</tool_call>'
            'Podniosłem głośność. '
            '<tool_call>\n{"name": "request_frame", "arguments": {}}\n</tool_call>'
        )
        clean, tcs = buf.feed(raw)
        assert clean == "Podniosłem głośność. "
        assert len(tcs) == 2


# ============================================================================
# SentenceBuffer: testy regresyjne
# ============================================================================

class TestSentenceBuffer:

    def test_basic_sentence_extraction(self):
        sb = SentenceBuffer(min_length=5)
        sentences = sb.add("Cześć wodzu. Jak leci?")
        assert "Cześć wodzu." in sentences

    def test_short_fragment_buffered(self):
        sb = SentenceBuffer(min_length=20)
        sentences = sb.add("OK.")
        assert sentences == []
        remaining = sb.flush()
        assert remaining == "OK."

    def test_markdown_sanitization(self):
        sb = SentenceBuffer(min_length=5)
        sentences = sb.add("**Cześć** _wodzu_. `Kod` tutaj.")
        # Gwiazdki, podkreślenia, backticki powinny być usunięte
        for s in sentences:
            assert "*" not in s
            assert "_" not in s
            assert "`" not in s

    def test_flush_returns_remainder(self):
        sb = SentenceBuffer(min_length=15)
        sb.add("Fragment bez kropki")
        result = sb.flush()
        assert result == "Fragment bez kropki"

    def test_flush_empty_returns_none(self):
        sb = SentenceBuffer()
        assert sb.flush() is None


# ============================================================================
# Safe API Whitelist
# ============================================================================

class TestSafeAPIWhitelist:

    def test_allowed_tools_complete(self):
        assert "desktop_action" in ALLOWED_TOOLS
        assert "request_frame" in ALLOWED_TOOLS
        # Brak run_command, exec, shell itp.
        assert "run_command" not in ALLOWED_TOOLS
        assert "exec" not in ALLOWED_TOOLS
        assert "shell" not in ALLOWED_TOOLS

    def test_allowed_actions_mapped_to_client(self):
        """Weryfikacja 1:1 mapowania z client_actions.py ActionExecutor."""
        expected = {
            "vol_up", "vol_down", "vol_mute", "vol_unmute",
            "open_browser", "open_calc", "open_notepad",
            "open_taskmgr", "open_powershell", "lock_system",
            "get_time", "get_date", "get_sys_status",
        }
        assert ALLOWED_ACTIONS == expected

    def test_unknown_tool_rejected(self):
        """Narzędzie spoza whitelist jest odrzucone."""
        assert "run_shell" not in ALLOWED_TOOLS

    def test_unknown_action_rejected(self):
        """Akcja spoza whitelist nie przechodzi walidacji."""
        assert "format_disk" not in ALLOWED_ACTIONS
        assert "rm_rf" not in ALLOWED_ACTIONS

    def test_double_bracket_tool_call(self):
        """Podwójny nawias trójkątny <<tool_call> nie gubi drugiego <."""
        buf = TagAwareBuffer()
        c, t = buf.feed('<<tool_call>\n{"name": "desktop_action", "arguments": {"action": "vol_up"}}\n</tool_call>')
        assert c == "<"
        assert len(t) == 1
        parsed = json.loads(t[0])
        assert parsed["arguments"]["action"] == "vol_up"
