from __future__ import annotations

from collections import deque
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import queue
import threading
import time
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from asr_interfaces.srv import GetStatus as AsrGetStatus
from reception_interfaces.msg import ConversationTrace
from reception_interfaces.msg import ExecutionCommand
from reception_interfaces.msg import ExecutionEvent
from reception_interfaces.msg import SessionStateV2
from ros2_chat_interfaces.msg import ChatBridgeStatus
from ros2_vllm_interfaces.msg import LlmStatus
from tts_msgs.action import Speak

from .conversation_trace import conversation_trace_to_dict
from .conversation_trace import ros_time_to_iso8601


HTML_PAGE = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reception Demo Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --bg2: #172554;
      --panel: rgba(15, 23, 42, 0.78);
      --panel-strong: rgba(15, 23, 42, 0.92);
      --line: rgba(148, 163, 184, 0.22);
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #22c55e;
      --accent-2: #38bdf8;
      --warn: #f59e0b;
      --danger: #ef4444;
      --shadow: 0 22px 60px rgba(2, 6, 23, 0.45);
      --radius: 22px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(34, 197, 94, 0.14), transparent 28%),
        linear-gradient(160deg, var(--bg) 0%, var(--bg2) 100%);
      font-family: "IBM Plex Sans JP", "Noto Sans JP", sans-serif;
      min-height: 100vh;
    }
    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero, .panel {
      background: var(--panel);
      backdrop-filter: blur(16px);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 24px;
      margin-bottom: 18px;
      display: grid;
      gap: 14px;
    }
    .hero-top {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .title {
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.02;
      letter-spacing: -0.04em;
      margin: 0;
      font-weight: 700;
    }
    .subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }
    .hero-status {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.72);
      border: 1px solid var(--line);
      font-size: 14px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--muted);
      box-shadow: 0 0 20px currentColor;
    }
    .dot.ready { color: var(--accent); background: var(--accent); }
    .dot.warn { color: var(--warn); background: var(--warn); }
    .dot.error { color: var(--danger); background: var(--danger); }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .status-card {
      padding: 16px;
      border-radius: 18px;
      background: rgba(15, 23, 42, 0.72);
      border: 1px solid var(--line);
      min-height: 132px;
    }
    .status-card h3 {
      margin: 0 0 10px;
      font-size: 13px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .status-main {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 8px;
    }
    .meta-list, .config-grid, .session-grid {
      display: grid;
      gap: 10px;
    }
    .meta-list {
      font-size: 13px;
      color: var(--muted);
    }
    .content-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(320px, 0.95fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      padding: 18px;
    }
    .panel h2 {
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: -0.02em;
    }
    .conversation-list, .event-list {
      display: grid;
      gap: 12px;
      max-height: 860px;
      overflow: auto;
      padding-right: 6px;
    }
    .bubble {
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 14px 16px;
      background: rgba(15, 23, 42, 0.72);
      position: relative;
    }
    .bubble.user {
      margin-right: 14%;
      background: linear-gradient(180deg, rgba(12, 74, 110, 0.72), rgba(8, 47, 73, 0.72));
    }
    .bubble.assistant {
      margin-left: 14%;
      background: linear-gradient(180deg, rgba(20, 83, 45, 0.72), rgba(20, 46, 32, 0.76));
    }
    .bubble.system {
      border-style: dashed;
      background: rgba(30, 41, 59, 0.76);
    }
    .bubble-head, .event-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .bubble-role {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(226, 232, 240, 0.92);
    }
    .bubble-text {
      font-size: 16px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--text);
      background: rgba(30, 41, 59, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 999px;
      padding: 6px 10px;
    }
    .config-grid, .session-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .kv {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(15, 23, 42, 0.62);
    }
    .kv .k {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .kv .v {
      font-size: 14px;
      line-height: 1.5;
      word-break: break-word;
    }
    .event-row {
      border: 1px solid var(--line);
      border-left-width: 4px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.68);
      padding: 14px;
    }
    .event-row.info { border-left-color: var(--accent-2); }
    .event-row.success { border-left-color: var(--accent); }
    .event-row.warn { border-left-color: var(--warn); }
    .event-row.error { border-left-color: var(--danger); }
    .event-title {
      font-size: 15px;
      font-weight: 600;
    }
    details {
      margin-top: 10px;
    }
    summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
    }
    pre {
      margin: 10px 0 0;
      padding: 12px;
      border-radius: 14px;
      background: rgba(2, 6, 23, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.15);
      color: #cbd5e1;
      font-size: 12px;
      line-height: 1.5;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .empty {
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 18px;
      padding: 18px;
      text-align: center;
    }
    @media (max-width: 1120px) {
      .status-grid, .content-grid, .config-grid, .session-grid {
        grid-template-columns: 1fr 1fr;
      }
    }
    @media (max-width: 760px) {
      .shell { padding: 14px; }
      .status-grid, .content-grid, .config-grid, .session-grid {
        grid-template-columns: 1fr;
      }
      .bubble.user, .bubble.assistant {
        margin-left: 0;
        margin-right: 0;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-top">
        <div>
          <h1 class="title">Reception Demo Dashboard</h1>
          <p class="subtitle">会話の流れ、セッション状態、依存ノードの生存状態をローカルブラウザで追跡</p>
        </div>
        <div id="hero-status" class="hero-status">
          <span class="dot"></span>
          <span>Loading</span>
        </div>
      </div>
      <div id="status-grid" class="status-grid"></div>
    </section>

    <section class="content-grid">
      <div class="panel">
        <h2>Conversation Timeline</h2>
        <div id="conversation-list" class="conversation-list"></div>
      </div>

      <div style="display:grid; gap:18px;">
        <section class="panel">
          <h2>Current Session</h2>
          <div id="session-grid" class="session-grid"></div>
        </section>
        <section class="panel">
          <h2>Runtime Config</h2>
          <div id="config-grid" class="config-grid"></div>
        </section>
      </div>
    </section>

    <section class="panel" style="margin-top:18px;">
      <h2>Pipeline Events</h2>
      <div id="event-list" class="event-list"></div>
    </section>
  </div>

  <script>
    const state = { snapshot: null, stream: null };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatValue(value) {
      if (value === null || value === undefined || value === "") return "—";
      if (typeof value === "boolean") return value ? "true" : "false";
      return String(value);
    }

    function formatJson(value) {
      if (value === null || value === undefined || value === "") return "";
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    }

    function formatStamp(value) {
      if (!value) return "—";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleTimeString("ja-JP", { hour12: false }) + "." + String(date.getMilliseconds()).padStart(3, "0");
    }

    function heroStatus(snapshot) {
      const ready = snapshot?.ready;
      const cls = ready ? "ready" : "warn";
      const text = ready ? "All Backends Ready" : "Waiting For Backends";
      document.getElementById("hero-status").innerHTML = `<span class="dot ${cls}"></span><span>${escapeHtml(text)}</span>`;
    }

    function renderStatusGrid(snapshot) {
      const order = ["asr", "llm", "tts", "chat"];
      const labels = { asr: "ASR", llm: "LLM", tts: "TTS", chat: "Chat" };
      const html = order.map((key) => {
        const item = snapshot.system.backends[key] || {};
        const dot = item.ready ? "ready" : (item.severity === "error" ? "error" : "warn");
        const extras = Object.entries(item.meta || {}).map(([k, v]) =>
          `<div>${escapeHtml(k)}: ${escapeHtml(formatValue(v))}</div>`
        ).join("");
        return `
          <article class="status-card">
            <h3>${labels[key]}</h3>
            <div class="status-main">
              <span class="dot ${dot}"></span>
              <span>${escapeHtml(item.label || "unknown")}</span>
            </div>
            <div class="meta-list">
              <div>${escapeHtml(item.message || "no status yet")}</div>
              ${extras}
            </div>
          </article>
        `;
      }).join("");
      document.getElementById("status-grid").innerHTML = html;
    }

    function renderConversation(snapshot) {
      const items = snapshot.conversation || [];
      if (!items.length) {
        document.getElementById("conversation-list").innerHTML = `<div class="empty">まだ会話トレースはありません。</div>`;
        return;
      }
      const html = items.slice().reverse().map((item) => {
        const chips = [
          item.turn_seq ? `turn ${item.turn_seq}` : "",
          item.dialog_act ? `act ${item.dialog_act}` : "",
          item.phase ? `phase ${item.phase}` : "",
          item.event_type ? `event ${item.event_type}` : "",
          item.utterance_id ? `utterance ${item.utterance_id}` : "",
          item.asr_confidence ? `conf ${item.asr_confidence}` : ""
        ].filter(Boolean).map((label) => `<span class="chip">${escapeHtml(label)}</span>`).join("");
        const payload = item.payload_json ? `
          <details>
            <summary>payload</summary>
            <pre>${escapeHtml(formatJson(item.payload || item.payload_json))}</pre>
          </details>` : "";
        return `
          <article class="bubble ${escapeHtml(item.role)}">
            <div class="bubble-head">
              <span class="bubble-role">${escapeHtml(item.role)}</span>
              <span class="bubble-role">${escapeHtml(formatStamp(item.timestamp))}</span>
            </div>
            <div class="bubble-text">${escapeHtml(item.text || "")}</div>
            <div class="chips">${chips}</div>
            ${payload}
          </article>
        `;
      }).join("");
      document.getElementById("conversation-list").innerHTML = html;
    }

    function renderSession(snapshot) {
      const session = snapshot.session_state || {};
      const working = session.working_info || {};
      const committed = session.committed_info || {};
      const entries = [
        ["phase", session.phase],
        ["session_id", session.session_id],
        ["response_language", session.response_language],
        ["focus_slot", session.focus_slot],
        ["last_system_act", session.last_system_act],
        ["pending_clarification_slot", session.pending_clarification_slot],
        ["working_name", working.name],
        ["working_affiliation", working.affiliation],
        ["working_purpose", working.purpose],
        ["committed_name", committed.name],
        ["committed_affiliation", committed.affiliation],
        ["committed_purpose", committed.purpose],
        ["chat_delivery_state", session.chat_delivery_state],
        ["discord_thread_id", session.discord_thread_id],
        ["latest_applied_turn", session.latest_applied_turn],
        ["version", session.version]
      ];
      document.getElementById("session-grid").innerHTML = entries.map(([k, v]) => `
        <div class="kv">
          <div class="k">${escapeHtml(k)}</div>
          <div class="v">${escapeHtml(formatValue(v))}</div>
        </div>
      `).join("");
    }

    function renderConfig(snapshot) {
      const config = snapshot.config || {};
      const topics = config.topics || {};
      const entries = [
        ["profile_name", config.profile_name],
        ["asr_profile", config.asr_profile],
        ["llm_profile", config.llm_profile],
        ["tts_profile", config.tts_profile],
        ["llm_provider", config.llm_provider],
        ["audio_backend", config.audio_backend],
        ["alsa_device", config.alsa_device],
        ["playback_enabled", config.playback_enabled],
        ["playback_device", config.playback_device],
        ["session_state_topic", topics.session_state],
        ["events_topic", topics.events],
        ["trace_topic", topics.conversation_trace],
        ["llm_status_topic", topics.llm_status],
        ["chat_status_topic", topics.chat_status]
      ];
      document.getElementById("config-grid").innerHTML = entries.map(([k, v]) => `
        <div class="kv">
          <div class="k">${escapeHtml(k)}</div>
          <div class="v">${escapeHtml(formatValue(v))}</div>
        </div>
      `).join("");
    }

    function renderEvents(snapshot) {
      const items = snapshot.events || [];
      if (!items.length) {
        document.getElementById("event-list").innerHTML = `<div class="empty">まだパイプラインイベントはありません。</div>`;
        return;
      }
      const html = items.slice().reverse().map((item) => `
        <article class="event-row ${escapeHtml(item.severity || "info")}">
          <div class="event-head">
            <div class="event-title">${escapeHtml(item.command_type_label)} / ${escapeHtml(item.status_label)}</div>
            <div class="bubble-role">${escapeHtml(formatStamp(item.timestamp))}</div>
          </div>
          <div class="meta-list">
            <div>turn=${escapeHtml(formatValue(item.turn_seq))} session=${escapeHtml(formatValue(item.session_id))}</div>
            <div>reason=${escapeHtml(item.reason_label)} command_id=${escapeHtml(item.command_id)}</div>
            <div>${escapeHtml(item.detail || "")}</div>
          </div>
          <details>
            <summary>details</summary>
            <pre>${escapeHtml(formatJson(item))}</pre>
          </details>
        </article>
      `).join("");
      document.getElementById("event-list").innerHTML = html;
    }

    function render(snapshot) {
      if (!snapshot) return;
      heroStatus(snapshot);
      renderStatusGrid(snapshot);
      renderConversation(snapshot);
      renderSession(snapshot);
      renderConfig(snapshot);
      renderEvents(snapshot);
    }

    async function loadInitial() {
      const response = await fetch("/api/snapshot", { cache: "no-store" });
      state.snapshot = await response.json();
      render(state.snapshot);
    }

    function connectStream() {
      if (state.stream) state.stream.close();
      const stream = new EventSource("/api/stream");
      state.stream = stream;
      stream.onmessage = (event) => {
        state.snapshot = JSON.parse(event.data);
        render(state.snapshot);
      };
      stream.onerror = () => {
        stream.close();
        setTimeout(connectStream, 1500);
      };
    }

    loadInitial().then(connectStream).catch((error) => {
      document.getElementById("conversation-list").innerHTML = `<div class="empty">${escapeHtml(String(error))}</div>`;
    });
  </script>
</body>
</html>
"""


def _clean_text(value: str) -> str | None:
    text = str(value or '').strip()
    return text or None


def _event_command_type_label(value: int) -> str:
    return {
        int(ExecutionCommand.COMMAND_UNKNOWN): 'pipeline',
        int(ExecutionCommand.COMMAND_TTS): 'tts',
        int(ExecutionCommand.COMMAND_DISCORD_CREATE): 'discord_create',
        int(ExecutionCommand.COMMAND_DISCORD_SEND): 'discord_send',
    }.get(int(value), f'unknown:{int(value)}')


def _event_status_label(value: int) -> str:
    return {
        int(ExecutionEvent.STATUS_STARTED): 'started',
        int(ExecutionEvent.STATUS_SUCCEEDED): 'succeeded',
        int(ExecutionEvent.STATUS_FAILED): 'failed',
        int(ExecutionEvent.STATUS_CANCELED): 'canceled',
    }.get(int(value), f'unknown:{int(value)}')


def _event_reason_label(value: int) -> str:
    return {
        int(ExecutionEvent.REASON_NONE): 'none',
        int(ExecutionEvent.REASON_STALE): 'stale',
        int(ExecutionEvent.REASON_VALIDATION_FAILED): 'validation_failed',
        int(ExecutionEvent.REASON_SERVICE_UNAVAILABLE): 'service_unavailable',
        int(ExecutionEvent.REASON_TIMEOUT): 'timeout',
        int(ExecutionEvent.REASON_INTERNAL_ERROR): 'internal_error',
        int(ExecutionEvent.REASON_REPLACED): 'replaced',
    }.get(int(value), f'unknown:{int(value)}')


def _event_severity(status: int, reason_code: int) -> str:
    if int(status) == ExecutionEvent.STATUS_FAILED:
        return 'error'
    if int(status) == ExecutionEvent.STATUS_CANCELED or int(reason_code) in {
        ExecutionEvent.REASON_TIMEOUT,
        ExecutionEvent.REASON_REPLACED,
        ExecutionEvent.REASON_STALE,
    }:
        return 'warn'
    if int(status) == ExecutionEvent.STATUS_SUCCEEDED:
        return 'success'
    return 'info'


def execution_event_to_dict(msg: ExecutionEvent) -> dict[str, Any]:
    status = int(msg.status)
    reason_code = int(msg.reason_code)
    return {
        'timestamp': ros_time_to_iso8601(msg.timestamp),
        'command_id': msg.command_id,
        'command_type': int(msg.command_type),
        'command_type_label': _event_command_type_label(msg.command_type),
        'session_id': msg.session_id,
        'turn_seq': int(msg.turn_seq),
        'status': status,
        'status_label': _event_status_label(status),
        'reason_code': reason_code,
        'reason_label': _event_reason_label(reason_code),
        'detail': msg.detail,
        'severity': _event_severity(status, reason_code),
    }


def session_state_to_dict(msg: SessionStateV2) -> dict[str, Any]:
    working_info = {
        'name': _clean_text(msg.working_info.name),
        'affiliation': _clean_text(msg.working_info.affiliation),
        'purpose': _clean_text(msg.working_info.purpose),
    }
    committed_info = {
        'name': _clean_text(msg.committed_info.name),
        'affiliation': _clean_text(msg.committed_info.affiliation),
        'purpose': _clean_text(msg.committed_info.purpose),
    }
    return {
        'timestamp': ros_time_to_iso8601(msg.timestamp),
        'session_id': msg.session_id,
        'phase': msg.phase,
        'response_language': _clean_text(msg.response_language),
        'working_info': working_info,
        'committed_info': committed_info,
        'visitor_info': working_info,
        'pending_confirmation': committed_info,
        'focus_slot': _clean_text(msg.focus_slot),
        'last_system_act': _clean_text(msg.last_system_act),
        'pending_clarification_slot': _clean_text(msg.pending_clarification_slot),
        'working_provenance': [
            {
                'slot': _clean_text(item.slot),
                'source_turn_seq': int(item.source_turn_seq),
                'grounded_text': _clean_text(item.grounded_text),
                'confidence': round(float(item.confidence), 3),
                'updated_at': _clean_text(item.updated_at),
            }
            for item in msg.working_provenance
        ],
        'chat_outbox': [
            {
                'cursor': int(item.cursor),
                'item_id': _clean_text(item.item_id),
                'turn_seq': int(item.turn_seq),
                'event_type': _clean_text(item.event_type),
                'thread_id': _clean_text(item.thread_id),
                'title': _clean_text(item.title),
                'text': _clean_text(item.text),
                'attempt_count': int(item.attempt_count),
                'status': _clean_text(item.status),
            }
            for item in msg.chat_outbox
        ],
        'chat_delivery_state': _clean_text(msg.chat_delivery_state),
        'discord_thread_id': _clean_text(msg.discord_thread_id),
        'discord_channel_id': _clean_text(msg.discord_channel_id),
        'latest_applied_turn': int(msg.latest_applied_turn),
        'version': int(msg.version),
    }


def llm_status_to_dict(msg: LlmStatus) -> dict[str, Any]:
    label = {
        int(LlmStatus.UNCONFIGURED): 'unconfigured',
        int(LlmStatus.STARTING): 'starting',
        int(LlmStatus.READY): 'ready',
        int(LlmStatus.BUSY): 'busy',
        int(LlmStatus.ERROR): 'error',
        int(LlmStatus.STOPPING): 'stopping',
    }.get(int(msg.status), 'unknown')
    return {
        'ready': int(msg.status) == int(LlmStatus.READY),
        'label': label,
        'message': msg.message,
        'severity': 'error' if int(msg.status) == int(LlmStatus.ERROR) else ('warn' if label in {'starting', 'busy', 'stopping'} else 'info'),
        'meta': {
            'model': _clean_text(msg.model_name),
            'version': _clean_text(msg.vllm_version),
        },
    }


def chat_status_to_dict(msg: ChatBridgeStatus) -> dict[str, Any]:
    label = {
        int(ChatBridgeStatus.STOPPED): 'stopped',
        int(ChatBridgeStatus.STARTING): 'starting',
        int(ChatBridgeStatus.READY): 'ready',
        int(ChatBridgeStatus.DEGRADED): 'degraded',
        int(ChatBridgeStatus.ERROR): 'error',
    }.get(int(msg.status), 'unknown')
    return {
        'ready': int(msg.status) == int(ChatBridgeStatus.READY),
        'label': label,
        'message': msg.message,
        'severity': 'error' if int(msg.status) == int(ChatBridgeStatus.ERROR) else ('warn' if label in {'starting', 'degraded'} else 'info'),
        'meta': {
            'gateway_connected': bool(msg.gateway_connected),
            'sidecar_reachable': bool(msg.sidecar_reachable),
            'adapters': ', '.join(msg.active_adapters) if msg.active_adapters else None,
            'state_backend': _clean_text(msg.state_backend),
        },
    }


class DashboardStore:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        conversation_limit: int = 50,
        event_limit: int = 200,
    ) -> None:
        self._lock = threading.RLock()
        self._conversation: deque[dict[str, Any]] = deque(maxlen=conversation_limit)
        self._events: deque[dict[str, Any]] = deque(maxlen=event_limit)
        self._session_state: dict[str, Any] = {}
        self._config = deepcopy(config)
        self._subscribers: list[queue.Queue[str]] = []
        self._version = 0
        self._last_fingerprints: dict[str, str] = {}
        self._system = {
            'backends': {
                'asr': {'ready': False, 'label': 'waiting', 'message': 'waiting for ASR', 'severity': 'warn', 'meta': {}},
                'llm': {'ready': False, 'label': 'waiting', 'message': 'waiting for LLM', 'severity': 'warn', 'meta': {}},
                'tts': {'ready': False, 'label': 'waiting', 'message': 'waiting for TTS', 'severity': 'warn', 'meta': {}},
                'chat': {'ready': False, 'label': 'waiting', 'message': 'waiting for chat bridge', 'severity': 'warn', 'meta': {}},
            }
        }

    def subscribe(self) -> queue.Queue[str]:
        subscriber: queue.Queue[str] = queue.Queue(maxsize=4)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[str]) -> None:
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item is not subscriber]

    def record_conversation(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conversation.append(payload)
            self._broadcast_locked()

    def record_event(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(payload)
            if payload.get('command_type_label') == 'tts':
                self._system['backends']['tts']['message'] = str(payload.get('detail') or 'tts event observed')
                self._system['backends']['tts']['severity'] = str(payload.get('severity') or 'info')
            self._broadcast_locked()

    def update_session_state(self, payload: dict[str, Any]) -> None:
        self._update_if_changed('session_state', payload, assign=lambda value: self._assign_session_state(value))

    def update_backend_status(self, key: str, payload: dict[str, Any]) -> None:
        self._update_if_changed(
            f'backend:{key}',
            payload,
            assign=lambda value: self._system['backends'].__setitem__(key, deepcopy(value)),
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _assign_session_state(self, value: dict[str, Any]) -> None:
        self._session_state = deepcopy(value)

    def _update_if_changed(
        self,
        fingerprint_key: str,
        payload: dict[str, Any],
        *,
        assign,
    ) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if self._last_fingerprints.get(fingerprint_key) == serialized:
                return
            self._last_fingerprints[fingerprint_key] = serialized
            assign(payload)
            self._broadcast_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        backends = deepcopy(self._system['backends'])
        ready = all(bool(item.get('ready')) for item in backends.values())
        return {
            'version': self._version,
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'ready': ready,
            'config': deepcopy(self._config),
            'system': {'backends': backends},
            'session_state': deepcopy(self._session_state),
            'conversation': list(self._conversation),
            'events': list(self._events),
        }

    def _broadcast_locked(self) -> None:
        self._version += 1
        payload = json.dumps(self._snapshot_locked(), ensure_ascii=False)
        active: list[queue.Queue[str]] = []
        for subscriber in self._subscribers:
            try:
                if subscriber.full():
                    subscriber.get_nowait()
                subscriber.put_nowait(payload)
                active.append(subscriber)
            except queue.Full:
                continue
        self._subscribers = active


class DashboardServer:
    def __init__(self, *, store: DashboardStore, host: str, port: int) -> None:
        self._store = store
        self._host = host
        self._port = int(port)
        self._server = ThreadingHTTPServer((self._host, self._port), self._make_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _make_handler(self):
        store = self._store

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == '/':
                    body = HTML_PAGE.encode('utf-8')
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == '/healthz':
                    self._send_json({'ok': True, 'version': store.snapshot().get('version', 0)})
                    return

                if self.path == '/api/snapshot':
                    self._send_json(store.snapshot())
                    return

                if self.path == '/api/stream':
                    self._serve_sse()
                    return

                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                del format, args

            def _send_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_sse(self) -> None:
                subscriber = store.subscribe()
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()

                try:
                    initial = json.dumps(store.snapshot(), ensure_ascii=False)
                    self.wfile.write(f'data: {initial}\n\n'.encode('utf-8'))
                    self.wfile.flush()
                    while True:
                        try:
                            payload = subscriber.get(timeout=15.0)
                            self.wfile.write(f'data: {payload}\n\n'.encode('utf-8'))
                        except queue.Empty:
                            self.wfile.write(b': keepalive\n\n')
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    return
                finally:
                    store.unsubscribe(subscriber)

        return Handler


class ReceptionDemoDashboardNode(Node):
    def __init__(self) -> None:
        super().__init__('reception_demo_dashboard')
        self._declare_parameters()
        self._load_parameters()

        config = {
            'profile_name': self._profile_name,
            'asr_profile': self._asr_profile,
            'llm_profile': self._llm_profile,
            'tts_profile': self._tts_profile,
            'llm_provider': self._llm_provider,
            'audio_backend': self._audio_backend,
            'alsa_device': self._alsa_device,
            'playback_enabled': self._playback_enabled,
            'playback_device': self._playback_device,
            'topics': {
                'session_state': self._session_state_topic,
                'events': self._execution_event_topic,
                'conversation_trace': self._conversation_trace_topic,
                'llm_status': self._llm_status_topic,
                'chat_status': self._chat_status_topic,
            },
        }
        self._store = DashboardStore(config=config)
        self._asr_status_client = self.create_client(AsrGetStatus, '/asr/get_status')
        self._tts_client = ActionClient(self, Speak, self._tts_action_name)
        self._asr_status_inflight = False

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            ConversationTrace,
            self._conversation_trace_topic,
            self._on_conversation_trace,
            50,
        )
        self.create_subscription(
            SessionStateV2,
            self._session_state_topic,
            self._on_session_state,
            30,
        )
        self.create_subscription(
            ExecutionEvent,
            self._execution_event_topic,
            self._on_execution_event,
            100,
        )
        self.create_subscription(
            LlmStatus,
            self._llm_status_topic,
            self._on_llm_status,
            status_qos,
        )
        self.create_subscription(
            ChatBridgeStatus,
            self._chat_status_topic,
            self._on_chat_status,
            status_qos,
        )

        self._status_timer = self.create_timer(1.0, self._on_status_timer)
        self._server = DashboardServer(store=self._store, host=self._host, port=self._port)
        self._server.start()
        self.get_logger().info(f'Reception demo dashboard serving http://{self._host}:{self._port}')

    def destroy_node(self) -> bool:
        self._server.stop()
        return super().destroy_node()

    def _declare_parameters(self) -> None:
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 8090)
        self.declare_parameter('profile_name', '')
        self.declare_parameter('asr_profile', 'qwen3_asr_0_6b_cpu')
        self.declare_parameter('llm_profile', 'qwen35_4b_text')
        self.declare_parameter('tts_profile', 'qwen3_tts_gpu')
        self.declare_parameter('llm_provider', 'vllm')
        self.declare_parameter('audio_backend', 'alsa_arecord')
        self.declare_parameter('alsa_device', 'default')
        self.declare_parameter('playback_enabled', False)
        self.declare_parameter('playback_device', '')
        self.declare_parameter('tts.action_name', '/tts/speak')
        self.declare_parameter('session.state_topic', '/reception/session_state')
        self.declare_parameter('execution.event_topic', '/reception/events')
        self.declare_parameter('conversation.trace_topic', '/reception/conversation_trace')
        self.declare_parameter('llm.status_topic', '/llm/status')
        self.declare_parameter('chat.status_topic', '/chat_bridge/status')

    def _load_parameters(self) -> None:
        self._host = str(self.get_parameter('host').value)
        self._port = int(self.get_parameter('port').value)
        self._profile_name = str(self.get_parameter('profile_name').value)
        self._asr_profile = str(self.get_parameter('asr_profile').value)
        self._llm_profile = str(self.get_parameter('llm_profile').value)
        self._tts_profile = str(self.get_parameter('tts_profile').value)
        self._llm_provider = str(self.get_parameter('llm_provider').value)
        self._audio_backend = str(self.get_parameter('audio_backend').value)
        self._alsa_device = str(self.get_parameter('alsa_device').value)
        self._playback_enabled = bool(self.get_parameter('playback_enabled').value)
        self._playback_device = str(self.get_parameter('playback_device').value)
        self._tts_action_name = str(self.get_parameter('tts.action_name').value)
        self._session_state_topic = str(self.get_parameter('session.state_topic').value)
        self._execution_event_topic = str(self.get_parameter('execution.event_topic').value)
        self._conversation_trace_topic = str(self.get_parameter('conversation.trace_topic').value)
        self._llm_status_topic = str(self.get_parameter('llm.status_topic').value)
        self._chat_status_topic = str(self.get_parameter('chat.status_topic').value)

    def _on_conversation_trace(self, msg: ConversationTrace) -> None:
        self._store.record_conversation(conversation_trace_to_dict(msg))

    def _on_session_state(self, msg: SessionStateV2) -> None:
        self._store.update_session_state(session_state_to_dict(msg))

    def _on_execution_event(self, msg: ExecutionEvent) -> None:
        self._store.record_event(execution_event_to_dict(msg))

    def _on_llm_status(self, msg: LlmStatus) -> None:
        self._store.update_backend_status('llm', llm_status_to_dict(msg))

    def _on_chat_status(self, msg: ChatBridgeStatus) -> None:
        self._store.update_backend_status('chat', chat_status_to_dict(msg))

    def _on_status_timer(self) -> None:
        asr_ready = (
            self.count_publishers('/asr/utterances') > 0
            and self.count_publishers('/asr/speech_events') > 0
            and self._asr_status_client.service_is_ready()
        )
        self._store.update_backend_status(
            'asr',
            {
                'ready': asr_ready,
                'label': 'ready' if asr_ready else 'waiting',
                'message': 'ASR topics and status service available' if asr_ready else 'Waiting for /asr/utterances, /asr/speech_events, /asr/get_status',
                'severity': 'info' if asr_ready else 'warn',
                'meta': {
                    'utterance_publishers': self.count_publishers('/asr/utterances'),
                    'speech_event_publishers': self.count_publishers('/asr/speech_events'),
                },
            },
        )

        tts_ready = self._tts_client.server_is_ready()
        self._store.update_backend_status(
            'tts',
            {
                'ready': tts_ready,
                'label': 'ready' if tts_ready else 'waiting',
                'message': 'TTS action server is ready' if tts_ready else f'Waiting for {self._tts_action_name}',
                'severity': 'info' if tts_ready else 'warn',
                'meta': {
                    'action_name': self._tts_action_name,
                },
            },
        )

        if self._asr_status_client.service_is_ready() and not self._asr_status_inflight:
            self._asr_status_inflight = True
            future = self._asr_status_client.call_async(AsrGetStatus.Request())
            future.add_done_callback(self._on_asr_status_response)

    def _on_asr_status_response(self, future) -> None:  # noqa: ANN001
        self._asr_status_inflight = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._store.update_backend_status(
                'asr',
                {
                    'ready': False,
                    'label': 'error',
                    'message': f'ASR status poll failed: {exc}',
                    'severity': 'error',
                    'meta': {},
                },
            )
            return

        last_error = _clean_text(response.last_error)
        publishers_ready = self.count_publishers('/asr/utterances') > 0 and self.count_publishers('/asr/speech_events') > 0
        ready = publishers_ready and last_error is None
        self._store.update_backend_status(
            'asr',
            {
                'ready': ready,
                'label': 'ready' if ready else ('error' if last_error else 'waiting'),
                'message': last_error or 'ASR running normally',
                'severity': 'error' if last_error else ('info' if ready else 'warn'),
                'meta': {
                    'backend': _clean_text(response.backend),
                    'device': _clean_text(response.device),
                    'frames_received': int(response.frames_received),
                    'frames_dropped': int(response.frames_dropped),
                    'rtf_ema': round(float(response.rtf_ema), 3),
                },
            },
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ReceptionDemoDashboardNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
