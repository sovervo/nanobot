"""Tests for the SimpleX Chat CLI WebSocket channel."""

from __future__ import annotations

import asyncio
import json

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.simplex import SimplexChannel, SimplexConfig


class _FakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def close(self) -> None:
        self.closed = True


def _channel(**config) -> SimplexChannel:
    return SimplexChannel(SimplexConfig(**config), MessageBus())


def _chat_item(*, chat_info: dict, direction: dict, text: str = "hello") -> dict:
    return {
        "chatInfo": chat_info,
        "chatItem": {
            "chatDir": direction,
            "meta": {"itemId": 42, "userMention": True},
            "content": {"type": "rcvMsgContent", "msgContent": {"type": "text", "text": text}},
        },
    }


def test_default_config_uses_local_cli_websocket():
    assert SimplexChannel.default_config() == {
        "enabled": False,
        "websocketUrl": "ws://localhost:5225",
        "allowFrom": [],
        "groupEnabled": False,
        "groupAllowFrom": [],
        "groupRequireMention": True,
        "commandTimeoutSeconds": 30.0,
    }


@pytest.mark.asyncio
async def test_send_builds_official_send_command_and_resolves_correlation():
    channel = _channel()
    ws = _FakeWebSocket()
    channel._ws = ws  # type: ignore[assignment]

    task = asyncio.create_task(channel.send(OutboundMessage(channel="simplex", chat_id="7", content="hi")))
    await asyncio.sleep(0)
    assert ws.sent == [
        {
            "corrId": "1",
            "cmd": '/_send @7 json [{"msgContent": {"type": "text", "text": "hi"}, "mentions": {}}]',
        }
    ]

    await channel._handle_websocket_message(json.dumps({"corrId": "1", "resp": {"type": "newChatItems"}}))
    await task


@pytest.mark.asyncio
async def test_send_raises_chat_command_error():
    channel = _channel()
    ws = _FakeWebSocket()
    channel._ws = ws  # type: ignore[assignment]

    task = asyncio.create_task(channel.send(OutboundMessage(channel="simplex", chat_id="@7", content="hi")))
    await asyncio.sleep(0)
    await channel._handle_websocket_message(
        json.dumps({"corrId": "1", "resp": {"type": "chatCmdError", "chatError": "no contact"}})
    )
    with pytest.raises(RuntimeError, match="no contact"):
        await task


@pytest.mark.asyncio
async def test_direct_message_is_forwarded_with_pairing_compatible_contact_id(monkeypatch):
    channel = _channel(allow_from=["7"])
    handled: list[dict] = []

    async def capture(**kwargs):
        handled.append(kwargs)

    monkeypatch.setattr(channel, "_handle_message", capture)
    item = _chat_item(chat_info={"type": "direct", "contact": {"contactId": 7}}, direction={"type": "directRcv"})
    await channel._handle_websocket_message(
        json.dumps({"corrId": "", "resp": {"type": "newChatItems", "chatItems": [item]}})
    )

    assert handled == [
        {
            "sender_id": "7",
            "chat_id": "@7",
            "content": "hello",
            "metadata": {"simplex_item_id": 42, "simplex_chat_type": "direct"},
            "is_dm": True,
        }
    ]


@pytest.mark.asyncio
async def test_group_message_requires_enabled_allowed_group_and_mention(monkeypatch):
    channel = _channel(group_enabled=True, group_allow_from=["9"])
    handled: list[dict] = []

    async def capture(**kwargs):
        handled.append(kwargs)

    monkeypatch.setattr(channel, "_handle_message", capture)
    item = _chat_item(
        chat_info={"type": "group", "groupInfo": {"groupId": 9}},
        direction={"type": "groupRcv", "groupMember": {"groupMemberId": 12}},
    )
    await channel._handle_chat_item(item)

    assert handled == [
        {
            "sender_id": "12",
            "chat_id": "#9",
            "content": "hello",
            "metadata": {"simplex_item_id": 42, "simplex_chat_type": "group", "simplex_group_id": "9"},
        }
    ]


@pytest.mark.asyncio
async def test_group_message_without_mention_is_ignored(monkeypatch):
    channel = _channel(group_enabled=True, group_allow_from=["9"])
    handled: list[dict] = []

    async def capture(**kwargs):
        handled.append(kwargs)

    monkeypatch.setattr(channel, "_handle_message", capture)
    item = _chat_item(
        chat_info={"type": "group", "groupInfo": {"groupId": 9}},
        direction={"type": "groupRcv", "groupMember": {"groupMemberId": 12}},
    )
    item["chatItem"]["meta"]["userMention"] = False
    await channel._handle_chat_item(item)
    assert handled == []


@pytest.mark.asyncio
async def test_unknown_events_and_invalid_json_are_ignored():
    channel = _channel()
    await channel._handle_websocket_message("not json")
    await channel._handle_websocket_message(json.dumps({"resp": {"type": "futureEvent", "other": True}}))


@pytest.mark.asyncio
async def test_stop_closes_websocket_and_fails_pending_command():
    channel = _channel()
    ws = _FakeWebSocket()
    channel._ws = ws  # type: ignore[assignment]
    task = asyncio.create_task(channel._send_command("/_show_address 1"))
    await asyncio.sleep(0)

    await channel.stop()

    assert ws.closed is True
    with pytest.raises(ConnectionError, match="stopped"):
        await task
