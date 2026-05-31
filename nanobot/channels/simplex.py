"""SimpleX Chat channel implementation using the CLI WebSocket bot API."""

from __future__ import annotations

import asyncio
import json
from itertools import count
from typing import Any

import websockets
from pydantic import Field
from websockets.asyncio.client import ClientConnection

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base


class SimplexConfig(Base):
    """SimpleX Chat CLI WebSocket channel configuration."""

    enabled: bool = False
    websocket_url: str = "ws://localhost:5225"
    allow_from: list[str] = Field(default_factory=list)  # Allowed contact IDs
    group_enabled: bool = False
    group_allow_from: list[str] = Field(default_factory=list)  # Allowed group IDs
    group_require_mention: bool = True
    command_timeout_seconds: float = Field(default=30.0, gt=0)


class SimplexChannel(BaseChannel):
    """SimpleX Chat channel backed by a locally running CLI WebSocket server."""

    name = "simplex"
    display_name = "SimpleX Chat"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return SimplexConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = SimplexConfig.model_validate(config)
        super().__init__(config, bus)
        self._ws: ClientConnection | None = None
        self._corr_ids = count(1)
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def start(self) -> None:
        """Connect to SimpleX Chat CLI and process events until stopped."""
        self._running = True
        reconnect_delay_s = 1.0
        max_reconnect_delay_s = 30.0

        while self._running:
            try:
                self.logger.info("Connecting to SimpleX Chat CLI at {}...", self.config.websocket_url)
                async with websockets.connect(self.config.websocket_url) as ws:
                    self._ws = ws
                    reconnect_delay_s = 1.0
                    self.logger.info("Connected to SimpleX Chat CLI")
                    async for raw_message in ws:
                        await self._handle_websocket_message(raw_message)
                if self._running:
                    raise ConnectionError("SimpleX Chat CLI WebSocket closed unexpectedly")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("SimpleX Chat channel error: {}", e)
            finally:
                self._ws = None
                self._fail_pending(ConnectionError("SimpleX Chat CLI WebSocket disconnected"))

            if self._running:
                self.logger.info(
                    "Reconnecting to SimpleX Chat CLI in {:.0f} seconds...", reconnect_delay_s
                )
                await asyncio.sleep(reconnect_delay_s)
                reconnect_delay_s = min(reconnect_delay_s * 2, max_reconnect_delay_s)

    async def stop(self) -> None:
        """Stop the channel and close its WebSocket connection."""
        self._running = False
        if self._ws is not None:
            await self._ws.close()
        self._fail_pending(ConnectionError("SimpleX Chat channel stopped"))

    async def send(self, msg: OutboundMessage) -> None:
        """Send a text message and optional file attachments through SimpleX Chat."""
        if not msg.content and not msg.media:
            return

        composed_messages: list[dict[str, Any]] = []
        if msg.content or not msg.media:
            composed_messages.append(self._composed_message(msg.content))
        for file_path in msg.media:
            composed_messages.append(self._composed_message("", file_path=file_path))

        cmd = f"/_send {self._normalize_chat_ref(msg.chat_id)} json {json.dumps(composed_messages)}"
        resp = await self._send_command(cmd)
        if resp.get("type") == "chatCmdError":
            raise RuntimeError(f"SimpleX Chat send failed: {resp.get('chatError', resp)}")

    async def _send_command(self, cmd: str) -> dict[str, Any]:
        """Send one CLI command and await its correlated response."""
        if self._ws is None:
            raise ConnectionError("SimpleX Chat CLI is not connected")

        corr_id = str(next(self._corr_ids))
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[corr_id] = future
        try:
            await self._ws.send(json.dumps({"corrId": corr_id, "cmd": cmd}))
            return await asyncio.wait_for(future, timeout=self.config.command_timeout_seconds)
        finally:
            self._pending.pop(corr_id, None)

    async def _handle_websocket_message(self, raw_message: str | bytes) -> None:
        """Resolve command replies or forward supported CLI events to nanobot."""
        try:
            message = json.loads(raw_message)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            self.logger.warning("Ignoring invalid SimpleX Chat WebSocket message")
            return
        if not isinstance(message, dict):
            return

        resp = message.get("resp")
        if not isinstance(resp, dict):
            return

        corr_id = message.get("corrId")
        future = self._pending.get(str(corr_id)) if corr_id is not None else None
        if future is not None:
            if not future.done():
                future.set_result(resp)
            return

        if resp.get("type") == "newChatItems":
            for chat_item in resp.get("chatItems", []):
                if isinstance(chat_item, dict):
                    await self._handle_chat_item(chat_item)

    async def _handle_chat_item(self, item: dict[str, Any]) -> None:
        """Forward one received direct or group text message to the shared bus."""
        chat_info = item.get("chatInfo")
        chat_item = item.get("chatItem")
        if not isinstance(chat_info, dict) or not isinstance(chat_item, dict):
            return

        direction = chat_item.get("chatDir")
        content = chat_item.get("content")
        if not isinstance(direction, dict) or not isinstance(content, dict):
            return
        if content.get("type") != "rcvMsgContent":
            return
        msg_content = content.get("msgContent")
        if not isinstance(msg_content, dict):
            return
        text = msg_content.get("text")
        if not isinstance(text, str):
            return

        meta = chat_item.get("meta") if isinstance(chat_item.get("meta"), dict) else {}
        item_id = meta.get("itemId")
        metadata: dict[str, Any] = {"simplex_item_id": item_id} if item_id is not None else {}

        if chat_info.get("type") == "direct" and direction.get("type") == "directRcv":
            contact = chat_info.get("contact")
            if not isinstance(contact, dict) or contact.get("contactId") is None:
                return
            contact_id = str(contact["contactId"])
            metadata["simplex_chat_type"] = "direct"
            await self._handle_message(
                sender_id=contact_id,
                chat_id=f"@{contact_id}",
                content=text,
                metadata=metadata,
                is_dm=True,
            )
            return

        if chat_info.get("type") != "group" or direction.get("type") != "groupRcv":
            return
        if not self.config.group_enabled:
            return

        group_info = chat_info.get("groupInfo")
        group_member = direction.get("groupMember")
        if not isinstance(group_info, dict) or not isinstance(group_member, dict):
            return
        if group_info.get("groupId") is None or group_member.get("groupMemberId") is None:
            return

        group_id = str(group_info["groupId"])
        if group_id not in self.config.group_allow_from and "*" not in self.config.group_allow_from:
            self.logger.warning("Ignoring message from SimpleX group {} not in groupAllowFrom", group_id)
            return
        if self.config.group_require_mention and not bool(meta.get("userMention")):
            return

        member_id = str(group_member["groupMemberId"])
        metadata.update({"simplex_chat_type": "group", "simplex_group_id": group_id})
        await self._handle_message(
            sender_id=member_id,
            chat_id=f"#{group_id}",
            content=text,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_chat_ref(chat_id: str) -> str:
        """Return a SimpleX CLI direct/group chat reference."""
        chat_ref = str(chat_id)
        if chat_ref.startswith(("@", "#")):
            return chat_ref
        return f"@{chat_ref}"

    @staticmethod
    def _composed_message(text: str, *, file_path: str | None = None) -> dict[str, Any]:
        """Build the JSON payload accepted by the CLI ``/_send`` command."""
        message: dict[str, Any] = {
            "msgContent": {"type": "text", "text": text},
            "mentions": {},
        }
        if file_path is not None:
            message["fileSource"] = {"filePath": file_path}
        return message

    def _fail_pending(self, exc: Exception) -> None:
        """Fail all commands waiting for a WebSocket response."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()
