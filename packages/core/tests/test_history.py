from __future__ import annotations

from clawde_core.context.history import History
from clawde_core.models import Message


def test_starts_empty() -> None:
    history = History()
    assert len(history) == 0
    assert history.messages == ()


def test_can_seed_from_existing_messages() -> None:
    history = History([Message.system("rules")])
    assert len(history) == 1
    assert history.messages == (Message.system("rules"),)


def test_append_grows_the_conversation() -> None:
    history = History()
    history.append(Message.user("hi"))
    history.append(Message.assistant("hello"))
    assert [m.content for m in history.messages] == ["hi", "hello"]


def test_messages_is_an_immutable_snapshot() -> None:
    history = History([Message.user("a")])
    snapshot = history.messages

    history.append(Message.user("b"))

    assert snapshot == (Message.user("a"),)  # the earlier snapshot is unaffected
    assert history.messages == (Message.user("a"), Message.user("b"))


def test_since_returns_messages_from_an_index() -> None:
    history = History([Message.user("a"), Message.assistant("b"), Message.user("c")])
    assert history.since(1) == (Message.assistant("b"), Message.user("c"))
    assert history.since(3) == ()  # past the end is empty, not an error


def test_truncate_drops_messages_from_an_index() -> None:
    history = History([Message.user("a"), Message.assistant("b"), Message.user("c")])

    history.truncate(1)

    assert history.messages == (Message.user("a"),)


def test_truncate_past_the_end_is_a_no_op() -> None:
    history = History([Message.user("a")])

    history.truncate(5)

    assert history.messages == (Message.user("a"),)
