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


def test_turns_of_empty_history_is_empty() -> None:
    assert History().turns() == ()


def test_turns_groups_messages_on_user_boundaries() -> None:
    history = History(
        [
            Message.user("a"),
            Message.assistant("a1"),
            Message.tool(tool_call_id="t1", content="result", name="echo"),
            Message.user("b"),
            Message.assistant("b1"),
        ]
    )

    turns = history.turns()

    assert len(turns) == 2
    assert [m.content for m in turns[0]] == ["a", "a1", "result"]  # call + result stay together
    assert [m.content for m in turns[1]] == ["b", "b1"]


def test_turns_keeps_a_leading_non_user_run_as_one_group() -> None:
    history = History([Message.assistant("recap"), Message.user("go on")])

    turns = history.turns()

    assert [m.content for m in turns[0]] == ["recap"]
    assert [m.content for m in turns[1]] == ["go on"]
