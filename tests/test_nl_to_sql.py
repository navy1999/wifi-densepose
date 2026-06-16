"""SQL safety validation is security-critical — test it hard."""

import pytest

from wifipose.llm.nl_to_sql import (
    UnsafeQueryError,
    rule_based_sql,
    validate_select,
)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE pose_events",
        "DELETE FROM pose_events",
        "UPDATE pose_events SET action='x'",
        "SELECT * FROM pose_events; DROP TABLE pose_events",
        "SELECT * FROM users",
        "SELECT * FROM pose_events -- comment",
        "INSERT INTO pose_events VALUES (1)",
        "SELECT pg_sleep(10)",
        "TRUNCATE pose_events",
        "SELECT * FROM information_schema.tables",
    ],
)
def test_rejects_unsafe(sql):
    with pytest.raises(UnsafeQueryError):
        validate_select(sql)


def test_allows_select_and_enforces_limit():
    out = validate_select("SELECT action, count(*) FROM pose_events GROUP BY action")
    assert out.lower().startswith("select")
    assert "limit" in out.lower()


def test_keeps_existing_limit():
    out = validate_select("SELECT * FROM pose_events LIMIT 5")
    assert out.lower().count("limit") == 1


def test_allows_cte():
    out = validate_select(
        "WITH x AS (SELECT action FROM pose_events) SELECT * FROM pose_events"
    )
    assert out.lower().startswith("with")


def test_strips_code_fence():
    out = validate_select("```sql\nSELECT * FROM pose_events\n```")
    assert "```" not in out


@pytest.mark.parametrize(
    "question,expect",
    [
        ("how many running events today?", "count(*)"),
        ("count events by action", "group by action"),
        ("average latency last hour", "avg(latency_ms)"),
        ("events per hour", "date_trunc"),
    ],
)
def test_rule_based_intents(question, expect):
    sql = rule_based_sql(question).lower()
    assert expect in sql
    # everything the rule-based generator emits must pass validation
    validate_select(sql)
