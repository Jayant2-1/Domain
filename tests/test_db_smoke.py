"""Quick smoke test for SQLite database layer."""
import asyncio
import os
from app.database import (
    init_db, close_db,
    get_or_create_user, get_or_create_topic,
    get_skill_rating, upsert_skill_rating,
    insert_interaction, update_interaction_feedback,
    get_all_skill_ratings, get_positive_interactions,
)

async def main():
    db_file = "data/test_smoke.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    await init_db(db_file)

    uid = await get_or_create_user("alice")
    print(f"[OK] User created: id={uid}")

    uid2 = await get_or_create_user("alice")
    assert uid == uid2, "Idempotent user creation failed"
    print("[OK] Idempotent user creation")

    tid = await get_or_create_topic("arrays")
    print(f"[OK] Topic created: id={tid}")

    sr = await get_skill_rating(uid, tid)
    assert sr["rating"] == 1000.0
    assert sr["matches"] == 0
    print(f"[OK] Default rating: {sr}")

    await upsert_skill_rating(uid, tid, 1020.0, 1)
    sr = await get_skill_rating(uid, tid)
    assert abs(sr["rating"] - 1020.0) < 0.1
    assert sr["matches"] == 1
    print(f"[OK] Upserted rating: {sr}")

    iid = await insert_interaction(
        uid, tid, "What is an array?", "Context here", "An array is...", 1000.0, 1000.0
    )
    print(f"[OK] Interaction inserted: id={iid}")

    await update_interaction_feedback(iid, 1, 1020.0)
    print("[OK] Feedback updated")

    positives = await get_positive_interactions()
    assert len(positives) >= 1
    print(f"[OK] Positive interactions: {len(positives)}")

    all_sr = await get_all_skill_ratings(uid)
    assert len(all_sr) == 1
    print(f"[OK] All skill ratings: {all_sr}")

    await close_db()
    os.remove(db_file)
    print("\n=== ALL DATABASE TESTS PASSED ===")

if __name__ == "__main__":
    asyncio.run(main())
