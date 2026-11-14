"""Transactional SQLite store for Gee Clipper users, quotas, jobs and favourites."""
import json
import os
import sqlite3
import uuid
from datetime import date, datetime, timezone
import config as C


def _connect():
    os.makedirs(os.path.dirname(C.DB_PATH) or ".", exist_ok=True)
    db = sqlite3.connect(C.DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=30000")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init():
    with _connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users(user_id INTEGER PRIMARY KEY, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS usage(user_id INTEGER NOT NULL, day TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(user_id,day));
        CREATE TABLE IF NOT EXISTS blocked_users(user_id INTEGER PRIMARY KEY, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, title TEXT NOT NULL DEFAULT 'Processing', url TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'processing', clips INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, finished_at TEXT);
        CREATE TABLE IF NOT EXISTS clips(clip_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, user_id INTEGER NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL, score REAL NOT NULL DEFAULT 0, hook TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '', hashtags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS favorites(user_id INTEGER NOT NULL, clip_id TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(user_id,clip_id), FOREIGN KEY(clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_id,created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_clips_user_created ON clips(user_id,created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_favorites_user_created ON favorites(user_id,created_at DESC);
        """)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_user(user_id):
    now = _now()
    with _connect() as db:
        db.execute("""INSERT INTO users VALUES(?,?,?) ON CONFLICT(user_id)
                      DO UPDATE SET last_seen=excluded.last_seen""", (int(user_id), now, now))


def is_blocked(user_id):
    with _connect() as db:
        return db.execute("SELECT 1 FROM blocked_users WHERE user_id=?", (int(user_id),)).fetchone() is not None


def set_blocked(user_id, blocked):
    with _connect() as db:
        if blocked:
            db.execute("INSERT OR IGNORE INTO blocked_users VALUES(?,?)", (int(user_id), _now()))
        else:
            db.execute("DELETE FROM blocked_users WHERE user_id=?", (int(user_id),))


def consume_quota(user_id, limit, day=None):
    if int(limit) <= 0:
        return True
    today = day or date.today().isoformat()
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT count FROM usage WHERE user_id=? AND day=?", (int(user_id), today)).fetchone()
        count = int(row["count"]) if row else 0
        if count >= int(limit):
            return False
        db.execute("""INSERT INTO usage(user_id,day,count) VALUES(?,?,1)
                      ON CONFLICT(user_id,day) DO UPDATE SET count=count+1""",
                   (int(user_id), today))
        return True


def create_job(user_id, url):
    job_id = uuid.uuid4().hex[:12]
    with _connect() as db:
        db.execute("INSERT INTO jobs(job_id,user_id,url,created_at) VALUES(?,?,?,?)",
                   (job_id, int(user_id), str(url), _now()))
    return job_id


def update_job(job_id, *, title=None, status=None, clips=None):
    fields, values = [], []
    if title is not None:
        fields.append("title=?"); values.append(str(title)[:160])
    if status is not None:
        fields.append("status=?"); values.append(str(status)[:32])
    if clips is not None:
        fields.append("clips=?"); values.append(max(0, int(clips)))
    if status in ("completed", "failed", "cancelled", "no clips", "skipped"):
        fields.append("finished_at=?"); values.append(_now())
    if fields:
        values.append(job_id)
        with _connect() as db:
            db.execute(f"UPDATE jobs SET {','.join(fields)} WHERE job_id=?", values)


def record_clip(user_id, job_id, url, clip):
    clip_id = uuid.uuid4().hex[:12]
    with _connect() as db:
        db.execute("""INSERT INTO clips VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (clip_id, job_id, int(user_id), str(clip.get("title","Clip"))[:160], str(url),
             float(clip.get("start",0)), float(clip.get("end",0)), float(clip.get("score",0)),
             str(clip.get("hook",""))[:100], str(clip.get("description",""))[:500],
             json.dumps(clip.get("hashtags",[])[:10]), _now()))
    return clip_id


def favorite(user_id, clip_id):
    with _connect() as db:
        exists = db.execute("SELECT 1 FROM clips WHERE clip_id=? AND user_id=?",
                            (str(clip_id), int(user_id))).fetchone()
        if not exists:
            return False
        db.execute("INSERT OR IGNORE INTO favorites VALUES(?,?,?)",
                   (int(user_id), str(clip_id), _now()))
        return True


def unfavorite(user_id, clip_id):
    with _connect() as db:
        cur = db.execute("DELETE FROM favorites WHERE user_id=? AND clip_id=?",
                         (int(user_id), str(clip_id)))
        return cur.rowcount > 0


def list_favorites(user_id, limit=20):
    with _connect() as db:
        rows = db.execute("""SELECT c.* FROM favorites f JOIN clips c ON c.clip_id=f.clip_id
                             WHERE f.user_id=? ORDER BY f.created_at DESC LIMIT ?""",
                          (int(user_id), max(1,min(int(limit),50)))).fetchall()
    return [dict(r) for r in rows]


def history(user_id, limit=10):
    with _connect() as db:
        rows = db.execute("SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                          (int(user_id), max(1,min(int(limit),30)))).fetchall()
    return [dict(r) for r in rows]


def user_analytics(user_id):
    today = date.today().isoformat()
    with _connect() as db:
        jobs = db.execute("SELECT COUNT(*) n FROM jobs WHERE user_id=?", (int(user_id),)).fetchone()["n"]
        clips = db.execute("SELECT COUNT(*) n FROM clips WHERE user_id=?", (int(user_id),)).fetchone()["n"]
        favs = db.execute("SELECT COUNT(*) n FROM favorites WHERE user_id=?", (int(user_id),)).fetchone()["n"]
        used = db.execute("SELECT count FROM usage WHERE user_id=? AND day=?", (int(user_id),today)).fetchone()
    return {"jobs":jobs,"clips":clips,"favorites":favs,"used_today":used["count"] if used else 0}


def global_analytics():
    today = date.today().isoformat()
    with _connect() as db:
        return {
            "users":db.execute("SELECT COUNT(*) n FROM users").fetchone()["n"],
            "jobs":db.execute("SELECT COUNT(*) n FROM jobs").fetchone()["n"],
            "clips":db.execute("SELECT COUNT(*) n FROM clips").fetchone()["n"],
            "today_jobs":db.execute("SELECT COUNT(*) n FROM jobs WHERE substr(created_at,1,10)=?",(today,)).fetchone()["n"],
            "active_jobs":db.execute("SELECT COUNT(*) n FROM jobs WHERE status IN ('processing','queued')").fetchone()["n"],
            "blocked":db.execute("SELECT COUNT(*) n FROM blocked_users").fetchone()["n"],
        }
