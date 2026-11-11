"""The shared brain's nightly merge (GitHub Action, free): the last brain + the inbox → a new versioned release.
Old releases are never deleted; a new one is published only when the gate passes.

  python merge.py --work DIR [--seed brain.sqlite] [--dry]

  env: REPO (owner/name), INBOX (the Worker's URL), MERGE_TOKEN, GH_TOKEN (the Action's own), OWNER_INSTALLS (optional,
       comma list of install ids that count as the owner's, besides the owner key the Worker checks)

Rules (PROXIMO-LOTE-LEITURA-CEREBRO.md, B and B4):
  a sound enters when the owner sent it, or 2 installs agree (same kind, vectors cosine >= 0.95);
  a category: the owner's newest vote wins (conf 1); else 2+ votes and 80% agreeing (conf = their share);
  votes are kept as evidence (evidence.sqlite in each release): when taxonomy.json changes, every vote is translated
  by its map and all categories are worked out again; a vote that can't be translated becomes a question;
  the gate: integrity, never fewer sounds than before, every owner vote honoured, and — with the taxonomy unchanged —
  at most 5% of the categories may move in one night.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.request

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS sound (fp INTEGER PRIMARY KEY, kind TEXT, category TEXT, bpm REAL, key TEXT, bconf REAL, est INTEGER, bars REAL, bad INTEGER, size INTEGER);
CREATE TABLE IF NOT EXISTS feat (fp INTEGER PRIMARY KEY, groove BLOB, melody BLOB, ihist BLOB);
CREATE TABLE IF NOT EXISTS tune (fp INTEGER PRIMARY KEY, melody BLOB, ihist BLOB, shape BLOB);
CREATE TABLE IF NOT EXISTS shot (fp INTEGER PRIMARY KEY, pitch REAL, decay REAL, bright REAL);
CREATE TABLE IF NOT EXISTS poly (fp INTEGER PRIMARY KEY, notes INTEGER);
CREATE TABLE IF NOT EXISTS waves (fp INTEGER PRIMARY KEY, data BLOB);
CREATE TABLE IF NOT EXISTS overrides (fp INTEGER PRIMARY KEY, category TEXT, kind TEXT, bpm REAL, at REAL, was_category TEXT, was_kind TEXT, was_bpm REAL, was_est INTEGER);
CREATE TABLE IF NOT EXISTS vec (fp INTEGER PRIMARY KEY, v BLOB);
CREATE TABLE IF NOT EXISTS timbre (fp INTEGER PRIMARY KEY, v BLOB);
CREATE TABLE IF NOT EXISTS info (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS ref (fp INTEGER PRIMARY KEY, grp TEXT, category TEXT, unit TEXT, dur REAL);
CREATE TABLE IF NOT EXISTS cat (fp INTEGER PRIMARY KEY, category TEXT, conf REAL, votes INTEGER);
"""
TABLES = ('sound', 'feat', 'tune', 'shot', 'poly', 'waves', 'overrides', 'vec', 'timbre', 'ref', 'cat')
EVIDENCE = """
CREATE TABLE IF NOT EXISTS votes (k INTEGER, install TEXT, owner INTEGER, cat TEXT, at REAL, PRIMARY KEY (k, install));
CREATE TABLE IF NOT EXISTS sounds (k INTEGER, install TEXT, owner INTEGER, kind TEXT, cat TEXT, bpm REAL, key TEXT, bconf REAL, est INTEGER, bars REAL, vec BLOB, PRIMARY KEY (k, install));
CREATE TABLE IF NOT EXISTS refs (fp TEXT, install TEXT, n INTEGER, vecs BLOB, starts TEXT, win REAL, mood TEXT, PRIMARY KEY (fp, install));
CREATE TABLE IF NOT EXISTS presets (h TEXT, kind INTEGER, install TEXT, owner INTEGER, product TEXT, vendor TEXT, name TEXT, cat TEXT, v BLOB, PRIMARY KEY (h, kind, install));
CREATE TABLE IF NOT EXISTS questions (k INTEGER PRIMARY KEY, was TEXT);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""
SHARD = 25000  # sounds per shard: a friend with a few packs fetches a few MB
AGREE, VOTES, SHARE, MOVE_MAX = 0.95, 2, 0.8, 0.05


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def latest(repo, work):
    """The newest release's assets → work/prev (None when there is none yet)."""
    try:
        tag = json.loads(sh('gh', 'release', 'view', '--repo', repo, '--json', 'tagName'))['tagName']
    except subprocess.CalledProcessError:
        return None, None
    d = os.path.join(work, 'prev')
    os.makedirs(d, exist_ok=True)
    sh('gh', 'release', 'download', tag, '--repo', repo, '--dir', d, '--clobber')
    return tag, json.load(open(os.path.join(d, 'index.json')))


def joined(prev_dir, index, dst, seed=None):
    """The last brain as one file (its shards together), or the owner's seed for the first release."""
    b = sqlite3.connect(dst)
    b.executescript(SCHEMA)
    for p in ([seed] if seed else []) + [os.path.join(prev_dir, s['name']) for s in (index or {}).get('shards', [])]:
        b.execute('ATTACH DATABASE ? AS o', (p,))
        for t in TABLES:
            if b.execute('SELECT 1 FROM o.sqlite_master WHERE name = ?', (t,)).fetchone():
                b.execute(f'INSERT OR IGNORE INTO {t} SELECT * FROM o.{t}')
        b.commit()
        b.execute('DETACH DATABASE o')
    return b


def _items(body):
    """A row's items: a whole push kept as one row (11/11: one row per item spent the free plan's 100 000 writes a day),
    or one item of the rows written before that."""
    b = json.loads(body)
    return b if isinstance(b, list) else [b]


def inbox(url, token, ev, owners):
    """Every unmerged item → the evidence (newer word from the same install replaces its older one).
    → (last id read, how many rows really changed): the cursor alone, or the same items sent again on every launch, are
    never news (09/11: it published every night)."""
    last = int((ev.execute("SELECT v FROM meta WHERE k = 'cursor'").fetchone() or [0])[0])
    top, fresh = last, 0

    def put(table, keys, row):  # → 1 only when the row is new or differs: the same word sent again is not news
        nonlocal fresh
        cols = [c[1] for c in ev.execute(f'PRAGMA table_info({table})')]
        old = ev.execute(f'SELECT * FROM {table} WHERE ' + ' AND '.join(f'{c} = ?' for c in cols[:len(keys)]), keys).fetchone()
        if old != tuple(row):
            ev.execute(f'INSERT OR REPLACE INTO {table} VALUES ({",".join("?" * len(row))})', row)
            fresh += 1
    while True:
        req = urllib.request.Request(f'{url.rstrip("/")}/pull?after={top}', headers={'Authorization': f'Bearer {token}', 'User-Agent': 'sherlock-merge'})
        rows = json.loads(urllib.request.urlopen(req, timeout=120).read())
        if not rows:
            break
        for r, it in ((r, it) for r in rows for it in _items(r['body'])):
            who, own = r['install'], int(r['owner'] or r['install'] in owners)
            if it['t'] == 'vote' and it.get('cat'):
                old = ev.execute('SELECT owner, cat, at FROM votes WHERE k = ? AND install = ?', (it['k'], who)).fetchone()
                at = it.get('at') or r['at']
                if not old or (at >= old[2] and (old[0], old[1]) != (own, it['cat'])):
                    put('votes', (it['k'], who), (it['k'], who, own, it['cat'], at))
            elif it['t'] == 'sound':
                put('sounds', (it['k'], who), (it['k'], who, own, it.get('kind'), it.get('cat'), it.get('bpm'), it.get('key'),
                    it.get('bconf'), it.get('est'), it.get('bars'), base64.b64decode(it['vec']) if it.get('vec') else None))
            elif it['t'] == 'ref':
                put('refs', (it['fp'], who), (it['fp'], who, it.get('n'), base64.b64decode(it['vecs']) if it.get('vecs') else None,
                    json.dumps(it.get('starts')), it.get('win'), json.dumps(it.get('mood'))))
            elif it['t'] == 'preset' and it.get('v'):
                put('presets', (it['h'], it.get('kind'), who), (it['h'], it.get('kind'), who, own, it.get('product'), it.get('vendor'),
                    it.get('name'), it.get('cat'), base64.b64decode(it['v'])))
            top = max(top, r['id'])
        ev.execute("INSERT OR REPLACE INTO meta VALUES ('cursor', ?)", (str(top),))
        ev.commit()
    return top, fresh


def taxonomy(path='taxonomy.json'):
    """{version, map: {old: new}}: a category renamed or folded; one that splits maps to null (asked again)."""
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {'version': 1, 'map': {}}


def tr(cat, tax):
    m = tax.get('map', {})
    return m[cat] if cat in m else cat


def cosine(a, b):
    a, b = np.frombuffer(a, np.float16).astype(np.float32), np.frombuffer(b, np.float16).astype(np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) if a.shape == b.shape else 0.0


def merge(b, ev, tax):
    """Sounds that earned their place, then every category worked out again from the votes. → (added, moved)."""
    added = 0
    for (k,) in ev.execute('SELECT DISTINCT k FROM sounds').fetchall():
        if b.execute('SELECT 1 FROM sound WHERE fp = ?', (k,)).fetchone():
            continue
        rows = ev.execute('SELECT owner, kind, cat, bpm, key, bconf, est, bars, vec FROM sounds WHERE k = ?', (k,)).fetchall()
        own = [r for r in rows if r[0]]
        agree = any(r[1] == rows[0][1] and r[8] and rows[0][8] and cosine(r[8], rows[0][8]) >= AGREE for r in rows[1:])  # a second install hears the same
        ok = own[:1] or (rows[:1] if agree else [])
        if not ok:
            continue
        r = ok[0]
        b.execute('INSERT INTO sound VALUES (?,?,?,?,?,?,?,?,0,NULL)', (k, r[1], tr(r[2], tax), r[3], r[4], r[5], r[6], r[7]))
        if r[8]:
            b.execute('INSERT OR IGNORE INTO vec VALUES (?, ?)', (k, r[8]))
        added += 1
    before = dict(b.execute('SELECT fp, category FROM cat'))
    b.execute('DELETE FROM cat')  # worked out again from all the evidence, with today's taxonomy
    ev.execute('DELETE FROM questions')
    for (k,) in ev.execute('SELECT DISTINCT k FROM votes').fetchall():
        vs = ev.execute('SELECT owner, cat, at FROM votes WHERE k = ? ORDER BY at DESC', (k,)).fetchall()
        own = [v for v in vs if v[0]]
        if own:
            c = tr(own[0][1], tax)
            if c is None:
                ev.execute('INSERT OR REPLACE INTO questions VALUES (?, ?)', (k, own[0][1]))
                continue
            b.execute('INSERT INTO cat VALUES (?, ?, 1.0, ?)', (k, c, len(vs)))
            continue
        cats = [tr(v[1], tax) for v in vs]
        if None in cats:
            ev.execute('INSERT OR REPLACE INTO questions VALUES (?, ?)', (k, vs[0][1]))
        cats = [c for c in cats if c]
        if len(cats) >= VOTES:
            top = max(set(cats), key=cats.count)
            share = cats.count(top) / len(cats)
            if share >= SHARE:
                b.execute('INSERT INTO cat VALUES (?, ?, ?, ?)', (k, top, round(share, 3), len(cats)))
    b.commit()
    ev.commit()
    after = dict(b.execute('SELECT fp, category FROM cat'))
    moved = sum(1 for k, c in before.items() if k in after and after[k] != c)
    return added, moved


def gate(b, ev, prev_sounds, moved, tax_changed, tax=None):
    """→ None when the new brain may be published, else why not."""
    if b.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        return 'integrity'
    n = b.execute('SELECT count(*) FROM sound').fetchone()[0]
    if n < prev_sounds:
        return f'fewer sounds ({n} < {prev_sounds})'
    for k, c, _ in ev.execute('SELECT k, cat, max(at) FROM votes WHERE owner = 1 GROUP BY k'):
        got = b.execute('SELECT category FROM cat WHERE fp = ?', (k,)).fetchone()
        if got and got[0] is not None and c and got[0] != tr(c, tax or {}) and not tax_changed:  # the vote as the taxonomy reads it today
            return f'an owner vote not honoured ({k})'
    cats = b.execute('SELECT count(*) FROM cat').fetchone()[0]
    if not tax_changed and cats and moved > MOVE_MAX * cats:
        return f'too many categories moved ({moved} of {cats})'
    return None


def shards(b, out):
    """The brain cut by key into shards (each a small brain of the same schema) → [{name, lo, hi, n, sha256, size}]."""
    keys = np.array([r[0] for r in b.execute('SELECT fp FROM sound ORDER BY fp')], np.int64)
    res = []
    for i in range(0, len(keys), SHARD):
        lo, hi = int(keys[i]), int(keys[min(len(keys), i + SHARD) - 1])
        name = f'shard-{i // SHARD:04d}.sqlite'
        p = os.path.join(out, name)
        s = sqlite3.connect(p)
        s.executescript(SCHEMA)
        s.close()
        b.execute('ATTACH DATABASE ? AS s', (p,))
        for t in TABLES:
            b.execute(f'INSERT INTO s.{t} SELECT * FROM main.{t} WHERE fp BETWEEN ? AND ?', (lo, hi))
        b.commit()
        b.execute('DETACH DATABASE s')
        res.append({'name': name, 'lo': lo, 'hi': hi, 'n': int(min(SHARD, len(keys) - i)), 'sha256': sha(p), 'size': os.path.getsize(p)})
    with open(os.path.join(out, 'keys.bin.gz'), 'wb') as f:
        f.write(gzip.compress(keys.astype('<i8').tobytes()))
    return res


def preset_shards(ev, prev_dir, index, out):
    """Presets' vectors by product (one shard each): the owner's, or 2 installs agreeing; the last release's kept."""
    by = {}
    for s in (index or {}).get('presets', []):
        c = sqlite3.connect(os.path.join(prev_dir, s['name']))
        by.setdefault(s['product'], {}).update({(h, k): v for h, k, v in c.execute('SELECT hash, kind, vec FROM vectors')})
    for h, kind, product in ev.execute('SELECT DISTINCT h, kind, product FROM presets').fetchall():
        rows = ev.execute('SELECT owner, v FROM presets WHERE h = ? AND kind = ?', (h, kind)).fetchall()
        own = [r for r in rows if r[0]]
        ok = own[:1] or ([rows[0]] if len(rows) >= 2 and cosine(rows[0][1], rows[1][1]) >= AGREE else [])
        if ok and product:
            by.setdefault(product, {})[(bytes.fromhex(h), kind)] = ok[0][1]
    res = []
    for n, (product, vs) in enumerate(sorted(by.items())):
        name = f'presets-{n:04d}.sqlite'
        p = os.path.join(out, name)
        c = sqlite3.connect(p)
        c.execute('CREATE TABLE vectors (hash BLOB NOT NULL, kind INTEGER NOT NULL, vec BLOB NOT NULL, PRIMARY KEY (hash, kind))')
        c.executemany('INSERT INTO vectors VALUES (?, ?, ?)', [(h, k, v) for (h, k), v in vs.items()])
        c.commit()
        c.close()
        res.append({'name': name, 'product': product, 'n': len(vs), 'sha256': sha(p), 'size': os.path.getsize(p)})
    return res


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while c := f.read(8 << 20):
            h.update(c)
    return h.hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--work', default='work')
    a.add_argument('--seed', help="the owner's brain (python -m hums.recall --export) for the first release")
    a.add_argument('--dry', action='store_true', help='build and check, publish nothing')
    a = a.parse_args()
    repo, url, token = os.environ.get('REPO', ''), os.environ.get('INBOX', ''), os.environ.get('MERGE_TOKEN', '')
    owners = set(filter(None, os.environ.get('OWNER_INSTALLS', '').split(',')))
    os.makedirs(a.work, exist_ok=True)
    tag, index = latest(repo, a.work) if repo else (None, None)
    prev = os.path.join(a.work, 'prev')
    b = joined(prev, index, os.path.join(a.work, 'brain.sqlite'), a.seed)
    evp = os.path.join(a.work, 'evidence.sqlite')
    if os.path.exists(os.path.join(prev, 'evidence.sqlite.gz')):
        with gzip.open(os.path.join(prev, 'evidence.sqlite.gz')) as f, open(evp, 'wb') as g:
            shutil.copyfileobj(f, g)
    ev = sqlite3.connect(evp)
    ev.executescript(EVIDENCE)
    top, fresh = inbox(url, token, ev, owners) if url and token else (0, 0)
    tax = taxonomy()
    tax_changed = bool(index) and index.get('taxonomy') != tax['version']
    prev_sounds = sum(s['n'] for s in (index or {}).get('shards', []))
    added, moved = merge(b, ev, tax)
    why = gate(b, ev, prev_sounds, moved, tax_changed, tax)
    print(f'merge: {fresh} new items, +{added} sounds, {moved} categories moved, gate: {why or "pass"}')
    if why:
        raise SystemExit(f'not published: {why}')
    if index and not (fresh or added or moved or tax_changed):  # a quiet night: the newest brain stays as it is
        print('nothing new: not published')
        return
    version = (index or {}).get('version', 0) + 1
    out = os.path.join(a.work, 'out')
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    new = {'version': version, 'tag': f'brain-v{version}', 'taxonomy': tax['version'], 'created': int(time.time()),
           'shards': shards(b, out), 'presets': preset_shards(ev, prev, index, out)}
    new['keys_sha256'] = sha(os.path.join(out, 'keys.bin.gz'))
    ev.close()
    with open(evp, 'rb') as f, gzip.open(os.path.join(out, 'evidence.sqlite.gz'), 'wb') as g:
        shutil.copyfileobj(f, g)
    json.dump(new, open(os.path.join(out, 'index.json'), 'w'))
    if a.dry:
        print(f'dry run: {new["tag"]} in {out}')
        return
    files = sorted(os.path.join(out, f) for f in os.listdir(out))
    sh('gh', 'release', 'create', new['tag'], '--repo', repo, '--title', f'Brain v{version}', '--notes', f'+{added} sounds · {moved} categories moved', *files)
    if url and token and fresh:
        urllib.request.urlopen(urllib.request.Request(f'{url.rstrip("/")}/ack', data=json.dumps({'upto': top}).encode(),
                               headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json', 'User-Agent': 'sherlock-merge'}), timeout=60)
    print(f'published {new["tag"]}')
    prune(repo)


def prune(repo, keep=2):
    """The server keeps the newest brain and the one before it (one step back); the owner's Mac mirrors every version."""
    tags = json.loads(sh('gh', 'release', 'list', '--repo', repo, '--limit', '200', '--json', 'tagName,createdAt') or '[]')
    tags = sorted((t for t in tags if t['tagName'].startswith('brain-v')), key=lambda t: int(t['tagName'][7:]), reverse=True)
    for t in tags[keep:]:
        sh('gh', 'release', 'delete', t['tagName'], '--repo', repo, '--yes', '--cleanup-tag')


if __name__ == '__main__':
    main()
