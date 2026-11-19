"""
Practice mode: the app shows one of your signs, you sign it. Signs you get
wrong more often come up more often, and per-sign stats are kept in the
profile so you can see which signs the app struggles with.

Per-sign stats: {"tries": int, "hits": int, "time": total secs to hit,
                 "confused": {other_label: count}}
"""

import random
from collections import Counter

from . import config as cfg


def accuracy(stat):
    return stat["hits"] / stat["tries"] if stat.get("tries") else None


class PracticeSession:
    def __init__(self, labels, stats, now, rng=None):
        self.labels = list(labels)
        self.stats = stats
        self.rng = rng or random.Random()
        self.results = []            # (label, hit, secs, confused_with)
        self.target = None
        self.phase = "prompt"        # prompt | feedback
        self.feedback = None         # ("hit", secs) or ("miss", confused_with)
        self.feedback_end = 0.0
        self._next_round(now)

    def _stat(self, label):
        return self.stats.setdefault(label, {"tries": 0, "hits": 0, "time": 0.0, "confused": {}})

    def _pick(self):
        choices = [l for l in self.labels if l != self.target] or self.labels
        weights = []
        for label in choices:
            acc = accuracy(self.stats.get(label, {}))
            weights.append(1.0 + 3.0 * (1.0 - (acc if acc is not None else 0.5)))
        return self.rng.choices(choices, weights=weights, k=1)[0]

    def _next_round(self, now):
        self.target = self._pick()
        self.started = now
        self.phase = "prompt"
        self.wrong = Counter()

    def time_left(self, now):
        return max(0.0, cfg.PRACTICE_TIMEOUT_SECS - (now - self.started))

    def update(self, recognition, now):
        """Feed one frame's Recognition. Returns "hit", "miss" or None."""
        if self.phase == "feedback":
            if now >= self.feedback_end:
                self._next_round(now)
            return None

        if recognition.shown and recognition.shown != self.target:
            self.wrong[recognition.shown] += 1

        stat = self._stat(self.target)
        if recognition.token == self.target:
            secs = now - self.started
            stat["tries"] += 1
            stat["hits"] += 1
            stat["time"] += secs
            self.results.append((self.target, True, secs, None))
            self._feedback(("hit", secs), now)
            return "hit"

        if now - self.started >= cfg.PRACTICE_TIMEOUT_SECS:
            confused = self.wrong.most_common(1)[0][0] if self.wrong else None
            stat["tries"] += 1
            if confused:
                stat["confused"][confused] = stat["confused"].get(confused, 0) + 1
            self.results.append((self.target, False, None, confused))
            self._feedback(("miss", confused), now)
            return "miss"
        return None

    def _feedback(self, fb, now):
        self.phase, self.feedback = "feedback", fb
        self.feedback_end = now + cfg.PRACTICE_FEEDBACK_SECS

    @property
    def score(self):
        return sum(1 for r in self.results if r[1]), len(self.results)

    def summary_lines(self):
        hits, total = self.score
        if not total:
            return ["No rounds played."]
        lines = [f"Score: {hits}/{total}  ({hits * 100 // total}%)", ""]
        per = {}
        for label, hit, _, confused in self.results:
            p = per.setdefault(label, [0, 0, Counter()])
            p[0] += hit
            p[1] += 1
            if confused:
                p[2][confused] += 1
        weakest = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])
        lines.append("Hardest signs this session:")
        for label, (h, t, conf) in weakest[:5]:
            line = f"  {label}: {h}/{t}"
            if conf:
                line += f"  (mistaken for {conf.most_common(1)[0][0]})"
            lines.append(line)
        return lines
