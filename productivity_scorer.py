import time

class ProductivityScorer:
    """
    Tracks state per person. Score 0-100.
    Productive behaviors raise score; unproductive behaviors lower it.
    """
    WEIGHTS = {
        'phone':     -30,
        'sleeping':  -40,
        'drowsy':    -15,
        'slouching': -10,
        'away_gaze': -20,
    }
    DECAY_PER_SEC = 2     # score drifts toward neutral if no signal
    AWAY_THRESHOLD = 10.0  # seconds before gaze-away is penalized

    def __init__(self):
        self.states = {}     # person_id -> dict of current flags
        self.scores = {}     # person_id -> float score
        self.away_since = {} # person_id -> timestamp or None

    def _init_person(self, pid):
        self.states[pid] = {k: False for k in self.WEIGHTS}
        self.scores[pid] = 80.0
        self.away_since[pid] = None

    def update(self, pid, posture, gaze_away, phone, sleep_state):
        if pid not in self.states:
            self._init_person(pid)

        now = time.time()

        # Gaze away with time threshold
        if gaze_away:
            if self.away_since[pid] is None:
                self.away_since[pid] = now
            elif now - self.away_since[pid] > self.AWAY_THRESHOLD:
                self.states[pid]['away_gaze'] = True
        else:
            self.away_since[pid] = None
            self.states[pid]['away_gaze'] = False

        self.states[pid]['phone']     = phone
        self.states[pid]['sleeping']  = (sleep_state == 'sleeping')
        self.states[pid]['drowsy']    = (sleep_state == 'drowsy')
        self.states[pid]['slouching'] = (posture == 'slouching')

        # Compute penalty
        penalty = sum(w for k, w in self.WEIGHTS.items() if self.states[pid][k])
        base = 60 if penalty < 0 else 80
        target = max(0, min(100, base + penalty))

        # Smooth score toward target (faster reaction)
        self.scores[pid] += (target - self.scores[pid]) * 0.25

        return self.scores[pid], self.states[pid]

    def get_label(self, pid):
        score = self.scores.get(pid, 80)
        if score >= 70:   return 'Productive', (0, 200, 80)
        elif score >= 40: return 'Low activity', (0, 165, 255)
        else:             return 'Unproductive', (0, 0, 220)