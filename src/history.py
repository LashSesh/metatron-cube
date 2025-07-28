
# src/history.py

import json
from datetime import datetime

class Event:
    def __init__(self, event_type, values, info=None):
        self.time = datetime.now().isoformat()
        self.event_type = event_type
        self.values = values
        self.info = info or {}

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "values": self.values,
            "info": self.info
        }

class HistoryLogger:
    """
    Universeller Event-/Trace-Logger für alle Blueprint-Komponenten.
    """
    def __init__(self):
        self.events = []

    def log(self, event_type, values, info=None):
        ev = Event(event_type, values, info)
        self.events.append(ev)

    def filter(self, event_type):
        return [ev for ev in self.events if ev.event_type == event_type]

    def export_json(self, path):
        with open(path, "w") as f:
            json.dump([ev.to_dict() for ev in self.events], f, indent=2)

    def export_csv(self, path):
        import csv
        if not self.events:
            return
        keys = list(self.events[0].to_dict().keys())
        with open(path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for ev in self.events:
                writer.writerow(ev.to_dict())

    def clear(self):
        self.events = []

    def as_list(self):
        return [ev.to_dict() for ev in self.events]
