"""Small Tk GUI that observes the same product orchestrator used by the CLI."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable

from .contracts import SessionResult
from .events import CallbackEventSink, EventSink

SessionRunner = Callable[[str, EventSink], SessionResult]


def run_gui(run_session: SessionRunner) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title("JEV Robot Control")
    root.geometry("1180x760")
    event_queue: queue.Queue[tuple[str, dict]] = queue.Queue()

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill=tk.BOTH, expand=True)
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    status_var = tk.StringVar(value="Ready")

    ttk.Label(frame, text="User goal").grid(row=0, column=0, sticky="w")
    goal_var = tk.StringVar()
    goal_entry = ttk.Entry(frame, textvariable=goal_var)
    goal_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 10))
    run_button = ttk.Button(frame, text="Start JEV control")
    run_button.grid(row=1, column=2, padx=(10, 0))
    ttk.Label(frame, textvariable=status_var).grid(
        row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0)
    )

    given_text = ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 9))
    choice_text = ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 9))
    log_text = ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 9))
    ttk.Label(frame, text="GIVEN THAT + persistent scene memory").grid(row=2, column=0, sticky="nw")
    ttk.Label(frame, text="JEV choices").grid(row=2, column=1, sticky="nw")
    ttk.Label(frame, text="Actions and robot feedback").grid(row=2, column=2, sticky="nw")
    given_text.grid(row=3, column=0, sticky="nsew", padx=(0, 8))
    choice_text.grid(row=3, column=1, sticky="nsew", padx=8)
    log_text.grid(row=3, column=2, sticky="nsew", padx=(8, 0))
    frame.rowconfigure(3, weight=1)
    frame.columnconfigure(2, weight=1)

    def enqueue(event: str, payload: dict) -> None:
        event_queue.put((event, payload))

    def worker(goal: str) -> None:
        try:
            run_session(goal, CallbackEventSink(enqueue))
        except Exception as exc:
            enqueue("session_error", {"type": type(exc).__name__, "message": str(exc)})

    def start() -> None:
        goal = goal_var.get().strip()
        if not goal:
            messagebox.showwarning("Missing goal", "Enter a natural-language robot goal.")
            return
        for widget in (given_text, choice_text, log_text):
            widget.delete("1.0", tk.END)
        run_button.configure(state=tk.DISABLED)
        status_var.set("Starting: capturing wrist RGB-D and asking JEV...")
        threading.Thread(target=worker, args=(goal,), daemon=True).start()

    def append(widget: ScrolledText, value: object) -> None:
        widget.insert(tk.END, json.dumps(value, ensure_ascii=False, indent=2) + "\n\n")
        widget.see(tk.END)

    def poll() -> None:
        try:
            while True:
                event, payload = event_queue.get_nowait()
                if event == "given_that_ready":
                    status_var.set(f"Turn {payload.get('turn')}: JEV is choosing the next action...")
                    given_text.delete("1.0", tk.END)
                    given_text.insert(tk.END, payload["rendered"])
                elif event == "jev_choice":
                    append(choice_text, payload)
                    status_var.set(
                        f"Turn {payload.get('turn')}: selected {payload.get('next_action')}"
                    )
                elif event == "action_started":
                    append(log_text, {"event": event, **payload})
                    status_var.set(
                        f"Executing {payload.get('action')} with the persistent sim..."
                    )
                else:
                    append(log_text, {"event": event, **payload})
                if event in {"session_finished", "session_error"}:
                    run_button.configure(state=tk.NORMAL)
                    status_var.set(
                        payload.get("message")
                        or payload.get("status")
                        or "Session finished"
                    )
        except queue.Empty:
            pass
        root.after(100, poll)

    run_button.configure(command=start)
    goal_entry.bind("<Return>", lambda _event: start())
    goal_entry.focus_set()
    root.after(100, poll)
    root.mainloop()
