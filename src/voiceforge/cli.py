from __future__ import annotations

import argparse
import json
import threading
import webbrowser

from .config import CorrectionConfig
from .engine import auto_tune_file, correct_file


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description="Local, inspectable lead-vocal pitch correction")
    app.add_argument("input", help="Input WAV or FLAC")
    app.add_argument("output", help="Output WAV")
    app.add_argument("--key", default="G", help="Song key, e.g. G, F#, Bb")
    app.add_argument("--scale", choices=("major", "minor", "chromatic"), default="minor")
    app.add_argument("--strength", type=float, default=0.72, help="0 natural to 1 locked")
    app.add_argument("--retune-ms", type=float, default=85.0, help="Lower is more robotic")
    app.add_argument("--min-hz", type=float, default=70.0)
    app.add_argument("--max-hz", type=float, default=600.0)
    return app


def auto_parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description="Song-aware local vocal autotune")
    app.add_argument("vocal", help="Dry vocal WAV/FLAC")
    app.add_argument("song", help="Instrumental or full song WAV/FLAC")
    app.add_argument("output", help="Corrected vocal WAV")
    app.add_argument(
        "--vibe", required=True, help="Words like intimate, warm, polished, pop, or robotic"
    )
    app.add_argument("--key-override", help="Use when detection is uncertain, e.g. 'F# minor'")
    app.add_argument(
        "--local-llm", action="store_true", help="Use configured local LM to choose a safe profile"
    )
    return app


def ui_parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description="Launch the private local CUTE Tuner studio")
    app.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    app.add_argument("--port", type=int, default=8765)
    app.add_argument("--no-open", action="store_true")
    return app


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        args = vars(auto_parser().parse_args(sys.argv[2:]))
        output = args.pop("output")
        vocal, song = args.pop("vocal"), args.pop("song")
        args["use_local_llm"] = args.pop("local_llm")
        print(json.dumps(auto_tune_file(vocal, song, output, **args), indent=2))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        args = ui_parser().parse_args(sys.argv[2:])
        import uvicorn

        if not args.no_open:
            threading.Timer(0.8, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
        uvicorn.run("voiceforge.api:app", host=args.host, port=args.port, log_level="warning")
        return
    args = vars(parser().parse_args())
    input_path, output_path = args.pop("input"), args.pop("output")
    print(json.dumps(correct_file(input_path, output_path, CorrectionConfig(**args)), indent=2))
