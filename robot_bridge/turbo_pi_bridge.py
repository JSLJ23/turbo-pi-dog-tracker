from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class MotionCommand:
    pan_delta: float
    tilt_delta: float
    forward: float
    turn: float
    stop: bool


class HiwonderAdapter:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.board: Any | None = None
        self.chassis: Any | None = None
        self.pan_pulse = 1500
        self.tilt_pulse = 1500
        self.pan_channel = 1
        self.tilt_channel = 2

        if dry_run:
            return

        try:
            import HiwonderSDK.Board as board  # type: ignore

            self.board = board
        except Exception as exc:  # pragma: no cover - hardware-only path
            print(f"warning: could not import HiwonderSDK.Board: {exc}", file=sys.stderr)

        try:
            from hiwonder import mecanum  # type: ignore

            self.chassis = mecanum.MecanumChassis()
        except Exception:
            try:
                import mecanum  # type: ignore

                self.chassis = mecanum.MecanumChassis()
            except Exception as exc:  # pragma: no cover - hardware-only path
                print(f"warning: could not import mecanum chassis: {exc}", file=sys.stderr)

    def stop(self) -> None:
        if self.dry_run:
            print("motion stop")
            return
        if self.chassis and hasattr(self.chassis, "set_velocity"):
            self.chassis.set_velocity(0, 0, 0)

    def apply(self, command: MotionCommand) -> None:
        if command.stop:
            self.stop()
            return

        if self.dry_run:
            print(
                "motion "
                f"pan_delta={command.pan_delta:.3f} "
                f"tilt_delta={command.tilt_delta:.3f} "
                f"forward={command.forward:.3f} "
                f"turn={command.turn:.3f}"
            )
            return

        if self.board and hasattr(self.board, "setPWMServoPulse"):
            self.pan_pulse = int(max(500, min(2500, self.pan_pulse - command.pan_delta * 45)))
            self.tilt_pulse = int(max(900, min(2100, self.tilt_pulse + command.tilt_delta * 35)))
            self.board.setPWMServoPulse(self.pan_channel, self.pan_pulse, 40)
            self.board.setPWMServoPulse(self.tilt_channel, self.tilt_pulse, 40)

        if self.chassis and hasattr(self.chassis, "set_velocity"):
            speed = max(0.0, min(35.0, abs(command.forward) * 35.0))
            direction = 90 if command.forward >= 0 else 270
            yaw_rate = max(-0.35, min(0.35, command.turn)) * 60.0
            self.chassis.set_velocity(speed, direction, yaw_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge tracker telemetry to HiWonder robot control")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-area", type=float, default=0.18)
    parser.add_argument("--deadband-x", type=float, default=0.08)
    parser.add_argument("--deadband-area", type=float, default=0.04)
    return parser.parse_args()


def command_from_telemetry(event: dict[str, Any], args: argparse.Namespace) -> MotionCommand:
    if event.get("state") != "tracked":
        return MotionCommand(0.0, 0.0, 0.0, 0.0, True)

    x_error = float(event.get("cx", 0.5)) - 0.5
    y_error = float(event.get("cy", 0.5)) - 0.5
    area_error = args.target_area - float(event.get("area", 0.0))

    turn = 0.0 if abs(x_error) < args.deadband_x else x_error * 1.4
    forward = 0.0 if abs(area_error) < args.deadband_area else area_error * 1.8

    return MotionCommand(
        pan_delta=max(-1.0, min(1.0, x_error * 2.0)),
        tilt_delta=max(-1.0, min(1.0, y_error * 2.0)),
        forward=max(-0.5, min(0.5, forward)),
        turn=max(-0.5, min(0.5, turn)),
        stop=False,
    )


def run() -> int:
    args = parse_args()
    adapter = HiwonderAdapter(dry_run=args.dry_run)

    while True:
        try:
            with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
                print(f"connected host={args.host} port={args.port}", file=sys.stderr)
                with sock.makefile("r", encoding="utf-8") as lines:
                    for line in lines:
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        adapter.apply(command_from_telemetry(event, args))
        except KeyboardInterrupt:
            adapter.stop()
            return 130
        except Exception as exc:
            print(f"bridge reconnecting after error: {exc}", file=sys.stderr)
            adapter.stop()
            time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(run())
