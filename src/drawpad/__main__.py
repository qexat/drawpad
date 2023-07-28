#!/usr/bin/env python3

from __future__ import annotations

import readline  # pyright: ignore[reportUnusedImport]  # noqa: F401
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Self, TypeGuard
from typing import TextIO

from coquille import apply
from coquille import Coquille
from coquille.sequences import background_truecolor
from coquille.sequences import cursor_back
from coquille.sequences import cursor_down
from coquille.sequences import cursor_forward
from coquille.sequences import cursor_position
from coquille.sequences import cursor_up
from coquille.sequences import default_background_color
from coquille.sequences import disable_alternative_screen_buffer
from coquille.sequences import enable_alternative_screen_buffer
from coquille.sequences import fg_red
from coquille.sequences import soft_reset


POS_MIN = (0, 0)
BG_COLOR = (255, 255, 255)
COLOR_MIN = (0, 0, 0)
COLOR_MAX = (255, 255, 255)

ERR_COQUILLE = Coquille.new(fg_red, file=sys.stderr)

Vec2 = tuple[int, int]
Vec3 = tuple[int, int, int]


def DEBUG(obj: object) -> None:
    apply(disable_alternative_screen_buffer)
    print("DEBUG:", obj)
    apply(enable_alternative_screen_buffer)


class ColorError(ValueError):
    @classmethod
    def new(cls, invalid_color: Vec3) -> Self:
        return cls(
            f"provided color {invalid_color}"
            f"was not between {COLOR_MIN} and {COLOR_MAX}"
        )


@dataclass(slots=True)
class Pad:
    buffer: TextIO

    @classmethod
    def new(cls, buffer: TextIO | None = None) -> Self:
        return cls(buffer or sys.stdout)

    def __enter__(self) -> Self:
        apply(enable_alternative_screen_buffer, self.buffer)
        apply(soft_reset, self.buffer)
        self.clear()
        self.move_to(1, 1)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        apply(disable_alternative_screen_buffer, self.buffer)

    def get_size(self) -> Vec2:
        return shutil.get_terminal_size()

    def move_to(self, x: int, y: int) -> None:
        apply(cursor_position(y + 1, x + 1), self.buffer)

    def move_from_bottom_to(self, x: int, y: int) -> None:
        height = self.get_size()[1]
        self.move_to(x, height - y)

    def move_by(self, *, x: int = 0, y: int = 0) -> None:
        if x >= 0:
            apply(cursor_down, self.buffer)
        else:
            apply(cursor_up, self.buffer)

        if y >= 0:
            apply(cursor_forward, self.buffer)
        else:
            apply(cursor_back, self.buffer)

    def apply_color(self, r: int, g: int, b: int) -> None:
        apply(background_truecolor, self.buffer, r, g, b)
        self.buffer.write(" ")  # pyright: ignore[reportUnusedCallResult]
        apply(default_background_color, self.buffer)

    def set_pixel(self, pos: Vec2, color: Vec3) -> None:
        width, height = self.get_size()

        if not (POS_MIN <= pos <= (width, height)):
            raise ValueError(
                f"provided position {pos} was not between"
                f"(0, 0) and {(width, height)}"
            )

        if not (COLOR_MIN <= color <= COLOR_MAX):
            raise ColorError.new(color)

        self.move_to(*pos)
        self.apply_color(*color)

    def set_pixels_from_function(
        self,
        func: Callable[[Vec2], Vec3],
        min: Vec2 | None = None,
        max: Vec2 | None = None,
    ) -> int:
        width, height = self.get_size()
        positions = tuple((x, y) for y in range(height + 1) for x in range(width + 1))

        for position in positions:
            self.set_pixel(position, func(position))

        return len(positions)

    def set_pixel_batch(
        self,
        pixels: dict[Vec2, Vec3],
    ) -> None:
        for pos, color in pixels.items():
            self.set_pixel(pos, color)

    def set_uniform_pixel_batch(
        self,
        *positions: Vec2,
        color: Vec3,
    ) -> None:
        for position in positions:
            self.set_pixel(position, color)

    def clear_pixel(self, pos: Vec2) -> None:
        self.set_pixel(pos, BG_COLOR)

    def clear_pixel_batch(self, *positions: Vec2) -> None:
        self.set_uniform_pixel_batch(*positions, color=BG_COLOR)

    def clear(self) -> None:
        width, height = self.get_size()

        for i in range(width + 1):
            for j in range(height + 1):
                self.clear_pixel((i, j))


@dataclass(slots=True)
class CommandHandler:
    pad: Pad
    functions: dict[str, Callable[[Vec2], Vec3]]

    @classmethod
    def new(cls, pad: Pad) -> Self:
        return cls(pad, {})

    def register(self, function: Callable[[Vec2], Vec3]) -> None:
        self.functions[function.__name__] = function

    def register_batch(self, *functions: Callable[[Vec2], Vec3]) -> None:
        for function in functions:
            self.register(function)

    def is_function(self, arg: str | int) -> TypeGuard[str]:
        return not isinstance(arg, int) and arg in self.functions

    def exec_function(self, name: str) -> None:
        self.pad.set_pixels_from_function(self.functions[name])

    def parse(self, command: str) -> tuple[str, tuple[str | int, ...]]:
        op, *args = command.split()

        return op, tuple(map(lambda arg: int(arg) if arg.isdecimal() else arg, args))

    def run(self, command: str) -> None:
        err_y = self.pad.get_size()[1] - 1

        if command:
            op, args = self.parse(command)

            match op:
                case "clear":
                    self.pad.clear()
                case "exec":
                    if len(args) != 1:
                        with ERR_COQUILLE as c:
                            c.apply(cursor_position(err_y, 0))
                            c.print(f"exec: expected 1 argument, got {len(args)}")

                    elif not self.is_function(args[0]):
                        with ERR_COQUILLE as c:
                            c.apply(cursor_position(err_y, 0))
                            c.print(f"exec: {args[0]!r} is not a function")
                    else:
                        self.exec_function(args[0])
                case "exit":
                    self.exit(*args[:1])
                case "help":
                    print("commands: clear, exec, exit, help")
                case _:
                    with ERR_COQUILLE as c:
                        c.print(f"command {op!r} does not exist")
        else:
            with ERR_COQUILLE as c:
                c.apply(cursor_position(err_y, 0))
                c.print("empty command", end="")

    def exit(self, code: str | int = 0) -> None:
        raise SystemExit(code)


def func1(pos: Vec2) -> Vec3:
    product = pos[0] * pos[1]

    return (
        ((product >> 16) & 0xFF),
        ((product >> 8) & 0xFF),
        (product & 0xFF),
    )


def func2(pos: Vec2) -> Vec3:
    width, height = shutil.get_terminal_size()

    c1 = pos[0] * (255 // width)
    c2 = pos[1] * (255 // height)
    c3 = 255

    return c3, c1, c2


def main() -> int:
    with Pad.new() as pad:
        handler = CommandHandler.new(pad)
        handler.register(func1)
        handler.register(func2)

        while True:
            pad.move_from_bottom_to(0, 0)
            command = input(":")

            handler.run(command)
