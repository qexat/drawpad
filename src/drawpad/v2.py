#!/usr/bin/env python3
# pyright: reportImplicitOverride = false

from __future__ import annotations


import argparse
import os
import readline  # pyright: ignore[reportUnusedImport]  # noqa: F401
import shutil
import sys
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import auto
from enum import IntEnum
import termios
import tty
from typing import Final, Iterable
from typing import cast
from typing import Self
from typing import TypeVar

from coquille import apply
from coquille.sequences import bg_truecolor
from coquille.sequences import erase_in_display
from coquille.sequences import erase_in_line
from coquille.sequences import cursor_position
from coquille.sequences import default_background_color
from coquille.sequences import disable_alternative_screen_buffer
from coquille.sequences import enable_alternative_screen_buffer
from coquille.sequences import soft_reset
from result import Err
from result import Ok
from result import Result


# *- TYPES -* #

_TNumber = TypeVar("_TNumber", int, float, complex)

Vec2 = tuple[_TNumber, _TNumber]
Vec3 = tuple[_TNumber, _TNumber, _TNumber]

Position = Vec2[int]
RGBColor = Vec3[int]
Canvas = list[list[RGBColor]]

# *- CONSTANTS -* #

WHITE: RGBColor = (255, 255, 255)
RED: RGBColor = (255, 0, 0)
ORANGE: RGBColor = (255, 127, 0)
YELLOW: RGBColor = (255, 255, 0)
CHARTREUSE: RGBColor = (127, 255, 0)
GREEN: RGBColor = (0, 255, 0)
SPRING: RGBColor = (0, 255, 127)
AQUA: RGBColor = (0, 255, 255)
AZURE: RGBColor = (0, 127, 255)
BLUE: RGBColor = (0, 0, 255)
VIOLET: RGBColor = (127, 0, 255)
MAGENTA: RGBColor = (255, 0, 255)
ROSE: RGBColor = (255, 0, 127)

BUFFER_DEFAULT = sys.stdout
POSITION_DEFAULT: Position = (0, 0)
COLOR_DEFAULT = WHITE

MANUAL_TITLE_DEFAULT = "HELP"

RGB_MIN = 0
RGB_MAX = 0xFFFFFF

SPECIAL_FN_KEY = "\x1b["


# *- GLOBALS -* #

_wid = 1


def get_window_id() -> int:
    """
    Util function to generate a new unique window id.
    """

    global _wid

    value = _wid
    _wid += 1

    return value


class KeyType(IntEnum):
    EOT = auto()
    TAB = auto()
    NEWLINE = auto()
    BACKWORD = auto()
    ESC = auto()
    ARROW_UP = auto()
    ARROW_DOWN = auto()
    ARROW_RIGHT = auto()
    ARROW_LEFT = auto()
    BACKSPACE = auto()

    CHARACTER = auto()


ARROW_TYPES = {
    65: KeyType.ARROW_UP,
    66: KeyType.ARROW_DOWN,
    67: KeyType.ARROW_RIGHT,
    68: KeyType.ARROW_LEFT,
}

C0_TYPES = {
    4: KeyType.EOT,
    9: KeyType.TAB,
    10: KeyType.NEWLINE,
    23: KeyType.BACKWORD,
    27: KeyType.ESC,
    127: KeyType.BACKSPACE,
}


@dataclass(slots=True)
class Key:
    type: KeyType
    ordinal: int
    value: str
    prefix: Iterable[str]

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> Self:
        ordinal = raw_bytes[-1]

        raw_string = raw_bytes.decode("utf-8")
        *prefix, value = raw_string

        lookup_dict = ARROW_TYPES if "".join(prefix) == SPECIAL_FN_KEY else C0_TYPES
        key_type = lookup_dict.get(ordinal, KeyType.CHARACTER)

        return Key(key_type, ordinal, value, prefix)

    def __repr__(self) -> str:
        return f"<Key {self.type.name} {self.get_raw()!r}>"

    def get_raw(self) -> str:
        return "".join(self.prefix) + self.value

    def get_raw_bytes(self) -> bytes:
        return self.get_raw().encode()

    def is_character(self, char: str) -> bool:
        return self.type is KeyType.CHARACTER and self.value == char


KEY_ARROW_UP = Key.from_bytes(b"\x1b[A")
KEY_ARROW_DOWN = Key.from_bytes(b"\x1b[B")
KEY_ARROW_RIGHT = Key.from_bytes(b"\x1b[C")
KEY_ARROW_LEFT = Key.from_bytes(b"\x1b[D")


@dataclass(slots=True)
class Window:
    """
    Window of a pad.

    - The ID is provided at creation. It must and will be unique.
    - The width and height should be pad's inner size.
    - `x` and `y` are the window cursor position. Should start at (0, 0).
    - `canvas` is the canvas used as the window buffer.
    """

    id: int
    width: int
    height: int
    x: int
    y: int
    canvas: Canvas

    @staticmethod
    def create_canvas(
        width: int,
        height: int,
        *,
        color: RGBColor = COLOR_DEFAULT,
    ) -> Canvas:
        """
        Util function to create a canvas of size `width`*`height`
        filled uniformly with color `color`.
        """

        return [[color] * width] * height

    @classmethod
    def new(cls, width: int, height: int) -> Self:
        """
        Create a new Window of size `width`*`height`.
        """

        return cls(
            get_window_id(),
            width,
            height,
            *POSITION_DEFAULT,
            cls.create_canvas(width, height),
        )

    def get_row(self, n: int) -> Result[list[RGBColor], IndexError]:
        """
        Return the array of pixels (colors) at the row `n` of the canvas.

        Note: rows index starts at 1.
        """

        if not (1 <= n <= self.height):
            return Err(IndexError(f"{n} must be between 1 and {self.height}"))

        return Ok(self.canvas[n - 1])

    def get_col(self, n: int) -> Result[list[RGBColor], IndexError]:
        """
        Return the array of pixels (colors) at the column `n` of the canvas.

        Note: columns index starts at 1.
        """

        if not (1 <= n <= self.width):
            return Err(IndexError(f"{n} must be between 1 and {self.width}"))

        return Ok([row[n - 1] for row in self.canvas])

    def move_up(self, n: int = 1) -> None:
        """
        Move the cursor up `n` times.

        Note: if it encounters the canvas limits, it does not move.
        """

        self.y = max(self.y - n, 0)

    def move_down(self, n: int = 1) -> None:
        """
        Move the cursor down `n` times.

        Note: if it encounters the canvas limits, it does not move.
        """

        self.y = min(self.y + n, self.height)

    def move_left(self, n: int = 1) -> None:
        """
        Move the cursor to the left `n` times.

        Note: if it encounters the canvas limits, it does not move.
        """

        self.x = max(self.x - n, 0)

    def move_right(self, n: int = 1) -> None:
        """
        Move the cursor to the right `n` times.

        Note: if it encounters the canvas limits, it does not move.
        """

        self.x = min(self.x + n, self.width)

    def fill_current(self, color: RGBColor) -> None:
        """
        Fill the current cell with a given color -- the cell at (x, y) in the canvas.
        """

        self.canvas[self.x][self.y] = color

    def clear_current(self) -> None:
        """
        Clear the current cell -- the cell at (x, y) in the canvas.
        """

        self.canvas[self.x][self.y] = COLOR_DEFAULT

    def fill(self, color: RGBColor) -> None:
        """
        Fill the whole window canvas with a given color.
        """

        self.canvas = self.create_canvas(self.width, self.height, color=color)

    def clear(self) -> None:
        """
        Clear the whole window canvas, i.e. fill with the default color.
        """

        self.canvas = self.create_canvas(self.width, self.height)

    def draw(self, sx: int, sy: int, ex: int, ey: int) -> None:
        for row in self.canvas[sy:ey]:
            for cell in row[sx:ex]:
                apply(bg_truecolor(*cell))
                sys.stdout.write(" ")  # pyright: ignore[reportUnusedCallResult]
        apply(bg_truecolor(*COLOR_DEFAULT))


@dataclass(slots=True)
class ManualPage:
    title: str
    content: str


class ManualEvent(IntEnum):
    QUIT = auto()
    PREV_PAGE = auto()
    NEXT_PAGE = auto()


@dataclass(slots=True)
class Manual:
    title: str
    pages: Iterable[ManualPage]
    triggers: Callable[[Key], ManualEvent | None]

    @staticmethod
    def event_trigger(key: Key) -> ManualEvent | None:
        match key.type:
            case KeyType.ARROW_LEFT:
                return ManualEvent.PREV_PAGE
            case KeyType.ARROW_RIGHT:
                return ManualEvent.NEXT_PAGE
            case KeyType.CHARACTER:
                if key.value == "q":
                    return ManualEvent.QUIT

        return None

    @classmethod
    def new(
        cls,
        title: str,
        pages: Iterable[ManualPage] | None = None,
        triggers: Callable[[Key], ManualEvent | None] | None = None,
    ) -> Self:
        return cls(title, pages or [], triggers or (lambda _: None))

    @classmethod
    def default(
        cls,
        *pages: ManualPage,
    ) -> Self:
        return cls(MANUAL_TITLE_DEFAULT, pages, cls.event_trigger)

    def show(self, reset_func: Callable[..., None]) -> None:
        ...


@dataclass(slots=True)
class Pad:
    """
    A pad is an interface with one or more windows which can be used to draw RGB cells.

    It is visually in two parts:
    - the top, where the current window canvas lives.
    - the bottom bar, which contains an interactive input and output.

    It should be noted that only one window can be shown at a time.
    They can be switched using the `win` command.
    """

    windows: MutableMapping[int, Window]
    current_wid: int

    @classmethod
    def new(cls) -> Self:
        """
        Create a new empty pad.
        """

        return cls({}, 0)

    @classmethod
    def default(cls) -> Self:
        """
        Return the default pad, with 8 blank windows, the first being selected.
        """

        obj = cls.new()
        first, *_ = obj.create_window_batch(n=8)
        # Manually set the current_wid ; otherwise it would make unnecessary checks.
        obj.current_wid = first.id

        return obj

    @classmethod
    def with_windows(cls, *windows: Window) -> Self:
        """
        Create a new pad with the given windows.
        """

        return cls({window.id: window for window in windows}, 0)

    def __enter__(self) -> Self:
        self.show()
        self.draw_window()

        return self

    def __exit__(self, *_) -> None:
        self.hide()

    def show(self) -> None:
        apply(enable_alternative_screen_buffer)
        apply(soft_reset)

    def hide(self) -> None:
        apply(soft_reset)
        apply(disable_alternative_screen_buffer)

    def get_size(self) -> tuple[int, int]:
        """
        Return the size of the pad.
        """

        return shutil.get_terminal_size()

    def get_canvas_size(self) -> tuple[int, int]:
        """
        Return the size of the canvas.
        """

        width, height = self.get_size()
        return width, height - 2

    def create_window(self) -> Window:
        """
        Create a new empty window and return it.
        """

        window = Window.new(*self.get_canvas_size())
        self.windows[window.id] = window

        return window

    def create_window_batch(self, n: int) -> list[Window]:
        """
        Create a `n`-sized batch of new empty windows and return it.
        """

        return [self.create_window() for _ in range(n)]

    def add_window(self, window: Window) -> None:
        """
        Add an existing window to the pad.
        """

        self.windows[window.id] = window

    def get_window(self, id: int) -> Result[Window, IndexError]:
        """
        Return the window with the given `id`.
        """

        if id not in self.windows:
            return Err(IndexError(f"no window with id {id}"))

        return Ok(self.windows[id])

    def delete_window(self, id: int) -> Result[Window, IndexError]:
        """
        Delete the window the given `id` and return it.
        """

        window = self.get_window(id)

        if isinstance(window, Ok):
            del self.windows[window.unwrap().id]

        return window

    def select_window(self, id: int) -> Result[Window, IndexError]:
        """
        Change the current window to another one, given its `id`.
        """

        result = self.get_window(id)

        if isinstance(result, Err):
            return result

        self.current_wid = id

        return result

    def current_window(self) -> Result[Window, ValueError]:
        if not self.current_wid:
            return Err(ValueError("no window is selected"))

        return Ok(self.windows[self.current_wid])

    def draw_window(self) -> None:
        apply(cursor_position(1, 1))
        cw, ch = self.get_canvas_size()
        window = self.windows[self.current_wid]
        window.draw(0, 0, cw, ch)
        apply(default_background_color)

    def draw_manual(self) -> None:
        apply(erase_in_display(2))
        apply(cursor_position(1, 1))
        sys.stdout.flush()

    def get_input(self) -> str:
        _, height = self.get_size()

        apply(cursor_position(height - 1, 1))
        apply(erase_in_line(2))

        return input(":")

    def write_output(self, message: str) -> None:
        _, height = self.get_size()

        apply(cursor_position(height, 1))
        apply(erase_in_line(2))

        _ = sys.stdout.write(message)

    def write_success(self, message: str) -> None:
        self.write_output(f"\x1b[32msuccess: {message}\x1b[39m")

    def write_error(self, message: str) -> None:
        self.write_output(f"\x1b[31merror: {message}\x1b[39m")

    def clear_output(self) -> None:
        self.write_output("")

    def open_manual(self) -> None:
        self.draw_manual()

        with InteractiveInput.new() as keyboard:
            while True:
                key = keyboard.get_key()

                if key.is_character("q"):
                    break

        self.draw_window()


@dataclass(slots=True)
class InteractiveInput:
    previous_state: list[int | list[bytes | int]]

    @classmethod
    def new(cls) -> Self:
        return cls(termios.tcgetattr(sys.stdin))

    def __enter__(self) -> Self:
        self.previous_state = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)

        return self

    def __exit__(self, *_) -> None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.previous_state)

    def get_key(self) -> Key:
        return Key.from_bytes(os.read(sys.stdin.fileno(), 3))


@dataclass(slots=True)
class App:
    testing: bool

    @classmethod
    def new(cls, *, testing: bool = False) -> Self:
        return cls(testing)

    def run(self) -> int:
        pad = Pad.default()
        handler = CommandHandler(pad)

        with pad:
            while True:
                command = pad.get_input()
                pad.clear_output()

                handler.run(command)
                pad.draw_window()


class color:
    @staticmethod
    def is_repr(value: str) -> bool:
        if not value:
            return False

        prefix, *hexdigits = value

        if prefix != "#" or len(hexdigits) not in {3, 6}:
            return False

        return all(ishex(digit) for digit in hexdigits)

    def __init__(self, value: str | int = 0) -> None:
        if isinstance(value, str):
            if not self.is_repr(value):
                raise ValueError(f"{value!r} is not a valid color")

            _, *hexdigits = value

            hexvalue = "".join(d for d in hexdigits for _ in range(6 // len(hexdigits)))
            value = int(hexvalue, base=16)

        if not (RGB_MIN <= value <= RGB_MAX):
            raise ValueError(f"{value!r} is not in the range of possible RGB colors")

        self.__integer = value
        self.__b = value % 256
        self.__g = ((value - self.__b) // 256) % 256
        self.__r = ((value - self.__b) // (256**2)) - self.__g // 256

    def __iter__(self) -> Iterator[int]:
        return iter(self.to_tuple())

    def __repr__(self) -> str:
        return f"#{self.__integer:X}"

    def to_tuple(self) -> tuple[int, int, int]:
        return self.__r, self.__g, self.__b


class CommandTokenType(IntEnum):
    IDENTIFIER = auto()
    STRING = auto()
    NATURAL = auto()
    VEC2 = auto()
    VEC3 = auto()
    COLOR = auto()

    # *- Keywords -* #
    ANIM = auto()
    CLEAR = auto()
    EXEC = auto()
    EXIT = auto()
    FILL = auto()
    HELP = auto()
    SOURCE = auto()
    WINDOW = auto()


KEYWORDS = {
    "anim": CommandTokenType.ANIM,
    "clear": CommandTokenType.CLEAR,
    "exec": CommandTokenType.EXEC,
    "exit": CommandTokenType.EXIT,
    "fill": CommandTokenType.FILL,
    "help": CommandTokenType.HELP,
    "source": CommandTokenType.SOURCE,
    "window": CommandTokenType.WINDOW,
}


@dataclass(slots=True)
class CommandToken:
    type: CommandTokenType
    lexeme: str
    literal: object


@dataclass(slots=True)
class CommandError:
    message: str


@dataclass(slots=True)
class CommandLexer:
    source: Final[str]
    tokens: Final[list[CommandToken]]
    start: int
    current: int
    errors: list[CommandError]

    @classmethod
    def new(cls, source: str) -> Self:
        return cls(source, [], 0, 0, [])

    def is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def add_error(self, message: str) -> None:
        self.errors.append(CommandError(message))

    def peek(self) -> str:
        if self.is_at_end():
            return "\0"

        return self.source[self.current]

    def peek_next(self) -> str:
        if (self.current + 1) >= len(self.source):
            return "\0"

        return self.source[self.current + 1]

    def advance(self) -> str:
        char = self.source[self.current]
        self.current += 1
        return char

    def match(self, expected: str) -> bool:
        if self.is_at_end():
            return False

        if self.source[self.current] != expected:
            return False

        self.current += 1

        return True

    def make_string(self) -> None:
        while self.peek() != '"' and not self.is_at_end():
            self.advance()

        if self.is_at_end():
            self.add_error("unterminated string literal")
            return

        self.advance()

        value = self.source[self.start + 1 : self.current - 1]
        self.add_token(CommandTokenType.STRING, value)

    def make_natural(self) -> None:
        while self.peek().isdecimal():
            self.advance()

        raw_value = self.source[self.start : self.current]
        self.add_token(CommandTokenType.NATURAL, int(raw_value))

    def make_identifier(self) -> None:
        while self.peek().isidentifier():
            self.advance()

        raw_value = self.source[self.start : self.current]
        type = KEYWORDS.get(raw_value, CommandTokenType.IDENTIFIER)

        self.add_token(type)

    def make_color(self) -> None:
        n = 0
        while ishex((char := self.peek())):
            if n >= 6:
                self.add_error(f"unexpected {char!r} after parsing color")
                return

            self.advance()
            n += 1

        raw_value = self.source[self.start : self.current]

        if (n % 3) != 0 or n == 0:
            self.add_error(f"{raw_value!r} is not a valid RGB color")
            return

        self.add_token(CommandTokenType.COLOR, color(raw_value))

    def make_vec(self) -> None:
        while self.peek() != ")" and not self.is_at_end():
            self.advance()

        if self.is_at_end():
            self.add_error("unterminated vector literal")
            return

        self.advance()

        substring = self.source[self.start + 1 : self.current - 1]

        values: list[int] = []
        substart = 0
        subcurrent = 0

        while subcurrent < len(substring):
            substart = subcurrent
            while subcurrent < len(substring) and substring[subcurrent].isdigit():
                subcurrent += 1

            values.append(int(substring[substart:subcurrent]))

            while substring[subcurrent] == " ":
                subcurrent += 1

            if not substring[subcurrent].isdigit():
                self.add_error(f"unexpected character {substring[subcurrent]!r}")
                return

        if len(values) == 2:
            self.add_token(CommandTokenType.VEC2, tuple(values))
        elif len(values) == 3:
            self.add_token(CommandTokenType.VEC3, tuple(values))
        else:
            self.add_error("invalid vector literal")

    def add_token(self, type: CommandTokenType, literal: object | None = None) -> None:
        lexeme = self.source[self.start : self.current]
        self.tokens.append(CommandToken(type, lexeme, literal))

    def scan_token(self) -> None:
        char = self.advance()

        match char:
            case "(":
                self.make_vec()
            case "#":
                self.make_color()
            case '"':
                self.make_string()
            case " " | "\r" | "\t":
                pass
            case _:
                if char.isdecimal():
                    self.make_natural()
                elif char.isidentifier():
                    self.make_identifier()
                else:
                    self.add_error(f"unexpected character {char!r}")

    def scan(self) -> list[CommandToken]:
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()

        return self.tokens


def ishex(char: str) -> bool:
    return char.isdecimal() or char.lower() in {"a", "b", "c", "d", "e", "f"}


def not_implemented(
    method: Callable[[CommandHandler, list[CommandToken]], None],
) -> Callable[[CommandHandler, list[CommandToken]], None]:
    def _(self: CommandHandler, args: list[CommandToken]) -> None:
        command_name = method.__name__.removeprefix("run_")
        self.pad.write_error(f"command {command_name!r} is not implemented")

    return _


@dataclass(slots=True)
class CommandHandler:
    pad: Pad

    def evaluate(self, tokens: list[CommandToken]) -> None:
        value, *others = tokens

        if others:
            self.pad.write_error(f"unexpected token {others[0].lexeme!r}")
            return

        self.pad.write_output(repr(value.literal))

    @not_implemented
    def run_anim(self, args: list[CommandToken]) -> None:
        ...

    @not_implemented
    def run_clear(self, args: list[CommandToken]) -> None:
        ...

    @not_implemented
    def run_exec(self, args: list[CommandToken]) -> None:
        ...

    def run_exit(self, args: list[CommandToken]) -> None:
        code = args[0].literal if args else 0
        raise SystemExit(code)

    def run_fill(self, args: list[CommandToken]) -> None:
        nb_args = len(args)

        if nb_args != 1:
            self.pad.write_error(f"expected 1 arg, received {nb_args}")
            return

        (arg,) = args

        match arg.type:
            case CommandTokenType.COLOR:
                value = cast(color, arg.literal).to_tuple()
            case CommandTokenType.VEC3:
                value = cast(tuple[int, int, int], arg.literal)
            case t:
                self.pad.write_error(f"expected color or vec3, received {t.name!r}")
                return

        match self.pad.current_window():
            case Ok(window):
                window.fill(value)
                self.pad.write_success(f"filled window {window.id} with color {value}")
            case Err(e):
                self.pad.write_error(e.args[0])

    def run_help(self, args: list[CommandToken]) -> None:
        self.pad.open_manual()

    @not_implemented
    def run_source(self, args: list[CommandToken]) -> None:
        ...

    @not_implemented
    def run_window(self, args: list[CommandToken]) -> None:
        ...

    def run(self, command: str) -> None:
        lexer = CommandLexer.new(command)
        tokens = lexer.scan()

        if lexer.errors:
            self.pad.write_error(lexer.errors[0].message)
            return

        if not tokens:
            return

        if tokens[0].type in KEYWORDS.values():
            args = tokens[1:]

            match tokens[0].type:
                case CommandTokenType.ANIM:
                    self.run_anim(args)
                case CommandTokenType.CLEAR:
                    self.run_clear(args)
                case CommandTokenType.EXEC:
                    self.run_exec(args)
                case CommandTokenType.EXIT:
                    self.run_exit(args)
                case CommandTokenType.FILL:
                    self.run_fill(args)
                case CommandTokenType.HELP:
                    self.run_help(args)
                case CommandTokenType.SOURCE:
                    self.run_source(args)
                case CommandTokenType.WINDOW:
                    self.run_window(args)
        elif tokens[0].type is CommandTokenType.IDENTIFIER:
            self.pad.write_error(f"unknown command {tokens[0].lexeme!r}")
        else:
            self.evaluate(tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--testing", action="store_true", help="activate testing mode")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = App.new(testing=args.testing)

    return app.run()
