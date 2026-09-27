#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_map.py —— 自动采集阶段 0：大地图实测探针
=============================================

对应 docs/自动采集规划.md 第 2 节「阶段 0」。只观测、只截图，
不做任何连续操作；全部动作都可逆（开地图 / 拖一下 / 关地图）。

两个子命令，按顺序跑：

    # ① 自动部分（约 30 秒）：0.1 开地图判据 / 0.3 视野稳定性 / 0.4 屏幕中心 / 0.6 拖动手感
    #    跑之前：游戏窗口化或无边框，角色站在清河或开封野外，不在战斗中
    python rpa/probe_map.py auto

    # ② 手动部分：按清单在游戏里操作，每步到位后按 F9 截图（F8 跳过这步，F12 结束）
    #    0.2 药材图标 / 0.5 识途按钮 / 0.7 F 采集提示 / 0.8 识途落点 ×5
    python rpa/probe_map.py manual

    # 不需要游戏：自检本文件里的纯算法（CI 也跑这个）
    python rpa/probe_map.py --selftest

产物全部在 rpa/shots/probe_map/<时间戳>/，含 report.json；
结束时自动打成同名 .zip —— **把这个 zip 发回来**即可。
截图里有游戏画面，rpa/shots/ 已在 .gitignore 里，不会误提交。
"""

from __future__ import annotations

# >>> utf8-guard >>>
# Windows 上往管道/重定向的 stdout 打中文会 UnicodeEncodeError 崩掉
# （Python 默认用系统 ANSI 代码页而不是 UTF-8）。不指望调用方设
# PYTHONIOENCODING —— 脚本自己保证输出编码。
# 自带 import 是刻意的：位置无关，也不依赖文件里其他 import 的先后。
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
# <<< utf8-guard <<<

import argparse
import json
import shutil
import struct
import sys
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("[x] 需要 numpy：pip install numpy", file=sys.stderr)
    raise SystemExit(2)

HERE = Path(__file__).resolve().parent
SHOTS_ROOT = HERE / "shots" / "probe_map"


# ---------------------------------------------------------------------------
# 纯算法（不依赖 Windows，--selftest 覆盖）
# ---------------------------------------------------------------------------
def save_png(img: np.ndarray, path: Path) -> None:
    """(h, w, 3) BGR uint8 → PNG。只用标准库，不依赖 Pillow。

    顺带修掉一个老坑：vision.grab 返回的是 **BGR**，直接 Image.fromarray 会存成
    红蓝互换的图（probe_world_input.py 就是这样）。裁模板时颜色错了会误导人。
    """
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"只接受 (h,w,3) uint8，拿到 {img.shape} {img.dtype}")
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR → 灰度 float64（与 vision.to_gray 同系数；这里自带一份是为了离开 Windows 也能自检）。"""
    if img.ndim == 2:
        return img.astype(np.float64)
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def center_crop(a: np.ndarray, ratio: float = 0.6) -> np.ndarray:
    """取中间 ratio 的区域。地图四周是固定 UI（按钮、图例），会把位移估计拖向 0。"""
    h, w = a.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return a[y0:y0 + ch, x0:x0 + cw]


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """相位相关：估计 b 相对 a 的整体平移 (dx, dy)，以及峰值置信度。

    约定：b 里的内容 = a 里的内容向右移 dx、向下移 dy。
    置信度是相关面峰值 / 均值：无关画面个位数，真实平移自检里 170~1800；
    低于 ~50 说明两帧不是简单平移（缩放变了、画面换了），dx/dy 不可信。
    """
    ga, gb = to_gray(a), to_gray(b)
    if ga.shape != gb.shape:
        raise ValueError(f"两帧尺寸不同：{ga.shape} vs {gb.shape}")
    h, w = ga.shape
    win = np.outer(np.hanning(h), np.hanning(w))   # 抑制边缘不连续带来的十字伪峰
    fa = np.fft.fft2((ga - ga.mean()) * win)
    fb = np.fft.fft2((gb - gb.mean()) * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-12
    corr = np.abs(np.fft.ifft2(r))
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    conf = float(corr[py, px] / (corr.mean() + 1e-12))
    return float(dx), float(dy), conf


def draw_crosshair(img: np.ndarray, *, size: int = 40, color=(0, 0, 255)) -> np.ndarray:
    """在画面正中画红色十字 + 方框，用来肉眼核对「角色图标是不是在屏幕中心」。"""
    out = img.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    out[max(0, cy - 1):cy + 2, max(0, cx - size):cx + size + 1] = color
    out[max(0, cy - size):cy + size + 1, max(0, cx - 1):cx + 2] = color
    r = size // 2
    out[cy - r:cy - r + 2, cx - r:cx + r] = color
    out[cy + r - 1:cy + r + 1, cx - r:cx + r] = color
    out[cy - r:cy + r, cx - r:cx - r + 2] = color
    out[cy - r:cy + r, cx + r - 1:cx + r + 1] = color
    return out


def selftest() -> int:
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'通过' if cond else '失败'}] {label}" + (f"  —— {detail}" if detail else ""))

    rng = np.random.default_rng(7)
    # 平滑纹理比白噪声更像地图，也更考验相位相关
    base = rng.integers(0, 256, (240, 320, 3)).astype(np.float64)
    for _ in range(3):
        base = (base + np.roll(base, 1, 0) + np.roll(base, 1, 1) + np.roll(base, -1, 0)) / 4
    base = base.astype(np.uint8)

    for dx, dy in [(0, 0), (37, -12), (-60, 25)]:
        moved = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
        ex, ey, conf = phase_shift(center_crop(base, 0.8), center_crop(moved, 0.8))
        check(f"相位相关找回平移 ({dx},{dy})", abs(ex - dx) <= 1 and abs(ey - dy) <= 1,
              f"估计 ({ex:+.0f},{ey:+.0f}) 置信 {conf:.0f}")

    other = rng.integers(0, 256, base.shape).astype(np.uint8)
    _, _, conf_bad = phase_shift(base, other)
    _, _, conf_good = phase_shift(base, np.roll(base, 9, axis=1))
    check("无关画面置信度明显低于真平移", conf_bad * 5 < conf_good,
          f"无关 {conf_bad:.0f} vs 平移 {conf_good:.0f}")

    try:
        phase_shift(base, base[:100])
        check("尺寸不同会报错", False, "没报错")
    except ValueError:
        check("尺寸不同会报错", True)

    x = draw_crosshair(base)
    check("十字画在正中", tuple(x[120, 160]) == (0, 0, 255) and x.shape == base.shape)

    tmp = HERE / "shots" / "_selftest_probe_map.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_png(base, tmp)
        blob = tmp.read_bytes()
        check("PNG 头与尺寸正确", blob[:8] == b"\x89PNG\r\n\x1a\n"
              and struct.unpack(">II", blob[16:24]) == (320, 240))
        try:
            from PIL import Image  # 装了 Pillow 才能做像素级回读对照
            back = np.asarray(Image.open(tmp).convert("RGB"))[:, :, ::-1]
            check("PNG 回读逐像素一致（含 BGR→RGB）", np.array_equal(back, base))
        except ImportError:
            print("  [跳过] PNG 回读对照 —— 未装 Pillow")
    finally:
        tmp.unlink(missing_ok=True)

    print("自检" + ("全部通过" if ok else "有失败"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 游戏侧（只在 Windows 上导入）
# ---------------------------------------------------------------------------
class Session:
    """一次探针运行：截图 + report.json + 收尾打包。"""

    def __init__(self, kind: str) -> None:
        sys.path.insert(0, str(HERE))
        import injector as inj  # noqa: PLC0415
        import vision as vis    # noqa: PLC0415
        self.inj, self.vis = inj, vis

        self.dir = SHOTS_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_{kind}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.report: dict = {"kind": kind, "startedAt": datetime.now().isoformat(timespec="seconds"),
                             "steps": {}}

        self.hwnd = inj.find_game_window()
        if not self.hwnd:
            raise SystemExit("[!] 没找到游戏窗口（进程 yysls + 标题「燕云十六声」）")
        if not inj.ensure_foreground(self.hwnd):
            raise SystemExit("[!] 无法把游戏调到前台 —— 游戏若以管理员运行，本脚本也要管理员")
        self.X, self.Y, self.W, self.H = self._client_rect()
        self.report["client"] = {"x": self.X, "y": self.Y, "w": self.W, "h": self.H}
        self.report["screen"] = list(vis.screen_size())
        self.report["isAdmin"] = inj.is_admin()
        print(f"游戏在前台 ✓  客户区 {self.W}x{self.H} @ ({self.X},{self.Y})")
        print(f"截图目录：{self.dir}\n")

    def _client_rect(self) -> tuple[int, int, int, int]:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415
        u32 = ctypes.WinDLL("user32", use_last_error=True)
        u32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        rc, pt = wintypes.RECT(), wintypes.POINT(0, 0)
        u32.GetClientRect(wintypes.HWND(self.hwnd), ctypes.byref(rc))
        u32.ClientToScreen(wintypes.HWND(self.hwnd), ctypes.byref(pt))
        return pt.x, pt.y, rc.right, rc.bottom

    @property
    def center(self) -> tuple[int, int]:
        return self.X + self.W // 2, self.Y + self.H // 2

    def grab(self) -> np.ndarray:
        return self.vis.grab(self.X, self.Y, self.W, self.H)

    def shot(self, name: str, img: np.ndarray | None = None) -> np.ndarray:
        img = self.grab() if img is None else img
        save_png(img, self.dir / f"{name}.png")
        return img

    def step(self, key: str, **data) -> None:
        self.report["steps"][key] = data

    def finish(self) -> Path:
        self.report["finishedAt"] = datetime.now().isoformat(timespec="seconds")
        (self.dir / "report.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
        z = Path(shutil.make_archive(str(self.dir), "zip", root_dir=self.dir))
        print(f"\n完成。把这个文件发回来：\n  {z}")
        return z

    # -- 基础动作 --------------------------------------------------------
    def open_map(self, key: str, wait: float) -> None:
        self.inj.press_key(key)
        time.sleep(wait)

    def close_map(self, wait: float) -> None:
        self.inj.press_key("esc")
        time.sleep(wait)

    def drag(self, dx: int, dy: int, *, steps: int = 20, hold: float = 0.15) -> None:
        """按住左键从屏幕中心拖 (dx, dy)。用 SetCursorPos 走路径，每步都真的挪光标。"""
        import ctypes  # noqa: PLC0415
        u32 = ctypes.windll.user32
        inj = self.inj
        cx, cy = self.center
        u32.SetCursorPos(cx, cy)
        time.sleep(0.1)
        inj._send_mouse(0, 0, inj.MOUSEEVENTF_LEFTDOWN)
        time.sleep(hold)
        for i in range(1, steps + 1):
            u32.SetCursorPos(cx + dx * i // steps, cy + dy * i // steps)
            time.sleep(0.015)
        time.sleep(hold)
        inj._send_mouse(0, 0, inj.MOUSEEVENTF_LEFTUP)


# ---------------------------------------------------------------------------
# ① 自动部分
# ---------------------------------------------------------------------------
def run_auto(args: argparse.Namespace) -> int:
    s = Session("auto")
    vis = s.vis

    # 0.1 开地图判据 ------------------------------------------------------
    print("== 0.1 开地图前后对比 ==")
    world = s.shot("01_world")
    s.open_map(args.map_key, args.open_wait)
    opened = s.shot("02_map_open")
    d_open = vis.frame_diff(to_gray(world), to_gray(opened))
    print(f"  开地图帧差 {d_open:.2f}  " + ("✓ 地图打开了" if d_open > 8 else "✗ 画面几乎没变，地图键可能不是 M"))
    s.step("0.1_open", mapKey=args.map_key, frameDiff=round(d_open, 2), opened=d_open > 8)
    if d_open <= 8:
        s.close_map(args.close_wait)
        s.finish()
        return 1

    # 0.4 屏幕中心 --------------------------------------------------------
    s.shot("03_map_center_crosshair", draw_crosshair(opened))
    print("  已存 03_map_center_crosshair.png：红十字应该正好压在角色图标上")
    s.step("0.4_center", note="人工看 03 号图：红十字是否压在角色图标上")

    # 0.6 拖动手感（地图还开着，顺手做） --------------------------------------
    print("\n== 0.6 拖动地图 ==")
    drags = []
    for i, (dx, dy) in enumerate([(args.drag, 0), (0, args.drag), (-args.drag, -args.drag)]):
        before = s.grab()
        s.drag(dx, dy)
        time.sleep(0.8)                    # 等惯性/缓动停下
        after = s.shot(f"04_drag_{i + 1}_after")
        ex, ey, conf = phase_shift(center_crop(before), center_crop(after))
        ratio = [round(ex / dx, 3) if dx else None, round(ey / dy, 3) if dy else None]
        print(f"  拖 ({dx:+},{dy:+}) → 画面移动 ({ex:+.0f},{ey:+.0f})  比例 {ratio}  置信 {conf:.0f}")
        drags.append({"drag": [dx, dy], "moved": [ex, ey], "ratio": ratio, "confidence": round(conf)})
    s.step("0.6_drag", drags=drags,
           note="比例≈1 说明地图跟手；≈0 说明拖动没生效；置信<50 说明不是纯平移（可能触发了缩放或点开了东西）")

    s.close_map(args.close_wait)
    back = s.shot("05_map_closed")
    d_back = vis.frame_diff(to_gray(world), to_gray(back))
    print(f"\n  关地图后与开图前帧差 {d_back:.2f}  " + ("✓ 回到世界" if d_back < 25 else "? 可能没完全关掉"))
    s.step("0.1_close", frameDiffVsWorld=round(d_back, 2))

    # 0.3 视野稳定性 ------------------------------------------------------
    print("\n== 0.3 关开地图 3 次，看视野是否每次一样 ==")
    frames = []
    for i in range(3):
        s.open_map(args.map_key, args.open_wait)
        frames.append(s.shot(f"06_reopen_{i + 1}"))
        s.close_map(args.close_wait)
    shifts = []
    for i in range(1, len(frames)):
        ex, ey, conf = phase_shift(center_crop(frames[0]), center_crop(frames[i]))
        shifts.append({"vs": f"1→{i + 1}", "shift": [ex, ey], "confidence": round(conf)})
        print(f"  第 1 次 vs 第 {i + 1} 次：偏移 ({ex:+.0f},{ey:+.0f})  置信 {conf:.0f}")
    stable = all(abs(x["shift"][0]) <= 2 and abs(x["shift"][1]) <= 2 and x["confidence"] >= 50
                 for x in shifts)
    print("  " + ("✓ 每次打开视野一致（拖过的位置被重置回角色中心）" if stable
                  else "✗ 每次打开不一致 —— 可能记住了上次拖动的位置，或缩放变了"))
    s.step("0.3_stability", shifts=shifts, stable=stable,
           note="注意 0.6 刚拖过地图：若这里仍稳定，说明重开会回到以角色为中心")

    s.finish()
    return 0


# ---------------------------------------------------------------------------
# ② 手动部分：热键清单
# ---------------------------------------------------------------------------
CHECKLIST: list[tuple[str, str]] = [
    ("0.2a_herb_icons", "在地图上追踪「佛泪参」（或别的稀有药材），打开大地图，能看到药材图标时"),
    ("0.2b_herb_icons_zoom", "用滚轮缩放到你平时会用的级别，画面里尽量有多个药材图标时"),
    ("0.5_shitu_button", "在地图上点一个药材图标，弹出带【识途】按钮的框时（先别点识途）"),
    ("0.7_f_prompt", "关掉地图，自己走到一株药材旁边，屏幕上出现 F 采集提示时（先别按 F）"),
] + [
    (f"0.8_arrive_{i}", f"识途落点 {i}/5：对一个药材图标点【识途】，角色自己停下后，**不碰任何键**直接按截图键")
    for i in range(1, 6)
]


def run_manual(args: argparse.Namespace) -> int:
    s = Session("manual")
    idx = 0
    done = threading.Event()
    lock = threading.Lock()
    t_last = [time.perf_counter()]

    def prompt() -> None:
        if idx >= len(CHECKLIST):
            done.set()
            return
        key, text = CHECKLIST[idx]
        print(f"[{idx + 1}/{len(CHECKLIST)}] {key}\n    {text}\n"
              f"    → 按 {args.shot_key.upper()} 截图 / {args.skip_key.upper()} 跳过 / "
              f"{args.quit_key.upper()} 结束")

    def on_shot() -> None:
        nonlocal idx
        with lock:
            if done.is_set():
                return
            key, _ = CHECKLIST[idx]
            now = time.perf_counter()
            img = s.shot(key)
            s.shot(f"{key}_crosshair", draw_crosshair(img))
            s.step(key, captured=True, secondsSincePrev=round(now - t_last[0], 1),
                   at=datetime.now().isoformat(timespec="seconds"))
            t_last[0] = now
            print(f"    ✓ 已存 {key}.png\n")
            idx += 1
            prompt()

    def on_skip() -> None:
        nonlocal idx
        with lock:
            if done.is_set():
                return
            key, _ = CHECKLIST[idx]
            s.step(key, captured=False, skipped=True)
            print(f"    - 跳过 {key}\n")
            idx += 1
            prompt()

    watcher = s.inj.HotkeyWatcher()
    watcher.add(args.shot_key, on_shot)
    watcher.add(args.skip_key, on_skip)
    watcher.add(args.quit_key, done.set)
    print(f"热键后端：{watcher.start()}")
    print("提示：截图键在游戏里若有别的功能，用 --shot-key 换一个（F1~F12）\n")
    prompt()
    try:
        while not done.wait(0.2):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
    s.report["checklistDone"] = idx
    s.report["checklistTotal"] = len(CHECKLIST)
    s.finish()
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="自动采集阶段 0：大地图实测探针")
    ap.add_argument("--selftest", action="store_true", help="只跑纯算法自检，不需要游戏")
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("auto", help="自动：开地图判据 / 视野稳定性 / 屏幕中心 / 拖动手感")
    a.add_argument("--map-key", default="m")
    a.add_argument("--open-wait", type=float, default=2.0, help="开地图后等动画的秒数")
    a.add_argument("--close-wait", type=float, default=1.6)
    a.add_argument("--drag", type=int, default=200, help="每次拖动的像素")

    m = sub.add_parser("manual", help="手动：按清单操作，热键截图")
    m.add_argument("--shot-key", default="f9")
    m.add_argument("--skip-key", default="f8")
    m.add_argument("--quit-key", default="f12")

    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if sys.platform != "win32":
        print("[x] auto / manual 只能在 Windows 上对着游戏跑；这里可以用 --selftest", file=sys.stderr)
        return 2
    if args.cmd == "auto":
        return run_auto(args)
    if args.cmd == "manual":
        return run_manual(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
