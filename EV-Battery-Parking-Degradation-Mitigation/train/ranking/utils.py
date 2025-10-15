
"""ランキング用の小さなユーティリティ関数群。"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable

import numpy as np


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2地点間のハーサイン距離[km]を計算する。"""

    try:
        r = 6371.0
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlambda = math.radians(float(lon2) - float(lon1))
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return float(r * c)
    except Exception:
        return float("nan")


def bearing_degree(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """出発点から到着点への方位角[deg]を計算する。"""

    try:
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        dlambda = math.radians(float(lon2) - float(lon1))
        y = math.sin(dlambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
        theta = math.atan2(y, x)
        deg = (math.degrees(theta) + 360) % 360
        return float(deg)
    except Exception:
        return float("nan")


def hours_range(start: datetime, end: datetime) -> Iterable[int]:
    """開始・終了日時に重なる時間帯(0-23)を列挙する。"""

    cur = start.replace(minute=0, second=0, microsecond=0)
    if cur > start:
        cur = cur - np.timedelta64(1, "h")
    while cur < end:
        yield cur.hour
        cur = cur + np.timedelta64(1, "h")


def gaussian_kernel(hour_delta: float, sigma: float) -> float:
    """時間差に対するガウシアンカーネル値を返す。"""

    if sigma <= 0:
        return 0.0
    return math.exp(-0.5 * (hour_delta / sigma) ** 2)
