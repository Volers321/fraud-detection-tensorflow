"""Synthetic card transactions with fraud, and point-in-time features.

This file is identical in all three repositories (sklearn / keras / tensorflow) and
``simulate(seed=0)`` returns the same data in each, so results are comparable across libraries.

What is simulated
-----------------
* Cards with their own habits: home city, typical amount, time of day, favourite
  categories, share of online purchases, one or two devices.
* Honest traffic with hard negatives: trips (payments far from home), a new phone (new
  device), rare large purchases. Without them the task would be far too easy.
* Four fraud patterns:
    - ``card_testing``  a burst of tiny online payments from a new device,
    - ``takeover``      account takeover: new device, large amount, night, foreign IP,
    - ``stolen_card``   physical use far from home, several payments in a row,
    - ``mule_cashout``  a series of ATM withdrawals; appears only in the last days, so it is
      a new attack type the models never saw in training.

The data is synthetic: it comes from no bank and does not reflect real fraud rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = [
    "grocery", "fuel", "restaurant", "clothing", "electronics", "travel",
    "pharmacy", "entertainment", "utilities", "gift_cards", "marketplace", "atm",
]  # fmt: skip
CAT_ID = {c: i for i, c in enumerate(CATEGORIES)}
CHANNELS = ["pos", "online", "atm"]
CATEGORICAL = ["channel", "category"]
NUMERIC = [
    "log_amount", "amount_z", "amount_ratio", "hour_sin", "hour_cos", "is_night",
    "is_foreign", "new_device", "new_category", "dist_home_km", "dist_prev_km",
    "speed_kmh", "log_minutes_since_prev", "cnt_10m", "cnt_1h", "cnt_24h",
    "sum_1h_ratio", "n_prev",
]  # fmt: skip
FEATURES = NUMERIC + CATEGORICAL

# lat, lon: Warsaw, Krakow, Lodz, Wroclaw, Poznan, Gdansk, Szczecin, Lublin,
# Katowice, Bialystok, Rzeszow, Olsztyn
CITIES = np.array(
    [
        [52.23, 21.01], [50.06, 19.94], [51.76, 19.46], [51.11, 17.04],
        [52.41, 16.93], [54.35, 18.65], [53.43, 14.55], [51.25, 22.57],
        [50.26, 19.02], [53.13, 23.16], [50.04, 22.00], [53.78, 20.49],
    ]
)  # fmt: skip
CAT_MULT = np.array([1.0, 2.2, 1.4, 1.8, 6.0, 8.0, 0.8, 1.1, 2.5, 3.0, 2.0, 5.0])

DAY = 86400.0


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _cards(rng, n):
    city = rng.integers(0, len(CITIES), n)
    return pd.DataFrame(
        {
            "card_id": np.arange(n),
            "home_lat": CITIES[city, 0] + rng.normal(0, 0.05, n),
            "home_lon": CITIES[city, 1] + rng.normal(0, 0.05, n),
            "amt_mu": rng.normal(3.6, 0.4, n),
            "amt_sigma": rng.uniform(0.6, 1.0, n),
            "rate": rng.gamma(2.0, 0.45, n),
            "online_share": rng.beta(2, 5, n),
            "peak_hour": rng.normal(15, 3, n),
            "two_devices": rng.random(n) < 0.5,
        }
    )


def _legit(rng, cards, days):
    n = len(cards)
    fav = rng.dirichlet(np.ones(len(CATEGORIES) - 1) * 0.7, n)  # without the ATM category
    counts = rng.poisson(cards["rate"].to_numpy() * days)
    cid = np.repeat(np.arange(n), counts)
    m = len(cid)
    c = cards.iloc[cid].reset_index(drop=True)
    day = rng.uniform(0, days, m)
    hour = (rng.normal(c["peak_hour"].to_numpy(), 4.0)) % 24
    t = np.floor(day) * DAY + hour * 3600 + rng.uniform(0, 3600, m)
    u = rng.random(m)
    channel = np.where(u < c["online_share"], 1, np.where(u < c["online_share"] + 0.06, 2, 0))
    cdf = np.cumsum(fav[cid], axis=1)
    cat = (rng.random(m)[:, None] > cdf).sum(axis=1).clip(0, len(CATEGORIES) - 2)
    cat = np.where(channel == 2, CAT_ID["atm"], cat)
    amount = np.exp(rng.normal(c["amt_mu"], c["amt_sigma"])) * CAT_MULT[cat] ** 0.6
    big = rng.random(m) < 0.004  # rare large purchase (hard negative)
    amount = np.where(big, amount * rng.uniform(5, 12, m), amount)
    atm_amount = rng.choice([100, 200, 300, 500, 800, 1000], m)
    amount = np.where(channel == 2, atm_amount, amount)
    lat = c["home_lat"].to_numpy() + rng.normal(0, 0.1, m)
    lon = c["home_lon"].to_numpy() + rng.normal(0, 0.1, m)
    # trips: 4% of cards have a 3-day trip to a random city
    trip_cards = rng.random(n) < 0.04
    trip_start = rng.uniform(0, days - 3, n)
    trip_city = rng.integers(0, len(CITIES), n)
    day_f = t / DAY
    on_trip = trip_cards[cid] & (day_f >= trip_start[cid]) & (day_f < trip_start[cid] + 3)
    lat = np.where(on_trip, CITIES[trip_city[cid], 0] + rng.normal(0, 0.05, m), lat)
    lon = np.where(on_trip, CITIES[trip_city[cid], 1] + rng.normal(0, 0.05, m), lon)
    foreign = (rng.random(m) < 0.012) | (on_trip & (rng.random(m) < 0.15))
    # devices: 2% of online payments come from a "new phone" (legitimate)
    dev = np.where(c["two_devices"].to_numpy() & (rng.random(m) < 0.4), 1, 0)
    new_phone = rng.random(m) < 0.02
    device = np.where(channel == 1, cid * 10 + dev + np.where(new_phone, 5, 0), -1)
    lat = np.where(channel == 1, np.nan, lat)
    lon = np.where(channel == 1, np.nan, lon)
    return pd.DataFrame(
        {
            "card_id": cid, "time": t, "amount": np.round(amount, 2), "channel": channel,
            "category": cat, "lat": lat, "lon": lon, "is_foreign": foreign.astype(int),
            "device": device, "is_fraud": 0, "pattern": "legit",
        }
    )  # fmt: skip


def _events(rng, cards, days, mule_from_day, scale):
    rows = []
    n = len(cards)

    def add(card, t, amount, channel, category, lat, lon, foreign, device, pattern):
        rows.append((card, t, round(float(amount), 2), channel, category, lat, lon,
                     int(foreign), device, 1, pattern))  # fmt: skip

    def typical(card):
        return float(np.exp(cards.at[card, "amt_mu"]))

    def far_city(card, min_km):
        for _ in range(20):
            j = rng.integers(0, len(CITIES))
            if haversine(cards.at[card, "home_lat"], cards.at[card, "home_lon"],
                         CITIES[j, 0], CITIES[j, 1]) > min_km:  # fmt: skip
                return CITIES[j]
        return CITIES[0]

    ev = 0
    for _ in range(int(28 * scale)):  # card testing
        card, ev = int(rng.integers(0, n)), ev + 1
        t0, dev = rng.uniform(0, days - 1) * DAY, 900000 + ev
        foreign = rng.random() < 0.6
        for k in range(int(rng.integers(5, 16))):
            t0 += rng.uniform(15, 120)
            add(card, t0, rng.uniform(1, 10), 1, CAT_ID[str(rng.choice(["entertainment", "marketplace", "utilities"]))],
                np.nan, np.nan, foreign, dev, "card_testing")  # fmt: skip
        if rng.random() < 0.3:
            add(card, t0 + 60, rng.uniform(200, 1500), 1, CAT_ID["gift_cards"], np.nan, np.nan, foreign, dev, "card_testing")  # fmt: skip
    for _ in range(int(170 * scale)):  # account takeover
        card, ev = int(rng.integers(0, n)), ev + 1
        d = rng.uniform(0, days - 1)
        t0 = np.floor(d) * DAY + (rng.normal(3, 3) % 24) * 3600
        dev, foreign = 800000 + ev, rng.random() < 0.7
        for k in range(int(rng.integers(1, 4))):
            t0 += rng.uniform(300, 1800)
            add(card, t0, typical(card) * rng.uniform(4, 12), 1,
                CAT_ID[str(rng.choice(["electronics", "gift_cards", "travel"]))], np.nan, np.nan, foreign, dev, "takeover")  # fmt: skip
    for _ in range(int(110 * scale)):  # stolen card used at a terminal
        card = int(rng.integers(0, n))
        loc = far_city(card, 150)
        t0 = rng.uniform(0, days - 1) * DAY
        for k in range(int(rng.integers(2, 5))):
            t0 += rng.uniform(600, 3600)
            atm = rng.random() < 0.3
            cat = "atm" if atm else str(rng.choice(["fuel", "electronics", "clothing", "grocery"]))
            amt = float(rng.choice([500, 800, 1000])) if atm else typical(card) * rng.uniform(2, 6)
            add(card, t0, amt, 2 if atm else 0, CAT_ID[cat], loc[0] + rng.normal(0, 0.03),
                loc[1] + rng.normal(0, 0.03), rng.random() < 0.2, -1, "stolen_card")  # fmt: skip
    for _ in range(int(26 * scale)):  # mule cash-out: only in the last days
        card = int(rng.integers(0, n))
        t0 = rng.uniform(mule_from_day, days - 0.2) * DAY
        near = rng.random() < 0.5  # half of them near home, so distance alone is not enough
        lat0, lon0 = (cards.at[card, "home_lat"], cards.at[card, "home_lon"]) if near else far_city(card, 100)
        for k in range(int(rng.integers(3, 8))):
            t0 += rng.uniform(180, 600)
            add(card, t0, float(rng.choice([1000, 1500, 2000])), 2, CAT_ID["atm"],
                lat0 + rng.normal(0, 0.01), lon0 + rng.normal(0, 0.01), False, -1, "mule_cashout")  # fmt: skip
    cols = ["card_id", "time", "amount", "channel", "category", "lat", "lon", "is_foreign",
            "device", "is_fraud", "pattern"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def simulate(n_cards: int = 2500, days: int = 90, seed: int = 0, mule_from_day: int = 76) -> pd.DataFrame:
    """Returns transactions sorted by time. Column ``pattern``: legit / card_testing /
    takeover / stolen_card / mule_cashout."""
    rng = np.random.default_rng(seed)
    cards = _cards(rng, n_cards)
    legit = _legit(rng, cards, days)
    fraud = _events(rng, cards, days, mule_from_day, n_cards / 2500)
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df[df["time"] < days * DAY].sort_values("time", kind="stable").reset_index(drop=True)
    df = df.merge(cards[["card_id", "home_lat", "home_lon"]], on="card_id", how="left")
    df["day"] = (df["time"] // DAY).astype(int)
    df.insert(0, "txn_id", np.arange(len(df)))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features of each transaction from what is known at authorisation time: only earlier
    transactions of the same card (time windows never include the current transaction)."""
    d = df.sort_values(["card_id", "time"], kind="stable").copy()
    card = d["card_id"].to_numpy()
    t = d["time"].to_numpy()
    amt = d["amount"].to_numpy()
    la = np.log1p(amt)
    bounds = np.r_[0, np.flatnonzero(card[1:] != card[:-1]) + 1, len(d)]

    n = len(d)
    n_prev = np.zeros(n)
    since = np.full(n, 60 * 24 * 30.0)
    c10, c1h, c24 = np.zeros(n), np.zeros(n), np.zeros(n)
    s1h, mean_la, std_la, mean_amt = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for a, b in zip(bounds[:-1], bounds[1:]):
        tg, ag, lg = t[a:b], amt[a:b], la[a:b]
        i = np.arange(b - a)
        n_prev[a:b] = i
        if b - a > 1:
            since[a + 1 : b] = np.minimum((tg[1:] - tg[:-1]) / 60.0, since[a])
        for w, out in ((600, c10), (3600, c1h), (DAY, c24)):
            out[a:b] = i - np.searchsorted(tg, tg - w, side="left")
        cs = np.r_[0.0, np.cumsum(ag)]
        left = np.searchsorted(tg, tg - 3600, side="left")
        s1h[a:b] = cs[i] - cs[left]
        cl, cl2 = np.r_[0.0, np.cumsum(lg)], np.r_[0.0, np.cumsum(lg**2)]
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(i > 0, cl[i] / np.maximum(i, 1), np.nan)
            v = np.where(i > 0, cl2[i] / np.maximum(i, 1) - m**2, np.nan)
            mean_la[a:b], std_la[a:b] = m, np.sqrt(np.maximum(v, 0))
            mean_amt[a:b] = np.where(i > 0, cs[i] / np.maximum(i, 1), np.nan)
    d["n_prev"], d["_since"] = n_prev, since
    d["cnt_10m"], d["cnt_1h"], d["cnt_24h"] = c10, c1h, c24
    z = (la - mean_la) / (std_la + 0.3)
    d["amount_z"] = np.where(n_prev >= 3, z, 0.0)
    d["amount_ratio"] = np.where(n_prev >= 3, amt / np.maximum(mean_amt, 1.0), 1.0)
    d["sum_1h_ratio"] = np.where(n_prev >= 3, s1h / np.maximum(mean_amt, 1.0), 0.0)

    # location of the previous terminal/ATM transaction (online has no position)
    g = d.groupby("card_id")
    geo_t = d["time"].where(d["lat"].notna())
    tmp = pd.DataFrame({"lat": d["lat"], "lon": d["lon"], "gt": geo_t, "card_id": d["card_id"]})
    prev = tmp.groupby("card_id")[["lat", "lon", "gt"]].shift().groupby(tmp["card_id"]).ffill()
    has = d["lat"].notna() & prev["lat"].notna()
    dist_prev = np.where(has, haversine(d["lat"], d["lon"], prev["lat"], prev["lon"]), 0.0)
    hours = np.maximum((d["time"] - prev["gt"]).to_numpy() / 3600.0, 0.05)
    d["dist_prev_km"] = np.nan_to_num(dist_prev)
    d["speed_kmh"] = np.minimum(np.nan_to_num(dist_prev / hours), 2000.0)
    dh = haversine(d["lat"], d["lon"], d["home_lat"], d["home_lon"])
    d["dist_home_km"] = np.nan_to_num(dh)

    online = d["channel"] == 1
    first_dev = (d.assign(_o=online).loc[online].groupby(["card_id", "device"]).cumcount() == 0)
    d["new_device"] = 0
    d.loc[first_dev.index[first_dev], "new_device"] = 1
    d["new_category"] = (g.cumcount() >= 0) & (d.groupby(["card_id", "category"]).cumcount() == 0)
    d["new_category"] = d["new_category"].astype(int)

    hr = (d["time"] % DAY) / 3600.0
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    d["is_night"] = ((hr < 5) | (hr >= 23)).astype(int)
    d["log_amount"] = np.log1p(d["amount"])
    d["log_minutes_since_prev"] = np.log1p(d["_since"])
    out = d.loc[:, ["txn_id"] + FEATURES].set_index("txn_id").sort_index()
    return out.astype({c: float for c in NUMERIC})


def split_by_day(df: pd.DataFrame, train_end: int = 60, val_end: int = 75):
    """Temporal split: days [0, train_end) train, [train_end, val_end) validation (threshold
    selection), the rest test. The ``mule_cashout`` pattern occurs only in the test period."""
    d = df["day"].to_numpy()
    return d < train_end, (d >= train_end) & (d < val_end), d >= val_end
