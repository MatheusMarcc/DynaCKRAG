"""Gera um mini-dataset sintético no formato OFICIAL do HotpotQA (perguntas de ponte de 2 saltos).

Usado pelo comando smoke para testar tudo sem baixar nada. Nomes inventados.
"""
from __future__ import annotations

import json
import os
import random

_A = ["Al", "Bren", "Cor", "Dal", "Ev", "Fen", "Gar", "Hal", "Ist", "Jor", "Kel", "Lor", "Mar", "Nor", "Os"]
_B = ["der", "wyn", "ric", "ton", "vel", "mar", "den", "sa", "lin", "ven"]
_SUR = ["Voss", "Hale", "Crane", "Morrow", "Pike", "Thorne", "Wade", "Ashby", "Quill", "Rowe"]
_CITY = ["moor", "haven", "field", "ford", "gate", "stead", "bury", "wick"]
_COUNTRY = ["Kestria", "Valonia", "Durmia", "Ostland", "Pelaria", "Zenthia", "Morvania", "Talistan"]
_JOB = ["sculptor", "painter", "novelist", "architect", "composer", "botanist"]


def _name(rng):
    return f"{rng.choice(_A)}{rng.choice(_B)} {rng.choice(_SUR)}"


def make_toy(out_dir: str, n_train: int = 80, n_dev: int = 40, seed: int = 13) -> tuple[str, str]:
    rng = random.Random(seed)
    n = n_train + n_dev
    people = list(dict.fromkeys(_name(rng) for _ in range(n * 4)))[:n]
    cities = list(dict.fromkeys(f"{rng.choice(_A)}{rng.choice(_B)}{rng.choice(_CITY)}" for _ in range(n * 4)))[:n]
    paras, items = {}, []
    for i, (p, c) in enumerate(zip(people, cities)):
        country, job = rng.choice(_COUNTRY), rng.choice(_JOB)
        paras[p] = [f"{p} is a {job} known for early work in the {1900 + rng.randint(0, 99)}s.",
                    f"{p} was born in {c}."]
        paras[c] = [f"{c} is a small city in {country}.", f"It lies near the {rng.choice(_SUR)} river."]
        items.append({"_id": f"toy{i:04d}", "question": f"In which country is the birthplace of the {job} {p}?",
                      "answer": country, "type": "bridge", "level": "easy",
                      "supporting_facts": [[p, 1], [c, 0]], "gold": [p, c]})
    titles = list(paras)
    for it in items:
        distract = rng.sample([t for t in titles if t not in it["gold"]], 8)
        ctx = [[t, paras[t]] for t in it["gold"] + distract]
        rng.shuffle(ctx)
        it["context"] = ctx
        del it["gold"]
    os.makedirs(out_dir, exist_ok=True)
    tr, dv = os.path.join(out_dir, "toy_train.json"), os.path.join(out_dir, "toy_dev.json")
    for path, part in ((tr, items[:n_train]), (dv, items[n_train:])):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(part, f)
    return tr, dv
