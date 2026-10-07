"""Dados sintéticos (multi-hop e single-hop) para o perfil ``smoke`` — nada é baixado."""
from __future__ import annotations

import random

_FIRST = ["Arlen", "Bea", "Cyril", "Dara", "Elio", "Fenna", "Goran", "Hilde", "Ivo", "Juno", "Kasimir", "Lorna",
          "Mauro", "Nadia", "Osric", "Petra", "Quill", "Rosa", "Silas", "Tamsin", "Ulric", "Vera", "Wendel", "Xenia"]
_LAST = ["Voss", "Marlow", "Quent", "Halloran", "Stroud", "Achterberg", "Lindqvist", "Okafor", "Brandt", "Castell",
         "Dunmore", "Everly", "Faraday", "Grisham", "Holtz", "Ingram"]
_CITY = ["Kestrel Bay", "Port Aldane", "Marrowgate", "Silverlake Falls", "Dunhollow", "Everreach", "Graywick",
         "Lunmoor", "Thornbury", "Westvale", "Ashcombe", "Brightwater"]
_ADJ = ["Blue", "Silent", "Golden", "Broken", "Hidden", "Crimson", "Frozen", "Distant", "Iron", "Paper"]
_NOUN = ["Harbor", "Orchard", "Mirror", "Lantern", "Horizon", "Garden", "Compass", "Letter", "Tide", "Bridge"]
_COUNTRY = ["Norland", "Estavia", "Kormland", "Veloria"]


def _world(rng: random.Random, n_people: int = 60, n_films: int = 60):
    people = {}
    names = [f"{f} {l}" for f in _FIRST for l in _LAST]
    rng.shuffle(names)
    for name in names[:n_people]:
        people[name] = {"born": rng.randint(1920, 1990), "city": rng.choice(_CITY)}
    films = {}
    titles = [f"{a} {n}" for a in _ADJ for n in _NOUN]
    rng.shuffle(titles)
    plist = list(people)
    for t in titles[:n_films]:
        films[t] = {"director": rng.choice(plist), "year": rng.randint(1950, 2020), "city": rng.choice(_CITY)}
    cities = {c: rng.choice(_COUNTRY) for c in _CITY}
    return people, films, cities


def _paragraphs(people, films, cities):
    paras = {}
    for p, v in people.items():
        paras[p] = (f"{p} (born {v['born']}) is a film director. {p} was born in {v['city']}. "
                    f"{p.split()[0]} began working in cinema at a young age.")
    for f, v in films.items():
        paras[f] = (f"{f} is a {v['year']} drama film directed by {v['director']}. "
                    f"The film was shot in {v['city']} and received mixed reviews.")
    for c, country in cities.items():
        paras[c] = f"{c} is a coastal city in {country}. It is known for its old harbour and markets."
    return paras


def make_toy(name: str, n_eval: int, n_train: int, n_extra: int, seed: int = 13):
    rng = random.Random(seed)
    people, films, cities = _world(rng)
    paras = _paragraphs(people, films, cities)
    all_titles = list(paras)

    def record(i: int) -> dict:
        kind = "single" if name == "toy_single" else ("comparison" if i % 3 == 2 else "bridge")
        if kind == "single":
            f = rng.choice(list(films))
            q, ans, sup = f"Who directed the film {f}?", films[f]["director"], [f]
        elif kind == "bridge":
            f = rng.choice(list(films))
            d = films[f]["director"]
            q, ans, sup = f"In which city was the director of {f} born?", people[d]["city"], [f, d]
        else:
            a, b = rng.sample(list(people), 2)
            while people[a]["born"] == people[b]["born"]:
                a, b = rng.sample(list(people), 2)
            ans = a if people[a]["born"] < people[b]["born"] else b
            q, sup = f"Who was born first, {a} or {b}?", [a, b]
        distract = [t for t in rng.sample(all_titles, 12) if t not in sup][: 10 - len(sup)]
        titles = sup + distract
        rng.shuffle(titles)
        return {"id": f"{name}-{i}", "dataset": name, "question": q, "answers": [ans], "supporting_titles": sup,
                "type": kind, "level": "", "hops": len(sup),
                "paragraphs": [{"title": t, "text": paras[t]} for t in titles]}

    recs = [record(i) for i in range(n_eval + n_train + n_extra)]
    return recs[:n_eval], recs[n_eval:n_eval + n_train], recs[n_eval + n_train:]
