"""Testes do DynaKRAG: validade (Tabela 5), transições, Eq. 1, features e controlador."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragbench.config import load_config  # noqa: E402
from ragbench.core.llm import LLM  # noqa: E402
from ragbench.core.retrieval import BM25Index, Retriever  # noqa: E402
from ragbench.core.toy import make_toy  # noqa: E402
from ragbench.core.datasets import build_corpus  # noqa: E402
from ragbench.dynakrag.controller import LearnedController, UniformController  # noqa: E402
from ragbench.dynakrag.env import DynaEnv, detect_bridges  # noqa: E402
from ragbench.dynakrag.features import CONT_FEATURE_NAMES, VALUE_FEATURE_NAMES, continuation_features, value_features  # noqa: E402
from ragbench.dynakrag.inference import VARIANTS, run_episode  # noqa: E402
from ragbench.dynakrag.state import (LEGACY_CONTROL, STOP, SUFF, control_signature, control_slug, row_control,  # noqa: E402
                                     support_recall)
from ragbench.dynakrag.validity import stop_allowed, valid_actions  # noqa: E402


def _env(tmp_path, *overrides):
    cfg = load_config("smoke", [f"paths.cache={tmp_path}/c.sqlite", *overrides], root=str(tmp_path))
    ev, tr, ex = make_toy("toy_multihop", 6, 6, 4, 13)
    corpus = build_corpus(ev, tr, ex)
    r = Retriever(corpus, BM25Index().build(corpus))
    return cfg, DynaEnv(cfg, "toy_multihop", r, LLM(cfg)), ev


def test_initial_state_and_validity(tmp_path):
    cfg, env, ev = _env(tmp_path)
    s = env.init_state(ev[0])
    # sem evidência só dá para avançar a fronteira; parar ainda não é válido (a sonda não rodou)
    assert valid_actions(s, env, 4) == ["retrieve_more"]
    assert valid_actions(s, env, 4, stop_rule="always") == ["retrieve_more", STOP]     # regra das versões antigas
    s1 = env.step(s, "retrieve_more")
    assert len(s1.docs) == 3 and s1.n_ret == 1 and s1.frontier >= 3
    assert len(s.docs) == 0  # transição sem efeito colateral
    A1 = valid_actions(s1, env, 4)
    assert "rewrite_query" in A1 and "sufficiency_check" in A1 and "gap_query" not in A1


def test_sufficiency_not_redundant_and_gap(tmp_path):
    cfg, env, ev = _env(tmp_path)
    s = env.step(env.init_state(ev[0]), "retrieve_more")
    s2 = env.step(s, "sufficiency_check")
    assert s2.n_checks == 1 and s2.docs_at_check == len(s2.docs)
    assert "sufficiency_check" not in valid_actions(s2, env, 4)  # sonda redundante no mesmo estado
    if s2.suff == 0:
        assert s2.gap and "gap_query" in valid_actions(s2, env, 4)


def test_stop_only_after_sufficient_probe(tmp_path):
    """Tabelas 6–8 do artigo: sem veredito "suficiente" ninguém para antes do limite."""
    cfg, env, ev = _env(tmp_path)
    s = env.step(env.init_state(ev[0]), "retrieve_more")
    assert STOP not in valid_actions(s, env, 4) and not stop_allowed(s, "sufficient")
    no = env.step(s, SUFF)
    no.suff, no.gap = 0, "missing fact"
    assert STOP not in valid_actions(no, env, 4)                 # sonda disse que falta evidência: segue adquirindo
    assert STOP in valid_actions(no, env, 4, stop_rule="probe")  # regra alternativa: qualquer veredito libera
    yes = env.step(s, SUFF)
    yes.suff, yes.gap = 1, ""
    assert STOP in valid_actions(yes, env, 4)
    # quando não sobra nenhuma ação executável, parar é a única saída mesmo sem "suficiente"
    assert valid_actions(no, env, 4, retrieval_cap=1) == [STOP]


def test_probe_runs_once_per_trajectory(tmp_path):
    """Tabela 7 do artigo: 0,98–1,00 sondas por pergunta — a sonda não volta depois de novas evidências."""
    cfg, env, ev = _env(tmp_path)
    s = env.step(env.step(env.init_state(ev[0]), "retrieve_more"), SUFF)
    s2 = env.step(s, "retrieve_more")                            # chegou evidência nova
    assert len(s2.docs) > len(s.docs)
    assert SUFF not in valid_actions(s2, env, 4)
    assert SUFF in valid_actions(s2, env, 4, max_probes=None)    # sem limite (versões antigas) ela voltaria
    assert SUFF in valid_actions(s2, env, 4, max_probes=2)


def test_one_step_probe_is_a_single_call_with_the_gap(tmp_path):
    """Tabela 6 do artigo: com limite de 1 retrieval são 3,000 chamadas ao LLM (sonda + compressão + resposta)."""
    cfg, env, ev = _env(tmp_path)
    assert env.probe_mode == "one_step"
    s = env.step(env.init_state(ev[0]), "retrieve_more")
    s2 = env.step(s, SUFF)
    assert s2.n_llm - s.n_llm == 1
    if s2.suff == 0:
        assert s2.gap                                            # a lacuna veio na mesma chamada

    class _PickFirst:
        name = "first"

        def choose(self, st, A, e):
            non_stop = [a for a in A if a != STOP]
            return (non_stop[0], {}) if non_stop else (STOP, {"forced": True})

    res = run_episode(env, _PickFirst(), ev[0], VARIANTS["dynakrag_cap1"], 4)
    assert res["actions"] == ["retrieve_more", SUFF] and res["n_ret"] == 1
    assert res["n_llm_actions"] == 3                             # sonda + compressão + resposta
    assert res["control"] == control_signature(cfg) and res["n_checks"] == 1


def test_no_probe_variant_never_stops_before_the_cap(tmp_path):
    """Tabela 8 do artigo: sem a sonda, "none selects stop_answer" e são 4,000 passos por pergunta."""
    cfg, env, ev = _env(tmp_path)

    class _WantsToStop:                                          # controlador que para sempre que puder
        name = "stopper"

        def choose(self, st, A, e):
            return (STOP, {}) if STOP in A else (A[0], {})

    for rec in ev[:4]:
        res = run_episode(env, _WantsToStop(), rec, VARIANTS["dynakrag_no_sufficiency"], 4)
        assert SUFF not in res["actions"] and res["steps"] == 4 and res["stopped_by"] == "cap"


def test_control_signature_and_controller_file(tmp_path):
    from ragbench.dynakrag.controller import controller_path

    cfg, env, ev = _env(tmp_path)
    assert control_signature(cfg) == "stop=sufficient;probes=1;probe=one_step" == env.control
    assert controller_path(cfg, "toy_multihop").endswith("controller_n16__suf-p1-one.pkl")
    old, env_old, _ = _env(tmp_path, "dynakrag.controller.stop_rule=always", "dynakrag.controller.max_probes=null",
                           "dynakrag.probe=two_step")
    assert control_signature(old) == LEGACY_CONTROL and control_slug(old) == ""
    assert controller_path(old, "toy_multihop").endswith("controller_n16.pkl")     # o nome que as versões antigas usavam
    assert env_old.stop_rule == "always" and env_old.max_probes is None and env_old.probe_mode == "two_step"
    assert row_control({}) == LEGACY_CONTROL and row_control({"control": "x"}) == "x"


def test_compression_drops_lines_without_text(tmp_path):
    cfg, env, ev = _env(tmp_path)
    s = env.step(env.init_state(ev[0]), "retrieve_more")

    class _LLM:
        def __init__(self, out):
            self.out = out

        def chat(self, *a, **k):
            return self.out

    env.llm = _LLM("- - -\n- - -\n| |")
    evidence, raw = env.compress(s, s.docs)
    assert evidence == env.passages(s.docs)                      # nada aproveitável: responde com os documentos
    env.llm = _LLM("- - -\n- Arlen Voss was born in Kestrel Bay.\n- -")
    assert env.compress(s, s.docs)[0] == "- Arlen Voss was born in Kestrel Bay."


def test_caps(tmp_path):
    cfg, env, ev = _env(tmp_path)
    s = env.step(env.init_state(ev[0]), "retrieve_more")
    A = valid_actions(s, env, 4, retrieval_cap=1)
    assert not ({"retrieve_more", "gap_query", "rewrite_query", "bridge_entity_expand"} & set(A))
    s.history = ["x"] * 4
    assert valid_actions(s, env, 4) == [STOP]


def test_support_recall():
    assert support_recall(["A", "b "], ["a", "B", "C"]) == 2 / 3
    assert support_recall([], []) == 0.0


def test_bridge_detection():
    docs = [{"title": "Blue Harbor", "text": "Blue Harbor is a film directed by Arlen Voss. It was shot in Kestrel Bay."}]
    b = detect_bridges(docs, "Where was the director of Blue Harbor born?", [])
    assert "Arlen Voss" in b and "Blue Harbor" not in b


def test_features_shapes(tmp_path):
    cfg, env, ev = _env(tmp_path)
    s = env.step(env.init_state(ev[0]), "retrieve_more")
    A = valid_actions(s, env, 4)
    X = value_features(s, A, env, 4)
    assert X.shape == (len(A), len(VALUE_FEATURE_NAMES))
    assert continuation_features(s, env, 4).shape == (len(CONT_FEATURE_NAMES),)


class _ConstModel:
    def __init__(self, vals):
        self.vals = vals

    def predict(self, X):
        import numpy as np
        return np.asarray(self.vals[: len(X)] + [0.0] * max(0, len(X) - len(self.vals)))


class _Cont:
    classes_ = [0, 1]

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        import numpy as np
        return np.asarray([[1 - self.p, self.p]])


def test_continuation_suppresses_stop(tmp_path):
    cfg, env, ev = _env(tmp_path)
    s = env.step(env.step(env.init_state(ev[0]), "retrieve_more"), SUFF)
    s.suff, s.gap = 1, ""                    # sonda disse "suficiente": parar passa a ser válido
    A = valid_actions(s, env, 4)
    assert A[-1] == STOP and len(A) > 1
    vals = [0.1] * (len(A) - 1) + [0.9]      # stop (último) tem o maior valor
    ctrl = LearnedController(_ConstModel(vals), _Cont(0.8), tau=0.5)
    a, info = ctrl.choose(s, A, env)
    assert a != STOP and info.get("stop_suppressed")
    ctrl2 = LearnedController(_ConstModel(vals), _Cont(0.2), tau=0.5)
    assert ctrl2.choose(s, A, env)[0] == STOP


def test_uniform_runs_to_cap(tmp_path):
    cfg, env, ev = _env(tmp_path)
    res = run_episode(env, UniformController(13), ev[0], VARIANTS["dynakrag_uniform"], 4)
    assert res["steps"] == 4 or res["stopped_by"] == "cap"
    assert res["pred"] is not None and "trace" in res


def test_threshold_calibration_only_where_stopping_is_allowed():
    import numpy as np

    from ragbench.dynakrag.training import DEFAULT_TAU, MIN_STOP_PREFIXES, calibrate_threshold

    n = 40
    groups = np.repeat(np.arange(n // 2), 2)
    f1_now = np.tile([0.0, 1.0], n // 2).astype(np.float32)      # parar no 1º prefixo é ruim, no 2º é bom
    f1_final = np.full(n, 0.5, dtype=np.float32)
    proba = np.tile([0.3, 0.3], n // 2)
    base = {"group": groups, "f1_now": f1_now, "f1_final": f1_final}
    # parada permitida em todos os prefixos (regra antiga): qualquer τ > 0,3 para no 1º prefixo, o pior
    tau_all, info_all = calibrate_threshold(proba, {**base, "can_stop": np.ones(n, dtype=np.int64)})
    assert info_all["n_can_stop"] == n and tau_all <= 0.3
    # parada permitida só no 2º prefixo de cada pergunta: agora parar ali é o melhor
    can = np.tile([0, 1], n // 2)
    tau, info = calibrate_threshold(proba, {**base, "can_stop": can})
    assert info["n_can_stop"] == n // 2 >= MIN_STOP_PREFIXES and tau > 0.3
    assert abs(info["curve"][tau] - 1.0) < 1e-9
    # quase nenhum prefixo com parada permitida: não há o que calibrar
    few = np.zeros(n, dtype=np.int64)
    few[:3] = 1
    tau_few, info_few = calibrate_threshold(proba, {**base, "can_stop": few})
    assert tau_few == DEFAULT_TAU and "não calibrado" in info_few["note"]


def test_results_from_other_control_rules_are_archived(tmp_path):
    from ragbench.core.io import read_jsonl, write_jsonl
    from ragbench.dynakrag import bench as B

    cfg, env, ev = _env(tmp_path)
    path = B.result_path(cfg, "toy_multihop", "dynakrag")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_jsonl(path, [{"id": "q1", "pred": "x", "em": 0.0}])                 # linha antiga: sem o campo control
    assert B.stale_control(cfg, "toy_multihop", "dynakrag") == LEGACY_CONTROL
    assert B.stale_control(cfg, "toy_multihop", "fixed_k3") is None           # baselines não dependem das regras
    dst = B.archive_stale(cfg, "toy_multihop", "dynakrag")
    assert dst and os.path.exists(dst) and not os.path.exists(path)
    assert read_jsonl(dst)[0]["id"] == "q1" and "controle_anterior" in dst
    assert B.archive_stale(cfg, "toy_multihop", "dynakrag") is None           # nada mais a guardar
    write_jsonl(path, [{"id": "q1", "pred": "x", "control": control_signature(cfg)}])
    assert B.stale_control(cfg, "toy_multihop", "dynakrag") is None           # resultado das regras atuais fica


def test_old_results_are_flagged_and_rerun(smoke_root, tmp_path):
    """Resultado e controlador gerados com as regras antigas: avisados nas tabelas e no painel, guardados e refeitos."""
    import glob
    import json
    import shutil

    from ragbench import cli
    from ragbench.core import datasets as D
    from ragbench.core import retrieval as R
    from ragbench.core.io import read_jsonl, write_jsonl
    from ragbench.dynakrag import bench as B
    from ragbench.dynakrag.controller import controller_path
    from ragbench.tables import build_tables
    from ragbench.ui import data as UI

    root = str(tmp_path / "copia")
    shutil.copytree(smoke_root, root)
    cfg = load_config("smoke", [], root=root)
    path = B.result_path(cfg, "toy_multihop", "dynakrag")
    rows = read_jsonl(path)
    assert rows and all(r["control"] == control_signature(cfg) for r in rows)
    # volta no tempo: linhas sem o campo control e controlador com o nome antigo
    write_jsonl(path, [{k: v for k, v in r.items() if k != "control"} for r in rows])
    new_ctrl = controller_path(cfg, "toy_multihop")
    os.replace(new_ctrl, new_ctrl.replace("__suf-p1-one", ""))
    assert B.stale_control(cfg, "toy_multihop", "dynakrag") == LEGACY_CONTROL and B.controller_missing(cfg, "toy_multihop")

    dk = {t["id"]: t for t in build_tables(cfg)}["dynakrag"]
    assert dk["rows"][-1][1][0] == "DynaKRAG †" and "regras de controle antigas" in dk["warn"]
    plan = UI.plan(cfg, ["toy_multihop"], ["dynakrag"], [1])
    stale = {j["name"] for j in plan["jobs"] if j["stale"]}
    assert stale == {"train_controller", "dynakrag"} and plan["n_stale"] == 2 and plan["n_todo"] == 2
    assert all(j["state"] == "pending" for j in plan["jobs"] if j["stale"])

    D.clear_cache()
    R.clear_cache()
    cli.main(["bench", "--profile", "smoke", "--root", root, "--paper", "dynakrag", "--tier", "1", "--no-report"], _exit=False)
    D.clear_cache()
    R.clear_cache()
    fresh = read_jsonl(path)
    assert len(fresh) == len(rows) and all(r["control"] == control_signature(cfg) for r in fresh)
    kept = glob.glob(os.path.join(cfg["paths"]["logs"], "controle_anterior", "smoke", "dynakrag", "toy_multihop", "dynakrag__*.jsonl"))
    assert len(kept) == 1 and len(read_jsonl(kept[0])) == len(rows)
    assert os.path.exists(new_ctrl) and os.path.exists(new_ctrl.replace("__suf-p1-one", ""))   # o antigo continua lá
    assert UI.plan(cfg, ["toy_multihop"], ["dynakrag"], [1])["n_stale"] == 0
    assert "warn" not in {t["id"]: t for t in build_tables(cfg)}["dynakrag"]
    with open(os.path.join(cfg["paths"]["results"], "status.json"), encoding="utf-8") as f:
        assert json.load(f)["dynakrag/toy_multihop/dynakrag"]["state"] == "done"
