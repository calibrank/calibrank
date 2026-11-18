"""CPU tests of the controls: window formats, the cross-entropy objective,
LoRA on factors, the side of square layers, and the output-Gram ridge."""

import torch

from conftest import random_orthonormal
from fsc import data
from fsc.arch import discover_units
from fsc.init import OutBasis
from fsc.project import LoraBank, merge_factors, swapped
from fsc.train import TrainConfig, ce_and_grad, train


def _windows(n, t=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 97, (n, t), generator=g)


def test_segment_windows_tile_with_and_without_bos():
    L, T = 7, 32                       # 4 segments of 7 tokens plus separators
    segs = [[100 + i] * L for i in range(8)]
    w = data._segment_windows(segs, 2, T, bos=1, eos=2)
    assert w.shape == (2, T)
    row = w[0].tolist()
    assert row[0] == 1 and row.count(2) == 3 and row.count(1) == 1
    w = data._segment_windows(segs, 2, T, bos=None, eos=2)
    assert w.shape == (2, T) and w[0].tolist().count(2) == 4 and w[0, -1] == 2


def test_ce_gradient_matches_autograd():
    torch.manual_seed(0)
    x = torch.randint(0, 11, (2, 9))
    s = torch.randn(2, 9, 11, requires_grad=True)
    scale = 1.0 / (2 * 8)
    total, g = ce_and_grad(s.detach(), x, scale, chunk=4)
    ref = torch.nn.functional.cross_entropy(s[:, :-1].reshape(-1, 11), x[:, 1:].reshape(-1),
                                            reduction="sum")
    ref.backward()
    assert abs(total - float(ref)) < 1e-4
    assert torch.allclose(g, s.grad * scale, atol=1e-6)


def _oblique_like_parts(model, units):
    """Exact factors of random removed subspaces, as (path, A, B, bias) with shared A."""
    parts = []
    for i, u in enumerate(units):
        U = random_orthonormal(u.d, max(1, u.d // 3), i)
        parts += merge_factors(model, u, U, torch.float32)[1]
    return parts


def test_lora_bank_starts_at_the_factors_and_merges_exactly(tiny_llama):
    units = discover_units(tiny_llama)
    parts = _oblique_like_parts(tiny_llama, units)
    x = _windows(2)
    with swapped(tiny_llama, parts, torch.float32):
        ref = tiny_llama(input_ids=x).logits
    bank = LoraBank(parts, r=2, alpha=4.0, dtype=torch.float32)
    bank.attach(tiny_llama)
    assert torch.allclose(tiny_llama(input_ids=x).logits, ref, atol=1e-5)
    # Perturb every LoRA matrix: the merged factors must reproduce the LoRA forward.
    with torch.no_grad():
        for d in (bank.la2, bank.lb2):
            for v in d.values():
                v.normal_(0, 0.1)
    lora_out = tiny_llama(input_ids=x).logits
    merged = bank.merged_parts(torch.float32)
    bank.detach(tiny_llama)
    with swapped(tiny_llama, merged, torch.float32):
        assert torch.allclose(tiny_llama(input_ids=x).logits, lora_out, atol=1e-4)
    # Tied members share one merged A.
    a_ids = {id(a) for _, a, _, _ in merged}
    assert len(a_ids) < len(merged)


def test_lora_cross_entropy_training_lowers_validation_ce(tiny_opt, tmp_path):
    units = discover_units(tiny_opt)
    bank = LoraBank(_oblique_like_parts(tiny_opt, units), r=2, alpha=4.0, dtype=torch.float32)
    bank.attach(tiny_opt)
    tr, va = _windows(16), _windows(4, seed=1)
    cfg = TrainConfig(mode="lora", epochs=3, lr=1e-2, batch_size=4, micro_batch=2, patience=5,
                      drop_p=0.0, lam_ort=0.0, loss="ce")
    out = train(tiny_opt, bank, tr, va, cfg, str(tmp_path), "cpu", torch.float32, log=lambda *_: None)
    assert out["criterion"] == "ce"
    assert out["best_val_kl"] < out["history"][0]["val_ce"]
    assert (tmp_path / "last_params.pt").exists()
    bank.detach(tiny_opt)


def test_square_side_out_moves_attention_output(tiny_llama):
    by = {u.name: u for u in discover_units(tiny_llama, square_side="out")}
    assert by["L0.o"].side == "out"
    by = {u.name: u for u in discover_units(tiny_llama)}
    assert by["L0.o"].side == "in"
    discover_units(tiny_llama)     # leave the default for the other tests


def test_output_ridge_keeps_the_eigenvectors():
    torch.manual_seed(0)
    M = torch.randn(6, 6, dtype=torch.float64)
    G = M @ M.T
    a, b = OutBasis(G), OutBasis(G, ridge=0.01)
    assert torch.allclose(a.removed(2).abs(), b.removed(2).abs(), atol=1e-8)


def test_state_dir_keeps_resume_state_out_of_the_run(tiny_llama, tmp_path):
    from fsc.project import ProjectorBank
    units = discover_units(tiny_llama)
    bases = {u.name: random_orthonormal(u.d, max(1, u.d // 3), i).float() for i, u in enumerate(units)}
    bank = ProjectorBank(units, bases)
    bank.attach(tiny_llama)
    run, state = tmp_path / "run", tmp_path / "state"
    run.mkdir()
    cfg = TrainConfig(epochs=2, lr=1e-2, batch_size=4, micro_batch=2, patience=5, drop_p=0.0)
    train(tiny_llama, bank, _windows(8), _windows(2, seed=1), cfg, str(run), "cpu", torch.float32,
          log=lambda *_: None, state_dir=str(state))
    assert (state / "train_state.pt").exists() and not (run / "train_state.pt").exists()
    assert (run / "best_params.pt").exists() and (run / "last_params.pt").exists()
    bank.detach(tiny_llama)


def test_span_pools_cap_documents_and_keep_validation_disjoint():
    L, cap = 7, data.SEGMENT_CAP
    docs = [[d] * (L * (cap + 10)) for d in range(10)]   # every document exceeds the cap
    tr, va, info = data._span_pools(iter(docs), L, 2 * cap, 10, seed=0)
    assert len(tr) == 2 * cap and len(va) == 10
    assert all(len(s) == L for s in tr + va)
    tr_docs, va_docs = {s[0] for s in tr}, {s[0] for s in va}
    assert not (tr_docs & va_docs)
    per_doc = {d: sum(s[0] == d for s in tr) for d in tr_docs}
    assert max(per_doc.values()) <= cap


def test_shared_factor_cache_changes_nothing(tiny_llama):
    """Tied members reuse z = A x; the output is the same as computing it per member."""
    units = discover_units(tiny_llama)
    parts = _oblique_like_parts(tiny_llama, units)
    x = _windows(2)
    with swapped(tiny_llama, parts, torch.float32):
        a = tiny_llama(input_ids=x).logits
    from fsc import project
    uncached = [(p, A.clone(), B, b) for p, A, B, b in parts]     # distinct A objects: no tie
    with project.swapped(tiny_llama, uncached, torch.float32):
        b = tiny_llama(input_ids=x).logits
    assert torch.allclose(a, b, atol=1e-6)
