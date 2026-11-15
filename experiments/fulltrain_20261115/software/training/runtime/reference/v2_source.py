"""Paired fine-tie pilot: seed0, 20 epochs, fit8263 -> existing tune459.

No confirmation or official-test scoring. No cache rebuilding or overwriting.
Training/model mathematics and ranking are preserved from the v1 pilot.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

POLICIES = ("legacy_raster", "spatial")
PILOT_VARIANTS = ("original_full", "joint_full", "joint_paths_off")
PAIR_SCHEMA = "pare.fine-tie-paired-cache.v1"
PRESERVED = ("scene", "nu", "c_emb", "c_mu", "c_t", "c_tf", "c_n", "L", "c_star")
V1_MODEL_SHA = "3fa643703187705c67892d7fb906842627b09d22822db2d6f94a5c430654cff6"
V1_BOUND_SHA = "6caba296b574e2ddd332adbcf6e97c19a37686719a9d0e3b74c353564d648c45"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def j(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def checked_pair(torch, audit_path, expected_audit_sha, requested_path, policy, parent_path):
    """Validate both NEW caches directly; never call schema-2 validate_cache.

    Caller supplies the reviewed manifest hash. The original parent is loaded
    only to check population identity and unchanged tensors, never for training.
    """
    if sha(audit_path) != expected_audit_sha:
        raise ValueError("Paired-cache audit hash differs from reviewed input")
    audit = j(audit_path)
    contract = audit["contract"]
    if audit["schema_version"] != PAIR_SCHEMA or contract["schema_version"] != PAIR_SCHEMA:
        raise ValueError("Wrong paired-cache schema")
    if contract["population"] != "train_9181" or contract["fine_crop_half_pixels"] != 12:
        raise ValueError("Only the full train9181, 24x24 paired cache is eligible")
    if contract["fine_slots"] != 8 or set(audit["outputs"]) != set(POLICIES):
        raise ValueError("Expected both policies and eight fine slots")
    parent_sha = sha(parent_path)
    if parent_sha != contract["parent_cache_sha256"]:
        raise ValueError("Wrong parent cache")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    parent_ids = parent["ids"]
    if len(parent_ids) != 9181 or len(set(parent_ids)) != 9181:
        raise ValueError("Parent must contain exactly 9181 unique train IDs")
    if contract["parent_extraction_signature"] != parent["provenance"]["signature"]:
        raise ValueError("Parent extraction lineage differs")
    preserved = {key: tensor_sha(parent[key]) for key in PRESERVED}
    if contract["preserved_tensor_sha256"] != preserved:
        raise ValueError("Recorded unchanged fields do not match the original cache")
    populations = {}
    for name in ("train", "fit", "val"):
        item = contract["population_id_files"][name]
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"Population ID file changed: {name}")
        populations[name] = j(item["path"])
    if populations["train"] != parent_ids:
        raise ValueError("Train-ID order differs from the original cache")
    fit, val = set(populations["fit"]), set(populations["val"])
    if len(fit) != len(populations["fit"]) or len(fit) != 8263 or len(val) != len(populations["val"]) or len(val) != 918:
        raise ValueError("Wrong fit/val population")
    if fit & val or fit | val != set(parent_ids):
        raise ValueError("fit/val do not partition the parent train population")
    caches, bindings = {}, {}
    for name in POLICIES:
        item = audit["outputs"][name]
        path = (Path(audit_path).resolve().parent / item["path"]).resolve()
        if path.parent != Path(audit_path).resolve().parent:
            raise ValueError("Cache must be a sibling of its audit manifest")
        if sha(path) != item["sha256"]:
            raise ValueError(f"Paired cache file hash mismatch: {name}")
        cache = torch.load(path, map_location="cpu", weights_only=True)
        provenance = cache["provenance"]
        signature = provenance["signature"]
        if provenance["schema_version"] != PAIR_SCHEMA:
            raise ValueError(f"Wrong active cache schema: {name}")
        if signature != dict(contract, fine_tie_selection=name):
            raise ValueError(f"New cache signature differs from its audited contract: {name}")
        if objsha(signature) != provenance["signature_sha256"] or objsha(signature) != item["signature_sha256"]:
            raise ValueError(f"New cache signature checksum mismatch: {name}")
        if cache["ids"] != parent_ids or provenance["ids_sha256"] != objsha(parent_ids):
            raise ValueError(f"Wrong train population or order: {name}")
        if signature["ids_sha256"] != objsha(parent_ids) or (cache["D"], cache["Mc"], cache["Mf"]) != (512, 6, 8):
            raise ValueError(f"Wrong cache dimensions or ID binding: {name}")
        if provenance["image_file_sha256"] != parent["provenance"]["image_file_sha256"]:
            raise ValueError(f"Input image hashes differ from the original cache: {name}")
        actual = {key: tensor_sha(cache[key]) for key in PRESERVED}
        if actual != preserved:
            raise ValueError(f"A scene/coarse/nu/L/c_star tensor changed: {name}")
        # The full-file hash above binds all tensors. These checks reject invalid
        # structure rather than silently passing malformed counts to move_docs.
        for branch, slots in (("c", 6), ("f", 8)):
            counts = cache[branch+"_n"]
            if counts.dtype != torch.long or tuple(counts.shape) != (9181,) or not ((counts >= 0) & (counts <= slots)).all():
                raise ValueError(f"Invalid region counts: {name}/{branch}")
        caches[name] = cache
        bindings[name] = {"path": str(path), "sha256": item["sha256"],
                          "signature_sha256": item["signature_sha256"], "preserved_tensor_sha256": actual}
    if Path(requested_path).resolve() != Path(bindings[policy]["path"]):
        raise ValueError("--cache does not match --policy in the reviewed audit")
    if bindings["legacy_raster"]["signature_sha256"] == bindings["spatial"]["signature_sha256"]:
        raise ValueError("The two policies must have distinct new signatures")
    return caches[policy], {"audit_path": str(Path(audit_path).resolve()), "audit_sha256": expected_audit_sha,
                           "parent_cache_sha256": parent_sha, "caches": bindings,
                           "population_id_files": contract["population_id_files"]}, populations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--paired-audit", required=True, type=Path)
    ap.add_argument("--paired-audit-sha256", required=True)
    ap.add_argument("--variant", required=True, choices=PILOT_VARIANTS)
    ap.add_argument("--gpu", required=True)
    args = ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    root = Path(__file__).resolve().parent
    workspace = root.parent
    study = workspace / "simplified_confirm_20261109_v1"
    validation = workspace / "method_validation_20261111_v1"
    code_hashes = {"runner": sha(__file__), "model": sha(root/"joint_model.py"),
                   "protocol": sha(root/"PROTOCOL.json"), "v1_inputs_manifest": sha(root/"INPUTS_BOUND.json")}
    if code_hashes["model"] != V1_MODEL_SHA or code_hashes["v1_inputs_manifest"] != V1_BOUND_SHA:
        raise ValueError("The v1 model and all v1 input bindings must be byte-identical")
    protocol = j(root/"PROTOCOL.json")
    if (protocol["epochs"], protocol["learning_rate"], protocol["kappa"], protocol["batch_size"], protocol["seeds"]) != (20, .002, .03, 64, [0]):
        raise ValueError("The frozen v1 training recipe changed")
    if tuple(protocol["policies"]) != POLICIES or tuple(protocol["variants"]) != PILOT_VARIANTS:
        raise ValueError("The six-run pilot grid changed")
    bound = j(root/"INPUTS_BOUND.json")
    for name, item in bound.items():
        if sha(item["path"]) != item["sha256"] or Path(item["path"]).stat().st_size != item["bytes"]:
            raise ValueError(f"V1 input binding changed: {name}")
        if (workspace/name).resolve() != Path(item["path"]).resolve():
            raise ValueError("Install this study alongside its original hash-bound input directories")
    sys.path[:0] = [str(study/"base_code"), str(study/"code"), str(validation), str(root)]
    import torch
    from torch.nn import functional as F
    from joint_model import JointPARE
    from train_saqre import load_captions, move_docs
    from study_core import state_sha

    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = root/"runs"/args.policy/args.variant
    out.mkdir(parents=True, exist_ok=False)
    try:
        parent_path = workspace/"formal_experiments_20261103/results/vitb32/cache_physics_train.pt"
        cache, paired_binding, populations = checked_pair(torch, args.paired_audit, args.paired_audit_sha256,
                                                          args.cache, args.policy, parent_path)
        write(out/"INPUTS_VERIFIED.json", {"v1_inputs": bound, "paired_cache_inputs": paired_binding, "code_sha256": code_hashes})
        caps_path = workspace/"data/ret_clean_20261007/Caption_train.csv"
        fit_path = workspace/"data/ret_clean_20261007/fit_ids.json"
        text_path = study/"shared/vitb32/text_embeddings.pt"
        split_path = validation/"split_manifest.json"
        caps = load_captions(caps_path)
        ids = cache["ids"]
        idrow = {name: i for i, name in enumerate(ids)}
        allpairs = [(idrow[name], caption) for name, captions in caps.items() if name in idrow for caption in captions]
        texts = sorted({caption for row, caption in allpairs})
        text = torch.load(text_path, map_location="cpu", weights_only=True)
        hashes = [hashlib.sha256(caption.encode()).hexdigest() for caption in texts]
        assert objsha(hashes) == text["training_text_sha256"]
        caprow = {caption: i for i, caption in enumerate(texts)}
        qmatrix = text["train"].cuda()
        del text  # No other tensor, including the existing unused test tensor, is used.
        fitset = set(j(fit_path))
        assert fitset == set(populations["fit"])
        fitrows = [idrow[name] for name in ids if name in fitset]
        pairs = [(row, caprow[caption]) for row, caption in allpairs if ids[row] in fitset]
        assert len(fitset) == len(fitrows) == 8263 and len(pairs) == 28759
        split = j(split_path)
        selected = [split["queries"][i] for i in split["parts"]["tune"]]
        assert len(selected) == 459 and len(split["gallery_ids"]) == 918
        assert set(split["gallery_ids"]) == set(populations["val"])
        assert fitset.isdisjoint(split["gallery_ids"])
        query_idx = []
        for query in selected:
            caption = caps[query["image_id"]][0]
            assert hashlib.sha256(caption.encode()).hexdigest() == query["caption_sha256"]
            assert split["gallery_ids"][query["gold_gallery_index"]] == query["image_id"]
            query_idx.append(caprow[caption])
        gallery_rows = torch.tensor([idrow[name] for name in split["gallery_ids"]], device="cuda")
        gold = torch.tensor([query["gold_gallery_index"] for query in selected], device="cuda")
        docs = move_docs(cache, torch.arange(len(ids)), torch.device("cuda"))
        ix = torch.tensor(fitrows, device="cuda")
        means = []
        for branch in ["c", "f"]:
            mask = docs[branch+"_mask"][ix]
            means.append(docs[branch+"_tf"][ix][mask].mean(0))
        assert all(torch.isfinite(mean).all() for mean in means)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = JointPARE(cache["D"], args.variant, *means).cuda()
        metadata = {"policy": args.policy, "variant": args.variant, "seed": 0, "epochs": 20, "backbone": "vitb32",
                    "initial_adapter_sha256": state_sha(model.adapter.state_dict()),
                    "initial_physics_sha256": state_sha(model.physics.state_dict()),
                    "code_sha256": code_hashes, "paired_cache_inputs": paired_binding,
                    "verified_inputs_sha256": sha(out/"INPUTS_VERIFIED.json"),
                    "train_images": 8263, "train_pairs": 28759, "tune_queries": 459, "gallery": 918,
                    "fit_crop_mean_c": means[0].tolist(), "fit_crop_mean_f": means[1].tolist(),
                    "gpu_name": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "stored_parameters": sum(p.numel() for p in model.parameters()),
                    "optimizer_parameters": sum(p.numel() for p in model.active_parameters()),
                    "disclosure": "Exploratory single-seed paired-tie pilot on previously used tune queries. No confirm/test scoring. Only the train tensor of the shared text artifact is used."}
        write(out/"run.json", metadata)
        optimizer = torch.optim.AdamW(model.active_parameters(), lr=.002, weight_decay=1e-4)
        pairrows = torch.tensor([row for row, caption in pairs], device="cuda")
        pairtexts = torch.tensor([caption for row, caption in pairs], device="cuda")
        order = list(range(len(pairs)))
        rng = random.Random(0)
        history = []
        start = time.monotonic()
        del cache
        for epoch in range(1, 21):
            rng.shuffle(order)
            loss_sum, nb = 0., 0
            model.train()
            for off in range(0, len(order), 64):
                batchidx = torch.tensor(order[off:off+64], device="cuda")
                rows = pairrows[batchidx]
                batch = {key: value[rows] for key, value in docs.items()}
                scores = model(qmatrix[pairtexts[batchidx]], batch)
                loss = F.cross_entropy(scores/.03, torch.arange(len(batchidx), device="cuda"))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if epoch == 1 and nb == 0:
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
                loss_sum += loss.item()
                nb += 1
            row = {"epoch": epoch, "loss": loss_sum/nb, "batches": nb, "pair_order_sha256": objsha(order),
                   "elapsed_seconds": time.monotonic()-start}
            history.append(row)
            write(out/"progress.json", {"history": history})
            print(json.dumps({"policy": args.policy, "variant": args.variant, **row}), flush=True)
            if epoch in [10, 20]:
                assert all(torch.isfinite(p).all() for p in model.parameters())
                checkpoint = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                              "optimizer": optimizer.state_dict(), "metadata": metadata, "history": list(history),
                              "order": order, "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                              "cuda_rng": torch.cuda.get_rng_state()}
                torch.save(checkpoint, out/f"epoch_{epoch}.pt")
                if epoch == 20:
                    model.eval()
                    parts = []
                    q = qmatrix[torch.tensor(query_idx, device="cuda")]
                    valdocs = {key: value[gallery_rows] for key, value in docs.items()}
                    with torch.no_grad():
                        for qi in range(0, len(q), 8):
                            scores = model(q[qi:qi+8], valdocs)
                            assert torch.isfinite(scores).all()
                            parts.append(scores.cpu())
                    scores = torch.cat(parts)
                    ranks = (torch.argsort(scores, dim=1, descending=True, stable=True) == gold.cpu()[:, None]).nonzero(as_tuple=True)[1]+1
                    metrics = {f"R@{k}": float((ranks <= k).float().mean()*100) for k in [1, 5, 10]}
                    metrics["MRR@10"] = float(torch.where(ranks <= 10, 1/ranks.double(), 0).mean()*100)
                    torch.save({"scores": scores, "ranks": ranks, "query_ids": [query["query_id"] for query in selected],
                                "gallery_ids": split["gallery_ids"], "gold": gold.cpu()}, out/"tune_scores.pt")
                    write(out/"tune.json", {"metrics_percent": metrics, "n_queries": 459, "n_gallery": 918,
                                           "split": "previously used tune only", "checkpoint_sha256": sha(out/"epoch_20.pt"),
                                           "score_sha256": sha(out/"tune_scores.pt"), "ranks": ranks.tolist()})
        if code_hashes != {"runner": sha(__file__), "model": sha(root/"joint_model.py"),
                           "protocol": sha(root/"PROTOCOL.json"), "v1_inputs_manifest": sha(root/"INPUTS_BOUND.json")}:
            raise ValueError("Code or protocol changed during training")
        for name, item in bound.items():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError(f"V1 input changed during training: {name}")
        if sha(args.paired_audit) != args.paired_audit_sha256:
            raise ValueError("Paired-cache audit changed during training")
        for item in paired_binding["caches"].values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("A paired cache changed during training")
        write(out/"complete.json", {"metrics_percent": metrics, "elapsed_seconds": time.monotonic()-start,
                                   "ended_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"COMPLETE": [args.policy, args.variant], "metrics": metrics}), flush=True)
    except BaseException as error:
        write(out/"failed.json", {"type": type(error).__name__, "message": str(error), "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
