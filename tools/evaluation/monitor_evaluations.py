#!/usr/bin/env python3
"""Compact read-only status and headline metrics for the five controllers."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control-dir", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for tag in ("vitb32", "vitb16", "remoteclip", "siglip", "georsclip"):
        path = args.control_dir / f"evaluation_{tag}_state.json"
        if not path.exists():
            rows.append({"tag": tag, "status": "not_started"})
            continue
        state = json.loads(path.read_text())
        alive = True
        try:
            os.kill(state["controller_pid"], 0)
        except ProcessLookupError:
            alive = False
        row = {"tag": tag, "status": state["status"], "controller_alive": alive,
               "child_pid": state.get("current_child_pid"), "output": state.get("output")}
        if "error" in state:
            row["error"] = state["error"]
        if state["status"] == "complete_verified_evaluation":
            summary = json.loads(Path(state["summary"]).read_text())
            full = summary["variants"]["full"]
            row["headline_metrics_percent"] = {
                "controlled_R10": full["100"]["controlled"]["PARE"]["R@10"],
                "natural_K100_candidate_recall": full["100"]["natural"]["candidate_recall_percent"],
                "natural_K100_R10": full["100"]["natural"]["PARE_end_to_end"]["R@10"],
                "full_gallery_R1": full["3688"]["natural"]["PARE_end_to_end"]["R@1"],
                "full_gallery_R10": full["3688"]["natural"]["PARE_end_to_end"]["R@10"],
                "scene_full_gallery_R10": summary["baselines"]["full_gallery"]["scene"]["R@10"],
                **{name + "_full_gallery_R10": summary["variants"][name]["3688"]["natural"]["PARE_end_to_end"]["R@10"]
                   for name in ("stats_off", "cfar_strict", "uniform_grid")}}
            row["verified_eval_json_count"] = summary["verified_eval_json_count"]
        rows.append(row)
    print(json.dumps({"all_complete": all(row["status"] == "complete_verified_evaluation" for row in rows),
                      "backbones": rows}, indent=2))


if __name__ == "__main__":
    main()
