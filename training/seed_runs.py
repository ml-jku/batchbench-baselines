"""Retrain a W&B sweep's best (or a chosen) configuration across several seeds.

Works for any method's sweep (``training/sweeps/*.yaml``) without
method-specific knowledge: it reads the sweep's own ``parameters:`` section
to learn exactly which Hydra overrides that sweep used (both the swept ones
and any fixed ``value:`` entries, e.g. ``trainer.pretrained_checkpoint`` for
TTT-MAE), looks up the best run's value for each, and reruns
``training.main`` with that same configuration once per seed, saving one
checkpoint per seed.

Usage:

    python -m training.seed_runs SWEEP_ID \\
        --project jumpcp-coral \\
        --save-dir models/seeds/coral \\
        --num-seeds 3

    python -m training.seed_runs entity/jumpcp-mae-probe/abcd1234 \\
        --seeds 0 42 645 \\
        --save-dir models/seeds/mae_ttt \\
        --epochs 30 --override trainer.ttt.eval_samples=500

    python -m training.seed_runs --run entity/jumpcp-vit-small-sweep/cab7z7dp \\
        --save-dir models/seeds/vit_small --epochs 100

``SWEEP`` is either a bare sweep ID (resolved against ``--entity``/
``--project``, falling back to the logged-in user's default entity) or a
full ``ENTITY/PROJECT/SWEEP_ID`` path. The Hydra config name
(``--config-name``) is inferred from the sweep's own ``command:`` list
unless given explicitly.

``--run RUN`` (a bare run ID or ``ENTITY/PROJECT/RUN_ID``) replaces the
sweep argument and retrains that run's configuration instead of the sweep's
best run. The run must belong to a sweep: its sweep still supplies the list
of overrides to copy and the config name.

Checkpoints are written to ``<save-dir>/<name>_<seed>.pt``, where ``<name>``
defaults to the save directory's own name (e.g. ``models/seeds/coral/`` ->
``coral_0.pt``, ``coral_42.pt``, ...), matching the ``<method>_seed_<seed>.pt``
convention already used under ``models/``.

With no ``--seeds``/``--num-seeds``, the three seeds already used for the
existing seed checkpoints are reused: 0, 42, 645.
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Set per seed by this script, never copied from the sweep's best run.
RESERVED_KEYS = {
    "seed",
    "trainer.output",
    "wandb_mode",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
}

DEFAULT_SEEDS = (0, 42, 645)

_BARE_TOKEN = re.compile(r"[A-Za-z0-9_./\-]*")


def format_override_value(value: Any) -> str:
    """Render a Python value as a Hydra override's right-hand side."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_override_value(item) for item in value) + "]"
    if isinstance(value, str):
        if _BARE_TOKEN.fullmatch(value):
            return value
        return "'" + value.replace("'", "\\'") + "'"
    return str(value)


def resolve_wandb_path(
    identifier: str,
    entity: str | None,
    project: str | None,
    api: Any,
    kind: str = "sweep",
) -> str:
    """Accept either a bare sweep/run ID or a full ENTITY/PROJECT/ID path."""
    full = f"ENTITY/PROJECT/{kind.upper()}_ID"
    parts = identifier.strip("/").split("/")
    if len(parts) == 3:
        return "/".join(parts)
    if len(parts) != 1:
        raise ValueError(
            f"{kind} must be a {kind.upper()}_ID or an {full} path; "
            f"got {identifier!r}."
        )

    resolved_entity = entity or api.default_entity
    if not resolved_entity:
        raise ValueError(
            "Could not determine the W&B entity; pass --entity or give the "
            f"full {full} path."
        )
    if not project:
        raise ValueError(
            f"A bare {kind} ID requires --project (or the full {full} path)."
        )
    return f"{resolved_entity}/{project}/{parts[0]}"


def infer_config_name(sweep_config: dict[str, Any]) -> str | None:
    """Recover ``--config-name=...`` from the sweep's own ``command:`` list."""
    command = sweep_config.get("command", [])
    for index, token in enumerate(command):
        token = str(token)
        if token.startswith("--config-name="):
            return token.split("=", 1)[1]
        if token == "--config-name" and index + 1 < len(command):
            return str(command[index + 1])
    return None


def collect_overrides(
    sweep_config: dict[str, Any],
    best_run_config: dict[str, Any],
    *,
    epochs: int | None,
    patience: int | None,
    extra: list[str],
) -> dict[str, str]:
    """Build the shared Hydra overrides, in precedence order (later wins).

    1. Every key the sweep itself defines under ``parameters:``, read from
       the best run's logged config (this covers both genuinely swept
       hyperparameters and fixed ``value:`` entries the sweep pinned, e.g. a
       checkpoint path).
    2. ``--epochs``/``--patience``, if given (sweeps often use fewer epochs
       than a final run should).
    3. ``--override key=value``, applied last.
    """
    overrides: dict[str, str] = {}

    for key in sweep_config.get("parameters", {}):
        if key in RESERVED_KEYS:
            continue
        if key == "epochs" and epochs is not None:
            continue
        if key == "patience" and patience is not None:
            continue
        if key not in best_run_config:
            print(
                f"warning: sweep parameter {key!r} not found in the best "
                "run's config; skipping it.",
                file=sys.stderr,
            )
            continue
        overrides[key] = format_override_value(best_run_config[key])

    if epochs is not None:
        overrides["epochs"] = str(epochs)
    if patience is not None:
        overrides["patience"] = str(patience)

    for item in extra:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"--override values must be key=value; got {item!r}.")
        overrides[key] = value

    return overrides


def build_command(
    *,
    python: str,
    config_name: str,
    overrides: dict[str, str],
    seed: int,
    output_path: Path,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str | None,
    run_name: str,
) -> list[str]:
    command = [python, "-m", "training.main", f"--config-name={config_name}"]
    command.extend(f"{key}={value}" for key, value in overrides.items())
    command.extend(
        (
            f"seed={seed}",
            f"trainer.output={output_path}",
            f"wandb_mode={wandb_mode}",
            f"wandb_project={wandb_project}",
            f"wandb_run_name={run_name}",
        )
    )
    if wandb_entity:
        command.append(f"wandb_entity={wandb_entity}")
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain a W&B sweep's best (or a chosen) configuration across "
            "several seeds."
        ),
    )
    parser.add_argument(
        "sweep",
        nargs="?",
        help="Sweep ID, or an ENTITY/PROJECT/SWEEP_ID path; its best run is used.",
    )
    parser.add_argument(
        "--run",
        help=(
            "Run ID, or an ENTITY/PROJECT/RUN_ID path, to use instead of a "
            "sweep's best run. The run must belong to a sweep."
        ),
    )
    parser.add_argument(
        "--entity",
        help="W&B entity for a bare sweep/run ID; defaults to the logged-in user.",
    )
    parser.add_argument(
        "--project",
        help="W&B project for a bare sweep/run ID.",
    )
    parser.add_argument(
        "--config-name",
        help=(
            "Hydra config name (e.g. experiment_coral). Inferred from the "
            "sweep's own command when omitted."
        ),
    )

    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help=f"Explicit seeds to train. Default: {list(DEFAULT_SEEDS)}.",
    )
    seeds_group.add_argument(
        "--num-seeds",
        type=int,
        help="Number of seeds to train (uses seeds 0..N-1) if --seeds is omitted.",
    )

    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory to save the seed checkpoints into (created if missing).",
    )
    parser.add_argument(
        "--name",
        help=(
            "Checkpoint filename prefix; files are saved as "
            "<name>_<seed>.pt. Default: the save directory's own name."
        ),
    )

    parser.add_argument("--epochs", type=int, help="Override the copied epoch count.")
    parser.add_argument("--patience", type=int, help="Override the copied patience.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra raw Hydra override, applied last. Repeatable.",
    )

    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--wandb-project",
        help="W&B project for the seed runs. Default: '<sweep-project>-seeds'.",
    )
    parser.add_argument(
        "--wandb-entity",
        help="W&B entity for the seed runs. Default: same as --entity/the sweep's.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter used to launch training.main.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected run, overrides, and per-seed commands; train nothing.",
    )
    args = parser.parse_args(argv)
    if (args.sweep is None) == (args.run is None):
        parser.error("give exactly one of SWEEP or --run.")
    return args


def main(argv: list[str] | None = None) -> None:
    import wandb

    args = parse_args(argv)
    seeds = args.seeds if args.seeds else list(range(args.num_seeds or 0)) or list(
        DEFAULT_SEEDS
    )

    api = wandb.Api()
    if args.run:
        run_path = resolve_wandb_path(
            args.run, args.entity, args.project, api, kind="run"
        )
        best_run = api.run(run_path)
        if best_run.sweep is None:
            raise ValueError(
                f"Run {run_path!r} is not part of a sweep; its sweep is needed "
                "to know which overrides to copy."
            )
        sweep_entity, sweep_project, _ = run_path.split("/")
        sweep_path = f"{sweep_entity}/{sweep_project}/{best_run.sweep.id}"
        sweep = api.sweep(sweep_path)
    else:
        sweep_path = resolve_wandb_path(args.sweep, args.entity, args.project, api)
        sweep = api.sweep(sweep_path)
        best_run = sweep.best_run()
        if best_run is None:
            raise RuntimeError(f"Sweep {sweep_path!r} has no run with metric data.")
        sweep_entity, sweep_project, _ = sweep_path.split("/")
    config_name = args.config_name or infer_config_name(sweep.config)
    if config_name is None:
        raise ValueError(
            "Could not infer --config-name from the sweep's command; pass "
            "it explicitly."
        )

    overrides = collect_overrides(
        sweep.config,
        dict(best_run.config),
        epochs=args.epochs,
        patience=args.patience,
        extra=args.override,
    )

    save_dir = Path(args.save_dir).expanduser().resolve()
    name = args.name or save_dir.name
    wandb_project = args.wandb_project or f"{sweep_project}-seeds"
    wandb_entity = args.wandb_entity or args.entity or sweep_entity

    print(f"Sweep: {sweep_path}")
    print(f"{'Run' if args.run else 'Best run'}: {best_run.name} ({best_run.id})")
    print(f"Config name: {config_name}")
    print("Overrides copied from the run:")
    for key, value in overrides.items():
        print(f"  {key}={value}")
    print(f"Seeds: {seeds}")
    print(f"Save dir: {save_dir}")

    commands = []
    for seed in seeds:
        output_path = save_dir / f"{name}_{seed}.pt"
        command = build_command(
            python=args.python,
            config_name=config_name,
            overrides=overrides,
            seed=seed,
            output_path=output_path,
            wandb_mode=args.wandb_mode,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            run_name=f"{name}_{seed}",
        )
        commands.append((seed, output_path, command))
        print(f"\n[seed {seed}] -> {output_path}")
        print(f"  {shlex.join(command)}")

    if args.dry_run:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    for seed, output_path, command in commands:
        print(f"\n=== training seed {seed} -> {output_path} ===")
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


if __name__ == "__main__":
    main()
