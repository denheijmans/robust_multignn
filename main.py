import argparse
import json
import logging
from collections.abc import Mapping
from itertools import product

from tqdm.contrib.logging import logging_redirect_tqdm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the task runner.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Robust Multi-GNN Tasks")
    parser.add_argument(
        "task",
        choices=("train", "attack", "adv_train", "adv_train_multigin"),
        help="type of task to run",
    )
    parser.add_argument(
        "config_path",
        help="path to JSON configuration",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="run without logging to wandb",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="run only the config at this index (for SLURM array jobs)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="print the number of expanded configs and exit",
    )
    parser.add_argument(
        "--project",
        default="robust_multignn",
        help="wandb project name (default: robust_multignn)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="override the config's device (e.g. 'cpu')",
    )
    return parser.parse_args()


def expand_config_grid(config: dict) -> list[dict]:
    """Expands a nested configuration dictionary into a list of configurations
    by taking the Cartesian product of all options.

    Parameters
    ----------
    config : dict
        Configuration dictionary to expand.

    Returns
    -------
    list[dict]
        List of expanded configuration dictionaries.
    """

    # Normalize values so all values become lists of options
    def normalize(value):
        if isinstance(value, Mapping):
            return [v for v in expand_config_grid(value)]
        elif isinstance(value, list):
            return value
        else:
            return [value]

    # Get keys and corresponding lists of options
    keys = list(config.keys())
    value_lists = [normalize(config[k]) for k in keys]

    # Take Cartesian product of all options and construct expanded configurations
    expanded = []
    for combo in product(*value_lists):
        item = {}
        for key, val in zip(keys, combo):
            item[key] = val
        expanded.append(item)

    return expanded


def init_wandb(config: dict, task: str, dryrun: bool, project: str = "robust_multignn"):
    """Initializes a Weights & Biases (wandb) run for task tracking.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing task parameters and settings.
    task : str
        Name of the task.
    dryrun : bool
        If True, disables logging to wandb.
    """
    import wandb

    def build_run_name() -> str:
        parts = [task]

        dataset = config.get("dataset")
        if dataset is not None:
            parts.append(str(dataset))

        if task == "attack":
            attack_type = config.get("attack_type")
            if attack_type is not None:
                parts.append(str(attack_type))
        elif task == "adv_train":
            train_type = config.get("train_type")
            if train_type is not None:
                parts.append(str(train_type))

        model_type = config.get("model_type")
        if model_type is not None:
            parts.append(str(model_type))

        file_name_attachment = config.get("file_name_attachment")
        if file_name_attachment:
            parts.append(str(file_name_attachment))

        return "-".join(parts)

    tags = config.get("wandb_tags")
    if tags is None:
        resolved_tags = None
    elif isinstance(tags, list):
        resolved_tags = [str(tag) for tag in tags]
    else:
        resolved_tags = [str(tags)]

    # Initialize a new wandb run with the task name and configuration
    wandb.init(
        project=project,
        name=build_run_name(),
        config=config,
        job_type=task,
        tags=resolved_tags,
        mode="disabled" if dryrun else None,
    )

    # Log all '.py' files in the current directory (and subdirectories) to wandb
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # Define callback function to finish the wandb run at the end of the task
    def finish():
        wandb.finish()

    return finish


def main():
    # Parse command-line arguments and load configuration
    args = parse_args()
    config = json.load(open(args.config_path))
    if args.device is not None:
        config["device"] = args.device

    # Expand nested configuration
    expanded_configs = expand_config_grid(config)

    if args.count:
        print(len(expanded_configs))
        return

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
    )

    # Select task module once
    logging.info("Task: %s", args.task)
    if args.task == "train":
        from experiments import experiment_train as task
    elif args.task == "attack":
        from experiments import experiment_attack as task
    elif args.task == "adv_train":
        from experiments import experiment_adv_train as task
    elif args.task == "adv_train_multigin":
        from experiments import experiment_adv_train_multigin as task

    configs_to_run = (
        [expanded_configs[args.index]] if args.index is not None else expanded_configs
    )

    # Run all expanded configurations
    for index, config in enumerate(configs_to_run, start=1):
        # Log configuration
        print("\n")
        logging.info(
            "Configuration %d/%d:\n%s",
            index,
            len(configs_to_run),
            json.dumps(config, indent=2),
        )

        # Initialize wandb for the current configuration
        finish_wandb = init_wandb(
            config,
            task=args.task,
            dryrun=args.dryrun,
            project=args.project,
        )

        # Run task with logging redirected to tqdm
        run_config = {
            k: v
            for k, v in config.items()
            if k not in ("wandb_tags", "file_name_attachment")
        }
        with logging_redirect_tqdm():
            task.run(**run_config)

        # Finish wandb run
        finish_wandb()


if __name__ == "__main__":
    main()
