import json
import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from autointent import Context
from .pipeline import Pipeline
from .utils import generate_name, get_db_dir

LoggingLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Path to a yaml configuration file that defines the optimization search space. "
        "Omit this to use the default configuration.",
    )
    parser.add_argument(
        "--multiclass-path",
        type=str,
        default="",
        help="Path to a json file with intent records. "
        'Set to "default" to use banking77 data stored within the autointent package.',
    )
    parser.add_argument(
        "--multilabel-path",
        type=str,
        default="",
        help="Path to a json file with utterance records. "
        'Set to "default" to use dstc3 data stored within the autointent package.',
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="",
        help="Path to a json file with utterance records. "
        "Skip this option if you want to use a random subset of the training sample as test data.",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="",
        help="Location where to save chroma database file. Omit to use your system's default cache directory.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="",
        help="Location where to save optimization logs that will be saved as "
        "`<logs_dir>/<run_name>_<cur_datetime>/logs.json`",
    )
    parser.add_argument(
        "--run-name", type=str, default="", help="Name of the run prepended to optimization logs filename"
    )
    parser.add_argument(
        "--mode",
        choices=["multiclass", "multilabel", "multiclass_as_multilabel"],
        default="multiclass",
        help="Evaluation mode. This parameter must be consistent with provided data.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Specify device in torch notation")
    parser.add_argument(
        "--regex-sampling",
        type=int,
        default=0,
        help="Number of shots per intent to sample from regular expressions. "
        "This option extends sample utterances within multiclass intent records.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Affects the data partitioning")
    parser.add_argument(
        "--log-level", type=str, default="ERROR", choices=LoggingLevelType.__args__, help="Set the logging level"
    )
    parser.add_argument(
        "--multilabel-generation-config",
        type=str,
        default="",
        help='Config string like "[20, 40, 20, 10]" means 20 one-label examples, '
        "40 two-label examples, 20 three-label examples, 10 four-label examples. "
        "This option extends multilabel utterance records.",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)  # TODO standardize logging
    logger = logging.getLogger(__name__)

    # configure the run and data
    run_name = get_run_name(args.run_name)
    db_dir = get_db_dir(args.db_dir, run_name)

    logger.debug("Run Name: %s", run_name)
    logger.debug("Chroma DB path: %s", db_dir)

    # create shared objects for a whole pipeline
    context = Context(
        load_data(args.multiclass_path),
        load_data(args.multilabel_path),
        load_data(args.test_path),
        args.device,
        args.mode,
        args.multilabel_generation_config,
        db_dir,
        args.regex_sampling,
        args.seed,
    )

    # run optimization
    pipeline = Pipeline(args.config_path, args.mode)
    pipeline.optimize(context)

    # save results
    pipeline.dump(args.logs_dir, run_name)


def load_data(data_path: str) -> list[dict[str, Any]]:
    """load data from the given path or load sample data which is distributed along with the autointent package"""
    path = Path(data_path)
    if not path.exists():
        msg = "Path not exists"
        raise ValueError(msg)
    with Path(data_path).open() as file:
        return json.load(file)


def get_run_name(run_name: str) -> str:
    if run_name == "":
        run_name = generate_name()
    return f"{run_name}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"  # noqa: DTZ005


def setup_logging(level: LoggingLevelType | None = None) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="{asctime} - {name} - {levelname} - {message}",
        style="{",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)
