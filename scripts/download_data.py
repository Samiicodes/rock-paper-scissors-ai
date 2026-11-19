'''Download the HaGRID annotations and the recorded human games into data/raw.'''

import argparse
from pathlib import Path

from rpsai.config import DEFAULT_CONFIG_PATH, configure_logging, load_config
from rpsai.downloads import fetch


def main() -> None:
    '''Fetch both datasets, skipping files that are already present and intact.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    configure_logging()
    config = load_config(arguments.config)
    fetch(config.downloads.hagrid, config.data.raw_dir)
    fetch(config.downloads.human_games, config.data.raw_dir)
    print('Done')


if __name__ == '__main__':
    main()
