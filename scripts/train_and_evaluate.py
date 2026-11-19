'''Train the gesture reader, tune and test the move predictor, and write the outputs.'''

import argparse
from pathlib import Path

from rpsai.config import DEFAULT_CONFIG_PATH, configure_logging, load_config
from rpsai.downloads import fetch
from rpsai.pipeline import run_pipeline


def main() -> None:
    '''Run the pipeline from the downloaded data.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        '--publish_docs',
        action='store_true',
        help='Copy the report and its figures into docs/.',
    )
    arguments = parser.parse_args()
    configure_logging()
    config = load_config(arguments.config)
    fetch(config.downloads.hagrid, config.data.raw_dir)
    fetch(config.downloads.human_games, config.data.raw_dir)
    run_pipeline(config, publish_docs=arguments.publish_docs)
    print('Done')


if __name__ == '__main__':
    main()
