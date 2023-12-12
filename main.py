import argparse
from copy import deepcopy

from common.scenario_factory import ScenarioFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL4Network_Sim')
    parser.add_argument('-p', '--process', default=None, help='Process name', type=str, required=True)
    args = deepcopy(parser.parse_args().__dict__)

    # Input arguments into variables
    process_name = args.get('process')

    p = ScenarioFactory().get_scenario(process_name)()
    p.run()
