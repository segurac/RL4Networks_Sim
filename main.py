from copy import deepcopy
import argparse
import os


from src.common.scenario_factory import ScenarioFactory
from src.common.utils import Conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL4Network_Sim')
    parser.add_argument('-p', '--process', default=None, help='Process name', type=str, required=True)
    parser.add_argument('-ns3', '--ns3_config', default=None, help='ns3 xml config file', type=str, required=True)
    parser.add_argument('-agent', '--agent_config', default=None, help='agent config file', type=str, required=True)
    args = deepcopy(parser.parse_args().__dict__)

    # Input arguments into variables
    process_name = args.get('process')
    ns3_xml_config_file = args.get('ns3_config')
    agent_config_file = args.get('agent_config')

    local_path = os.environ['PWD']
    ns3_xml_config_file = os.path.join(local_path, ns3_xml_config_file)
    agent_config_file = os.path.join(local_path, agent_config_file)

    conf = Conf(ns3_xml_config_file, agent_config_file)

    p = ScenarioFactory().get_scenario(process_name)(conf)
    p.run()
