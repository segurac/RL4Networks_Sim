from typing import Dict, List
import json
import pandas as pd
import os
import torch
import xml.etree.ElementTree as ET


def load_json(json_files: List[str]) -> List[Dict]:
    """Load json files

    :param json_files: List[str], list of json path to load
    :return: List[Dict]
    """
    to_return = []
    for path in json_files:
        with open(path) as f:
            to_return.append(json.load(f))
    return to_return


def read_xml_config(path: str):
    """Load xml config file

    :param path: str, xml file path
    :return xml root
    """
    tree = ET.parse(path)
    return tree.getroot()


class Conf(object):

    def __init__(self, ns3_config_xml_path: str, agent_config_path: str):
        # =========================================
        # NS3 PARAMS
        # =========================================
        self.nMacroEnbSites = None
        self.nUEs = None
        self.envStepTime = None
        self.openGymPort = None

        xml_root = read_xml_config(ns3_config_xml_path)
        for child in xml_root:
            if child.tag == 'global':
                if child.attrib.get('name', None) == 'nMacroEnbSites':
                    self.nMacroEnbSites = int(child.attrib.get('value', None))
                if child.attrib.get('name', None) == 'nUEs':
                    self.nUEs = int(child.attrib.get('value', None))
                if child.attrib.get('name', None) == 'envStepTime':
                    self.envStepTime = float(child.attrib.get('value', None))
                if child.attrib.get('name', None) == 'openGymPort':
                    self.openGymPort = int(child.attrib.get('value', None))
                if child.attrib.get('name', None) == 'eNBAdjacencyMatrixFile':
                    self.eNBAdjacencyMatrixFile = str(child.attrib.get('value', None))

        # =========================================
        # AGENT PARAMS
        # =========================================
        self.n_episodes = None
        self.max_steps_per_episode = None
        self.evaluate_agent_each_n_steps = None
        self.n_eval_episodes = None
        self.max_troughput_normalization = None
        self.device = None
        self.agent_params = None
        self.max_msc_idx = None
        self.sum_up_mcs = None

        tmp = load_json([agent_config_path])[0]
        self.n_episodes = tmp.get('number_of_episodes', None)
        self.max_steps_per_episode = tmp.get('max_steps_per_episode', None)
        self.evaluate_agent_each_n_steps = tmp.get('evaluate_agent_each_n_steps', None)
        self.n_eval_episodes = tmp.get('n_eval_episodes', None)
        self.max_troughput_normalization = tmp.get('max_troughput_normalization', None)
        self.device = tmp.get('device', None)
        self.agent_type = tmp['agent_type']
        self.agent_params = tmp['agents'].get(tmp['agent_type'], None)
        self.max_msc_idx = tmp.get('max_msc_idx', None)
        self.sum_up_mcs = tmp.get('sum_up_mcs', None)
        self.step_CIO = tmp.get('step_CIO', None)
        self.ns3_path = tmp.get('ns3_path', None)
        self.launch_sim_file = tmp.get('launch_sim_file', '')
        self.model_path = tmp.get('model_path', None)
        self.tensorboard_log_dir = tmp.get('tensorboard_log_dir', None)

        # Adjacency Matrix
        path = os.path.join(self.ns3_path, self.eNBAdjacencyMatrixFile)
        self.adjacency_matrix = torch.tensor(pd.read_csv(path, sep=";", header=None).values)
