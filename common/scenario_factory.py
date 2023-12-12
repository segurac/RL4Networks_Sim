from typing import Callable

from scenarios.examples.Proof_of_concept.agent.poc_process import POC_Process


class ScenarioFactory(object):

    def __init__(self):
        # Here new processes have to be included
        # All the processes executing some scenario should inherit from base Process class
        self.processes = {
            'poc': POC_Process
        }

    def get_scenario(self, name: str):
        p = self.processes.get(name, None)
        if p is None:
            print(f'Process  <{name}> does not exist. \n'
                  f'Only the following processes are currently implemented:')
            print(list(self.processes.keys()))
            raise Exception
        return p
