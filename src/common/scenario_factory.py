from src.processes.poc_process import POC_Process
from src.processes.stable_baseline_process import StableBaselineProcess


class ScenarioFactory(object):

    def __init__(self):
        # Here new processes have to be included
        # All the processes should inherit from base Process class
        self.processes = {
            POC_Process.name: POC_Process,
            StableBaselineProcess.name: StableBaselineProcess
        }

    def get_scenario(self, name: str):
        p = self.processes.get(name, None)
        if p is None:
            print(f'Process  <{name}> does not exist. \n'
                  f'Only the following processes are currently implemented:')
            print(list(self.processes.keys()))
            raise Exception
        return p
