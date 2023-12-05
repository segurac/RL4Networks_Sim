# RL4Networks_Sim



Docker Build instructions

cd docker/

docker build -t ns3.30-mobility:latest ./

## General workflow
The actors involved in the workflows are the followings:
* **NS3 Simulator**: It is executed inside a docker container
* **Agent**: It is executed locally

A set of examples can be found in the directory /scenario. 
A **simulator** and an **agent** are required for each scenario.
* agent: Directory including all the python files needed to solve the RL workflow
* <simulator_name>: Directory including:
  * *.h/.*cc files needed to fulfill the GYM API requirements (GetActionSpace, GetObservationSpace, GetObservation, GetReward, etc)
  * *.txt: file with simulation parameters (optional)
  * *.sh: script to launch the simulator

## Build Docker image and create ns3gym whl

```console
cd docker/
./build_image.sh
```

At the end of scripts: 
* A new docker image has been created. You can check it using *docker image ls*
* A copy of ns3gym whl is created in the current working directory. You can check it using *ls*

## Create and activate venv for Agent 

Python version: >= 3.9  
(This workflow has been tested on Kronos with Python 3.11.2)

```console
python3 -m venv agent_env
source ./agent_env/bin/activate
cd <project_dir>
pip install -r requirements.txt
pip install docker/ns3gym-0.1.0-py3-none-any.whl
```

Open the following file:

<agent_env>/lib/python3.11/site-packages/ns3gym/ns3env.py

Modify line 119

np.float -> np.float64

Modify line 121

np.float -> np.float32

## Launch scenario

Example: Proof_of_concept


**STEP 1: AGENT** 

Open a console for the agent. 

Active agent_env and move to the corresponding agent path
```console
source ./agent_env/bin/activate
cd <project_dir>/scenarios/examples/Proof_of_concept/agent
```

Launch the agent main 
```console
python3 ddqn_agent.py
```

**STEP 2: SIMULATOR**

Open a console for the simulator and move to the scenario path

```console
cd <project_dir>/scenarios/examples/Proof_of_concept
```

Launch the simulator 

```console
./lauch_simulator.sh
```


