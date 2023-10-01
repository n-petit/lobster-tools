from concurrent.futures import ProcessPoolExecutor
import subprocess

def execute_commands(cmd_tuple):
    step_1, step_2, step_3, step_4 = cmd_tuple
    subprocess.run(step_1, shell=True)
    subprocess.run(step_2, shell=True)
    subprocess.run(step_3, shell=True)
    subprocess.run(step_4, shell=True)

with open('step1.txt', 'r') as f:
    step_1s = [line.strip() for line in f.readlines()]

with open('step2.txt', 'r') as f:
    step_2s = [line.strip() for line in f.readlines()]

with open('step3.txt', 'r') as f:
    step_3s = [line.strip() for line in f.readlines()]

with open('step4.txt', 'r') as f:
    step_4s = [line.strip() for line in f.readlines()]

all_cmds = list(zip(step_1s, step_2s, step_3s, step_4s))

batch_size = 20
with ProcessPoolExecutor(max_workers=20) as executor:
    for i in range(0, len(all_cmds), batch_size):
        batch = all_cmds[i:i+batch_size]
        executor.map(execute_commands, batch)