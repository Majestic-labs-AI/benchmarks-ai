# Example 
# python automated_batch_size_testing.py -p large -ot 128 -t 96 -pc 8B-Instruct -v 3.1 -m Meta-Llama-3.1

import argparse
import subprocess
# import psutil
from tqdm import tqdm
from datetime import datetime

# def get_memory_usage():
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     return mem_info.rss/(1024**3)

today = datetime.today().isoformat()
today = today.replace(':', '.')
today = today.replace('T', '_')
today = today[:-7]

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--prompt_length", help='desired length of prompt', choices=['small', 'medium', 'large'], required=True)
parser.add_argument('-ot', "--output_tokens", help='number of tokens to output', required=True)
parser.add_argument('-t', '--threads', help='number of threads to use', required=True)
parser.add_argument('-pc', '--parameter_count', help='desired parameter size to test', choices=['8B-Instruct', '27B-Instruct', '70B-Instruct', '405B'], required=True)
parser.add_argument('-m', '--model', help='desired model to test', choices=['Meta-Llama-3.1','Gemma-2'], required=True)

batch_sizes=[
    '0',
    '8',
    '16',
    '24',
    '32',
    '40',
    '48',
    '56',
    '64',
    '72',
    '80',
    '88',
    '96',
]

prompts={'small': "Write a short story about a cat who discovers a hidden world in its backyard.", 'medium':"Write a short story about a young girl who discovers a hidden door in her grandmother’s attic. The door leads to a magical world where she meets talking animals and learns about her family’s secret history.", 'large':"Compose a detailed research paper on the impact of climate change on global agriculture. Include sections on the following: 1.Introduction to climate change and its causes. 2.Effects of climate change on crop yields and food security. 3.Case studies from different regions around the world. 4.Mitigation strategies and technological advancements. 5.Conclusion summarizing the findings and suggesting future research directions."}

args=parser.parse_args()

output_file = '/tmp/'+args.model+'_'+args.parameter_count+'_Benchmark_'+today+'.txt'

with open(output_file, 'w') as f:
    for batch_size in tqdm(batch_sizes):
        command = [
            'build/bin/llama-batched',
            '-m', 'models/'+args.model+'-'+args.parameter_count+'-Instruct.gguf',
            '-p', prompts[args.prompt_length],
            '-b', batch_size,
            '-np', '192',
            '-t', '96',
            '-n', args.output_tokens
        ]
        # command = "for i in {1..10000}; do openssl rand -base64 100; echo; done"
        result = subprocess.run(command, capture_output=True, universal_newlines=True)
        output = result.stderr
        f.write(output)

print()
print('DONE TESTING')

