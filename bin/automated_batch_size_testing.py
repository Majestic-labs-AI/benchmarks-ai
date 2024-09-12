# Example 
# python automated_batch_size_testing.py -p small -tn 128 -t 2 -m 8B -v 3.1

import argparse
import subprocess
from tqdm import tqdm
from datetime import datetime

today = datetime.today().isoformat()
today = today.replace(':', '.')
today = today.replace('T', '_')
today = today[:-7]

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--prompt_length", help='desired length of prompt', choices=['small', 'medium', 'large'], required=True)
parser.add_argument('-tn', "--token_number", help='number of tokens to output', required=True)
parser.add_argument('-t', '--threads', help='number of threads to use', required=True)
parser.add_argument('-m', '--model', help='desired model to test', choices=['8B', '70B', '405B'], required=True)
parser.add_argument('-v', '--version', help='desired Llama version to test', choices=['3.1'], required=True)

batch_sizes=[
    '0',
    '16',
    '32',
    '48',
    '64',
    '80',
    '96',
    '112',
    '128',
    '144',
    '160',
    '176',
    '192',
]

prompts={'small': "Write a short story about a cat who discovers a hidden world in its backyard.", 'medium':"Write a short story about a young girl who discovers a hidden door in her grandmother’s attic. The door leads to a magical world where she meets talking animals and learns about her family’s secret history.", 'large':"Compose a detailed research paper on the impact of climate change on global agriculture. Include sections on the following: 1.Introduction to climate change and its causes. 2.Effects of climate change on crop yields and food security. 3.Case studies from different regions around the world. 4.Mitigation strategies and technological advancements. 5.Conclusion summarizing the findings and suggesting future research directions."}

args=parser.parse_args()

output_file = '/tmp/Llama_'+args.version+'_'+args.model+'_Benchmark_'+today+'.txt'

with open(output_file, 'w') as f:
    for batch_size in tqdm(batch_sizes):
        print(f"BATCH_SIZE: {batch_size}")
        command = [
            'build/bin/llama-batched',
            '-m', 'models/Meta-Llama-3.1-'+args.model+'-Instruct.gguf',
            '-p', prompts[args.prompt_length],
            '-np', batch_size,
            '-t', args.threads,
            '-n', args.token_number
        ]
        # command = "for i in {1..10000}; do openssl rand -base64 100; echo; done"
        result = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
        output = result.stdout.read()
        f.write(output)

print()
print('DONE TESTING')

