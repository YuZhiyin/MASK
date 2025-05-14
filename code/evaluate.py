import os
import json
import argparse
import re
import numpy as np
from tqdm import tqdm 
from collections import defaultdict

import pandas as pd

def load_data(filename):
    
    #obtain folder name
    folder_name = filename.split('/')[1]
    dataset_name = folder_name

    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data, dataset_name

def parse_mmlu(text, task_info):

    pattern = r"\((\w+)\)|(\w+)\)"
    matches = re.findall(pattern, text)
    matches = [match[0] or match[1] for match in matches]

    solution_by_re = None
    for match_str in matches[::-1]:
        solution_by_re = match_str.upper()
        if solution_by_re >= 'A' and solution_by_re <= 'D':
            break
        else:
            solution_by_re = None

    
    solution_by_item = [-1,-1,-1,-1]
    idx = 0
    for item in task_info[1:-1]:
        pos = text.lower().rfind(item.lower().strip())
        if pos >= 0:
            solution_by_item[idx] = pos
        idx += 1

    if max(solution_by_item) == -1:
        solution_by_item = None
    else:
        solution_by_item = ["A","B","C","D"][
            solution_by_item.index(max(solution_by_item))
        ]

    if solution_by_item is None and solution_by_re is not None:
        return solution_by_re
    elif solution_by_item is not None and solution_by_re is None:
        return solution_by_item
    elif solution_by_item is None and solution_by_re is None:
        return None
    elif solution_by_item is not None and solution_by_re is not None:
        if solution_by_item == solution_by_re:
            return solution_by_item
        else:
            return solution_by_item
        

def parse_truthfulqa(text, task_info):

    pattern = r"\((\w+)\)|(\w+)\)"
    matches = re.findall(pattern, text)
    matches = [match[0] or match[1] for match in matches]

    answers_raw = task_info['mc1_targets']
    answers = [(chr(97 + i).upper(), answer) for i, answer in enumerate(answers_raw)]

    solution_by_re = None

    for match_str in matches[::-1]:
        solution_by_re = match_str.upper()
        if solution_by_re >= 'A' and solution_by_re <= answers[-1][0]:
            break
        else:
            solution_by_re = None

    
    solution_by_item = [-1 for _ in range(len(answers))]
    idx = 0
    for item in [a[1] for a in answers]:
        pos = text.lower().rfind(item.lower().strip('., '))
        if pos >= 0:
            solution_by_item[idx] = pos
        idx += 1

    if max(solution_by_item) == -1:
        solution_by_item = None
    else:
        solution_by_item = [a[0] for a in answers][
            solution_by_item.index(max(solution_by_item))
        ]

    if solution_by_item is None and solution_by_re is not None:
        return solution_by_re
    elif solution_by_item is not None and solution_by_re is None:
        return solution_by_item
    elif solution_by_item is None and solution_by_re is None:
        return None
    elif solution_by_item is not None and solution_by_re is not None:
        if solution_by_item == solution_by_re:
            return solution_by_item
        else:
            return solution_by_item

def parse_medmcqa(text, task_info):

    pattern = r"\((\w+)\)|(\w+)\)"
    matches = re.findall(pattern, text)
    matches = [match[0] or match[1] for match in matches]

    answers_raw = [task_info['opa'], task_info['opb'], task_info['opc'], task_info['opd']]
    answers = [(chr(97 + i).upper(), answer) for i, answer in enumerate(answers_raw)]

    solution_by_re = None

    for match_str in matches[::-1]:
        solution_by_re = match_str.upper()
        if solution_by_re >= 'A' and solution_by_re <= answers[-1][0]:
            break
        else:
            solution_by_re = None

    solution_by_item = [-1 for _ in range(len(answers))]
    idx = 0
    for item in [a[1] for a in answers]:
        pos = text.lower().rfind(item.lower().strip('., '))
        if pos >= 0:
            solution_by_item[idx] = pos
        idx += 1

    if max(solution_by_item) == -1:
        solution_by_item = None
    else:
        solution_by_item = [a[0] for a in answers][
            solution_by_item.index(max(solution_by_item))
        ]

    if solution_by_item is None and solution_by_re is not None:
        return solution_by_re
    elif solution_by_item is not None and solution_by_re is None:
        return solution_by_item
    elif solution_by_item is None and solution_by_re is None:
        return None
    elif solution_by_item is not None and solution_by_re is not None:
        if solution_by_item == solution_by_re:
            return solution_by_item
        else:
            return solution_by_item
        

def parse_scalr(text, task_info):

    pattern = r"\((\w+)\)|(\w+)\)"
    matches = re.findall(pattern, text)
    matches = [match[0] or match[1] for match in matches]

    answers_raw = [task_info['choice_0'], task_info['choice_1'], task_info['choice_2'], task_info['choice_3'], task_info['choice_4']]
    answers = [(chr(97 + i).upper(), answer) for i, answer in enumerate(answers_raw)]

    solution_by_re = None
    for match_str in matches[::-1]:
        solution_by_re = match_str.upper()
        if solution_by_re >= 'A' and solution_by_re <= answers[-1][0]:
            break
        else:
            solution_by_re = None

    
    solution_by_item = [-1 for _ in range(len(answers))]
    idx = 0
    for item in [a[1] for a in answers]:
        pos = text.lower().rfind(item.lower().strip('., '))
        if pos >= 0:
            solution_by_item[idx] = pos
        idx += 1

    if max(solution_by_item) == -1:
        solution_by_item = None
    else:
        solution_by_item = [a[0] for a in answers][
            solution_by_item.index(max(solution_by_item))
        ]

    if solution_by_item is None and solution_by_re is not None:
        return solution_by_re
    elif solution_by_item is not None and solution_by_re is None:
        return solution_by_item
    elif solution_by_item is None and solution_by_re is None:
        return None
    elif solution_by_item is not None and solution_by_re is not None:
        if solution_by_item == solution_by_re:
            return solution_by_item
        else:
            return solution_by_item


def parse_answer(dataset, text, raw_task):
    if dataset == "mmlu":
        parsed_answer = parse_mmlu(text, raw_task)
    elif dataset == "truthfulqa":
        parsed_answer = parse_truthfulqa(text, raw_task)
    elif dataset == "medmcqa":
        parsed_answer = parse_medmcqa(text, raw_task)
    elif dataset == "scalr":
        parsed_answer = parse_scalr(text, raw_task)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    return parsed_answer

def most_frequent_answer(answers):
    counter = 0
    if answers is None:
        return None
    num = [answers[0]]

    for i in answers:
        current_frequency = answers.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = [i]
        elif current_frequency == counter:
            num.append(i)

    num = list(set(num))
    if counter == 1:
        return None
    elif len(num) != 1:
        return None
    else:
        return num[0]
    
def check_answer_correctness(dataset, answer, gt):
    if answer is None:
        return 0
    if dataset == 'mmlu':
        return (answer.lower() == gt.lower()) * 1
    elif dataset == 'truthfulqa':
        return (answer.lower() == gt[0][0].lower()) * 1
    elif dataset == 'medmcqa':
        return (answer.lower() == gt.lower()) * 1
    elif dataset == 'scalr':
        return (answer.lower() == gt.lower()) * 1
    else:
        raise ValueError(f"Dataset {dataset} not supported")


def eval(args):

    # Read files for evaluation
    eval_files = []
    if os.path.isdir(args.eval_address):
        # it is a directory
        # read all files in the directory
        for file in os.listdir(args.eval_address):
            if file.endswith('.jsonl'):
                eval_files.append(os.path.join(args.eval_address, file))
    elif os.path.isfile(args.eval_address):
        # it is a file
        if not args.eval_address.endswith('.jsonl'):
            raise ValueError("File must be a jsonl file")
        
        eval_files.append(args.eval_address)
    else:
        raise ValueError("File does not exist")
    
    # Loop over each file
    n_reps = len(eval_files)
    for i_file, file in enumerate(eval_files):

        agent_responses, dataset = load_data(file)
        n_samples = len(agent_responses)
        n_agents = len(agent_responses[0]['agent_responses'])
        n_turns = len(agent_responses[0]['agent_responses'][0])//2
        if i_file == 0:
            majority_vote_accs = np.zeros((n_reps, n_turns))
            agent_turn_accs = np.zeros((n_reps, n_agents, n_turns))

        agent_turn_correct = np.zeros((n_agents, n_turns))
        agent_adversary = np.zeros(n_agents)
        majority_vote = np.zeros(n_turns)
        for agent_response in tqdm(agent_responses):

            question = agent_response['question']
            gt = agent_response['answer']
            raw_task = agent_response['raw_task']
            agents_conv = agent_response['agent_responses']

            all_answers = defaultdict(list)
            for agent, agent_conv in enumerate(agents_conv):
                adversary = False
                for context in agent_conv:
                    if context['role'] == 'system':
                        adversary = True
                        agent_adversary[agent] = 1

                    if context['role'] == 'assistant':
                        tmp_answer = parse_answer(dataset, context['content'], raw_task)
                        all_answers[agent].append(tmp_answer)

            np_all_answers = np.array(list(all_answers.values()))
            # compute answer correctness per round/agent
            for agent in range(n_agents):
                answers = np_all_answers[agent]
                for turn in range(n_turns):
                    ans = answers[turn]
                    agent_turn_correct[agent][turn] += check_answer_correctness(dataset, ans, gt)

            # compute majority vote per round
            for turn in range(n_turns):
                answers = np_all_answers[:, turn].tolist()
                final_answer = most_frequent_answer(answers)
                majority_vote[turn] += check_answer_correctness(dataset, final_answer, gt)
        
        # print results
        print(f"Results file: {file}")
        # Majority vote per turn 
        majority_vote_acc = majority_vote/n_samples
        print(pd.DataFrame(majority_vote_acc, columns=['Majority Vote per Turn']))

        # Accuracy per agent per turn
        agent_turn_acc = agent_turn_correct/n_samples
        print("Accuracy per agent per turn")
        print(pd.DataFrame(agent_turn_acc.transpose(), columns=[f'Agent {i+1}' for i in range(n_agents)]))

        # update for all files
        majority_vote_accs[i_file] = majority_vote_acc
        agent_turn_accs[i_file] = agent_turn_acc

    print('-'*25)
    print("Final Results")
    print("Majority Vote per Turn")
    print(pd.DataFrame({"mean": majority_vote_accs.mean(axis=0), "std": majority_vote_accs.std(axis=0)}))

    print("Accuracy per agent per turn")
    print("Mean: ")
    print(pd.DataFrame(agent_turn_accs.mean(axis=0).transpose(), columns=[f'Agent {i+1}' for i in range(n_agents)]))
    print("Std: ")
    print(pd.DataFrame(agent_turn_accs.std(axis=0).transpose(), columns=[f'Agent {i+1}' for i in range(n_agents)]))


if __name__ == "__main__":
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument("--eval_address", type=str, default='results/truthfulqa/adv_100_3_3_1-gpt-4o-gpt-3.5-turbo')
    args = argsparser.parse_args()

    eval(args)