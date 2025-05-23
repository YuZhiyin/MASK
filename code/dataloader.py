from glob import glob
import pandas as pd
import json
import random 
import datasets

random.seed(0)

class MMLU:
    def __init__(self, dataset_name='mmlu', n_samples=50, data_dir='data'):
        self.dataset_name = dataset_name
        self.data_path = f'{data_dir}/{dataset_name}/test/*.csv'
        self.data_files = glob(self.data_path)
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()

    def load_data(self):
        
        dfs = [pd.read_csv(file, header=None) for file in self.data_files]
        self.data = pd.concat(dfs, ignore_index=True)
        self.selected_data = self.data.sample(self.n_samples)
            

    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()

    def __len__(self):
        return len(self.selected_data)

class TruthfulQA:
    def __init__(self, dataset_name='truthfulqa', n_samples=50, data_dir='data', context=False):
        self.dataset_name = dataset_name
        self.data_path = f'{data_dir}/{dataset_name}'
        self.data_file = self.data_path + '/mc_task.json'
        self.context = context
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()
    
    def load_data(self):
        self.data = pd.read_json(self.data_file)
        if self.context:
            # donwload context
            self.context_data = datasets.load_dataset('portkey/truthful_qa_context', split='train')
            self.context_data = self.context_data.to_pandas()
            # select only relevant fields
            self.context_data = self.context_data[['question', 'context', 'source']]
            # clean context from errors
            self.context_data = self.context_data[~self.context_data.context.str.lower().str.contains('error')]
            # limit context length to 2000 chars (represent ~700 samples)
            self.context_data = self.context_data[self.context_data.context.str.len() < 2000]
            # merge context with data
            self.merge_df = pd.merge(self.data, self.context_data, on='question')
            # change data df
            self.data = self.merge_df.copy()

        self.selected_data = self.data.sample(self.n_samples, random_state=42)

    
    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()
    
    def __len__(self):
        return len(self.selected_data)


class MedMCQA:
    def __init__(self, dataset_name='medmcqa', n_samples=50, data_dir='data'):
        self.dataset_name = dataset_name
        self.data_path = f'{data_dir}/{dataset_name}'
        self.data_file = self.data_path + '/dev.json'
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()
    
    def load_data(self):
        self.data = pd.read_json('data/medmcqa/dev.json', lines=True)
        self.filtered_data = self.data[ self.data['choice_type'] == 'single']
        self.selected_data = self.filtered_data.sample(self.n_samples)
    
    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()
    
    def __len__(self):
        return len(self.selected_data)


class Scalr:
    def __init__(self, dataset_name='scalr', n_samples=50, data_dir='data'):
        self.dataset_name = dataset_name
        self.data_path = f'{data_dir}/{dataset_name}'
        self.data_file = self.data_path + '/test.jsonl'
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()
    
    def load_data(self):
        self.data = pd.read_json('data/scalr/test.jsonl', lines=True )
        self.selected_data = self.data.sample(self.n_samples)
    
    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()
    
    def __len__(self):
        return len(self.selected_data)


def get_dataset(dataset_name='mmlu', n_samples=50, data_dir='data', context=False):

    if dataset_name == 'mmlu':
        return MMLU(dataset_name, n_samples, data_dir)
    elif dataset_name == 'truthfulqa':
        return TruthfulQA(dataset_name, n_samples, data_dir, context)
    elif dataset_name == 'medmcqa':
        return MedMCQA(dataset_name, n_samples, data_dir)
    elif dataset_name == 'scalr':
        return Scalr(dataset_name, n_samples, data_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
        