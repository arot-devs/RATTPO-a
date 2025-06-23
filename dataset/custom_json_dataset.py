from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, json_path):
        super().__init__()
        import json
        with open(json_path, 'r') as f:
            self.df = json.load(f)
        
        self.len_df = len(self.df) - 1
        self.df['Prompts'] = []
        
        for i in range(self.len_df):
            self.df['Prompts'].append(self.df[str(i)])
        self.all_prompts = self.df['Prompts']

    def __len__(self):
        return self.len_df
    
    def __getitem__(self, i):
        return dict(self.df.iloc[i])

def get_custom_list(json_path):
    return CustomDataset(json_path).all_prompts
