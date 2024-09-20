import yaml

class Config:
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_paths(self):
        return self.config['paths']

    def get_dataset_params(self):
        return self.config['dataset']

    def get_training_params(self):
        return self.config['training']
