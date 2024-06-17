import numpy as np

class InstanceDEA:
    def __init__(self, an_inputs: list, an_outputs: list, an_goals: list = [], a_M = 0.2, is_input_oriented = True) -> None:
        self.inputs = np.array(an_inputs)
        self.outputs = np.array(an_outputs)
        self.goals = np.array(an_goals)
        self.M = a_M
        self.input_oriented = is_input_oriented
        self.test_valid_data()
        
        self.num_dmus = len(self.inputs[0])
        self.num_inputs = len(self.inputs)
        self.num_outputs = len(self.outputs)
        
        self.dmu_set = np.array(range(self.num_dmus))
        self.input_set = np.array(range(self.num_inputs))
        self.output_set = np.array(range(self.num_outputs))
        
        self.inputs_dict = self.convert_to_dict(self.inputs)
        self.outputs_dict = self.convert_to_dict(self.outputs)
        self.goals_dict = self.convert_to_dict(self.goals)
        
    
    def convert_to_dict(self, matrix):
        result = {}
        for i in range(matrix.shape[0]):
            for j in self.dmu_set:
                result[i, j] = matrix[i][j]
        
        return result
    
    
    def test_valid_data(self):
        assert self.inputs.shape[1] == self.outputs.shape[1], '''Inputs hasn't the same shape as outputs'''
        assert self.goals.shape[0] == 0 or self.has_goals(), '''Goals hasn't the same shape as inputs'''
    
    def has_goals(self):
        return self.outputs.shape == self.goals.shape if self.input_oriented else self.inputs.shape == self.goals.shape