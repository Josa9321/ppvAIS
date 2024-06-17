import pandas as pd

from src.instance import *
from src.model import *
from src.tradeoff_model import *
from src.security_model import *

######################
# Report
######################

# def efficiency_report(inputs, outputs, is_VRS = False):
#     result = pd.DataFrame()
#     efficiency_inputs_report(inputs, outputs, goals, is_VRS, result)
#     efficiency_outputs_report(inputs, outputs, goals, is_VRS, result)
#     return result

def efficiency_inputs_report(inputs, outputs, goals = [], is_VRS = False, is_tradeoff = False):
    result = pd.DataFrame()
    for (i, x_i) in enumerate(inputs):
        result[f'Input {i+1}'] = x_i
        
    efficiency = DEA(inputs, outputs, goals,
                     is_input_oriented = True, is_VRS = is_VRS, is_tradeoff_model = is_tradeoff)
    result['Efficiency Input'] = efficiency
    for (i, x_i) in enumerate(inputs):
        # result[f'Target of Input {i+1}'] = x_i * efficiency
        result[f'Resources saving of Input {i+1}'] = x_i - x_i * efficiency
    
    return result

def efficiency_outputs_report(inputs, outputs, goals = [], is_VRS = False, is_tradeoff = False):
    result = pd.DataFrame()
    for (r, y_r) in enumerate(outputs):
        result[f'Outputs {r+1}'] = y_r
        
    efficiency = DEA(inputs, outputs, goals,
                     is_input_oriented = False, is_VRS = is_VRS, is_tradeoff_model = is_tradeoff)
    result['Efficiency Output'] = efficiency
    
    for (r, y_r) in enumerate(outputs):
        # result[f'Target of Output {r+1}'] = y_r * efficiency
        result[f'Improvement in Output {r+1}'] = y_r * efficiency - y_r
    
    return result


######################
# Solving
######################

def DEA(inputs, outputs, goals = [], is_input_oriented = True, is_VRS = False, is_tradeoff_model = False):
    instance = InstanceDEA(inputs, outputs, goals)
    mDEA = pyoDEA(instance, is_input_oriented, is_VRS) if not(is_tradeoff_model) else pyoTradeoffDEA(instance, is_input_oriented, is_VRS)
    result = run_DEA(mDEA)
    return result

def run_DEA(mDEA):
    opt = pyo.SolverFactory('cplex')
    result = mDEA.run(opt)
    return result
    