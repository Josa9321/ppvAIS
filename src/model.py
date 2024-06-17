import pyomo.environ as pyo
from src.instance import *
import cplex

class pyoDEA(pyo.ConcreteModel):
    def __init__(self, an_instance, is_input: bool, is_VRS: bool = False, a_dmu_to_focus = 0):
        super().__init__()
        self.has_goals = an_instance.has_goals()
        
        self.j = pyo.Set(initialize=an_instance.dmu_set, doc='DMUs')
        self.i = pyo.Set(initialize=an_instance.input_set, doc='Inputs')
        self.r = pyo.Set(initialize=an_instance.output_set, doc='Outputs')
        self.k = a_dmu_to_focus
        
        self.inputs = pyo.Param(self.i, self.j, initialize=an_instance.inputs_dict, domain=pyo.NonNegativeReals)
        self.outputs = pyo.Param(self.r, self.j, initialize=an_instance.outputs_dict, domain=pyo.NonNegativeReals)
        if self.has_goals:
            g = self.r if is_input else self.i
            self.goals = pyo.Param(g, self.j, initialize=an_instance.goals_dict, domain=pyo.NonNegativeReals)

        self.theta = pyo.Var(domain = pyo.NonNegativeReals)
        self.l = pyo.Var(self.j, domain = pyo.NonNegativeReals)
        
        self.is_input_oriented = is_input
        self.set_constraint_BCC_if_necessary(is_VRS)
        self.set_objective()
        
    def set_constraint_BCC_if_necessary(self, is_VRS):
        if is_VRS:
            self.cVRS = pyo.Constraint(rule=ruleVRS)
            
    def set_objective(self):
        objective_type = pyo.minimize if self.is_input_oriented else pyo.maximize
        self.efficiency = pyo.Objective(
            expr = self.theta,
            sense = objective_type
        )
        
            
    def run(self, optimizer):
        result = np.zeros(len(self.j))
        for k in self.j:
            self.set_DMU_k(k)
            optimizer.solve(self)
            result[k] = self.theta.get_values()[None]
        return result
        
        
    def set_DMU_k(self, k):
        self.reset()
        self.k = k
        self.set_orientation_constraints()
        
    def set_orientation_constraints(self):
        self.set_inputs_constraints_CCR() if self.is_input_oriented else self.set_outputs_constraints_CCR()

    def set_inputs_constraints_CCR(self):
        self.cInputs = pyo.Constraint(self.i, rule=ruleInput1CCR, doc='input-oriented inputs CCR constraints')
        self.cOutputs = pyo.Constraint(self.r, rule=(ruleInputGoals2CCR if self.has_goals else ruleInput2CCR), doc='input-oriented outputs CCR constraints')

    def set_outputs_constraints_CCR(self):
        self.cInputs = pyo.Constraint(self.i, rule=(ruleOutputGoals1CCR if self.has_goals else ruleOutput1CCR), doc='input-oriented inputs CCR constraints')
        self.cOutputs = pyo.Constraint(self.r, rule=ruleOutput2CCR, doc='input-oriented outputs CCR constraints')
        
    def reset(self):
        try:
            self.del_component(self.cInputs)
            self.del_component(self.cOutputs)
        except:
            None
    
        
#############################
# Rules of constraints
#############################

# Variable Return to Scale's rules
def ruleVRS(mDEA):
    return sum(mDEA.l[j] for j in mDEA.j) == 1 

# Input oriented constraint's rules 
def ruleInput1CCR(mDEA, i):
    k = mDEA.k
    return sum(mDEA.l[j] * mDEA.inputs[i, j] for j in mDEA.j) <= mDEA.theta * mDEA.inputs[i, k]

def ruleInput2CCR(mDEA, r):
    k = mDEA.k
    return sum(mDEA.l[j] * mDEA.outputs[r, j] for j in mDEA.j) >= mDEA.outputs[r, k]
#or
def ruleInputGoals2CCR(mDEA, r):
    k = mDEA.k
    return sum(mDEA.l[j] * mDEA.outputs[r, j] for j in mDEA.j) >= mDEA.goals[r, k]

# Output oriented constraint's rules
def ruleOutput1CCR(mDEA, i):
    k = mDEA.k
    return sum(mDEA.l[j] * mDEA.inputs[i, j] for j in mDEA.j) <= mDEA.inputs[i, k]
#or
def ruleOutputGoals1CCR(mDEA, i):
    k = mDEA.k
    return sum(mDEA.l[j] * mDEA.inputs[i, j] for j in mDEA.j) <= mDEA.goals[i, k]

def ruleOutput2CCR(mDEA, r):
    k = mDEA.k
    return sum(mDEA.l[j] * mDEA.outputs[r, j] for j in mDEA.j) >= mDEA.theta * mDEA.outputs[r, k]