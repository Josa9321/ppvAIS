import pyomo.environ as pyo
from src.instance import *
import cplex

class pyoSecurityDEA(pyo.ConcreteModel):
    def __init__(self, an_instance, is_VRS: bool = False):
        super().__init__() 
        assert an_instance.has_goals(), '''The model should have goals'''
        
        self.j = pyo.Set(initialize=an_instance.dmu_set, doc='DMUs')
        self.i = pyo.Set(initialize=an_instance.input_set, doc='Inputs')
        self.r = pyo.Set(initialize=an_instance.output_set, doc='Outputs')
        
        self.inputs = pyo.Param(self.i, self.j, initialize=an_instance.inputs_dict, domain=pyo.NonNegativeReals)
        self.outputs = pyo.Param(self.r, self.j, initialize=an_instance.outputs_dict, domain=pyo.NonNegativeReals)
        self.goals = pyo.Param(self.r, self.j, initialize=an_instance.goals_dict, domain=pyo.NonNegativeReals)

        self.phi = pyo.Var(self.j, domain = pyo.NonNegativeReals)
        self.l = pyo.Var(self.j, self.j, domain = pyo.NonNegativeReals)
        self.d = pyo.Var(self.i, self.j, self.j, domain = pyo.NonNegativeIntegers)
        for i in self.i:
            for j in self.j:
                self.d[i, j, j].fix(0)

        self.set_orientation_constraints()
        self.set_constraint_BCC_if_necessary(is_VRS)
        self.set_tradeoff_constraints()
        self.set_objective()
        
    # CONSTRAINTS
    def set_orientation_constraints(self):
        self.set_inputs_constraints_CCR()

    def set_inputs_constraints_CCR(self):
        self.cInputs = pyo.Constraint(self.i, self.j, rule=ruleInput1CCR, doc='input-oriented inputs CCR constraints')
        self.cOutputs = pyo.Constraint(self.r, self.j, rule=ruleInput2CCR, doc='input-oriented outputs CCR constraints')
        
    def set_constraint_BCC_if_necessary(self, is_VRS):
        if is_VRS:
            self.cVRS = pyo.Constraint(self.j, rule=ruleVRS)
            
    def set_tradeoff_constraints(self):
        self.cNumRealocations = pyo.Constraint(self.i, self.j, rule=ruleNumRealocations)
            
    # OBJECTIVE
    def set_objective(self):
        objective_type = pyo.minimize
        self.efficiency = pyo.Objective(
            expr = sum(self.phi[k] for k in self.j),
            sense = objective_type
        )
    
    # RUN ALGORITHM
    def run(self, optimizer):
        optimizer.solve(self)
        result = {}
        result['phi'] = np.zeros(len(self.j))
        phi_values = self.phi.get_values()
        for k in self.j:
            result['phi'][k] = phi_values[k]
        result['d'] = self.d.get_values()
        return result
        
#############################
# Rules of constraints
#############################

# Variable Return to Scale's rules
def ruleVRS(mDEA, k):
    return sum(mDEA.l[j, k] for j in mDEA.j) == 1 

# Input oriented constraint's rules 
def ruleInput1CCR(mDEA, i, k):
    return sum(mDEA.l[j, k] * mDEA.inputs[i, j] for j in mDEA.j) <= mDEA.phi[k] * (mDEA.inputs[i, k] + sum(mDEA.d[i, j, k] for j in mDEA.j))

def ruleInput2CCR(mDEA, r, k):
    return sum(mDEA.l[j, k] * mDEA.outputs[r, j] for j in mDEA.j) >= mDEA.goals[r, k]

# Tradeoff constraint's rules
def ruleNumRealocations(mDEA, i, k):
    return (1 - mDEA.phi[k]) * mDEA.inputs[i, k] == sum(mDEA.d[i, k, j] for j in mDEA.j)

def ruleTransport(mDEA, i, k):
    return sum(mDEA.d[i, k, j] for j in mDEA.j) == sum(mDEA.d[i, k, j] for j in mDEA.j)