import pyomo.environ as pyo
from src.instance import *
import cplex

class pyoTradeoffDEA(pyo.ConcreteModel):
    def __init__(self, an_instance, is_input: bool, is_VRS: bool = False):
        super().__init__()
        self.has_goals = an_instance.has_goals()
        self.M = an_instance.M
        
        self.j = pyo.Set(initialize=an_instance.dmu_set, doc='DMUs')
        self.i = pyo.Set(initialize=an_instance.input_set, doc='Inputs')
        self.r = pyo.Set(initialize=an_instance.output_set, doc='Outputs')
        g = self.r if is_input else self.i
        
        self.inputs = pyo.Param(self.i, self.j, initialize=an_instance.inputs_dict, domain=pyo.NonNegativeReals)
        self.outputs = pyo.Param(self.r, self.j, initialize=an_instance.outputs_dict, domain=pyo.NonNegativeReals)
        self.goals = pyo.Param(g, self.j, initialize=an_instance.goals_dict, domain=pyo.NonNegativeReals)

        self.phi = pyo.Var(self.j, domain = pyo.NonNegativeReals)
        self.l = pyo.Var(self.j, self.j, domain = pyo.NonNegativeReals)
        
        self.dl = pyo.Var(g, self.j, domain = pyo.NonNegativeReals)
        self.du = pyo.Var(g, self.j, domain = pyo.NonNegativeReals)
        self.z = pyo.Var(g, self.j, domain = pyo.Binary)

        self.is_input_oriented = is_input

        
        self.set_orientation_constraints()
        self.set_constraint_BCC_if_necessary(is_VRS)
        self.cDeviationU = pyo.Constraint(g, self.j, rule=ruleTradeoffDeviationU)
        self.cDeviationL = pyo.Constraint(g, self.j, rule=ruleTradeoffDeviationL)
        self.cDeviationTotal = pyo.Constraint(g, rule=ruleTradeoffDeviationTotal)

        self.set_objective()
        
    def set_constraint_BCC_if_necessary(self, is_VRS):
        if is_VRS:
            self.cVRS = pyo.Constraint(self.j, rule=ruleTradeoffVRS)
            
    def set_objective(self):
        objective_type = pyo.minimize if self.is_input_oriented else pyo.maximize
        self.efficiency = pyo.Objective(
            expr = sum(self.phi[k] for k in self.j),
            sense = objective_type
        )
        
    def run(self, optimizer):
        optimizer.solve(self)
        result = {}
        result['phi'] = np.zeros(len(self.j))
        phi_values = self.phi.get_values()
        for k in self.j:
            result['phi'][k] = phi_values[k]
        result['du'] = self.du.get_values()
        result['dl'] = self.dl.get_values()
        return result #np.array(self.phi.get_values()) # result
        
        
        
    def set_orientation_constraints(self):
        self.set_inputs_constraints_CCR() if self.is_input_oriented else self.set_outputs_constraints_CCR()

    def set_inputs_constraints_CCR(self):
        self.cInputs = pyo.Constraint(self.i, self.j, rule=ruleTradeoffInput1CCR, doc='input-oriented inputs CCR constraints')
        self.cOutputs = pyo.Constraint(self.r, self.j, rule=(ruleTradeoffInputGoals2CCR if self.has_goals else ruleTradeoffInput2CCR), doc='input-oriented outputs CCR constraints')

    def set_outputs_constraints_CCR(self):
        self.cInputs = pyo.Constraint(self.i, self.j, rule=(ruleTradeoffOutputGoals1CCR if self.has_goals else ruleTradeoffOutput1CCR), doc='input-oriented inputs CCR constraints')
        self.cOutputs = pyo.Constraint(self.r, self.j, rule=ruleTradeoffOutput2CCR, doc='input-oriented outputs CCR constraints')
         
        
#############################
# Rules of constraints
#############################

# Variable Return to Scale's rules
def ruleTradeoffVRS(mDEA, k):
    return sum(mDEA.l[j, k] for j in mDEA.j) == 1 

# Input oriented constraint's rules 
def ruleTradeoffInput1CCR(mDEA, i, k):
    return sum(mDEA.l[j, k] * mDEA.inputs[i, j] for j in mDEA.j) <= mDEA.phi[k] * mDEA.inputs[i, k]

def ruleTradeoffInput2CCR(mDEA, r, k):
    return sum(mDEA.l[j, k] * mDEA.outputs[r, j] for j in mDEA.j) >= mDEA.outputs[r, k]
#or
def ruleTradeoffInputGoals2CCR(mDEA, r, k):
    return sum(mDEA.l[j, k] * mDEA.outputs[r, j] for j in mDEA.j) >= mDEA.goals[r, k] + mDEA.du[r, k] - mDEA.dl[r, k]

# Output oriented constraint's rules
def ruleTradeoffOutput1CCR(mDEA, i, k):
    return sum(mDEA.l[j, k] * mDEA.inputs[i, j] for j in mDEA.j) <= mDEA.inputs[i, k]
#or
def ruleTradeoffOutputGoals1CCR(mDEA, i, k):
    return sum(mDEA.l[j, k] * mDEA.inputs[i, j] for j in mDEA.j) <= mDEA.goals[i, k] + mDEA.du[i, k] - mDEA.dl[i, k]

def ruleTradeoffOutput2CCR(mDEA, r, k):
    return sum(mDEA.l[j, k] * mDEA.outputs[r, j] for j in mDEA.j) >= mDEA.phi[k] * mDEA.outputs[r, k]

def ruleTradeoffDeviationU(mDEA, g, k):
    return mDEA.du[g, k] <= mDEA.M * mDEA.goals[g, k] * mDEA.z[g, k]

def ruleTradeoffDeviationL(mDEA, g, k):
    return mDEA.dl[g, k] <= mDEA.M * mDEA.goals[g, k] * (1 - mDEA.z[g, k])

def ruleTradeoffDeviationTotal(mDEA, g):
    return sum(mDEA.du[g, k] for k in mDEA.j) == sum(mDEA.dl[g, k] for k in mDEA.j) # mDEA.dl[g, k]

