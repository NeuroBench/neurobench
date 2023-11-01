from torch.nn import functional as F

def discrete_Actor_Critic(output):
    '''returns action based on output
    output expected to be 
    critic, actor'''
    # find probabilities of certain actions
    policy = output[1]
    prob = F.softmax(policy, dim=-1)

    # choose the action and detach from computational graph
    action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization
    
    # action
    return int(action.squeeze(0))