from nodes import MCTSNode
import numpy as np
import torch
from treelib import Tree
import os
from RL_interface import evaluate_state


class MCTS:
    def __init__(self, node: MCTSNode):
        self.root = node
        self.params = node.params
        self.next_node_candidates = []


    def tree_expantion(self):
        tree_id = 0

        for _ in range(self.params['num_expansion']):
            ### Node selection 
            curr_node = self.root
            while(curr_node.number_of_visits > 1):
                #### UCB of the children and selecting a child node
                UCB_children = []
                for c in curr_node.children:
                    if c:
                        UCB = self.compute_UCB(c)
                        UCB_children.append(UCB)
                    else:
                        UCB_children.append(-np.inf)

                curr_node = curr_node.children[np.argmax(UCB_children)]  

            ### at this point current node is a leaf node
            ### Node expantion
            
            while not curr_node.is_fully_expanded():
                child_node = curr_node.expand()
                if child_node == None:
                    continue

                child_node.value = self.evaluate_node(child_node)
                child_node.backpropagate()
                child_node.tree_id = tree_id+1
                tree_id +=1
            
            # self.draw_tree()

        return self.best_child_node()


    def compute_UCB(self, c):
        c_param= 1.
        prob = 1.
        # if c.state.next_to_move == 0: 
        #     if c.action == 'left':
        #         prob = 0.1
        #     elif c.action == 'straight':
        #         prob = 0.8
        #     elif c.action == 'right':
        #         prob = 0.1

        # else:
        #     prob = 1./ len(self.params['robot_acts'])      

        UCB = (c.value/c.n) + c_param * np.sqrt((np.log(c.parent.n) / c.n))  * prob

        return UCB


    def draw_tree(self):
        self.tree = Tree()
        node = self.root
        self.tree.create_node("parent_("+str(self.root.value/self.root.number_of_visits)+")", 0)
        self.extent_draw_tree(node)

        os.remove("/home/sahar/Follow-ahead-3/MCTS_reaction/scripts/tree.txt")
        self.tree.save2file('/home/sahar/Follow-ahead-3/MCTS_reaction/scripts/tree.txt')

    def extent_draw_tree(self, parent):
        for node in parent.children:
            if node:
                val = node.val if not node.number_of_visits else node.value/node.number_of_visits
                self.tree.create_node(node.action+'_('+str(val)+')',  node.tree_id , node.parent.tree_id)
                self.extent_draw_tree(node)

        return
    
    def evaluate_node(self, node):
        device = 'cpu'
        state = node.state.state
        obs = np.concatenate([
            state[0,:2] - state[1,:2] , [state[1,2]], [state[0,2]]
        ])
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device=device)
        policy='a2c'
        value = evaluate_state(self.params['RL_model'], obs, policy=policy)[0][0]
        value = value.detach().cpu().numpy()
        return value
    
    def best_child_node(self):
        visit = []
        for node in self.root.children:
            if node:
                visit.append(node.n)
            else:
                visit.append(0)
      
        return self.root.children[np.argmax(visit)]        