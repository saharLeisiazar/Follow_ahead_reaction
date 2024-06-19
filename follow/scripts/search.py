from nodes import MCTSNode
import numpy as np
import torch
from treelib import Tree
import os


class MCTS:
    def __init__(self, node: MCTSNode, human_prob):
        self.root = node
        self.params = node.params
        self.next_node_candidates = []
        self.human_prob = human_prob


    def tree_expantion(self):
        tree_id = 0

        for _ in range(self.params['num_expansion']):
            ### Node selection 
            curr_node = self.root
            while(curr_node.number_of_visits > 1):
                #### UCB of the children and selecting a child node
                UCB_children = []
                for c in curr_node.children:
                    UCB_children.append(self.compute_UCB(c))

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
        if not c:
            return -np.inf
        
        c_param= 1.
        prob = 1.
        if c.state.next_to_move == 0: 
            if c.action == 'left':
                prob = self.human_prob['left']
            elif c.action == 'straight':
                prob = self.human_prob['straight']
            elif c.action == 'right':
                prob = self.human_prob['right']

        else:
            prob = 1.      

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
        state = node.state.state
        obs = np.concatenate([
            state[0,:2] - state[1,:2] , [state[1,2]], [state[0,2]]
        ])
        obs = torch.FloatTensor(obs).unsqueeze(0)
        policy='a2c'
        value = self.params['RL_model'].evaluate_state(obs, policy=policy)
        return value.item()
    
    def best_child_node(self):
        visit = []
        for node in self.root.children:
            if node:
                visit.append(node.n)
            else:
                visit.append(0)
      
        # print(visit)
        return self.root.children[np.argmax(visit)]        