from neuralnetwork import *
from threading import Lock

# the permanent tree data structure
class PermaTree:
    def __init__(self, checker, is_cuda):
        self.is_cuda = is_cuda
        self.node_count = 0
        self.root = PermaNode(self, checker.state)
        self.last_capture = 0

    def move_root(self, node):
        # move from root to a immediate child
        # update parent to None
        if not node.is_root():
            if node.from_edge.is_capture:
                self.last_capture = 0
            else:
                self.last_capture += 1
        self.root = node
        self.root.parent = None

    # def update(self, edge):
    #     pass
    #
    # def update(self, node):
    #     pass


class PermaEdge:
    """
    Guarantees that from_node and to_node are not None
    """

    def __init__(self, perma_tree, action, from_node):
        self.perma_tree = perma_tree
        # an Action object of the checker program
        self.action = action
        # a Node object for where the action comes from
        self.from_node = from_node
        # initialize node whenever an edge is created, guarantees the data structure property
        self.to_node = PermaNode(perma_tree, action.get_flipped_state(), self.from_node, self)
        self.is_capture = action.is_capture
        # self.to_node = None  # create new child node in expand() and update this

        # # these values are initialized at expand() and updated in backup()
        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        # from neural network queue
        self.value = None
        self.logit = None
        # computed after the from_node is ready
        self.prior_probability = None

    def checker_to_tensor(self):
        return binary_board(self.to_node.checker_state.board)

    def assign_value(self, nn):
        tensors = states_to_batch_tensor([self.to_node.checker_state], self.perma_tree.is_cuda)
        value = nn(tensors)
        self.value = value.cpu().numpy().tolist()[0][0]
        return value


class PermaNode:
    """
    Guarantees that from_edge is not None. May not have self.edges
    """

    def __init__(self, perma_tree, checker_state, parent=None, from_edge=None):
        self.perma_tree = perma_tree
        # every element in self.edges is an Edge object
        self.checker_state = checker_state
        self.parent = parent  # parent is an edge, None when root
        self.from_edge = from_edge
        # adjacency list implementation
        self.edges = []
        # locked if the children prior probability is being calculated by the neural network or selection is in progress
        self.lock = Lock()
        self.unassigned=0
        self.probability_ready=False
        perma_tree.node_count += 1

    def is_leaf(self):
        return len(self.edges) == 0

    def is_root(self):
        return self.from_edge is None

    def construct_edges(self):
        """

        :return: there are no edges
        """
        # call get_legal_actions from checker
        actions, _ = self.checker_state.get_legal_actions()
        if len(actions) == 0:
            return True
        # init and add edges into node
        for action in actions:
            new_edge = PermaEdge(self.perma_tree, action, self)  # prior_prob will be updated in expand()
            self.edges.append(new_edge)
        return False

    def get_children_checker_states(self):
        return [edge.to_node.checker_state for edge in self.edges]

    def put_children_on_nn_queue(self, nn_queue):
        """
        self is a leaf node that is being expanded
        This function will asynchronously assign the child edge value and prior probability
        A worker thread with neural network is constantly scanning the nn_queue to assign edges in batch
        :param parallel_nn_queue:
        :return:
        """
        self.lock.acquire()
        for edge in self.edges:
            self.unassigned+=1
            nn_queue.put(edge)

        #
        # # this lock will be released when all of its children are evaluated
        # states = [edge.to_node.checker_state for edge in self.edges]
        # input_tensor = states_to_batch_tensor(states, self.perma_tree.is_cuda)
        # value_tensor = nn(input_tensor)
        # value_array = value_tensor.cpu().numpy()
        # value_array = np.squeeze(value_array, axis=1)
        # value_list = value_array.tolist()
        # for edx, edge in enumerate(self.edges):
        #     edge.value = value_list[edx]
        #
        # # initialize the prior probability of all children
        # p = nn.children_values_to_probability(value_tensor)
        # # assert that all edges must not be shuffled
        # # this should only be used for MCTS, not training. no gradient is here.
        # npp = p.cpu().numpy().tolist()
        # for edx, edge in enumerate(self.edges):
        #     edge.prior_probability = npp[edx]


