import nltk


class DependencyTree(nltk.Tree):
    """A dependency tree

    Extends the nltk.Tree class by adding public methods for get the
    indices of the root, children of a particular node, and parents of
    a particular node.
    In general, the initializer should not be used directly; rather, the
    class method fromstring should be used to build a tree from a parse
    represented as a string using Penn TreeBank annotation conventions

    """
    def __init__(self, node, children=None):
        super().__init__(node, children)

        # precompute positions
        self.positions = self.treepositions()
        self.token = None
        self.sentence = None

    def root_idx(self):
        return [()]

    def children_idx(self, idx):
        if idx:
            return [i for i in self.positions
                if (len(idx) + 1) == len(i) and i[0] == idx[0]]
        else:
            return [i for i in self.positions
                if (len(idx) + 1) == len(i)]

    def parents_idx(self, idx):
        if idx:
            return [idx[:-1]]
        else:
            return []

    def words(self):
        return [i.label() for i in self.subtrees()] + self.leaves()

    def word_index(self, idx):
        '''input: tree index ex: (), (1,)
           returns: word index in sequence
        '''
        if isinstance(self[idx], str):
            return self.words().index(self[idx])
        else:
            return self.words().index(self[idx].label())
