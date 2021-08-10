# Description of Decision Forest Dialect

## Dialect
The dialect is called decisionforest (see src/include/Ops.td). The subsequent sections describe its contents.

## Types
* **TreeNode Type:** The node type contains the type of the feature index.
    * **Numerical TreeNode Type:** Is a subclass of the TreeNode type. Specifies that the node is a numerical node. Additionally, has the type of the threshold.
    * **Leaf type:** Is a subclass of the TreeNode type. Specifies that a node is a leaf. Has the type of the prediction.
    * **Categorical TreeNode Type:** (Currently unimplemented) Specifies the node computes a predicate on a categorical feature.
* **Tree Type:** Specifies general properties of the decision tree. Currently only has the return type of the tree (the type of the prediction). 
    * TODO We currently assume that all nodes in a tree have the same feature index and threshold type. Should the tree type just contain those details?
* **Tree Ensemble Type:** Contains details of the full forest. Currently has the return type, number of trees, the input type and the reduction type.

## Attributes
* 

## Operations
### High-level IR

### Mid-level IR

### Low-level 