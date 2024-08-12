1. We will add a **discussion section** to clarify several contextual and system related issues that came up during the review. This section will address the following. 

    * **Scheduling language:** We will add details about how SilvanForge's scheduling language compares to the scheduling languages in Halide and TVM. Additionally, we try to clearly bring out concrete ways in which our scheduling language differs. This will include all the material that we wrote in our rebuttal. 
    * **Optimizations in other systems:** This section will discuss how the optimizations performed by other systems differ from those enabled by SilvanForge. Firstly, we will add the information from the rebuttal which shows how SilvanForge compares to HummingBird and PyTorch. Second, we will add comparisons to other related work pointed out by the reviewers (lleaves and Yggdrasil).

2. We will add any required important context related to the above to the **Introduction**.
3. In **Section 9.5 (A Note on Batch Size)**, we will add additional context about the tradeoff between latency and batch size.
4. The following are some other changes we intend to make
    * **Figure 2:** Simplify the figure as much as possible.
    * **Section 4 (Overview):** Elaborate what is shared between different compilation pipelines and how. Specifically, make the portability of SilvanForge's scheduling language, HIR and MIR clear.
    * **⁠Section 5 (Reductions):** Make it clearer what reduction support MLIR currently has and what we've added. Also, make the current limitations clear. 