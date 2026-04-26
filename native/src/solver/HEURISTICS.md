A* Heuristic Functions for Sokoban and Grid‑Based Puzzle Games

Introduction

Sokoban and other grid‑based puzzle games present challenging planning problems: the state space grows exponentially with the number of movable objects, solutions may be hundreds of steps deep, and wrong moves can create dead states (deadlocks) that make the puzzle unsolvable.  Single‑agent search algorithms like A* and Iterative Deepening A* (IDA*) are commonly used to find optimal or near‑optimal solutions, but their efficiency depends critically on the quality of the heuristic function (the function h(n) estimating the cost from state n to a goal).  An admissible heuristic (one that never over‑estimates the true cost) guarantees optimality, while a more informed heuristic drastically reduces the number of nodes expanded.  This report summarises general heuristics used in grid‑based search (e.g., sliding‑tile puzzles) and domain‑specific heuristics developed for Sokoban.  It then discusses their advantages and disadvantages, providing guidance for implementing an A* solver in PuzzleScript or similar environments.

General A* Heuristics for Grid‑Based Puzzles

Zero, L∞, Manhattan and Euclidean distances

* Zero heuristic: always returns 0.  It is admissible for any problem but provides no guidance; A* degenerates into breadth‑first search and expands all reachable nodes up to the solution depth.  Its only advantage is constant‑time computation.
* L∞ (Chebyshev) distance: the heuristic value for node n at (x_n,y_n) to a goal at (x_g,y_g) is max(|x_n–x_g|, |y_n–y_g|).  The MIT Principles of Autonomy and Decision‑Making notes that this “max distance” is admissible because the steps required to reach the goal are at least the maximum travel in either direction .  It is slightly more informative than the zero heuristic but ignores obstacles and allows diagonal movement that is impossible in Sokoban; thus it underestimates heavily.
* Manhattan (L1) distance: the sum of horizontal and vertical distances: |x_n–x_g| + |y_n–y_g|.  For problems without diagonal moves, the Manhattan distance equals the shortest path length in empty space; obstacles can only increase the cost, so the heuristic is admissible .  It is widely used in sliding‑tile puzzles and basic Sokoban solvers.  Manhattan distance is more informative than the L∞ heuristic, and the computational cost is essentially constant .  However, it treats each object independently and does not account for interference between objects or walls, often producing loose lower bounds.
* Euclidean distance: the straight‑line distance \sqrt{(x_n–x_g)^2+(y_n–y_g)^2}.  Euclidean distance is an admissible heuristic for problems that allow diagonal moves, but for grid puzzles with orthogonal moves it systematically underestimates the cost more than Manhattan distance and offers no advantage over Manhattan.

Pros: very cheap to compute; guaranteed admissibility; simple to implement.  Cons: ignore interactions between movable elements and obstacles; lead to large search trees on complex puzzles.

Misplaced tiles (Hamming) heuristic

For sliding puzzles like the 8‑puzzle or 15‑puzzle, the misplaced tiles heuristic counts the number of tiles that are not in their goal location.  It is admissible because each misplaced tile requires at least one move.  Its simplicity is a strength, but it provides very little guidance compared with Manhattan distance.  Hamming distance is not generally useful for Sokoban because boxes are identical and only their positions relative to goals matter.

Linear conflict heuristic

In sliding‑tile puzzles, tiles often block each other along a row or column.  The linear conflict heuristic extends the Manhattan heuristic: if two tiles in the same row or column both need to travel past each other to reach their goal positions, an additional cost of 2 is added .  Because each conflict requires at least one tile to move out of the way, this heuristic remains admissible while improving accuracy.  It is easy to check for conflicts in sliding puzzles, but in Sokoban the interplay of boxes and the player complicates the definition.  The idea of accounting for conflicts is used in Sokoban’s domain‑specific heuristics (see Enhanced Minimum Matching below).

Pros: more accurate than pure Manhattan distance; still admissible.  Cons: limited to problems where conflicts are easily defined (e.g., sliding tiles); only accounts for simple interactions; additional computational overhead.

Pattern database heuristics (PDB)

Pattern databases precompute exact solution costs for subproblems and store them in a lookup table.  Culberson & Schaeffer’s slides describe them as a human‑like way to solve complex puzzles: solve part of the problem optimally and use that cost as a lower bound for the full problem .  When solving the sliding‑tile puzzle, one can store the minimal number of moves required to place a subset of tiles (a pattern) and sum multiple PDB values (under appropriate cost partitioning) to obtain strong admissible heuristics .  Additive pattern databases partition the problem into disjoint subproblems so that their costs can be added without overestimating .  Pattern databases can also use symmetry: by reflecting the puzzle, one can obtain additional cost bounds and take the maximum over all reflections .

Pros: extremely informative; substantially reduce the number of nodes expanded; can combine multiple patterns additively for stronger heuristics.  Cons: large memory requirements (tables can contain millions of entries); time required to build PDBs; patterns must be carefully chosen; for Sokoban, the mapping of boxes to goals is not fixed, making direct application difficult .

Weighted and dynamically weighted heuristics

Classic A* guarantees optimal solutions when using an admissible heuristic but often explores many nodes with equal f‑costs.  Weighted A* multiplies the heuristic by a constant w>1 to bias the search towards nodes believed to be closer to the goal.  The TU Berlin thesis notes that using a modified heuristic h'(n)=w\times h(n) allows A* to find a solution whose cost is at most w times optimal .  Dynamic weighting adjusts the weight during the search, for example using \hat{h}(n)=h(n)\cdot(1+\epsilon(n)) with a function \epsilon(n) that decreases as the depth increases .  Both methods are \epsilon-admissible: they relinquish strict admissibility to gain speed.

Pros: often dramatically reduce search time; tunable trade‑off between optimality and speed.  Cons: solutions may be suboptimal; need to choose appropriate weights; not suitable when optimal solutions are required.

Symmetry and reflection heuristics

Symmetries in grid puzzles create equivalent states.  The pattern‑database slides show that reflecting the puzzle horizontally, vertically or diagonally can provide additional cost bounds; one takes the maximum of the heuristic values over all reflections .  Symmetry detection can also be used to prune duplicate states.  Implementing symmetry heuristics requires mapping actions under reflections and handling coordinate transforms.

Pros: prunes symmetrical branches; improves heuristic values.  Cons: increases lookup complexity; benefit depends on the amount of symmetry; not always applicable to asymmetric levels.

Domain‑Specific Heuristics for Sokoban

Deadlock detection and dead squares

Many Sokoban states are unsolvable because a box is pushed into a corner or trapped by other boxes and walls.  A deadlock detection heuristic assigns infinite cost to these states to prune them.  The Rolling Stone solver and later works identify dead squares (squares where pushing a box permanently blocks it) and check whether any box is on a dead square .  Deadlock detection can be simple (corner and edge squares) or complex, involving patterns of multiple boxes; the Learning Deadlocks thesis notes that a deadlock can involve a subset of stones such that, if the stones are removed, the remaining puzzle becomes solvable .  Retrograde analysis can precompute deadlocks, but building these tables is expensive .  Recent approaches train neural networks to detect deadlocks, achieving performance comparable to pattern databases .

Pros: dramatically reduces search by eliminating unsolvable branches; simple dead square detection is easy to implement.  Cons: full deadlock tables require significant preprocessing; more complex deadlock detection may misclassify states if learned heuristics are used; admissibility is lost if deadlocks are incorrectly identified.

Minimum matching and Enhanced Minimum Matching (EMM)

Sokoban differs from sliding puzzles because the mapping of boxes to goals is not fixed.  A heuristic must match each box to a distinct goal in a way that minimises total pushes.  The Minimum Matching heuristic computes, for each box–goal pair, the minimum number of pushes needed to move the box to that goal, ignoring other boxes.  It then solves a minimum‑cost perfect matching problem to assign boxes to goals (typically via the Hungarian algorithm) and uses the sum as a lower bound.  This heuristic is admissible but does not account for interactions.

Junghanns & Schaeffer introduced Enhanced Minimum Matching (EMM).  The Learning Deadlocks thesis describes EMM as the standard heuristic for Sokoban .  EMM adds two improvements:

1. Backout conflicts: if a box must be moved away from the goal before it can be pushed towards it (because the player’s position matters), the extra effort is reflected in the precomputed push distance .
2. Linear conflicts: analogous to sliding puzzles, if two boxes block each other’s optimal paths, an additional cost of 2 is added .  Figure 2.3b in the thesis shows an example where two stones in row 3 prevent each other from reaching their goals .

EMM detects simple deadlocks: a stone in a dead square or a matching that cannot assign all stones to unique goals implies a deadlock .  EMM is consistent (monotonic), making it suitable for A*.  However, computing the matching requires solving a weighted bipartite matching problem at each node; the Hungarian algorithm runs in O(k^3) time for k boxes and can become a bottleneck.  Optimisations include caching matching results and incrementally updating matchings as boxes move.

Pros: provides a strong admissible lower bound; accounts for interactions between boxes; detects many deadlocks; widely used in state‑of‑the‑art solvers.  Cons: expensive to compute; requires precomputed push distances; still ignores complex multi‑box interactions; memory‑intensive if large tables are cached.

Intermediate and multiple goal state pattern databases (IPDB/MPDB)

Pereira et al. observed that traditional PDB heuristics struggle with transportation domains like Sokoban because the correspondence between boxes and goals is not fixed .  To address this, they introduced Intermediate Pattern Databases (IPDB).  The idea is to decompose the puzzle into a maze zone and a goal zone, use a PDB to guide the search through the maze zone towards an intermediate goal, and then apply minimum matching within the goal zone.  IPDB provides strong lower bounds but only detects deadlocks in the maze zone .

Later, Multiple goal state PDBs (MPDB) were proposed to detect deadlocks directly.  An MPDB abstracts the positions of up to p boxes (e.g., MPDB‑4 uses four boxes) and stores whether the abstracted configuration is a deadlock .  During search, if the current configuration matches an entry in the MPDB, the state is pruned.  MPDBs can detect deadlocks of order up to p but require memory exponential in p.  Combining EMM with MPDB‑4 deadlock pruning allowed A* to solve two additional standard Sokoban instances while expanding an order of magnitude fewer states .

Pros: produces very informative heuristics; improves deadlock detection; IPDB/MPDB heuristics remain admissible.  Cons: building PDBs is computationally expensive; table sizes grow rapidly with the number of boxes included; IPDB works only when an instance can be decomposed cleanly into maze and goal zones; MPDB heuristics alone may produce weaker estimates than EMM .

Feature‑based heuristics and the FESS algorithm

The Festival Solver’s Feature Space Search (FESS) uses domain‑specific “features” instead of a single heuristic value.  Shoham and Schaeffer introduced the Connectivity Feature, which increases the heuristic when the room is divided by boxes into disconnected areas that the player cannot reach .  FESS selects moves using these features rather than estimating remaining pushes.  It is complete but prioritises finding any solution quickly rather than the optimal solution .  Feature‑based heuristics are similar to greedy best‑first search and can incorporate additional domain knowledge (e.g., avoiding tunnels or pushing boxes into walls).

Pros: can solve all standard Sokoban levels quickly ; features capture rich domain knowledge; easy to tailor to specific puzzle layouts.  Cons: not admissible; may miss optimal solutions; designing effective features is non‑trivial; behaviour may vary significantly between levels.

Macro moves, tunnels and goal rooms

Many Sokoban levels contain long corridors or tunnels where the player has only one way to push a box.  Macro moves collapse a sequence of forced pushes into a single action, reducing the branching factor.  Rolling Stone’s macro‑move mechanism includes goal macros (pushing a box into a goal along a tunnel) and tunnel macros (pushing a box through a corridor) .  When a goal macro is available, all other pushes from that state are ignored because placing a box on a goal is always beneficial .  Macros are not heuristics per se but search enhancements; however, they interact with heuristics by reducing the number of successor states.

Pros: drastically reduces branching factor in constrained areas; accelerates search; does not affect admissibility.  Cons: requires detection of tunnels and goal rooms; may not provide benefits on open levels; implementation complexity.

Weighted A* for Sokoban

Weighted and dynamically weighted A* can be used in Sokoban to trade solution quality for speed.  The TU Berlin thesis describes using static weighting h'(n)=w\times h(n) and dynamic weighting h'(n)=h(n)(1+\epsilon(n)) to find solutions within a factor of w of optimal .  These techniques are applicable to any heuristic; for instance, one can weight EMM to produce faster but potentially suboptimal solutions.  Dynamic weighting reduces the weight as the search progresses, often improving solution quality.

Pros: faster search when exact optimality is unnecessary; simple to implement.  Cons: cannot guarantee optimal solutions; choosing appropriate weight functions is problem‑dependent.

Learning‑based heuristics

Machine learning has been explored to improve deadlock detection.  Boelter’s thesis trains neural networks to classify Sokoban states as deadlocks or alive and achieves performance comparable to MPDB‑4 while expanding an order of magnitude fewer states .  Learning‑based heuristics can also predict cost‑to‑go estimates, but training data generation and generalisation remain challenging.  Furthermore, learned heuristics may not be admissible and thus cannot be used with optimal A* without modification.

Pros: automatically capture complex patterns; can outperform hand‑crafted heuristics on specific instances.  Cons: require a large training set; may misclassify states or overestimate costs; typically non‑admissible.

Practical Considerations and Recommendations

1. Use a simple admissible baseline.  For many small PuzzleScript puzzles, a heuristic based on precomputed push distances (Manhattan distance modified to account for walls) and minimum matching may suffice.  Tim Wheeler’s blog suggests precomputing the minimum pushes from each tile to the nearest goal, ignoring other boxes but respecting walls; summing these values yields a better lower bound than a simple sum of straight‑line distances .  This heuristic also detects unreachable boxes (infinite distance) and thus prunes trivial deadlocks .
2. Incorporate matching and conflicts for stronger admissibility.  Implement EMM by computing minimum push distances, solving a minimum‑cost perfect matching, and adding backout and linear conflict penalties.  Cache results and update incrementally to reduce overhead.  For puzzle types where boxes and goals are distinguishable, assign each box to its specific goal; for puzzles with interchangeable goals, use minimum matching.
3. Detect deadlocks aggressively.  Mark obvious dead squares (non‑goal corners and dead corridors) and prune states with boxes on those squares.  Consider building small pattern databases (e.g., MPDB‑2) or training a classifier to detect multi‑box deadlocks if puzzles are large.  Even simple deadlock detection yields huge performance gains by avoiding fruitless branches.
4. Exploit structural features.  Identify tunnels and goal rooms and create macro moves.  Macro moves reduce the search depth and branching factor; they also help heuristics by skipping intermediate states that contribute no meaningful information.
5. Balance optimality and speed.  If optimal solutions are necessary (e.g., counting pushes in Sokoban), stick to admissible heuristics like EMM or IPDB/MPDB.  Otherwise, consider weighted A* or feature‑based search (e.g., FESS) to obtain solutions quickly.  Weighted heuristics can often solve puzzles that strict A* cannot finish within practical time limits.
6. Consider pattern databases for repeated puzzles.  When solving many instances of the same puzzle type, investing time to build PDBs may pay off.  For unique PuzzleScript levels, the overhead may not justify the benefits unless the instances share structure.  Use automatically generated patterns (Haslum & Botea’s domain‑independent method ) or manually select patterns that capture key interactions.

Conclusion

Heuristic quality largely determines the success of an A* solver in Sokoban and other grid‑based puzzles.  Simple heuristics like Manhattan distance provide inexpensive guidance but often fail to prune deep search trees.  Enhancing heuristics to account for box–goal assignments, interactions (linear and backout conflicts) and deadlocks leads to significantly better performance.  Pattern database techniques offer powerful admissible heuristics but require substantial memory and precomputation.  In domains like Sokoban, where the assignment of boxes to goals is not fixed and deadlocks abound, Enhanced Minimum Matching combined with deadlock detection remains the state of the art for optimal solving.  For time‑constrained or approximate solving, weighted A* and feature‑based methods like FESS provide attractive alternatives.  Ultimately, the choice of heuristic should balance computational cost, memory usage, domain knowledge and the need for optimality.