# escpy: Necessary and sufficient elastic stability conditions in various crystal systems

A crystalline structure is stable, in the absence of external load and in the harmonic approximation, if and only if

1. all its phonon modes have positive frequencies for all wave vectors (dynamical stability);
2. the elastic energy is always positive. This condition is called the *elastic stability criterion*. As first noted by Born (See Ref. 2), it is mathematically equivalent to the following **necessary and sufficient stability conditions**:
   1. The matrix $\mathrm{ C }$ is definite positive;
   2. all eigenvalues of $\mathrm{ C }$ are positive;
   3. all the leading principal minors of $\mathrm{ C }$ (determinants of its upper-left $k \times k$ submatrix, where $1 \le k \le 6$) are positive, a property known as *Sylvester’s criterion*;
   4. an arbitrary set of minors of $\mathrm{ C }$ are all positive. It can be useful to choose, for example, the trailing minors, or any other set. 

These are 4 possible formulations of the generic Born elastic stability conditions for an **unstressed** crystal. They are valid regardless of the symmetry of the crystal studied, and are not linear. 

Please check Ref. 1 for more information.

## References

1. Mouhat, F. & Coudert, F.-X. Necessary and sufficient elastic stability conditions in various crystal systems. *Physical Review B* **90,** 224104 (2014).
2. Born, M. On the stability of crystal lattices. I. *Mathematical Proceedings of the Cambridge Philosophical Society* **36,** 160–172 (1940).