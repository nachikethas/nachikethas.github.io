---
layout: post
title: Essential PCA
date: 2016-10-29
tags: [statistics, learning, algorithms]
---

Assume that we are given a matrix $$X \in \mathbb{R}^{n \times p}$$.
Each row of the matrix $$X$$ is considered to be an observation
represented by a data vector that measures $$p$$ features of some
phenomenon. We can think of Principal Component Analysis (PCA) as trying
to trying to solve two related problems.

1. [*Compression*](#compression): How do we represent the data matrix $$X$$
   succinctly? In other words, is there an efficient representation of
   $$X$$ that uses less space while not sacrificing too much accuracy?
2. [*Prediction*](#prediction): Which (linear) combinations of the $$p$$
   features best explain or influence the data?

<!--more-->

Mathematically, the compression problem can be formulated as trying to
identify a low rank approximation of $$X$$ that minimizes its [Frobenius
norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm). Assume
that the rank of $$X$$ is given by $$r \le \min\{n, p\}.$$ Then for some
given $$l \le r,$$ we want to find $$X_l \in \mathbb{R}^{n \times p}$$
such that $$X_l$$ is the solution to the optimization problem

$$
\begin{align}
& \underset{L \in \mathbb{R}^{n \times p}}{\text{minimize}}
& & || X - L ||_F^2 \\
& \text{subject to}
& & \text{rank}(L) \le l.
\end{align}
$$

As it happens, the solution to the above optimization problem will allow
us also to identify the combination of the predictors that best capture
the variability in the data. It also
happens to have a nice analytical solution.

### Low Rank Matrix Approximation Theorem
Represent $$X$$ using its [singular value
decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition). Then,

$$
\begin{align}
X &= UXV' \\
  &= \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top,
\end{align}
$$

where we assume that the singular values are sorted in descending
order. For some $$ l \le r $$ define the matrix

$$
X_l = \sum_{i=1}^l \sigma_i \mathbf{u}_i \mathbf{v}_i^\top.
$$

Then the [low rank matrix approximation
theorem](http://link.springer.com/article/10.1007%2FBF02288367) tells us
that $$ X_l $$ minimizes the squared Frobenius norm $$ || X - L ||_F^2
$$ among all matrices $$L$$ with rank less than or equal to $$l$$.
Moreover, we know that the following properties hold:
* $$X_l$$ is unique if and only if $$\sigma_l > \sigma_{l-1}$$.
* The minimum is given by $$\sum_{i = l+1}^r \sigma_i^2$$.

### Compression {#compression}

Represent the data matrix $$X$$ as $$[\mathbf{x}_1 \dots
\mathbf{x}_n]^\top$$ where $$\mathbf{x}_i$$ is a column vector that
represents the data corresponding to observation $$i$$ where $$i \in
\{1\dots n\}$$. Then,

$$
\begin{align}
X_l &= \sum_{i=1}^l \sigma_i \mathbf{u}_i \mathbf{v}_i^\top\\
    &= \sum_{i=1}^l (X\mathbf{v}_i)\mathbf{v}_i^\top,
\end{align}
$$

where the second equality follows from the fact that the columns of
$$V$$ are orthonormal. Looking at the above a bit more closely, we see
that the $$j$$-th row of $$X_l$$ can be represented as

$$
(\mathbf{x}_j^l)^\top = \sum_{i=1}^l (\mathbf{x}_j^\top\mathbf{v}_i)\mathbf{v}_i^\top.
$$

The above equation gives us precisely the lower dimensional
representation of the observation $$j$$. In fact, the above equation
represents the observation $$j$$ as the sum of its projections along the
basis vectors $$\{\mathbf{v}_i : i \in 1\dots l\}$$, which can be
interpreted as the reduced set of features that we use for representing
our observations. In short, we represent each observation by its
projection onto the subspace spanned by the basis vectors
$$\{\mathbf{v}_i : i \in 1\dots l\}$$. Since the projections lie in a
lower dimensional subspace ($$l \le r$$), we obtain a compressed representation of the
original data.

### Prediction {#prediction}

At the outset we stated that PCA identifies the (linear)
combinations of features that best explain or influence the data. Let's
make this notion slightly more precise. Consider a random vector
$$\mathbf{X}$$ with mean zero that takes values in $$\mathbb{R}^p$$
according to some unknown distribution $$\mathbb{P}$$. Assume that the
covariance matrix of $$\mathbf{X}$$ exists and is given by
$$\mathrm{E}[\mathbf{X}^\top\mathbf{X}] = \Sigma$$. Think of each data
vector $$\mathbf{x}_j$$ as a realization of $$\mathbf{X}$$ that we get
to observe. We have $$n$$ such data vectors which constitutes our data
matrix $$X$$.

Identifying the direction along which $$\mathbf{X}$$ varies the most can
be formulated as identifying the vector $$\mathbf{b} \in \mathbb{R}^p$$
that maximizes $$\mathrm{Var}(\mathbf{b}^\top \mathbf{X})$$. This
can be written down as the optimization problem,

$$
\max_{||\mathbf{b}||_2 = 1} \mathrm{Var}(\mathbf{b}^\top \mathbf{X}) =
\max_{||\mathbf{b}||_2 = 1} \mathbf{b}^\top \Sigma \mathbf{b}.
$$

The [Rayleigh-Ritz
Theorem](http://www.cis.upenn.edu/~cis515/cis515-15-spectral-clust-appA.pdf)
tells us that the maximum value of the above optimization is
the largest eigenvalue, say $$\lambda_r$$, of the covariance matrix $$
\Sigma $$. This value is obtained when $$\mathbf{b}$$ is the
normalized eigenvector, say $$\mathbf{v}_r$$, corresponding to the
largest eigenvalue. Indeed, we can obtain the second largest eigenvalue
by adding the constraint that $$\mathbf{b}$$ should be perpendicular to
$$\mathbf{v}_r$$. We can obtain the other eigenvalues and eigenvectors
in a similar fashion.

In the singular value decomposition of $$X$$, the column vectors of
$$V$$ correspond to the eigenvectors of $$X^\top X$$, which is an
approximation of $$\Sigma$$ scaled by $$n$$. In fact, by [the law of
large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers),

$$
\lim_{n \to \infty} \frac{1}{n} \left(X^\top X\right) = \Sigma.
$$

Thus the $$l$$ directions that
PCA chooses are precisely the directions along which the data is most
variable.
