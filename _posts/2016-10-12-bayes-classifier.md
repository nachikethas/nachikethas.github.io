---
layout: post
title: Bayes Classifier with Asymmetric Costs
date: 2016-10-12
tags: [statistics, bayes, classification]
---

Thanks to [Prof. Larry](http://www-bcf.usc.edu/~larry/) for this problem!

Consider the following binary classification problem. Every individual
of a population is associated with an independent replicate of the pair
$$ (\mathbf{X}, Y) $$, having known joint distribution and where the
(observed) covariate $$\mathbf{X}$$ has a (marginal) distribution
$$\pi$$, and the (unobserved) response $$Y \in \{-1, 1\} $$. Suppose
the costs of misclassifying an individual with $$Y = 1$$ and $$Y =
-1$$ are $$a > 0$$ and $$b > 0$$, respectively. What's the Bayes
decision rule?

A classification rule, say $$g$$, is a function of $$\mathbf{X}$$
taking values in $$\{-1, 1\}$$. We incur a loss when,

* $$Y=1$$ and we predict $$-1$$ (i.e $$g(\mathbf{X}) = -1$$).
  The loss in this case is $$a$$.
* $$Y=-1$$ and we predict $$1$$ (i.e $$g(\mathbf{X}) = 1$$).
  The loss in this case is $$b$$.

Thus, the expected loss or cost $$L(g)$$ of using the classification
rule $$g$$ may be expressed as

$$
    L(g) = aP[Y=1, g(\mathbf{X}) = -1] + bP[Y=-1, g(\mathbf{X}) = 1].
$$

To compute the above expected loss, it is useful to define the following
quantities. Define the random variable,

$$
Z =
\begin{cases}
    a & \text{if } Y = 1 \\
    -b & \text{if } Y = -1.
\end{cases}
$$

Moreover,

* $$\eta(\mathbf{X}) = \mathbb{E}[Z | \mathbf{X}]$$
* $$R_1$$ denotes the set of $$\mathbf{X}$$'s on which $$g$$ takes
  the value $$1$$.
* Similarly, $$R_{-1}$$ denotes the set of $$\mathbf{X}$$'s on which $$g$$ takes
  the value $$-1$$.

A touch of algebra yields,
$$ \begin{align}
P[Y=1 | \mathbf{X}] &= \frac{\eta(\mathbf{X}) + b}{a + b}, \\\\
P[Y=-1 | \mathbf{X}] &= \frac{a - \eta(\mathbf{X})}{a + b}.
\end{align} $$

This enables us to compute the expected cost as
$$ \begin{align}
L(g) &= a \int_{R_{-1}} P[Y=1 | \mathbf{x}] \pi(\mathbf{x}) \,
    \mathrm{d}\mathbf{x} + b \int_{R_1} P[Y= -1 | \mathbf{x}]
    \pi(\mathbf{x}) \, \mathrm{d}\mathbf{x} \\\\
    &= \frac{1}{a + b}
    \left(a \int_{R_{-1}} \eta(\mathbf{x}) \pi(\mathbf{x}) \,\mathrm{d}\mathbf{x} -
    b\int_{R_1}\eta(\mathbf{x})\pi(\mathbf{x})\,\mathrm{d}\mathbf{x}\right)
    + \frac{ab}{a + b}.
\end{align} $$

Thus the Bayes decision rule, $$g^*$$ that minimizes the cost is the
one with regions $$R_1, R_{-1}$$ chosen to minimize

$$
    \left(a \int_{R_{-1}} \eta(\mathbf{x}) \pi(\mathbf{x}) \,\mathrm{d}\mathbf{x} -
    b\int_{R_1}\eta(\mathbf{x})\pi(\mathbf{x})\,\mathrm{d}\mathbf{x}\right).
$$

How do we choose these regions? Pick any $$\mathbf{x}$$. If
$$\eta(\mathbf{x}) < 0,$$ then we want that $$\mathbf{x}$$ to be
part of $$R_{-1}$$ since otherwise that $$\mathbf{x}$$ would only
serve to increase the above expression. This yields

$$
R_1 = \{ \mathbf{x} : \eta(\mathbf{x}) \ge 0 \},
$$

and

$$
R_{-1} = \{ \mathbf{x} : \eta(\mathbf{x}) < 0 \}.
$$

Since by definition

$$
\eta(\mathbf{x}) = a P[Y=1|\mathbf{x}] - b\left(1 - P[Y=1|\mathbf{x}]\right),
$$

checking if $$\eta(\mathbf{x}) < 0,$$ amounts to checking if

$$
P[Y=1|\mathbf{x}] < \frac{b}{a + b}.
$$

Similarly, checking if $$\eta(\mathbf{x}) \ge 0,$$ amounts to checking if

$$
P[Y=1|\mathbf{x}] \ge \frac{b}{a + b}.
$$

This yields the optimum decision rule,

$$
g^*(\mathbf{x}) =
\begin{cases}
    1 & \text{if } P[Y=1|\mathbf{x}] \ge \frac{b}{a + b} \\
   -1 & \text{if } P[Y=1|\mathbf{x}] < \frac{b}{a + b}.
\end{cases}
$$

Intuitively, if $$a$$ is much larger than $$b$$, then we care much
more about (not) misclassifying $$Y = 1$$ which makes us more likely
to classify a given covariate $$\mathbf{x}$$ as $$1$$. The decision
rule derived above satisfies this intuition. In the limit $$a \to
\infty$$, it is easy to see that the classifier will always classify an
observation as $$1$$. Finally, when $$a$$ and $$b$$ are the same,
we recover the original Bayes classifier which simply looks at which
response is most likely:

$$
g^*(\mathbf{x}) =
\begin{cases}
    1 & \text{if } P[Y=1|\mathbf{x}] \ge \frac{1}{2} \\
   -1 & \text{if } P[Y=1|\mathbf{x}] <   \frac{1}{2}.
\end{cases}
$$
