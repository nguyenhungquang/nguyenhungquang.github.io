---
layout: post
title: Introductory PAC-Bayes
---
Instead of choosing an optimal parameter $$\phi$$ in parameter space, one could instead infer with a probablity measure on this space. Let $$(A, \mathcal{A})$$ be a measurable space of parameter, our target is finding a p.m $$q$$ on this space. This p.m can be computed by Bayesian approach as posterior distribution with training dataset $$\mathcal{D}_n$$ and prior $$p$$

\begin{aligned}
q( \phi|\mathcal{D}_n) \propto \mathcal{L}(\mathcal{D}_n|\phi) \times p(\phi)
\end{aligned}

where $$\mathcal{L}(\mathcal{D}_n|\phi)$$ is loss with parameter $$\phi$$ on training data, more particular it is average of loss on every data point. One also has $$\mathcal{L}(\mathcal{D}|\phi)$$ is loss when intergrating over data distribution.\\
Generalization ability of model can be measured by difference between loss on training data and loss over data distribution. In classical PAC framework, it can be computed from complexity of hypothesis class term (VC-dimension, Rademacher complexity, ...) and the number of sample. For this approach, complexity is substituted with distance between posterior and prior distribution, which is computed by KL divergence. Thus, all possible posterior must be absolutely continuous wrt prior distribution.

**Theorem** Given a loss function $$l$$ bounded between $$0$$ and $$1$$, a prior distribution $$p$$ over hypothesis space, a real number $$\delta \in (0,1]$$, $$\forall q$$ such that $$q \ll p$$, one has

\begin{aligned}
\mathbb{E}_{q}\mathcal{L}(\mathcal{D}|\phi) \leq \frac{1}{1-e^{-1}}(1-e^{-\mathbb{E}_q\mathcal{L}(\mathcal{D}_n|\phi)-\frac{1}{n}(KL(q||p)+ln\frac{1}{\delta})})
\end{aligned} 

 $$\text{with probablity at least }1-\delta$$
 
 This results suggests minimizing loss over training data as well as distance between posterior and prior. Concretely, one minimizes the following function
 \begin{aligned}
 n\mathbb{E}_q\mathcal{L}(\mathcal{D}_n|\phi)+KL(q||p)
 \end{aligned}
 The unique solution of this objective function is Gibbs posterior
 \begin{aligned}
 p^*(\phi) = \frac{1}{Z}p(\phi)e^{-n\mathcal{L}(D_n|\phi)}
 \end{aligned}
 One can also multiply likelihood term with a positive number $$\lambda$$ to obtain general tempered Bayesian learning objective function. The following lemma gives a more general result on the correctness of Gibbs posterior.
 
**Lemma** Given a measurable function $$h: A \rightarrow \mathbb{R}$$ such that $$\mathbb{E}_p[exp \circ h] < \infty$$ and $$h$$ is upper bounded on the support of $$p$$, one has 
 
 \begin{aligned}
 log\mathbb{E}_p[exp\circ h] = sup\_{q \in M(p)}(\mathbb{E}_qh-KL(q||p))
 \end{aligned}
  
 where $$M(p)$$ is the set of p.m which is absolutely continuous wrt $$p$$. The supremum on the right hand side is obtained for the Gibbs distribution $$g$$
 
 \begin{aligned}
 \frac{dg}{dp} = \frac{exp\circ h}{\mathbb{E}_p[{exp\circ h}]}
 \end{aligned}
 
**Proof** \\
$$g$$ has the same support as $$p$$, thus $$q \ll g$$ 

 $$\begin{aligned}
 -KL(q||g) &= -\mathbb{E}_q[log(\frac{dq}{dg})] \\
 &= -\mathbb{E}_q[log(\frac{dq}{dp})] + \mathbb{E}_q[log(\frac{dg}{dp})] \\\
 &= -KL(q||p) + \mathbb{E}_q[h]-log\mathbb{E}_p[exp\circ h] 
 \end{aligned}$$

The above Gibbs posterior is obtained by substituting $$-h$$ with empirical loss (minimizing loss is equivalent to maximizing $$h$$), however it only works for bounded loss function. For unbounded case, like negative log likelihood, one can set value outside a bound by zero to apply this result.

In practice, Gibbs posterior is obtained by MCMC methods or variational inference methods.

This approach also has connection with Bayesian inference method, which computes posterior 

\begin{aligned}
p(\phi|\mathcal{D}\_n) \propto p(\phi)p(\mathcal{D}\_n|\phi)
\end{aligned}

In PAC-Bayes approach, considering negative log likelihood loss 

\begin{aligned}
l_{nll}(x,y,\phi) = -logp(y|x,\phi) \quad (x,y) \sim \mathcal{D}
\end{aligned}

 Assuming each data point is independent, loss over training data is computed as
 
 \begin{aligned}
 \mathcal{L}_{nll}(\mathcal{D}_n|\phi) = -\frac{1}{n}\prod logp(y|x,\phi) = -\frac{1}{n}p(\mathcal{D}_n|\phi)
 \end{aligned}
 Gibbs posterior is exactly Bayesian posterior, where $$Z$$ corresponds to marginal likelihood. 
 
 \begin{aligned}
 p^*(\phi|\mathcal{D}\_n) = \frac{1}{Z}p(\phi)e^{-n\mathcal{L}\_{nll}(D_n|\phi)} = \frac{1}{p(\mathcal{D}_n)}p(\phi)p(\mathcal{D}\_n|\phi)
 \end{aligned}
 
From the above lemma, one can see that optimal value of objective function in PAC-Bayes becomes marginal likelihood $$Z = p(\mathcal{D}_n)$$

## Reference
- [PAC-Bayesian Theory Meets Bayesian Inference](https://papers.nips.cc/paper/2016/file/84d2004bf28a2095230e8e14993d398d-Paper.pdf)
- [A Primer on PAC-Bayesian Learning](https://arxiv.org/pdf/1901.05353.pdf)