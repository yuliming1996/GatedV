"""Exact original conditional-bootstrap and seed-interval kernels.

Medium and small use their distinct predeclared RNG seeds/reductions.
"""
import math
import statistics
import numpy as np
T_CRITICAL=2.7764451051977987

def require(ok, message):
    if not ok: raise ValueError(message)

def medium_cluster_interval(base,candidate,counts,labels):
    groups={}
    for i,label in enumerate(labels):groups.setdefault(label,[]).append(i)
    difference=np.mean(np.asarray(candidate)-np.asarray(base),axis=0)
    nums=np.array([math.fsum(float(difference[i]) for i in ids) for ids in groups.values()])
    dens=np.array([sum(int(counts[i]) for i in ids) for ids in groups.values()])
    rng=np.random.Generator(np.random.PCG64(2026092901));samples=[]
    for _ in range(2000):
        ids=rng.integers(0,len(groups),size=len(groups));samples.append(float(nums[ids].sum()/dens[ids].sum()))
    ci=np.quantile(samples,[.025,.975],method='linear').tolist()
    return {'groups':len(groups),'seed':2026092901,'replicates':2000,'nll_difference_ci95':ci,
            'pooled_nll_exponential_change_pct_ci95':[100*math.expm1(x) for x in ci],
            'scope':'Conditional on these three seed pairs; not a CI for arithmetic mean PPL ratio.'}

def quantile(values,p):
    ordered=sorted(values);z=(len(ordered)-1)*p;l=math.floor(z);u=math.ceil(z)
    return ordered[l]+(z-l)*(ordered[u]-ordered[l])


def small_cluster_interval(base,candidate,counts,labels):
    """Groups in first appearance of frozen document order, never label sort."""
    grouped={}
    for i,label in enumerate(labels):grouped.setdefault(label,[]).append(i)
    require(len(grouped)>=2,'Need >=2 clusters')
    numerator=[];denominator=[]
    for indexes in grouped.values():
        numerator.append(math.fsum((math.fsum(float(row[i]) for row in candidate)-
                                    math.fsum(float(row[i]) for row in base))/5 for i in indexes))
        denominator.append(sum(int(counts[i]) for i in indexes))
    rng=np.random.Generator(np.random.PCG64(2026092801))
    samples=[]
    for _ in range(2000):
        draw=rng.integers(0,len(grouped),size=len(grouped)).tolist()
        samples.append(math.fsum(numerator[k] for k in draw)/sum(denominator[k] for k in draw))
    bounds=[quantile(samples,.025),quantile(samples,.975)]
    return {'groups':len(grouped),'replicates':2000,'seed':2026092801,
            'group_order':'first appearance in frozen doc_id order',
            'nll_difference_ci95':bounds,'pooled_nll_exponential_change_pct_ci95':[100*math.expm1(v) for v in bounds],
            'scope':'Conditional on these five training seeds; not an interval for the ratio of arithmetic mean PPL and not independent training replicates.'}


def seed_uncertainty(differences):
    require(len(differences)==5 and all(math.isfinite(x) for x in differences),'Need five finite paired differences')
    mean=statistics.mean(differences);sd=statistics.stdev(differences);se=sd/math.sqrt(5)
    return {'paired_nll_differences':list(differences),'mean':mean,'sample_sd':sd,'standard_error':se,
            'df':4,'t_critical_two_sided_95':T_CRITICAL,'mean_nll_difference_t_ci95':[mean-T_CRITICAL*se,mean+T_CRITICAL*se],
            'scope':'Student-t approximation assumes approximately normal paired training-seed effects; only five independent seed pairs. Interpret alongside raw differences.'}


