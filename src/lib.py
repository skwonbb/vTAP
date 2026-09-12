"""vTAP and the comparison methods."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _t(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEV)


def standardize(Ztr, Zte):
    sc = StandardScaler().fit(Ztr)
    return sc.transform(Ztr), sc.transform(Zte)


def make_logits(Utr, ytr, k, n_classes, seed=0):
    head = _fit_linear(Utr, ytr, n_classes, seed=seed, epochs=300)

    def f(X):
        with torch.no_grad():
            return head(_t(X)).cpu().numpy()

    return f


def make_between_only(Utr, ytr, k, n_classes, seed=0):
    mu = Utr.mean(0)
    cls = np.unique(ytr)
    M = np.stack([Utr[ytr == c].mean(0) - mu for c in cls])
    n = np.array([(ytr == c).sum() for c in cls], dtype=float)
    SB = (M * n[:, None]).T @ M
    _, V = np.linalg.eigh(SB)                 
    W = np.ascontiguousarray(V[:, ::-1][:, :k])
    return lambda X: (X - mu) @ W


def make_lda(Utr, ytr, k, n_classes, seed=0):
    lda = LinearDiscriminantAnalysis(n_components=k).fit(Utr, ytr)
    return lda.transform


def make_lda_shrink(Utr, ytr, k, n_classes, seed=0):
    lda = LinearDiscriminantAnalysis(
        n_components=k, solver="eigen", shrinkage="auto").fit(Utr, ytr)
    return lda.transform


def make_lda_rff(Utr, ytr, k, n_classes, seed=0, D=2000):
    g = _median_gamma(Utr)
    lda = LinearDiscriminantAnalysis(n_components=k).fit(
        rff_transform(Utr, g, D, seed), ytr)
    return lambda X: lda.transform(rff_transform(X, g, D, seed))


def make_bottleneck(Utr, ytr, k, n_classes, seed=0, epochs=60):
    torch.manual_seed(seed)
    enc = nn.Linear(Utr.shape[1], k).to(DEV)
    head = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, n_classes)).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    X, y = _t(Utr), _t(ytr, torch.long)
    lossf = nn.CrossEntropyLoss()
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 512):
            idx = perm[i : i + 512]
            opt.zero_grad()
            lossf(head(enc(X[idx])), y[idx]).backward()
            opt.step()

    def f(Z):
        with torch.no_grad():
            return enc(_t(Z)).cpu().numpy()

    return f


def make_vib(Utr, ytr, k, n_classes, seed=0, beta=1e-2, epochs=80):
    torch.manual_seed(seed)
    d = Utr.shape[1]
    enc = nn.Linear(d, 2 * k).to(DEV)          
    head = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, n_classes)).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    X, y = _t(Utr), _t(ytr, torch.long)
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes))
    n = len(X)
    for ep in range(epochs):

        b = beta * min(1.0, (ep + 1) / (epochs * 0.3))
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 512):
            idx = perm[i : i + 512]
            h = enc(X[idx])
            mu, logvar = h[:, :k], h[:, k:].clamp(-8, 8)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
            opt.zero_grad()
            (lossf(head(z), y[idx]) + b * kl).backward()
            opt.step()

    def f(Z):
        with torch.no_grad():
            return enc(_t(Z))[:, :k].cpu().numpy()

    return f


def make_random(Utr, ytr, k, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(Utr.shape[1], k))
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return lambda X: X @ W


def make_pca(Utr, ytr, k, n_classes, seed=0):
    return PCA(n_components=k, random_state=seed).fit(Utr).transform


def _as_attr_cols(b):
    b = np.asarray(b)
    return b[:, None] if b.ndim == 1 else b


def inlp_projection(Utr, b_tr, n_iter=100, seed=0):
    from sklearn.linear_model import LogisticRegression

    Bc = _as_attr_cols(b_tr)
    d = Utr.shape[1]
    P = np.eye(d)
    for _ in range(n_iter):
        X = Utr @ P
        rows = []
        for j in range(Bc.shape[1]):
            y = Bc[:, j]
            if len(np.unique(y)) < 2:
                continue
            clf = LogisticRegression(max_iter=500, random_state=seed).fit(X, y)
            rows.append(np.atleast_2d(clf.coef_))
        if not rows:
            break
        W = np.vstack(rows)
        _, s, vt = np.linalg.svd(W, full_matrices=False)
        B = vt[s > 1e-10]
        if len(B) == 0:
            break
        P = P @ (np.eye(d) - B.T @ B)
    return P


def make_inlp(Utr, ytr, k, n_classes, seed=0, blabels=None):
    P = inlp_projection(Utr, blabels, seed=seed)
    return lambda X: X @ P


def make_inlp_bottleneck(Utr, ytr, k, n_classes, seed=0, blabels=None):
    P = inlp_projection(Utr, blabels, seed=seed)
    inner = make_bottleneck(Utr @ P, ytr, k, n_classes, seed=seed)
    return lambda X: inner(X @ P)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def make_adversarial(Utr, ytr, k, n_classes, seed=0, blabels=None,
                     lam=1.0, epochs=80):
    torch.manual_seed(seed)
    B = np.atleast_2d(np.asarray(blabels))
    if B.shape[0] != len(Utr):
        B = B.T
    n_adv = [int(B[:, j].max()) + 1 for j in range(B.shape[1])]

    enc = nn.Linear(Utr.shape[1], k).to(DEV)
    head = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, n_classes)).to(DEV)
    advs = [nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, c)).to(DEV)
            for c in n_adv]
    params = list(enc.parameters()) + list(head.parameters())
    for a in advs:
        params += list(a.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)

    X, y = _t(Utr), _t(ytr, torch.long)
    Bt = [_t(B[:, j], torch.long) for j in range(B.shape[1])]
    task_loss = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes))
    adv_loss = [nn.CrossEntropyLoss(weight=class_weights(B[:, j], n_adv[j]))
                for j in range(B.shape[1])]
    n = len(X)
    for ep in range(epochs):

        cur = lam * min(1.0, (ep + 1) / (epochs * 0.3))
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 512):
            idx = perm[i : i + 512]
            z = enc(X[idx])
            loss = task_loss(head(z), y[idx])
            zr = _GradReverse.apply(z, cur)
            for j, a in enumerate(advs):
                loss = loss + adv_loss[j](a(zr), Bt[j][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    def f(Z):
        with torch.no_grad():
            return enc(_t(Z)).cpu().numpy()

    return f


def leace_projection(Utr, b_tr, eps=1e-6):
    mu = Utr.mean(0)
    Xc = Utr - mu
    d = Xc.shape[1]
    S = (Xc.T @ Xc) / max(len(Xc) - 1, 1) + eps * np.eye(d)
    ev, V = np.linalg.eigh(S)
    ev = np.maximum(ev, eps)
    W = V @ np.diag(ev ** -0.5) @ V.T          
    Winv = V @ np.diag(ev ** 0.5) @ V.T


    Bc = _as_attr_cols(b_tr)
    Zoh = np.hstack([np.eye(int(Bc[:, j].max()) + 1)[Bc[:, j]]
                     for j in range(Bc.shape[1])])
    Zc = Zoh - Zoh.mean(0)
    Sxz = (Xc.T @ Zc) / max(len(Xc) - 1, 1)    

    M = W @ Sxz
    U_, s_, _ = np.linalg.svd(M, full_matrices=False)
    Bs = U_[:, s_ > eps]                       
    P = np.eye(d) - Winv @ Bs @ Bs.T @ W
    return mu, P


def make_leace(Utr, ytr, k, n_classes, seed=0, blabels=None):
    mu, P = leace_projection(Utr, blabels)
    return lambda X: (X - mu) @ P.T + mu


METHODS = {

    "logits": make_logits, "lda": make_lda, "lda_shrink": make_lda_shrink,
    "between": make_between_only,          
    "lda_rff": make_lda_rff,
    "bottleneck": make_bottleneck, "random": make_random, "pca": make_pca,

    "inlp": make_inlp, "inlp_bottleneck": make_inlp_bottleneck,
    "leace": make_leace,
}


VIB_VARIANTS = {f"vib_b{str(b).replace('.', '').replace('0', '', 1)}": b
                for b in (0.001, 0.01, 0.1)}
for _name, _b in VIB_VARIANTS.items():
    METHODS[_name] = (
        lambda Utr, ytr, k, nc, seed=0, _beta=_b:
        make_vib(Utr, ytr, k, nc, seed=seed, beta=_beta)
    )


ADV_VARIANTS = {
    f"adv{n}attr_lam{str(l).replace('.', '')}": (n, l)
    for n in (1, 2) for l in (0.3, 1.0, 3.0)
}
for _name, (_n, _l) in ADV_VARIANTS.items():
    METHODS[_name] = (
        lambda Utr, ytr, k, nc, seed=0, blabels=None, _lam=_l:
        make_adversarial(Utr, ytr, k, nc, seed=seed, blabels=blabels, lam=_lam)
    )


METHODS["inlp2attr"] = make_inlp
METHODS["leace2attr"] = make_leace

USES_B_LABELS = {"inlp", "inlp2attr", "inlp_bottleneck",
                 "leace", "leace2attr", *ADV_VARIANTS}
N_ADV_ATTRS = {k: v[0] for k, v in ADV_VARIANTS.items()}
N_ADV_ATTRS.update({"inlp2attr": 2, "leace2attr": 2})   


def rff_transform(X, gamma, D=2000, seed=0, batch=20000):
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = X.shape[1]
    W = (torch.randn(d, D, generator=g) * np.sqrt(2 * gamma)).to(DEV)
    b = (torch.rand(D, generator=g) * 2 * np.pi).to(DEV)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = _t(X[i : i + batch])
            out.append((torch.cos(xb @ W + b) * np.sqrt(2.0 / D)).cpu().numpy())
    return np.concatenate(out, 0)


def _median_gamma(X, n=2000, seed=0, block=16):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    S = X[idx]
    d2 = np.empty((len(S), len(S)), dtype=S.dtype)
    for i in range(0, len(S), block):
        d2[i:i + block] = ((S[i:i + block, None, :] - S[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    return 1.0 / med


def class_weights(y, n_classes):
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(cnt > 0, len(y) / np.maximum(cnt, 1), 0.0)
    return _t(w / w[w > 0].mean())


def _fit_linear(Xtr, ytr, n_classes, seed=0, wd=1e-4, epochs=300):
    torch.manual_seed(seed)
    m = nn.Linear(Xtr.shape[1], n_classes).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=wd)
    X, y = _t(Xtr), _t(ytr, torch.long)
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes))
    for _ in range(epochs):
        opt.zero_grad()
        lossf(m(X), y).backward()
        opt.step()
    return m


def _fit_mlp(Xtr, ytr, n_classes, hidden=256, seed=0, wd=1e-4, epochs=None):


    if epochs is None:
        steps_per_epoch = max(1, int(np.ceil(len(Xtr) / 2048)))
        epochs = int(max(40, min(300, 3000 // steps_per_epoch)))
    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
    ).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
    X, y = _t(Xtr), _t(ytr, torch.long)
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes))
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 2048):
            idx = perm[i : i + 2048]
            opt.zero_grad()
            lossf(m(X[idx]), y[idx]).backward()
            opt.step()
    return m


def _pred(m, X):
    with torch.no_grad():
        return m(_t(X)).argmax(1).cpu().numpy()


def probe_fitted(Xtr, ytr, seed_base=0, kernel_cap=40_000):
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    n_classes = int(ytr.max()) + 1
    out = []

    for wd in (1e-5, 1e-3, 1e-1):
        m = _fit_linear(Xs, ytr, n_classes, seed=seed_base, wd=wd)
        out.append(("lr", f"wd={wd}",
                    lambda X, m=m: _pred(m, sc.transform(X))))

    for hid in (256, 1024):
        for s in range(2):
            m = _fit_mlp(Xs, ytr, n_classes, hidden=hid, seed=seed_base + s)
            out.append(("mlp", f"h={hid},seed={s}",
                        lambda X, m=m: _pred(m, sc.transform(X))))

    rng = np.random.default_rng(seed_base)
    idx = rng.choice(len(Xs), min(kernel_cap, len(Xs)), replace=False)
    g0 = _median_gamma(Xs[idx])
    for mult in (0.5, 2.0):
        gam = g0 * mult
        E = rff_transform(Xs[idx], gam, 2000, seed_base)
        m = _fit_linear(E, ytr[idx], n_classes, seed=seed_base, wd=1e-4)
        out.append(("rff", f"gamma x{mult}",
                    lambda X, m=m, gam=gam: _pred(
                        m, rff_transform(sc.transform(X), gam, 2000, seed_base))))

    from sklearn.ensemble import HistGradientBoostingClassifier
    cw = {c: w for c, w in enumerate(
        len(ytr) / np.maximum(np.bincount(ytr, minlength=n_classes), 1))}
    for depth in (None, 6):
        gb = HistGradientBoostingClassifier(
            max_depth=depth, max_iter=100, early_stopping=True,
            n_iter_no_change=8, validation_fraction=0.15,
            random_state=seed_base, class_weight=cw).fit(Xs, ytr)
        out.append(("tree", f"depth={depth}",
                    lambda X, gb=gb: gb.predict(sc.transform(X))))

    from sklearn.neighbors import KNeighborsClassifier
    for nn in (15, 50):
        kn = KNeighborsClassifier(n_neighbors=nn, weights="distance").fit(Xs, ytr)
        out.append(("knn", f"k={nn}",
                    lambda X, kn=kn: kn.predict(sc.transform(X))))

    return out


def probe(Ztr, ytr, Zte, yte, seed_base=0, families=None,
          max_train=60_000, kernel_cap=40_000, svm_cap=4000):
    Ztr, Zte = standardize(Ztr, Zte)
    n_classes = int(max(ytr.max(), yte.max())) + 1
    runs = []
    families = families or ("lr", "mlp", "rff", "tree", "knn")

    rng = np.random.default_rng(seed_base)
    if len(Ztr) > max_train:
        sub = rng.choice(len(Ztr), max_train, replace=False)
        Ztr_p, ytr_p = Ztr[sub], ytr[sub]
    else:
        Ztr_p, ytr_p = Ztr, ytr

    if "lr" in families:
        for wd in (1e-5, 1e-3, 1e-1):
            t0 = time.time()
            m = _fit_linear(Ztr_p, ytr_p, n_classes, seed=seed_base, wd=wd)
            runs.append(_score("lr", f"wd={wd}", m, Ztr_p, ytr_p, Zte, yte, t0))

    if "mlp" in families:

        for hid in (256, 1024):
            for s in range(2):
                t0 = time.time()
                m = _fit_mlp(Ztr_p, ytr_p, n_classes, hidden=hid, seed=seed_base + s)
                runs.append(_score("mlp", f"h={hid},seed={s}", m,
                                   Ztr_p, ytr_p, Zte, yte, t0))

    if "rff" in families:
        idx = rng.choice(len(Ztr), min(kernel_cap, len(Ztr)), replace=False)
        g = _median_gamma(Ztr[idx])
        for mult in (0.5, 2.0):                      
            t0 = time.time()
            Etr = rff_transform(Ztr[idx], g * mult, 2000, seed_base)
            Ete = rff_transform(Zte, g * mult, 2000, seed_base)
            m = _fit_linear(Etr, ytr[idx], n_classes, seed=seed_base, wd=1e-4)
            runs.append(_score("rff", f"gamma x{mult}", m, Etr, ytr[idx], Ete, yte, t0))

    if "tree" in families:


        from sklearn.ensemble import HistGradientBoostingClassifier

        cw = {c: w for c, w in enumerate(
            len(ytr_p) / np.maximum(np.bincount(ytr_p, minlength=n_classes), 1))}


        for depth in (None, 6):
            t0 = time.time()
            g = HistGradientBoostingClassifier(
                max_depth=depth, max_iter=100, early_stopping=True,
                n_iter_no_change=8, validation_fraction=0.15,
                random_state=seed_base, class_weight=cw).fit(Ztr_p, ytr_p)
            runs.append(_score_sk("tree", f"depth={depth},it={g.n_iter_}", g,
                                  Ztr_p, ytr_p, Zte, yte, t0))

    if "knn" in families:
        from sklearn.neighbors import KNeighborsClassifier

        for nn in (15, 50):
            t0 = time.time()
            kn = KNeighborsClassifier(n_neighbors=nn, weights="distance").fit(
                Ztr_p, ytr_p)
            runs.append(_score_sk("knn", f"k={nn}", kn, Ztr_p, ytr_p, Zte, yte, t0))

    if "svm" in families:
        idx = rng.choice(len(Ztr), min(svm_cap, len(Ztr)), replace=False)
        for C in (1.0, 10.0):
            t0 = time.time()
            sv = SVC(C=C, kernel="rbf", gamma="scale").fit(Ztr[idx], ytr[idx])
            runs.append(
                dict(
                    family="svm", cfg=f"C={C},n={len(idx)}",
                    train_acc=accuracy_score(ytr[idx], sv.predict(Ztr[idx])),
                    train_bal=balanced_accuracy_score(ytr[idx], sv.predict(Ztr[idx])),
                    test_acc=accuracy_score(yte, sv.predict(Zte)),
                    test_bal=balanced_accuracy_score(yte, sv.predict(Zte)),
                    sec=time.time() - t0,
                )
            )

    best = max(runs, key=lambda r: r["test_bal"])
    return best, runs


def _score_sk(family, cfg, model, Ztr, ytr, Zte, yte, t0):
    ptr, pte = model.predict(Ztr), model.predict(Zte)
    return dict(
        family=family, cfg=cfg,
        train_acc=accuracy_score(ytr, ptr),
        train_bal=balanced_accuracy_score(ytr, ptr),
        test_acc=accuracy_score(yte, pte),
        test_bal=balanced_accuracy_score(yte, pte),
        sec=time.time() - t0,
    )


def _score(family, cfg, m, Ztr, ytr, Zte, yte, t0):
    ptr, pte = _pred(m, Ztr), _pred(m, Zte)
    return dict(
        family=family, cfg=cfg,
        train_acc=accuracy_score(ytr, ptr),
        train_bal=balanced_accuracy_score(ytr, ptr),
        test_acc=accuracy_score(yte, pte),
        test_bal=balanced_accuracy_score(yte, pte),
        sec=time.time() - t0,
    )


def probe_regression(Ztr, Ytr, Zte, Yte, seed=0, epochs=400):
    Ztr, Zte = standardize(Ztr, Zte)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8
    Ytr_n, Yte_n = (Ytr - mu) / sd, (Yte - mu) / sd
    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.Linear(Ztr.shape[1], 256), nn.ReLU(), nn.Linear(256, Ytr.shape[1])
    ).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    X, Y = _t(Ztr), _t(Ytr_n)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(m(X), Y).backward()
        opt.step()
    with torch.no_grad():
        pred = m(_t(Zte)).cpu().numpy()
    mse = float(((pred - Yte_n) ** 2).mean())

    r2 = float(1 - mse / ((Yte_n - Yte_n.mean(0)) ** 2).mean())
    return dict(landmark_mse=mse, landmark_r2=r2)


def reconstruct(Ztr, Utr, Zte, Ute, seed=0, epochs=400):
    Ztr_s, Zte_s = standardize(Ztr, Zte)
    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.Linear(Ztr.shape[1], 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, Utr.shape[1]),
    ).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    X, Y = _t(Ztr_s), _t(Utr)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 2048):
            idx = perm[i : i + 2048]
            opt.zero_grad()
            nn.functional.mse_loss(m(X[idx]), Y[idx]).backward()
            opt.step()
    with torch.no_grad():
        Uhat_tr = m(_t(Ztr_s)).cpu().numpy()
        Uhat = m(_t(Zte_s)).cpu().numpy()
    rel = float(np.linalg.norm(Uhat - Ute) / np.linalg.norm(Ute))
    cos = float(
        np.mean(
            (Uhat * Ute).sum(1)
            / (np.linalg.norm(Uhat, axis=1) * np.linalg.norm(Ute, axis=1) + 1e-9)
        )
    )
    ev = float(1 - ((Uhat - Ute) ** 2).sum() / ((Ute - Ute.mean(0)) ** 2).sum())
    return dict(recon_rel_error=rel, recon_cosine=cos, recon_explained_var=ev), Uhat_tr, Uhat
