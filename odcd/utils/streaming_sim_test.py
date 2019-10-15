import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from creme.stats import Mean, Var
import numpy as np


def sliding_window(vae, X_orig, y_orig,  X_cd, y_cd, window_size, cd_start, nb_samples_tot, start):
    cd_full = 2 * cd_start

    acc = []
    mu = []
    var = []
    ps = []
    entrs = []

    ps_tmp = []
    xs = []
    ys = []
    y_orig = np.argmax(y_orig, axis=1)
    y_cd = np.argmax(y_cd, axis=1)
    for i in range(nb_samples_tot):
        if i % 1000 == 0:
            print('Sample {} of {}'.format(i, nb_samples_tot))
        if i < cd_start:
            p = 0
        else:
            p = (i - cd_start) / cd_start
        if p > 1:
            p = 1

        pp = np.random.choice([0, 1], p=[p, 1 - p])
        if pp == 1:
            idx = np.random.choice(range(len(X_orig)))
            x = X_orig[idx]
            y = y_orig[idx]
        else:
            idx = np.random.choice(range(len(X_cd)))
            x = X_cd[idx]
            y = y_cd[idx]

        if i % window_size == 0 and i != 0:
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            ps_tmp = np.asarray(ps_tmp)

            vae_outs_test = vae.vae.predict(xs)
            symm_samples_test = vae_outs_test[0]
            orig_preds_test = vae_outs_test[1]
            trans_preds_test = vae_outs_test[2]

            kl_test = entropy(orig_preds_test.T, trans_preds_test.T)
            pred = np.argmax(orig_preds_test, axis=1)
            pred_prob = orig_preds_test[:, pred[0]]
            r = accuracy_score(pred, ys)

            mu.append(kl_test.mean())
            var.append(kl_test.var())
            acc.append(r)
            ps.append(ps_tmp.mean())
            xs = []
            ps_tmp = []
            ys = []
        else:
            xs.append(x)
            ys.append(y)
            ps_tmp.append(p)

        if i == cd_start:
            window_start = len(mu)
        if i == cd_full:
            window_full = len(mu)

    df = pd.DataFrame()
    df['accuracy'] = acc[start:]
    df['contamination'] = ps[start:]
    df['entropy_mean'] = mu[start:]
    df['entropy_var'] = var[start:]

    return df, window_start, window_full


def rolling_stats(vae, X_orig, y_orig, X_cd, y_cd, cd_start, nb_samples_tot, start):
    cd_full = 2 * cd_start
    m = Mean()
    a = Mean()
    v = Var()

    rws = []
    entrs = []
    probs = []

    acc = []
    mu = []
    var = []
    ps = []
    y_orig = np.argmax(y_orig, axis=1)
    y_cd = np.argmax(y_cd, axis=1)
    for i in range(nb_samples_tot):
        if i % 1000 == 0:
            print('Sample {} of {}'.format(i, nb_samples_tot))
        if i < cd_start:
            p = 0
        else:
            p = (i - cd_start) / cd_start
        if p > 1:
            p = 1

        pp = np.random.choice([0, 1], p=[p, 1 - p])
        if pp == 1:
            idx = np.random.choice(range(len(X_orig)))
            x = X_orig[idx].reshape((1, ) + X_orig.shape[1:])
            y = y_orig[idx]

        else:
            idx = np.random.choice(range(len(X_cd)))
            x = X_cd[idx].reshape((1, ) + X_cd.shape[1:])
            y = y_cd[idx]

        vae_outs_test = vae.vae.predict(x)
        symm_samples_test = vae_outs_test[0]
        orig_preds_test = vae_outs_test[1]
        trans_preds_test = vae_outs_test[2]

        kl_test = entropy(orig_preds_test.T, trans_preds_test.T)[0]
        pred = np.argmax(orig_preds_test, axis=1)
        pred_prob = orig_preds_test[:, pred[0]]
        r = is_good(pred, y)

        a.update(r)
        m.update(kl_test)
        v.update(kl_test)
        mm, vv, aa = m.get(), v.get(), a.get()

        mu.append(mm)
        var.append(vv)
        acc.append(aa)
        ps.append(p)

        rws.append(r)
        entrs.append(kl_test)
        probs.append(pred_prob)

    cd_start = cd_start - start
    cd_full = cd_full - start
    df = pd.DataFrame()
    df['rw'] = rws[start:]
    # df['rw'] = df['rw'].apply(lambda x:'r' if x == 0 else 'b')
    df['entropy'] = entrs[start:]
    df['pred_prob'] = probs[start:]
    df['accuracy'] = acc[start:]
    df['contamination'] = ps[start:]
    df['entropy_mean'] = mu[start:]
    df['entropy_var'] = var[start:]

    return df


def is_good(y_pred, y_true):
    print(y_pred == y_true)
    if y_pred == y_true:
        return 1
    else:
        return 0
