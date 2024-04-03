

def test_pp_LeidaCenSC(k_evecs, xtest, n_states, ages, train_idx, test_idx):
    trlab = est.predict(k_evecs, n_states)
    telab = est.predict_on_unseen(xtest)
    trlab  = trlab.reshape(train_idx.size,-1)
    telab  = telab.reshape(test_idx.size,-1)
    fp = fingerprinting.Fingerprinting()
    fp.fit(trlab, telab)
    tror, teor = fp.transform(trlab, telab)
    pp = predictive_power.Predict()
    y = ages[train_idx]
    pp.fit(tror, y)
    predictors, r_values, p_values = pp.predict(teor, ages[test_idx])
    return tror, predictors, r_values, p_values

# getting recurrent objects for xtrain
def test_pp(xtrain, xtest, n_states, ages, train_idx, test_idx, random_state=seed):
    k = clustering.Cluster()
    kmeans = MiniBatchKMeans(n_clusters=n_states, n_init='auto', random_state=random_state)
    assert xtrain.shape[-1] == xtest.shape[-1]
    xtrain = xtrain.reshape(-1, xtrain.shape[-1])
    xtest = xtest.reshape(-1, xtrain.shape[-1])
    k.fit(xtrain, kmeans)
    trlab_grouped  = k.predict(xtrain).reshape(train_idx.size,-1)
    telab_grouped  = k.predict(xtest).reshape(test_idx.size,-1)
    fp = fingerprinting.Fingerprinting()
    fp.fit(trlab_grouped, telab_grouped)
    tror, teor = fp.transform(trlab_grouped, telab_grouped)
    pp = predictive_power.Predict()
    y = ages[train_idx]
    pp.fit(tror, y)
    predictors, r_values, p_values = pp.predict(teor, ages[test_idx])
    return tror, predictors, r_values, p_values

# getting recurrent objects for xtrain
def test_fp(xtest, xretest, n_states, num_permutations=1000, random_state=seed):
    k = clustering.Cluster()
    kmeans = MiniBatchKMeans(n_clusters=n_states, n_init='auto',random_state=random_state)
    xtest = xtest.reshape(-1, est_zudah.n_features)
    xretest = xretest.reshape(-1, est_zudah.n_features)
    k.fit(xtest, kmeans)
    X  = k.predict(xtest).reshape(n_participants,-1)
    y  = k.predict(xretest).reshape(n_participants,-1)
    fp = fingerprinting.Fingerprinting()
    fp.fit(X, y)
    Xor, yor = fp.transform(X, y)
    permst, obsst, pv = fp.score_intrasim_vs_intrasim(num_permutations=num_permutations)
    diff = fp.intrasim.mean() - fp.intersim.mean()
    return diff