from transfer import TransferUtils
import numpy as np

def augment_features(datasets, zeros_idx_words, zeros_idx_casing):
    # copy everything so that function is immutable
    working_ds = []
    no_feat = len(datasets)
    window_size = datasets[0]['train']['input'][0].shape[1]
    for idx, source_dataset in enumerate(datasets):
        # ----- X ----- # 
        inter_train = np.full((source_dataset['train']['input'][0].shape[0], (no_feat + 1) * window_size), zeros_idx_words, dtype=np.int)
        for j, observation in enumerate(inter_train):
            observation[0: window_size] = source_dataset['train']['input'][0][j]
            observation[(idx + 1) * window_size: (idx + 1) * window_size + window_size] = source_dataset['train']['input'][0][j]

        inter_dev = np.full((source_dataset['dev']['input'][0].shape[0], (no_feat + 1) * window_size), zeros_idx_words, dtype=np.int)
        for j, observation in enumerate(inter_dev):
            observation[0: window_size] = source_dataset['dev']['input'][0][j]
            observation[(idx + 1) * window_size: (idx + 1) * window_size + window_size] = \
            source_dataset['dev']['input'][0][j]

        inter_test = np.full((source_dataset['test']['input'][0].shape[0], (no_feat + 1) * window_size), zeros_idx_words, dtype=np.int)
        for j, observation in enumerate(inter_test):
            observation[0: window_size] = source_dataset['test']['input'][0][j]
            observation[(idx + 1) * window_size: (idx + 1) * window_size + window_size] = \
            source_dataset['test']['input'][0][j]

        # ----- CASING ----- #
        inter_casing_train = np.full((source_dataset['train']['input'][1].shape[0], (no_feat + 1) * window_size), zeros_idx_casing,
                                     dtype=np.int)
        for j, observation in enumerate(inter_casing_train):
            observation[0: window_size] = source_dataset['train']['input'][1][j]
            observation[(idx + 1) * window_size: (idx + 1) * window_size + window_size] = \
            source_dataset['train']['input'][1][j]

        inter_casing_dev = np.full((source_dataset['dev']['input'][1].shape[0], (no_feat + 1) * window_size), zeros_idx_casing, dtype=np.int)
        for j, observation in enumerate(inter_casing_dev):
            observation[0: window_size] = source_dataset['dev']['input'][1][j]
            observation[(idx + 1) * window_size: (idx + 1) * window_size + window_size] = \
                source_dataset['dev']['input'][1][j]

        inter_casing_test = np.full((source_dataset['test']['input'][1].shape[0], (no_feat + 1) * window_size), zeros_idx_casing, dtype=np.int)
        for j, observation in enumerate(inter_casing_test):
            observation[0: window_size] = source_dataset['test']['input'][1][j]
            observation[(idx + 1) * window_size: (idx + 1) * window_size + window_size] = \
                source_dataset['test']['input'][1][j]

        cur_dataset = source_dataset.copy()
        cur_dataset.update({
            'train': {
                'input':  [inter_train, inter_casing_train],
                'y': source_dataset['train']['y']
            },
            'dev': {
                'input': [inter_dev, inter_casing_dev],
                'y': source_dataset['dev']['y']
            },
            'test': {
                'input': [inter_test, inter_casing_test],
                'y': source_dataset['test']['y']
            }
        })
        working_ds.append(cur_dataset)
        # set features of source_dataset to general features
        # set features of source_dataset to
        print '----- SANITY CHECKS -----'
        print '----- X -----'
        print 'train\n', np.array_str(source_dataset['train']['input'][0][0:5]), '\n==>\n', np.array_str(cur_dataset['train']['input'][0][0:5])
        print 'dev\n', np.array_str(source_dataset['dev']['input'][0][0:5]), '\n==>\n', np.array_str(cur_dataset['dev']['input'][0][0:5])
        print 'test\n', np.array_str(source_dataset['test']['input'][0][0:5]), '\n==>\n', np.array_str(cur_dataset['test']['input'][0][0:5])

        print '----- CASING -----'
        print 'train\n', np.array_str(source_dataset['train']['input'][1][0:5]), '\n==>\n', np.array_str(cur_dataset['train']['input'][1][0:5])
        print 'dev\n', np.array_str(source_dataset['dev']['input'][1][0:5]), '\n==>\n', np.array_str(cur_dataset['dev']['input'][1][0:5])
        print 'test\n', np.array_str(source_dataset['test']['input'][1][0:5]), '\n==>\n', np.array_str(cur_dataset['test']['input'][1][0:5])

    return working_ds

def apply_coral(datasets):
    # copy everything so that function is immutable
    working_ds = list(datasets)
    working_ds = map(lambda ds: ds.copy(), working_ds)

    # calculate coloring recoloring matrix
    target_dataset = working_ds[0]
    R = TransferUtils.get_recolering_matrix(target_dataset['train']['input'][0])
    # whiten all source domains
    # recolor every source domain
    for source_dataset in working_ds[1:]:
        # calculate whitening matrices for each source domain
        # whiten word input
        X, W = TransferUtils.whiten(source_dataset['train']['input'][0])
        # recolor source
        source_dataset['train']['input'][0] = TransferUtils.recoloring(X, R)
    return working_ds

def convertToEmbeddingValues(datasets, embeddings):
    # copy everything so that function is immutable
    working_ds = list(datasets)
    working_ds = map(lambda ds: ds.copy(), working_ds)

    for idx, ds in enumerate(datasets):
        input = ds['train']['input']
        windows_words = input[0]
        working_ds[0]['train']['input'][0] = map(lambda window: np.array(map(lambda word: embeddings[word], window)).flatten(), windows_words)
    return working_ds
