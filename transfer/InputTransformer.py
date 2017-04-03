from transfer import TransferUtils

def augment_features(input):
    return input

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