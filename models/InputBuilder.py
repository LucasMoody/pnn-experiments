from keras.layers import Input, Embedding, Flatten, merge

def buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing, train_word_embeddings=False):
    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, casing]
    inputs = [words_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    return input_layers_merged, inputs

def buildPipelineModelInput(config):
    input_layers = []
    inputs = []

    if 'words' in config:
        n = config['words']['n']
        idx = config['words']['idx']
        words_input = Input(shape=(n,), dtype='int32', name='words_input')
        wordEmbeddingLayer = Embedding(
            output_dim=idx.shape[1],
            input_dim=idx.shape[0],
            input_length=n,
            weights=[idx],
            trainable=False)
        words = wordEmbeddingLayer(words_input)
        words = Flatten(name='words_flatten')(words)
        inputs.append(words_input)
        input_layers.append(words)

    if 'casing' in config:
        input, input_layer = buildInputNodes(config, 'casing')
        inputs.append(input)
        input_layers.append(input_layer)

    if 'pos' in config:
        input, input_layer = buildInputNodes(config, 'pos')
        inputs.append(input)
        input_layers.append(input_layer)

    if 'ner' in config:
        input, input_layer = buildInputNodes(config, 'ner')
        inputs.append(input)
        input_layers.append(input_layer)

    if 'chunking' in config:
        input, input_layer = buildInputNodes(config, 'chunking')
        inputs.append(input)
        input_layers.append(input_layer)

    if 'ace' in config:
        input, input_layer = buildInputNodes(config, 'ace')
        inputs.append(input)
        input_layers.append(input_layer)

    if 'ecb' in config:
        input, input_layer = buildInputNodes(config, 'ecb')
        inputs.append(input)
        input_layers.append(input_layer)

    if 'tac' in config:
        input, input_layer = buildInputNodes(config, 'tac')
        inputs.append(input)
        input_layers.append(input_layer)

    if 'tempeval' in config:
        input, input_layer = buildInputNodes(config, 'tempeval')
        inputs.append(input)
        input_layers.append(input_layer)


    return input_layers, inputs

def buildInputNodes(config, name):
    n = config[name]['n']
    idx = config[name]['idx']
    input = Input(shape=(n,), dtype='int32', name=name + '_input')
    embeddingLayer = Embedding(
        output_dim=len(idx),
        input_dim=len(idx),
        input_length=n,
        trainable=True)
    input_layer = embeddingLayer(input)
    input_layer = Flatten(name=name + '_flatten')(input_layer)
    return input, input_layer

def buildCoralModelInput(case2Idx, n_in_x, n_in_casing, train_word_embeddings=False):
    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, casing]
    inputs = [words_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    return input_layers_merged, inputs

