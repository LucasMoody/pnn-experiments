from optimizer import OptimizedModels

def buildTransferModels(input_layers, inputs, params, config):
    transfer_models = []
    if 'pos' in config:
        transfer_models.append(OptimizedModels.getWSJPOSModelGivenInput(input_layers, inputs, params))

    if 'ud_pos' in config:
        transfer_models.append(OptimizedModels.getUDPOSModelGivenInput(input_layers, inputs, params))

    if 'ner' in config:
        transfer_models.append(OptimizedModels.getNERModelGivenInput(input_layers, inputs, params))

    if 'chunking' in config:
        transfer_models.append(OptimizedModels.getChunkingModelGivenInput(input_layers, inputs, params))

    if 'ace' in config:
        transfer_models.append(OptimizedModels.getAceEDModelGivenInput(input_layers, inputs, params))

    if 'ecb' in config:
        transfer_models.append(OptimizedModels.getEcbEDModelGivenInput(input_layers, inputs, params))

    if 'tac' in config:
        transfer_models.append(OptimizedModels.getTacEDModelGivenInput(input_layers, inputs, params))

    if 'tempeval' in config:
        transfer_models.append(OptimizedModels.getTempevalEDModelGivenInput(input_layers, inputs, params))
    return transfer_models