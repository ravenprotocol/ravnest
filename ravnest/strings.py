class NodeTypes(object):
    LEAF = "leaf"
    MID = "mid"
    ROOT = "root"

class ActionTypes(object):
    FORWARD = "forward"
    BACKWARD = "backward"
    FIND_LOSS = "find_loss"
    NO_GRAD_FORWARD = "no_grad_forward"
    ACCURACY = "accuracy"
    PREDICTION = "prediction"
    SAVE_SUBMODEL = "save_submodel"

class BufferStatus(object):
    SEND_BUFFER = "send_buffer"
