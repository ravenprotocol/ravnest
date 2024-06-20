class NodeTypes(object):
    LEAF = "leaf"
    STEM = "stem"
    ROOT = "root"

class ActionTypes(object):
    ROOT_FORWARD = "root_forward"
    FORWARD = "forward"
    BACKWARD = "backward"
    FIND_LOSS = "find_loss"
    NO_GRAD_FORWARD = "no_grad_forward"
    ACCURACY = "accuracy"
    VAL_ACCURACY = "val_accuracy"
    PREDICTION = "prediction"
    SAVE_SUBMODEL = "save_submodel"

class BufferStatus(object):
    SEND_BUFFER = "send_buffer"

class NodeStatus(object):
    FORWARD = "forward"
    BACKWARD = "backward"
    REDUCING = "reducing"
    GATHERING = "gathering"
    IDLE = "idle"