import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import os

tf.keras.backend.clear_session()
tf.keras.backend.set_learning_phase(0) 
# Clear any previous session.
# def generate_pb(model_path,model_name) :
#print('222222222222222222222',os.listdir(save_pb_dir))

def get_tensor_name(model_path,model_name) :
    model = load_model(model_path)

    input_names = [t.op.name for t in model.inputs]
    output_names = [t.op.name for t in model.outputs]

    print('{} input : {} , output : {}'.format(model_name,input_names,output_names))

    return input_names,output_names

def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='tmp', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name+'.pb', as_text=save_pb_as_text)
        return graphdef_frozen

if __name__ == '__main__' :
    save_pb_dir = './models/xception/'
    model_name = 'xception'
    model_path = save_pb_dir+'best_gobot_model_acc.h5'
    print(model_path)
    # This line must be executed before loading Keras model.
    model = load_model(model_path)

    session = tf.keras.backend.get_session()

    input_names = [t.op.name for t in model.inputs]
    output_names = [t.op.name for t in model.outputs]
    # Prints input and output nodes names, take notes of them.
    print(input_names, output_names)

    if not os.path.exists(save_pb_dir+'/'+model_name+'.pb') :
        print('Create TensorRT Model')
        print(model_name)
        print(session.graph)
        frozen_graph = freeze_graph(
            session.graph,
            session, 
            [out.op.name for out in model.outputs], 
            save_pb_dir=save_pb_dir,
            save_pb_name=model_name
            )