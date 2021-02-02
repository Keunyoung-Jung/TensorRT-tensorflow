import tensorflow as tf

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


class Tf_TRT :
    def __init__(self,path,model_name,input_names,output_names) :
        # input_names = ['conv2d_1_input']
        # output_names = ['dense_2/Sigmoid']

        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_sess = tf.Session(config=self.tf_config)

        self.trt_graph = get_frozen_graph(path+model_name+'/'+model_name+'.pb')
        tf.import_graph_def(self.trt_graph, name='')

        self.input_tensor_name = input_names[0] + ":0"
        self.output_tensor_name = output_names[0] + ":0"

        self.output_tensor = self.tf_sess.graph.get_tensor_by_name(self.output_tensor_name)

    def predict(self,image) :
        return self.tf_sess.run(self.output_tensor, { self.input_tensor_name : image})
