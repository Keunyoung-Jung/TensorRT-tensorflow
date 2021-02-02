import os
import numpy as np
from pred_trt import Tf_TRT

def predict(model,frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame,(150,150))
    frame = img_to_array(frame)
    x = np.expand_dims(frame, axis=0)
    x /=255.
    array = model.predict(x)
    result = array[0]
    #print(result)
    answer = np.argmax(result)
    name = [k for k,v in pizza_dict.items() if v == answer][0]
    acc = round(max(result),3)

    return name,acc

def voting(array):
    vote_result = Counter(array).most_common(1)
    return vote_result[0][0]


def make_json(output_data,accuracy):
    json_dict = {
        'inference' : output_data,
        'accuracy' : accuracy
        }
    return json.dumps(json_dict,default=str)


    
if __name__=='__main__' :
    st = time.time()
    model_name = 'resnet'
    resnet = Tf_TRT(model_path,model_name,['input_1'],['probs/Softmax'])
    model_dict['resnet']=resnet
    print(f'Loaded {model_name} model',time.time()-st,'sec')

    model_name = 'mobilenet'
    mobilenet = Tf_TRT(model_path,model_name,['input_1'],['probs/Softmax'])
    model_dict['mobilenet']=mobilenet
    print(f'Loaded {model_name} model',time.time()-st,'sec')

    print('loaded model complete')

    print('-'*20)

    image = cv2.imread('test.jpg')
    predict_arr = []
    for name,model in model_dict.items() :
        pizza_name,acc = predict(model,image)
        predict_arr.append([name,output_data,accuracy])

    voting_output_data = voting([x[1] for x in predict_arr])
    max_accuracy = max([x[2] for x in predict_arr])

    json_result = make_json(output_data,accuracy)
    print(json_result)


