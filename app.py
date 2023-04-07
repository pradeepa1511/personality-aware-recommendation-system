import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('indexx.html')

@app.route('/ils')
def ils():
    #print(prediction_text)
    return render_template('ils.html')  

@app.route('/ils_predict',methods=['POST'])
def ils_predict():
    initial_features = [int(y) for y in request.form.values()]    
    #end_features = [np.array(initial_features)]
    print(initial_features)
    act_ref = initial_features[0:11]
    sns_int = initial_features[11:22]
    vis_vrb = initial_features[22:33]
    seq_glo = initial_features[33:44]
    print(act_ref)
    act_ref_a_count = act_ref.count(1)
    b1_counter = 11 - act_ref_a_count
    sns_int_a_count = sns_int.count(1)
    b2_counter = 11 - sns_int_a_count
    vis_vrb_a_count = vis_vrb.count(1)
    b3_counter = 11 - vis_vrb_a_count
    seq_glo_a_count = seq_glo.count(1)
    b4_counter = 11 - seq_glo_a_count
    print(act_ref_a_count)
    print(sns_int_a_count)
    print(vis_vrb_a_count)
    print(seq_glo_a_count)
    a1b1 = abs(b1_counter-act_ref_a_count)
    a2b2 = abs(b2_counter- sns_int_a_count)
    a3b3 = abs(b3_counter-vis_vrb_a_count)
    a4b4 = abs(b4_counter-seq_glo_a_count)
    if b1_counter < act_ref_a_count :
      r1 = 'active'
    else:
      r1 = 'reflective'

    if b2_counter<sns_int_a_count :
      r2='sensing'
    else:
      r2='intuitive'

    if b3_counter<vis_vrb_a_count :
      r3 = 'visual'
    else:
      r3='verbal'

    if b4_counter<seq_glo_a_count  :
      r4 = 'sequential'
    else:
      r4= 'global'  

    print('Active-Reflective ',abs(b1_counter-act_ref_a_count),'a' if b1_counter < act_ref_a_count else 'b')
    print('Sensing-Intuitive ',abs(b2_counter- sns_int_a_count),'a' if b2_counter<sns_int_a_count else 'b')
    print('Visual-Verbal ',abs(b3_counter-vis_vrb_a_count),'a' if b3_counter<vis_vrb_a_count else 'b')
    print('Sequential-Global ',abs(b4_counter-seq_glo_a_count),'a' if b4_counter<seq_glo_a_count else 'b')
    
    dic = {r1 : a1b1 , r2 : a2b2, r3 : a3b3 , r4 : a4b4 }
    print(dic)
    print(a1b1,a2b2,a3b3,a4b4)
    
    res={}
    global lis
    lis=[]
    for k,v in dic.items():
        if v == max(dic.values()):
            res[k] = v
            lis.append(k)
    print(res)
    print(lis)
    # dic['max_dic']=res 
    #recommendation(lis)
    #print(prediction_text)
    return render_template('table.html',result = dic,res=res,lis = lis)            

@app.route('/recommendation',methods=['GET'])
def recommendation():
    #final_lis = ils_predict()
    #lis1 = request.args.getlist(lis)
    #personality = predict()
    #personality1 = request.args.get(personality)
    print(lis)
    print(request)
    return render_template('recommendation.html',final_lis=lis)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    #output = round(prediction[0], 2)
    if prediction==0 :
        personality = 'Extroversion'
        #personality = 'Extroversion - A person high in extraversion are outgoing and tend to gain energy in social situations. '
    elif prediction==1:
        personality = 'Neuroticism'
        #personality =  'Neuroticism - Individuals who are high in neuroticism tend to experience mood swings, anxiety, irritability, and sadness.'
    elif prediction==2:
        personality = 'Agreeableness'
        #personality = 'Agreeableness - People who are high in agreeableness tend to be more cooperative while those low in this personality trait tend to be more competitive and sometimes even manipulative.'
    elif prediction==3:
        personality = 'Conscientiousness'
        #personality = 'Conscientiousness - Highly conscientious people tend to be organized and mindful of details. They plan ahead and are mindful of deadlines.'
    else:
        personality = 'Openness'
        #personality = 'Openness - People who are high in openness tend to have a broad range of interests. They are curious about the world and are eager to learn new things and enjoy new experiences.'

    #recommendation(personality)
    return render_template('sucess.html', prediction_text=personality)



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]  
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

