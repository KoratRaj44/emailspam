import time
from ML import model
import streamlit as st
from DP import *
import matplotlib.pyplot as plt
import seaborn as sns
inputs=[0,1]
@st.cache_data
def create_model():
    mode=model()
    return mode
col1,col2,col3,col4,col5=st.columns(5)
with col3:
    st.title("Spade")
st.write('welcome to Spade...')
st.write('A Spam Detection algorithm based on Machine Learning and Natural Language Processing')

text=st.text_area('please provide email/text you wish to classify',height=400,placeholder='type/paste more than 50 characters here')
file=st.file_uploader("please upload file with your text.. (only .txt format supported")

if len(text)>20:
    inputs[0]=1
if file is None:
    inputs[1]=0
if inputs.count(1)>1:
    st.error('multiple inputs given please select only one option')
else:
    if inputs[0]==1:
        e=text
        given_email = e
    if inputs[1]==1:
        bytes_data = file.getvalue()

        given_email = bytes_data



predictions=[]
probs=[]

col1,col2,col3,col4,col5=st.columns(5)
with col3:
    clean_button = st.button('Detect')
st.caption("In case of a warning it's probably related to caching of your browser")
st.caption("please hit the detect button again....")

if clean_button:
    if inputs.count(0)>1:
        st.error('No input given please try after giving the input')
    else:
        with st.spinner('Please wait while the model is running....'):
            mode = create_model()
        given_email,n=clean(given_email)
        vector = mode.get_vector(given_email)
        predictions.append(mode.get_prediction(vector))
        probs.append(mode.get_probabilities(vector))
        col1, col2, col3 = st.columns(3)
        with col2:
            st.header(f"{predictions[0]}")
        probs_pos = [i[1] for i in probs[0]]
        probs_neg = [i[0] for i in probs[0]]
        if predictions[0] == 'Spam':
            # st.caption(str(probs_pos))
            plot_values = probs_pos
        else:
            # st.caption(str(probs_neg))
            plot_values = probs_neg
        plot_values=[int(i) for i in plot_values]
        st.header(f'These are the results obtained from the models')
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader('predicted Accuracies of models')
            with st.expander('Technical Details'):
                st.write('Model-1 : Naive Bayes')
                st.write('Model-2 : Random Forest')
                st.write('Model-3 : Logistic Regression')
                st.write('Model-4 : K-Nearest Neighbors')
                st.write('Model-5 : Support Vector Machines')
        with col2:
            st.write('Model-1', plot_values[0])
            bar1 = st.progress(0)
            for i in range(plot_values[0]):
                time.sleep(0.01)
                bar1.progress(i)
            st.write('Model-2', plot_values[1])
            bar2 = st.progress(0)
            for i in range(plot_values[1]):
                time.sleep(0.01)
                bar2.progress(i)
            st.write('Model-3', plot_values[2])
            bar3 = st.progress(0)
            for i in range(plot_values[2]):
                time.sleep(0.01)
                bar3.progress(i)
            st.write('Model-4', plot_values[3])
            bar4 = st.progress(0)
            for i in range(plot_values[3]):
                time.sleep(0.01)
                bar4.progress(i)
            st.write('Model-5', plot_values[4])
            bar5 = st.progress(0)
            for i in range(plot_values[4]):
                time.sleep(0.01)
                bar5.progress(i)
        st.header('These are some insights from the given text.')
        entities=ents(text)
        col1,col2=st.columns([2,3])
        with col1:
            st.subheader('These are the named entities extracted from the text')
            st.write('please expand each category to view the entities')
            st.write('a small description has been included with entities for user understanding')
        with col2:
            if entities=='no':
                st.subheader('No Named Entities found.')
            else:
                renames = {'CARDINAL': 'Numbers', 'TIME': 'Time', 'ORG': 'Companies/Organizations', 'GPE': 'Locations',
                           'PERSON': 'People', 'MONEY': 'Money', 'FAC': 'Factories'}
                for i in renames.keys():
                    with st.expander(renames[i]):
                         st.caption(spacy.explain(i))
                         values = list(set(entities[i]))
                         strin = ',  '.join(values)
                         st.write(strin)







