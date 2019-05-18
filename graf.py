# coding: utf8;

import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import neighbors, ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import scipy as sp
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import word2vec
import feedparser
import re
import gensim
from gensim.models.word2vec import Word2Vec

morph = pymorphy2.MorphAnalyzer()


def text_file():
    lf3 = [tuple(map(str.strip, l.split('\t'))) for l in open('test2.tsv').readlines()]
    print('loaded %d lines'%len(lf3))
    return lf3

def dist_raw(vl, v2):
	""" расстояние между фразами """
	delta = vl - v2
	return sp.linalg.norm(delta.toarray())

def dist_norm(v1, v2):
	v1_normalized = v1 / sp.linalg.norm(v1.toarray())
	v2_normalized = v2 / sp.linalg.norm(v2.toarray())
	delta = v1_normalized - v2_normalized
	return sp.linalg.norm(delta.toarray())



def vectot_text(data):
    # векторизованный текст
    vectorizer = CountVectorizer(min_df=1)
    V = vectorizer.fit_transform(data)
    num_samples, num_features = V.shape
    print("# samples : %d , # features : %d" % (num_samples, num_features))
    print('len(feature_names)=%d {уник.слов признаков}' % len(vectorizer.get_feature_names()))

    return vectorizer, V, num_samples

def claster_text(vector, num_samples):
    if num_samples >= 50:
        num_clusters = 50
    else:
        num_clusters = num_samples
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1, random_state=3)
    km.fit(vector)

    return km



def pred_morphe(content):
    pred_sp = []
    for pred_i in range(0, len(content), 1):
        pred1 = content[pred_i].split(None)
        pred2 = [x.rstrip(',.!?;') for x in pred1]
        pred_sp.append(pred2)

    return pred_sp


def create_graf(slovo):
    #промежуточные списки
    sp1 = []#части речи
    sp2 = []#слова
    sp3 = []#падежи
    for j in range(0, len(slovo), 1):
        st = slovo[j]
        pr = morph.parse(st)[0]
        pos = pr.tag.POS
        case = pr.tag.case
        sp1.append(pos)
        sp2.append(st)
        sp3.append(case)

    return sp1, sp2, sp3


def add_node_edge(dG, wr, nx_wr, weight):
	dG.add_node(wr)
	dG.node[wr]['count'] = 1
	dG.add_edge(wr, nx_wr, weight=weight)

def search_graf(spisok_POS, spisok_slovo, spisok_case):
    podleg = ""
    skaz = ""
    prep_word = ""
    noun_dop = ""
    prep_flag = False
    skaz_infn = ""
    prcl_word = ""

    dG = nx.Graph()

    sl = 0
    sr_pred = 0

    for i, word in enumerate(spisok_slovo):
        try:
            if (spisok_POS[i] == 'NPRO' or spisok_POS[i] == 'NOUN') and (spisok_case[i] == 'nomn'):
                podleg = spisok_slovo[i]
                # add_node_edge(dG, podleg, skaz, 5)
                if (spisok_POS[i - 1] == 'ADJF'):
                    ADJF_word = spisok_slovo[i - 1]
                    add_node_edge(dG, ADJF_word, podleg, 5)
            if (spisok_POS[i] == 'VERB'):
                skaz = spisok_slovo[i]
                if (spisok_POS[i - 1] == 'ADVB'):
                    advb_word = spisok_slovo[i - 1]
                    if (spisok_POS[i - 2] == 'PRCL'):
                        prcl_word_2 = spisok_slovo[i - 2]
                        add_node_edge(dG, prcl_word_2, advb_word, 5)
                    add_node_edge(dG, advb_word, skaz, 5)
                if i < len(spisok_POS) - 1:
                    if (spisok_POS[i + 1] == 'ADVB'):
                        advb_word = spisok_slovo[i + 1]
                        add_node_edge(dG, advb_word, skaz, 5)

                if (spisok_POS[i - 1] == 'PRCL'):
                    prcl_word = spisok_slovo[i - 1]
                    if podleg != "":
                        add_node_edge(dG, podleg, prcl_word, 5)
                    add_node_edge(dG, prcl_word, skaz, 5)
                else:
                    add_node_edge(dG, podleg, skaz, 5)
                if (spisok_POS[i - 1] == 'INFN'):
                    skaz_infn = spisok_slovo[i - 1]
                    add_node_edge(dG, skaz_infn, skaz, 5)
                if i < len(spisok_POS) - 1:
                    if (spisok_POS[i + 1] == 'INFN'):
                        skaz_infn = spisok_slovo[i + 1]
                        add_node_edge(dG, skaz, skaz_infn, 5)

            if (spisok_POS[i] == 'NOUN') and (spisok_case[i] != 'nomn'):
                noun_dop = spisok_slovo[i]
                if (spisok_POS[i - 2] == 'PREP'):
                    prep_flag = True
                    prep_word = spisok_slovo[i - 2]
                    add_node_edge(dG, skaz, prep_word, 10)
                    if (spisok_POS[i - 1] == 'ADJF'):
                        ADJF_word = spisok_slovo[i - 1]
                        add_node_edge(dG, ADJF_word, noun_dop, 5)

                if prep_flag == True:
                    if (spisok_POS[i - 1] == 'ADJF'):
                        ADJF_word = spisok_slovo[i - 1]
                        add_node_edge(dG, prep_word, ADJF_word, 5)
                        add_node_edge(dG, ADJF_word, noun_dop, 5)
                    add_node_edge(dG, prep_word, noun_dop, 10)
                else:
                    if (spisok_POS[i - 1] == 'ADJF'):
                        ADJF_word = spisok_slovo[i - 1]
                        add_node_edge(dG, skaz, ADJF_word, 5)
                        add_node_edge(dG, ADJF_word, noun_dop, 10)
                    add_node_edge(dG, skaz, noun_dop, 10)

        except IndexError:
            if not dG.has_node(word):
                dG.add_node(word)
                dG.node[word]['count'] = 1
            else:
                dG.node[word]['count'] += 1
        except:
            raise

    print(dG.nodes())
    print(dG.edges())
    spl = nx.all_pairs_shortest_path_length(dG)
    for node1 in spl:
        for node2 in node1:
            if node2 != "":
                for val in range(0, len(spisok_slovo), 1):
                    try:
                        try:
                            sl = sl + int(node2[spisok_slovo[val]])
                        except TypeError:
                            pass
                    except KeyError:
                        pass



    if len(spisok_slovo) > 0:
        sr_pred = (sl / len(spisok_slovo))
    elif sr_pred == 0.0 or sr_pred == 0:
        sr_pred = 1.0

    nx.draw(dG, with_labels=True)
    # plt.show()
    return sr_pred



def median_slovo(text, data):
    text_pr = []
    text_sp1 = []
    text_sp2 = []
    text_sp3 = []

    data_pred = []

    text_pr = pred_morphe([text])
    print('text!!!!!!!!!!!!!!!!', text_pr)
    text_sp1, text_sp2, text_sp3 = create_graf(text_pr[0])
    print('sp1', text_sp1, 'sp2', text_sp2, 'sp3', text_sp3)

    text_sr_pred = search_graf(text_sp1, text_sp2, text_sp3)
    print('text_sr_pred', text_sr_pred)

    for data_i in range(0, len(data), 1):
        if (float(data[data_i][2]) == float(text_sr_pred)):
            print('Похожие по структуре предложения фразы', data[data_i][1])
            data_pred.append(data[data_i])

    return data_pred


def screach_claster(text, data_file):
    data_file_train = median_slovo(text, data_file)
    train_data = [x[1] for x in data_file_train if len(x[0]) > 0]
    vector, V, num_samples = vectot_text(train_data)
    km = claster_text(V, num_samples)
    new_post = text
    text_otvet = ""
    new_post_vec = vector.transform([new_post])
    new_post_label = km.predict(new_post_vec)[0]

    similar_indices = (km.labels_ == new_post_label).nonzero()[0]
    similar = []
    print(len(similar_indices))
    print(len(data_file_train))

    for i in similar_indices:
        dist = sp.linalg.norm((new_post_vec - V[i]).toarray())
        similar.append((dist, i, data_file_train[i][1], data_file_train[i][2], data_file_train[i][3], data_file_train[i][5], data_file_train[i][6]))
    similar = sorted(similar)
    print(len(similar))


    if similar!='':
        text_otvet = similar[0][4]
    print('dist otvet', similar[0][4])
    for i_sim in range(0, len(similar), 1):
        if (similar[i_sim][0] >= 0.0 and similar[i_sim][0]<= 1.0):
            if (similar[i_sim][3] == similar[i_sim][5]):
                if (similar[i_sim][6] == 'good' or similar[i_sim][6] == 'neutral'):
                    print('dist = ', similar[i_sim][0], 'graf_rhrase', similar[i_sim][3], 'graf_otvet', similar[i_sim][5], 'ocenka = ', similar[i_sim][6], 'otvet = ', similar[i_sim][4])
                    text_otvet = similar[i_sim][4]
                    print('Ответ = ', text_otvet)
                    break

        elif text_otvet == '' and similar[i_sim][6] != 'bad':
            text_otvet = similar[0][4]


        elif text_otvet == '':
            text_otvet = similar[0][4]

    return text_otvet